"""
argus-sec — Retrieval module.

Implements hybrid retrieval pipeline:
  1. Embed query (dense + sparse) via FastEmbed
  2. Qdrant hybrid search with optional pre-filtering
     (BM25 sparse + dense cosine, fused via RRF)
  3. Cross-encoder rerank (ms-marco-MiniLM-L-6-v2)
  4. Confidence scoring from top-1 rerank score

Design decisions:
  - k_candidates=10 before reranking, k_final=5 after — standard RAG
    practice; fetching 2× final k gives reranker enough signal without
    bloating the synthesis prompt.
  - RRF fusion in Qdrant (not client-side) — single network round-trip,
    Qdrant's native implementation is correct and efficient.
    Ref: https://qdrant.tech/documentation/concepts/hybrid-queries/
  - Cross-encoder: ms-marco-MiniLM-L-6-v2 — trained on MS MARCO passage
    ranking, strong on factual Q&A over short technical passages.
    Ref: https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2
  - Confidence thresholds (high ≥ 0.7, medium 0.4–0.69, low < 0.4) based
    on rerank score, not dense similarity — cross-encoder score is a direct
    relevance judgement, more reliable as a confidence signal than cosine.
  - Pre-filtering (Option A): Qdrant payload filters applied before vector
    search, not post-retrieval. More efficient; avoids fetching irrelevant
    results that would otherwise consume reranker capacity.
    Ref: https://qdrant.tech/documentation/concepts/filtering/

Supported filter fields (all optional, passed as a dict):
  severity    : "CRITICAL" | "HIGH" | "MEDIUM" | "LOW" | "UNKNOWN"
  package     : e.g. "LangChain" — matches if present in packages array
  cisa_kev    : True | False
  fix_available: True | False
  attack_vector: "NETWORK" | "ADJACENT" | "LOCAL" | "PHYSICAL"
  multi_package: True | False
"""

from __future__ import annotations

import logging
import os
import math
import time
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    models,
    Prefetch,
    SparseVector,
)
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

QDRANT_URL        = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY    = os.environ.get("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "cve_chunks")
FASTEMBED_MODEL   = os.environ.get("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5")
FASTEMBED_CACHE   = os.environ.get("FASTEMBED_CACHE_DIR", "./data/.fastembed_cache")

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Confidence thresholds (cross-encoder score, not cosine similarity)
CONFIDENCE_HIGH   = 0.7
CONFIDENCE_MEDIUM = 0.4


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    cve_id:          str
    narrative_text:  str
    dense_score:     float          # cosine similarity from Qdrant
    rerank_score:    float          # cross-encoder score (post-reranking)
    payload:         dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    chunks:          list[RetrievedChunk]
    confidence:      str            # "high" | "medium" | "low"
    top_score:       float          # rerank score of top-1 chunk
    retrieval_ms:    int
    rerank_ms:       int


# ---------------------------------------------------------------------------
# Lazy-loaded singletons — models loaded once, reused across requests
# ---------------------------------------------------------------------------

_dense_model:   TextEmbedding | None        = None
_sparse_model:  SparseTextEmbedding | None  = None
_cross_encoder: CrossEncoder | None         = None
_qdrant_client: QdrantClient | None         = None


def _get_dense_model() -> TextEmbedding:
    global _dense_model
    if _dense_model is None:
        logger.info("Loading dense embedding model: %s", FASTEMBED_MODEL)
        _dense_model = TextEmbedding(
            model_name=FASTEMBED_MODEL,
            cache_dir=FASTEMBED_CACHE,
        )
    return _dense_model


def _get_sparse_model() -> SparseTextEmbedding:
    global _sparse_model
    if _sparse_model is None:
        logger.info("Loading sparse (BM25) embedding model")
        _sparse_model = SparseTextEmbedding(
            model_name="Qdrant/bm25",
            cache_dir=FASTEMBED_CACHE,
        )
    return _sparse_model


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        logger.info("Loading cross-encoder: %s", CROSS_ENCODER_MODEL)
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder


def _get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY or None,
        )
    return _qdrant_client


# ---------------------------------------------------------------------------
# Filter builder
# ---------------------------------------------------------------------------

def _build_filter(filters: dict | None) -> Filter | None:
    """
    Convert a filters dict into a Qdrant Filter object.

    Supported keys:
      severity     → MatchValue on severity field
      package      → MatchValue on packages array (Qdrant matches if value
                     is present in the array)
      cisa_kev     → MatchValue (bool)
      fix_available→ MatchValue (bool)
      attack_vector→ MatchValue
      multi_package→ MatchValue (bool)

    Multiple filters are combined with AND (must = all conditions).
    Returns None if no valid filters provided — Qdrant skips filtering.
    """
    if not filters:
        return None

    conditions = []

    if severity := filters.get("severity"):
        conditions.append(
            FieldCondition(key="severity", match=MatchValue(value=severity.upper()))
        )

    if package := filters.get("package"):
        # Qdrant array field: MatchValue matches if the value is in the array
        conditions.append(
            FieldCondition(key="packages", match=MatchValue(value=package))
        )

    if "cisa_kev" in filters:
        conditions.append(
            FieldCondition(key="cisa_kev", match=MatchValue(value=bool(filters["cisa_kev"])))
        )

    if "fix_available" in filters:
        conditions.append(
            FieldCondition(
                key="fix_available",
                match=MatchValue(value=bool(filters["fix_available"]))
            )
        )

    if attack_vector := filters.get("attack_vector"):
        conditions.append(
            FieldCondition(
                key="attack_vector",
                match=MatchValue(value=attack_vector.upper())
            )
        )

    if "multi_package" in filters:
        conditions.append(
            FieldCondition(
                key="multi_package",
                match=MatchValue(value=bool(filters["multi_package"]))
            )
        )

    if not conditions:
        return None

    return Filter(must=conditions)


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def _score_confidence(top_rerank_score: float) -> str:
    """
    Map cross-encoder rerank score to confidence tier.

    ms-marco-MiniLM-L-6-v2 scores are unbounded but practically range
    from ~-10 (irrelevant) to ~10 (highly relevant). Thresholds are set
    conservatively given the technical specificity of CVE queries.

    Rationale for thresholds:
      ≥ 0.7  → model is confident the chunk directly answers the query
      0.4–0.69 → partial match; answer may be incomplete
      < 0.4  → poor match; synthesis would likely hallucinate
    """
    if top_rerank_score >= CONFIDENCE_HIGH:
        return "high"
    elif top_rerank_score >= CONFIDENCE_MEDIUM:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Core retrieval function
# ---------------------------------------------------------------------------

def retrieve(
    query:        str,
    k_candidates: int = 10,
    k_final:      int = 5,
    filters:      dict | None = None,
) -> RetrievalResult:
    """
    Full hybrid retrieval pipeline for a natural language query.

    Args:
        query:        Natural language query string.
        k_candidates: Number of candidates to fetch from Qdrant before
                      reranking. Default 10 (2× k_final).
        k_final:      Number of chunks to return after reranking. Default 5.
        filters:      Optional dict of payload filters (see _build_filter).

    Returns:
        RetrievalResult with ranked chunks, confidence, and timing.

    Pipeline:
        1. Embed query → dense vector + sparse BM25 vector
        2. Qdrant hybrid search with RRF fusion + optional pre-filter
        3. Cross-encoder rerank top-k_candidates → top-k_final
        4. Confidence scoring from top-1 rerank score
    """
    t_retrieval_start = time.monotonic()

    # Step 1 — embed query
    dense_model  = _get_dense_model()
    sparse_model = _get_sparse_model()

    dense_vec  = list(dense_model.embed([query]))[0].tolist()
    sparse_emb = list(sparse_model.embed([query]))[0]
    sparse_vec = SparseVector(
        indices=sparse_emb.indices.tolist(),
        values=sparse_emb.values.tolist(),
    )

    # Step 2 — Qdrant hybrid search with RRF fusion
    # Prefetch dense and sparse separately, fuse with RRF.
    # Ref: https://qdrant.tech/documentation/concepts/hybrid-queries/
    qdrant = _get_qdrant_client()
    qdrant_filter = _build_filter(filters)

    results = qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        prefetch=[
            Prefetch(
                query=dense_vec,
                using="dense",
                limit=k_candidates,
                filter=qdrant_filter,
            ),
            Prefetch(
                query=sparse_vec,
                using="sparse",
                limit=k_candidates,
                filter=qdrant_filter,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=k_candidates,
        with_payload=True,
    ).points

    retrieval_ms = int((time.monotonic() - t_retrieval_start) * 1000)

    if not results:
        return RetrievalResult(
            chunks=[],
            confidence="low",
            top_score=0.0,
            retrieval_ms=retrieval_ms,
            rerank_ms=0,
        )

    # Step 3 — cross-encoder rerank
    t_rerank_start = time.monotonic()
    cross_encoder  = _get_cross_encoder()

    pairs  = [(query, r.payload.get("narrative_text", "")) for r in results]
    scores = cross_encoder.predict(pairs)

    ranked = sorted(
        zip(results, scores),
        key=lambda x: x[1],
        reverse=True,
    )[:k_final]

    rerank_ms = int((time.monotonic() - t_rerank_start) * 1000)

    # Step 4 — build output + confidence
    # Normalise rerank score to [0, 1] for confidence threshold comparison.
    # ms-marco scores practical range: [-10, 10] → sigmoid normalisation.
    def sigmoid(x: float) -> float:
        return 1 / (1 + math.exp(-x))

    top_raw_score    = float(ranked[0][1]) if ranked else 0.0
    top_norm_score   = sigmoid(top_raw_score)
    confidence       = _score_confidence(top_norm_score)

    chunks = [
        RetrievedChunk(
            cve_id         = r.payload.get("cve_id", ""),
            narrative_text = r.payload.get("narrative_text", ""),
            dense_score    = float(r.score),
            rerank_score   = sigmoid(float(score)),
            payload        = r.payload,
        )
        for r, score in ranked
    ]

    logger.info(
        "Retrieved %d chunks (retrieval=%dms, rerank=%dms, confidence=%s, top_score=%.3f)",
        len(chunks), retrieval_ms, rerank_ms, confidence, top_norm_score,
    )

    return RetrievalResult(
        chunks       = chunks,
        confidence   = confidence,
        top_score    = top_norm_score,
        retrieval_ms = retrieval_ms,
        rerank_ms    = rerank_ms,
    )
