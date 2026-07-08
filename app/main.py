#  app.py

"""
argus-sec — RAG API.

Two endpoints:
  POST /query  — full pipeline: retrieve → rerank → synthesise → return answer + citations + chunks
  POST /search — retrieval only: retrieve → rerank → return ranked chunks, no synthesis

Synthesis model: gpt-5.4-nano (primary), gemma3:1b via Ollama (fallback)
Retrieval:       hybrid BM25 + dense (Qdrant RRF), cross-encoder rerank
Grounding:       strict — LLM answers only from retrieved CVE context (Option A)

Design decisions:
  - Models loaded at startup (lifespan) not per-request — avoids 10s first-call
    penalty on cross-encoder and FastEmbed models.
  - Synthesis prompt enforces strict grounding: no external knowledge.
  - Confidence gate: if top rerank score < 0.4 (low confidence), synthesis
    is skipped and a "insufficient corpus" message is returned — avoids
    hallucination on out-of-corpus queries.
  - Citations extracted from retrieved chunk CVE IDs, not LLM output —
    prevents the LLM from fabricating CVE references.
  - gpt-5.4-nano fallback to gemma3:1b: if OpenAI call fails, local Ollama
    model is used with an explicit disclaimer added to the response.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv(override=True)

from fastapi import FastAPI, HTTPException, Query
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from services.retrieval import (
    RetrievalResult,
    RetrievedChunk,
    _get_cross_encoder,
    _get_dense_model,
    _get_qdrant_client,
    _get_sparse_model,
    retrieve,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL      = os.environ.get("OPENAI_MODEL", "gpt-5.4-nano")
OLLAMA_URL        = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL      = os.environ.get("OLLAMA_MODEL", "gemma3:1b")
MAX_TOKENS        = int(os.environ.get("SYNTHESIS_MAX_TOKENS", "800"))

LOW_CONFIDENCE_MSG = (
    "Insufficient information in the CVE corpus to answer this query confidently. "
    "The retrieved records do not closely match the question. "
    "Try rephrasing, broadening the query, or removing filters."
)

SYSTEM_PROMPT = """You are a cybersecurity analyst assistant specialising in vulnerability analysis.

You will be given a set of CVE (Common Vulnerabilities and Exposures) records retrieved from a corpus of real NVD data.

Rules you must follow without exception:
1. Answer ONLY from the provided CVE records. Do not add any information not present in the context.
2. If the CVE records do not contain enough information to answer the question, say so explicitly.
3. When citing specific CVSS scores, affected versions, or CVE IDs, quote them exactly as they appear in the context — do not paraphrase or estimate.
4. Be concise: maximum 3-4 sentences per CVE. Lead with the direct answer, then one line of key detail.
5. End your answer with a "Sources:" line listing the CVE IDs you used.

Do not invent CVE IDs, CVSS scores, version numbers, or vendor names."""


# ---------------------------------------------------------------------------
# Lifespan — pre-load all models at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Pre-load all heavyweight models at startup so the first request
    does not incur a 10-second model-loading penalty.
    Models are cached as module-level singletons in retrieval.py.
    """
    logger.info("Pre-loading retrieval models...")
    _get_dense_model()
    _get_sparse_model()
    _get_cross_encoder()
    _get_qdrant_client()
    logger.info("All models loaded — API ready")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="argus-sec RAG API",
    description="Hybrid CVE retrieval with LLM synthesis. Corpus: 912 CVEs across 31 packages.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query:         str   = Field(..., min_length=3, max_length=500,
                                 description="Natural language security question")
    k:             int   = Field(default=5, ge=1, le=20,
                                 description="Number of chunks to retrieve and synthesise over")
    severity:      str | None = Field(default=None,
                                      description="Filter by severity: CRITICAL|HIGH|MEDIUM|LOW|UNKNOWN")
    package:       str | None = Field(default=None,
                                      description="Filter by package name e.g. 'LangChain'")
    cisa_kev:      bool | None = Field(default=None,
                                       description="Filter to CISA KEV-listed CVEs only")
    fix_available: bool | None = Field(default=None,
                                       description="Filter to CVEs with a known fix")
    attack_vector: str | None  = Field(default=None,
                                       description="Filter by attack vector: NETWORK|ADJACENT|LOCAL|PHYSICAL")
    multi_package: bool | None = Field(default=None,
                                       description="Filter to CVEs affecting multiple packages")


class ChunkOut(BaseModel):
    cve_id:        str
    rerank_score:  float
    dense_score:   float
    severity:      str | None
    packages:      list[str]
    cisa_kev:      bool
    fix_available: bool
    attack_vector: str | None
    published:     str | None
    nvd_url:       str
    narrative_text: str


class QueryResponse(BaseModel):
    answer:       str
    citations:    list[str]          # CVE IDs used in synthesis
    confidence:   str                # "high" | "medium" | "low"
    top_score:    float
    model_used:   str                # which synthesis model was used
    fallback:     bool               # True if Ollama fallback was used
    chunks:       list[ChunkOut]
    retrieval_ms: int
    rerank_ms:    int
    synthesis_ms: int
    total_ms:     int


class SearchResponse(BaseModel):
    chunks:       list[ChunkOut]
    confidence:   str
    top_score:    float
    retrieval_ms: int
    rerank_ms:    int
    total_ms:     int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_filters(
    severity:      str | None,
    package:       str | None,
    cisa_kev:      bool | None,
    fix_available: bool | None,
    attack_vector: str | None,
    multi_package: bool | None,
) -> dict | None:
    filters = {}
    if severity:      filters["severity"]      = severity.upper()
    if package:       filters["package"]       = package
    if cisa_kev      is not None: filters["cisa_kev"]      = cisa_kev
    if fix_available is not None: filters["fix_available"] = fix_available
    if attack_vector: filters["attack_vector"] = attack_vector.upper()
    if multi_package is not None: filters["multi_package"] = multi_package
    return filters or None


def _chunk_to_out(chunk: RetrievedChunk) -> ChunkOut:
    p = chunk.payload
    return ChunkOut(
        cve_id         = chunk.cve_id,
        rerank_score   = round(chunk.rerank_score, 4),
        dense_score    = round(chunk.dense_score, 4),
        severity       = p.get("severity"),
        packages       = p.get("packages", []),
        cisa_kev       = p.get("cisa_kev", False),
        fix_available  = p.get("fix_available", False),
        attack_vector  = p.get("attack_vector"),
        published      = p.get("published"),
        nvd_url        = p.get("nvd_url", ""),
        narrative_text = chunk.narrative_text,
    )


def _build_context(chunks: list[RetrievedChunk]) -> str:
    """Build the context block passed to the synthesis LLM."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"--- CVE Record {i} ---\n{chunk.narrative_text}\n")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Synthesis — gpt-5.4-nano primary, gemma3:1b fallback
# ---------------------------------------------------------------------------

async def _synthesise_openai(query: str, context: str) -> tuple[str, str]:
    """
    Call gpt-5.4-nano for synthesis.
    Returns (answer_text, model_name).
    Raises on failure so the caller can fall back to Ollama.
    """
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    response = await client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Question: {query}\n\nCVE Context:\n{context}"},
        ],
        max_completion_tokens=MAX_TOKENS,
        temperature=0.1,   # low temperature for factual synthesis
    )
    return (response.choices[0].message.content or "").strip(), OPENAI_MODEL


async def _synthesise_ollama(query: str, context: str) -> tuple[str, str]:
    """
    Call gemma3:1b via Ollama as fallback synthesis model.
    Adds a disclaimer to the response noting local model limitations.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Question: {query}\n\nCVE Context:\n{context}"},
        ],
        "stream": False,
        "options": {"num_predict": MAX_TOKENS, "temperature": 0.1},
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
        resp.raise_for_status()
        answer = resp.json()["message"]["content"].strip()

    disclaimer = (
        "\n\n⚠ Note: This answer was generated by a local fallback model (gemma3:1b). "
        "Verify CVSS scores and version ranges against the source CVE records linked above."
    )
    return answer + disclaimer, OLLAMA_MODEL


async def synthesise(query: str, chunks: list[RetrievedChunk]) -> tuple[str, str, bool]:
    """
    Attempt OpenAI synthesis, fall back to Ollama on failure.
    Returns (answer, model_used, fallback_used).
    """
    context = _build_context(chunks)
    try:
        answer, model = await _synthesise_openai(query, context)
        return answer, model, False
    except Exception as e:
        logger.warning("OpenAI synthesis failed (%s) — falling back to Ollama", str(e))
        try:
            answer, model = await _synthesise_ollama(query, context)
            return answer, model, True
        except Exception as e2:
            raise HTTPException(
                status_code=503,
                detail=f"Both synthesis providers failed. OpenAI: {e}. Ollama: {e2}",
            )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/query", response_model=QueryResponse, summary="RAG query — retrieve + synthesise")
async def query_endpoint(req: QueryRequest) -> QueryResponse:
    """
    Full RAG pipeline:
      1. Hybrid retrieval (BM25 + dense, RRF fusion, optional pre-filters)
      2. Cross-encoder rerank
      3. Confidence gate — returns early if confidence is low
      4. LLM synthesis (gpt-5.4-nano → gemma3:1b fallback)
      5. Citations extracted from retrieved CVE IDs (not LLM output)
    """
    t_total = time.monotonic()

    filters = _build_filters(
        req.severity, req.package, req.cisa_kev,
        req.fix_available, req.attack_vector, req.multi_package,
    )

    result: RetrievalResult = retrieve(
        query=req.query,
        k_candidates=req.k * 2,   # fetch 2× before reranking
        k_final=req.k,
        filters=filters,
    )

    chunks_out = [_chunk_to_out(c) for c in result.chunks]

    # Confidence gate — skip synthesis if corpus match is poor
    if result.confidence == "low":
        return QueryResponse(
            answer       = LOW_CONFIDENCE_MSG,
            citations    = [],
            confidence   = "low",
            top_score    = round(result.top_score, 4),
            model_used   = "none",
            fallback     = False,
            chunks       = chunks_out,
            retrieval_ms = result.retrieval_ms,
            rerank_ms    = result.rerank_ms,
            synthesis_ms = 0,
            total_ms     = int((time.monotonic() - t_total) * 1000),
        )

    t_synthesis = time.monotonic()
    answer, model_used, fallback = await synthesise(req.query, result.chunks)
    synthesis_ms = int((time.monotonic() - t_synthesis) * 1000)

    citations = [c.cve_id for c in result.chunks]

    logger.info(
        "Query: %r | confidence=%s | model=%s | fallback=%s | synthesis=%dms",
        req.query[:60], result.confidence, model_used, fallback, synthesis_ms,
    )

    return QueryResponse(
        answer       = answer,
        citations    = citations,
        confidence   = result.confidence,
        top_score    = round(result.top_score, 4),
        model_used   = model_used,
        fallback     = fallback,
        chunks       = chunks_out,
        retrieval_ms = result.retrieval_ms,
        rerank_ms    = result.rerank_ms,
        synthesis_ms = synthesis_ms,
        total_ms     = int((time.monotonic() - t_total) * 1000),
    )


@app.post("/search", response_model=SearchResponse, summary="Retrieval only — no synthesis")
async def search_endpoint(req: QueryRequest) -> SearchResponse:
    """
    Retrieval-only pipeline — no LLM synthesis.
    Returns ranked chunks with scores for programmatic access
    (e.g. kronos-agent pulling raw CVE context without synthesis cost).
    """
    t_total = time.monotonic()

    filters = _build_filters(
        req.severity, req.package, req.cisa_kev,
        req.fix_available, req.attack_vector, req.multi_package,
    )

    result: RetrievalResult = retrieve(
        query=req.query,
        k_candidates=req.k * 2,
        k_final=req.k,
        filters=filters,
    )

    return SearchResponse(
        chunks       = [_chunk_to_out(c) for c in result.chunks],
        confidence   = result.confidence,
        top_score    = round(result.top_score, 4),
        retrieval_ms = result.retrieval_ms,
        rerank_ms    = result.rerank_ms,
        total_ms     = int((time.monotonic() - t_total) * 1000),
    )


@app.get("/health", summary="Health check")
async def health() -> dict:
    return {
        "status":     "ok",
        "collection": os.environ.get("QDRANT_COLLECTION", "cve_chunks"),
        "version":    app.version,
    }