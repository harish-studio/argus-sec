"""
argus-sec — Ingestion script.

Reads data/chunks.json and ingests into:
  1. PostgreSQL 16 — structured metadata for exact-match queries
  2. Qdrant v1.17.1 — dense + sparse vectors for hybrid retrieval

Hybrid retrieval design:
  Dense:  BAAI/bge-small-en-v1.5 (384-dim, cosine) via FastEmbed
          → semantic queries ("remote code execution in web frameworks")
  Sparse: Qdrant/bm25 via FastEmbed SparseTextEmbedding
          → exact-term queries ("CVE-2024-3400", "CWE-89")
  Fusion: Qdrant native RRF at query time (no separate rank_bm25 needed)
          Ref: https://qdrant.tech/documentation/concepts/hybrid-queries/

Why BM25 in Qdrant rather than rank_bm25 pickle:
  - BM25 index lives in Qdrant alongside dense vectors → single data store
  - RRF fusion is a native Qdrant query operation (prefetch + Fusion.RRF)
  - No pickle file to manage, version, or keep in sync with the corpus
  - Built-in BM25 support confirmed in qdrant-client v1.9.x+
    Ref: https://github.com/qdrant/qdrant-client/releases

PostgreSQL schema:
  See CREATE TABLE statement below — matches the final schema agreed in
  the design session (severity, fix_available, attack_vector,
  multi_package, cisa_kev, ssvc_exploitation, score_disagreement, etc.)

Idempotency:
  - PostgreSQL: CREATE TABLE IF NOT EXISTS + INSERT ... ON CONFLICT DO NOTHING
  - Qdrant: upsert (creates or replaces by point ID)
  Both stores can be re-ingested safely without duplication.

Environment variables (all required — set in docker-compose.yml from .env):
  POSTGRES_URL, QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION,
  CHUNKS_PATH, FASTEMBED_MODEL, EMBEDDING_DIM, FASTEMBED_CACHE_DIR,
  QDRANT_BATCH_SIZE, PG_BATCH_SIZE
"""

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

import psycopg2
import psycopg2.extras
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseIndexParams,
    SparseVectorParams,
    SparseVector,
    VectorParams,
)
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

def require_env(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        raise RuntimeError(f"Required environment variable {key!r} is not set")
    return val


POSTGRES_URL      = require_env("POSTGRES_URL")
QDRANT_URL        = require_env("QDRANT_URL")
QDRANT_API_KEY    = require_env("QDRANT_API_KEY")
QDRANT_COLLECTION = require_env("QDRANT_COLLECTION")
CHUNKS_PATH       = Path(require_env("CHUNKS_PATH"))
FASTEMBED_MODEL   = os.environ.get("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM     = int(os.environ.get("EMBEDDING_DIM", "384"))
FASTEMBED_CACHE   = os.environ.get("FASTEMBED_CACHE_DIR", "/app/.fastembed_cache")
QDRANT_BATCH      = int(os.environ.get("QDRANT_BATCH_SIZE", "64"))
PG_BATCH          = int(os.environ.get("PG_BATCH_SIZE", "100"))


# ---------------------------------------------------------------------------
# PostgreSQL — schema
# ---------------------------------------------------------------------------

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS cve_chunks (
    -- Identity
    cve_id              TEXT PRIMARY KEY,

    -- Core retrieval columns — indexed for direct filtering
    severity            TEXT        NOT NULL DEFAULT 'UNKNOWN',
    published           DATE,
    last_modified       DATE,
    status              TEXT,
    cisa_kev            BOOLEAN     NOT NULL DEFAULT FALSE,
    score_disagreement  BOOLEAN     NOT NULL DEFAULT FALSE,
    ssvc_exploitation   TEXT,           -- NULL | 'poc' | 'active'
    fix_available       BOOLEAN     NOT NULL DEFAULT FALSE,
    attack_vector       TEXT,           -- 'NETWORK'|'ADJACENT'|'LOCAL'|'PHYSICAL'
    multi_package       BOOLEAN     NOT NULL DEFAULT FALSE,

    -- Array columns — GIN indexed for containment queries
    packages            TEXT[]      NOT NULL DEFAULT '{}',
    cwes                TEXT[]      NOT NULL DEFAULT '{}',

    -- Embedding input — stored here AND in Qdrant payload
    narrative_text      TEXT        NOT NULL,

    -- JSONB remainder — rich detail, not directly filtered
    cvss_scores         JSONB       NOT NULL DEFAULT '[]',
    affected_versions   JSONB       NOT NULL DEFAULT '[]',
    ref_links           JSONB       NOT NULL DEFAULT '[]',

    -- Provenance
    nvd_url             TEXT        NOT NULL,
    ingested_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_cve_severity   ON cve_chunks(severity);",
    "CREATE INDEX IF NOT EXISTS idx_cve_published  ON cve_chunks(published);",
    "CREATE INDEX IF NOT EXISTS idx_cve_urgency    ON cve_chunks(cisa_kev, ssvc_exploitation);",
    "CREATE INDEX IF NOT EXISTS idx_cve_attack     ON cve_chunks(attack_vector);",
    "CREATE INDEX IF NOT EXISTS idx_cve_fix        ON cve_chunks(fix_available);",
    "CREATE INDEX IF NOT EXISTS idx_cve_packages   ON cve_chunks USING GIN(packages);",
    "CREATE INDEX IF NOT EXISTS idx_cve_cwes       ON cve_chunks USING GIN(cwes);",
]

INSERT_SQL = """
INSERT INTO cve_chunks (
    cve_id, severity, published, last_modified, status,
    cisa_kev, score_disagreement, ssvc_exploitation,
    fix_available, attack_vector, multi_package,
    packages, cwes, narrative_text,
    cvss_scores, affected_versions, ref_links,
    nvd_url
) VALUES %s
ON CONFLICT (cve_id) DO NOTHING;
"""


def setup_postgres(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(CREATE_TABLE_SQL)
        for idx_sql in CREATE_INDEXES_SQL:
            cur.execute(idx_sql)
    conn.commit()
    logger.info("PostgreSQL schema ready")


def chunk_to_pg_row(chunk: dict) -> tuple:
    m = chunk["metadata"]

    _signal = m.get("exploitation_signal") or ""
    

    return (
        chunk["cve_id"],
        m.get("severity", "UNKNOWN"),
        m.get("published") or None,
        m.get("last_modified") or None,
        m.get("status"),
        m.get("cisa_kev", False),
        m.get("score_disagreement", False),
        "poc" if "proof-of-concept" in _signal.lower() else "active" if "active exploitation" in _signal.lower() else None,
        m.get("fix_available", False),
        m.get("attack_vector"),
        m.get("multi_package", False),
        m.get("packages", []),
        m.get("cwes", []),
        chunk["narrative_text"],
        json.dumps(m.get("cvss_scores", [])),
        json.dumps(m.get("affected_versions", [])),
        json.dumps(m.get("references", [])),
        m.get("nvd_url", ""),
    )


def ingest_postgres(chunks: list[dict], conn) -> int:
    """
    Batch-insert chunks into PostgreSQL.
    Uses execute_values for efficient bulk insert.
    ON CONFLICT DO NOTHING ensures idempotency — re-running
    ingestion won't duplicate records.

    Note on ssvc_exploitation: the metadata field stores the full
    human-readable label (e.g. "Proof-of-concept code exists (SSVC)").
    PostgreSQL stores this as-is in the TEXT column for display;
    the retrieval layer filters on the Qdrant payload's shorter
    exploitation_signal value instead.
    """
    rows = [chunk_to_pg_row(c) for c in chunks]
    inserted = 0

    with conn.cursor() as cur:
        for i in tqdm(range(0, len(rows), PG_BATCH), desc="PostgreSQL insert", unit="batch"):
            batch = rows[i : i + PG_BATCH]
            psycopg2.extras.execute_values(cur, INSERT_SQL, batch)
            inserted += cur.rowcount
        conn.commit()

    return inserted


# ---------------------------------------------------------------------------
# Qdrant — collection + upsert
# ---------------------------------------------------------------------------

def stable_point_id(cve_id: str) -> int:
    """
    Derive a stable integer point ID from the CVE ID string.
    Qdrant supports both UUID and integer point IDs; integer is simpler
    for a corpus where the natural key (CVE ID) is a known string.
    Uses first 8 bytes of SHA-256 — collision probability negligible at 912 points.
    """
    return int(hashlib.sha256(cve_id.encode()).hexdigest()[:16], 16) % (2**63)


def setup_qdrant_collection(client: QdrantClient) -> None:
    """
    Create Qdrant collection with dense + sparse vector configuration.
    Skips creation if the collection already exists (idempotent).

    Dense vector:  BAAI/bge-small-en-v1.5, 384-dim, cosine distance
    Sparse vector: Qdrant/bm25 — stored as sparse vector with IDF modifier
                   Required for native BM25 scoring in Qdrant.
                   Ref: https://qdrant.tech/documentation/concepts/vectors/#sparse-vectors
    """
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION in existing:
        logger.info("Qdrant collection %r already exists — skipping creation", QDRANT_COLLECTION)
        return

    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config={
            "dense": VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(
                    on_disk=False,   # keep in RAM for fast BM25 lookups at 912 points
                ),
            ),
        },
    )
    logger.info(
        "Qdrant collection %r created (dense=%d-dim cosine + sparse BM25)",
        QDRANT_COLLECTION, EMBEDDING_DIM,
    )


def chunk_to_payload(chunk: dict) -> dict:
    """
    Build the Qdrant point payload — fields needed to render a result
    card in the Streamlit UI without a PostgreSQL roundtrip.
    Excludes large JSONB fields (cvss_scores, affected_versions,
    references) — those are fetched from PostgreSQL on detail view.
    """
    m = chunk["metadata"]
    _signal = m.get("exploitation_signal") or ""

    return {
        "cve_id":             chunk["cve_id"],
        "narrative_text":     chunk["narrative_text"],
        "severity":           m.get("severity", "UNKNOWN"),
        "packages":           m.get("packages", []),
        "cisa_kev":           m.get("cisa_kev", False),
        "ssvc_exploitation":  "poc" if "proof-of-concept" in _signal.lower() else "active" if "active exploitation" in _signal.lower() else None,
        "fix_available":      m.get("fix_available", False),
        "attack_vector":      m.get("attack_vector"),
        "multi_package":      m.get("multi_package", False),
        "score_disagreement": m.get("score_disagreement", False),
        "published":          m.get("published"),
        "nvd_url":            m.get("nvd_url", ""),
    }


def ingest_qdrant(
    chunks: list[dict],
    client: QdrantClient,
    dense_model: TextEmbedding,
    sparse_model: SparseTextEmbedding,
) -> int:
    """
    Embed narrative_text with both dense and sparse models,
    then upsert into Qdrant in batches.

    Upsert is idempotent — running ingestion again replaces existing
    points with the same ID rather than duplicating them.
    """
    texts = [c["narrative_text"] for c in chunks]
    upserted = 0

    logger.info("Generating dense embeddings for %d chunks...", len(chunks))
    dense_embeddings = list(
        tqdm(dense_model.embed(texts), total=len(texts), desc="Dense embeddings", unit="chunk")
    )

    logger.info("Generating sparse (BM25) embeddings for %d chunks...", len(chunks))
    sparse_embeddings = list(
        tqdm(sparse_model.embed(texts), total=len(texts), desc="Sparse embeddings", unit="chunk")
    )

    for i in tqdm(range(0, len(chunks), QDRANT_BATCH), desc="Qdrant upsert", unit="batch"):
        batch_chunks = chunks[i : i + QDRANT_BATCH]
        batch_dense  = dense_embeddings[i : i + QDRANT_BATCH]
        batch_sparse = sparse_embeddings[i : i + QDRANT_BATCH]

        points = [
            PointStruct(
                id=stable_point_id(chunk["cve_id"]),
                vector={
                    "dense":  dense_vec.tolist(),
                    "sparse": SparseVector(
                        indices=sparse_vec.indices.tolist(),
                        values=sparse_vec.values.tolist(),
                    ),
                },
                payload=chunk_to_payload(chunk),
            )
            for chunk, dense_vec, sparse_vec in zip(batch_chunks, batch_dense, batch_sparse)
        ]

        client.upsert(collection_name=QDRANT_COLLECTION, points=points)
        upserted += len(points)

    return upserted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"{CHUNKS_PATH} not found — run chunk_cves.py first"
        )

    chunks: list[dict] = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    logger.info("Loaded %d chunks from %s", len(chunks), CHUNKS_PATH)

    # Load embedding models
    # cache_dir= is required — FASTEMBED_CACHE_PATH env var is ignored in 0.4.x
    # Ref: https://github.com/qdrant/fastembed/issues/217
    logger.info("Loading FastEmbed dense model: %s", FASTEMBED_MODEL)
    dense_model = TextEmbedding(
        model_name=FASTEMBED_MODEL,
        cache_dir=FASTEMBED_CACHE,
    )

    logger.info("Loading FastEmbed sparse model: Qdrant/bm25")
    sparse_model = SparseTextEmbedding(
        model_name="Qdrant/bm25",
        cache_dir=FASTEMBED_CACHE,
    )

    # PostgreSQL ingestion
    logger.info("Connecting to PostgreSQL: %s", POSTGRES_URL.split("@")[-1])
    pg_conn = psycopg2.connect(POSTGRES_URL)
    try:
        setup_postgres(pg_conn)
        t0 = time.time()
        pg_inserted = ingest_postgres(chunks, pg_conn)
        pg_elapsed = time.time() - t0
        logger.info(
            "PostgreSQL: %d rows inserted (%.1fs, %d skipped on conflict)",
            pg_inserted, pg_elapsed, len(chunks) - pg_inserted,
        )
    finally:
        pg_conn.close()

    # Qdrant ingestion
    logger.info("Connecting to Qdrant: %s", QDRANT_URL)
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    setup_qdrant_collection(qdrant_client)

    t0 = time.time()
    qdrant_upserted = ingest_qdrant(chunks, qdrant_client, dense_model, sparse_model)
    qdrant_elapsed = time.time() - t0

    logger.info(
        "Qdrant: %d points upserted (%.1fs)",
        qdrant_upserted, qdrant_elapsed,
    )

    logger.info("─" * 60)
    logger.info("Ingestion complete.")
    logger.info("  PostgreSQL rows: %d", pg_inserted)
    logger.info("  Qdrant points:   %d (dense + sparse)", qdrant_upserted)
    logger.info("  Collection:      %s", QDRANT_COLLECTION)


if __name__ == "__main__":
    main()
