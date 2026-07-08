# argus-sec

**Hybrid RAG system for CVE security intelligence.**

Answers natural language questions about vulnerabilities affecting a production GenAI stack — 912 real CVEs across 31 packages, retrieved with BM25 + dense hybrid search, reranked with a cross-encoder, and synthesised with strict grounding via gpt-5.4-nano.

Part of a 5-project GenAI Solution Architect portfolio demonstrating AI technical depth, enterprise delivery discipline, and EU governance fluency.

---

## What it does

```
Query: "Which critical LangChain vulnerabilities can be exploited without authentication?"

→ Hybrid retrieval (BM25 sparse + BAAI/bge-small-en-v1.5 dense, RRF fusion)
→ Cross-encoder rerank (ms-marco-MiniLM-L-6-v2)
→ Confidence gate (skips synthesis if top rerank score < 0.4)
→ Synthesis (gpt-5.4-nano, strict grounding — no external knowledge)
→ Citations from retrieved chunks, not LLM output

Answer: CVE-2024-7042 (CRITICAL 9.8, SQL injection, fix in 0.3.1) ...
        CVE-2024-8309 (CRITICAL 9.8, SQL injection via prompt injection) ...
Sources: CVE-2024-7042, CVE-2024-8309, CVE-2025-46059, CVE-2024-2057, CVE-2024-7774
```

---

## Corpus

| Attribute | Value |
|---|---|
| Source | NVD API 2.0 (National Vulnerability Database) |
| CVEs | 912 unique, deduplicated |
| Packages | 31 (FastAPI, LangChain, LangGraph, Redis, PostgreSQL, Docker, MLflow, Grafana, Ollama, LiteLLM, and more) |
| Time window | Last 24 months, `lastModStartDate` sync |
| Severity | CRITICAL: 127 · HIGH: 334 · MEDIUM: 297 · LOW: 39 · UNKNOWN: 115 |
| Multi-package CVEs | 22 (same CVE affecting multiple stack components) |
| CISA KEV flagged | 9 (confirmed active exploitation in the wild) |
| Score disagreements | 320 (NVD vs vendor CVSS scoring divergence) |
| SSVC exploitation signal | 281 (PoC or active exploitation confirmed) |

**Corpus integrity:** CPE-based post-filtering applied to Docker, PostgreSQL, and Redis (high-volume packages where keyword search over-counts via incidental description mentions). All other packages use keyword search.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        POST /query                              │
│                   natural language question                     │
└───────────────────────────┬─────────────────────────────────────┘
                            │
              ┌─────────────▼─────────────┐
              │     FastEmbed embed       │
              │  BAAI/bge-small-en-v1.5   │
              │  dense (384-dim) + BM25   │
              └─────────────┬─────────────┘
                            │
              ┌─────────────▼─────────────┐
              │    Qdrant hybrid search   │
              │  prefetch dense + sparse  │
              │     RRF fusion, k=10      │
              │   optional pre-filters    │
              └─────────────┬─────────────┘
                            │
              ┌─────────────▼─────────────┐
              │   Cross-encoder rerank    │
              │ ms-marco-MiniLM-L-6-v2    │
              │     top 10 → top 5        │
              └─────────────┬─────────────┘
                            │
              ┌─────────────▼─────────────┐
              │    Confidence gate        │
              │  sigmoid(rerank) < 0.4    │
              │  → return "insufficient"  │
              └─────────────┬─────────────┘
                            │
              ┌─────────────▼─────────────┐
              │   gpt-5.4-nano synthesis  │
              │  strict grounding prompt  │
              │  gemma3:1b local fallback │
              └─────────────┬─────────────┘
                            │
              ┌─────────────▼─────────────┐
              │  Citations from retrieval │
              │  (not from LLM output)    │
              └───────────────────────────┘
```

### Why hybrid retrieval?

CVE queries have two distinct patterns that require different retrieval mechanisms:

- **Exact-term queries** — `"CVE-2024-3400"`, `"CWE-89"`, `"LangChain 0.3.1"` — where BM25 keyword matching is essential. Dense embeddings handle semantic meaning but often miss exact identifiers.
- **Semantic queries** — `"remote code execution in Python web frameworks"` — where dense embeddings capture intent that keyword search misses.

RRF fusion combines both scores without requiring manual weight tuning. The cross-encoder then re-scores (query, chunk) pairs directly — a more accurate relevance signal than vector similarity alone.

### Why strict grounding?

For a security tool, a hallucinated CVSS score (e.g. 7.5 instead of 9.8) or wrong affected version range causes direct operational harm. The synthesis prompt enforces: *"Answer ONLY from the provided CVE records. Do not add information not present in the context."* Citations are extracted from retrieved chunk IDs, not from LLM output, preventing the model from fabricating CVE references.

---

## RAGAS Evaluation

Evaluated against 25 hand-curated golden questions spanning 6 query types (exact CVE lookup, package-scoped, severity-filtered, cross-package, weakness-type, exploitation-signal). All metrics are reference-free — no human-written ground truth answers required.

| Metric | Score | Threshold | Status |
|---|---|---|---|
| Faithfulness | 0.9265 | 0.85 | ✅ PASS |
| Answer Relevancy | 0.7971 | 0.75 | ✅ PASS |
| Context Precision | 0.8245 | 0.75 | ✅ PASS |

**Judge LLM:** gpt-4o-mini · **Embedding model:** text-embedding-3-small

**Per-type breakdown:**

| Query type | Faithfulness | Relevancy | Precision |
|---|---|---|---|
| exact_cve_lookup | 0.933 | 0.790 | 0.889 |
| package_scoped | 1.000 | 0.785 | 0.967 |
| cross_package | 0.738 | 0.941 | 0.833 |
| exploitation_signal | 0.944 | 0.900 | 0.333 |
| severity_filtered | 1.000 | 0.280 | 1.000 |
| weakness_type | 1.000 | 0.857 | 0.857 |

**Known limitation:** Answer Relevancy scores lower on filter-type queries (`severity_filtered`: 0.280) because RAGAS penalises list-style responses relative to question scope. This is a metric characteristic, not a retrieval quality issue — Context Precision on the same queries scores 1.000, confirming the retriever is surfacing the right CVEs.

**2 questions skipped** (low confidence): the system correctly identified queries where corpus match quality was insufficient and returned "Insufficient information" rather than a low-confidence answer.

---

## Latency Budget

Measured on Apple M-series (CPU inference, no GPU):

| Stage | Latency |
|---|---|
| FastEmbed query embedding (dense + sparse) | ~15ms |
| Qdrant hybrid search (RRF, k=10, 912 points) | ~96ms |
| Cross-encoder rerank (10 pairs, MPS) | ~240ms |
| gpt-5.4-nano synthesis (~600 tokens) | ~6,000ms |
| **Total (warm, models loaded)** | **~6,400ms** |

Model loading at startup (~15s, one-time): dense embedding, sparse BM25, cross-encoder, Qdrant client — all loaded via FastAPI lifespan hook, never per-request.

---

## API

### `POST /query` — Full RAG pipeline

```bash
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Which critical LangChain CVEs have no fix available?",
    "k": 5,
    "severity": "CRITICAL",
    "package": "LangChain",
    "fix_available": false
  }'
```

**Response fields:** `answer`, `citations`, `confidence`, `model_used`, `fallback`, `chunks` (with `rerank_score`, `dense_score`, payload), `retrieval_ms`, `rerank_ms`, `synthesis_ms`, `total_ms`

**Optional filters:** `severity`, `package`, `cisa_kev`, `fix_available`, `attack_vector`, `multi_package`

### `POST /search` — Retrieval only (no synthesis)

Same request schema as `/query`. Returns ranked chunks with scores — no LLM call, no cost. Used by downstream agents (e.g. `kronos-agent`) that synthesise their own answers.

### `GET /health`

```json
{"status": "ok", "collection": "cve_chunks", "version": "0.1.0"}
```

---

## Storage Design

| Store | Purpose | Schema |
|---|---|---|
| **Qdrant** | Dense + sparse vectors, payload for result cards | Named vectors: `dense` (384-dim cosine) + `sparse` (BM25) |
| **PostgreSQL 16** | Structured metadata for exact-match filtering | `cve_chunks` table: severity, packages[], cwes[], cisa_kev, fix_available, attack_vector, ssvc_exploitation, score_disagreement |

**Why two stores?** Qdrant serves retrieval without a PostgreSQL round-trip (payload contains all fields needed to render a result card). PostgreSQL enables SQL-level aggregations and audit queries that Qdrant's payload filtering can't express efficiently (e.g. `GROUP BY severity`, `JOIN` with other tables).

---

## CVE Narrative Format

Each CVE is stored as a composed narrative text — the single field fed to both the embedding model and BM25 index:

```
CVE-2026-42208 | Published: 2026-03-15 | Last Modified: 2026-05-08 | Status: Analyzed
Packages: LiteLLM, OpenAI SDK
⚠ CISA KEV: Actively exploited — "BerriAI LiteLLM SQL Injection Vulnerability" (added 2026-05-08)
⚠ Exploitation: Active exploitation confirmed in the wild (SSVC)

Description: A SQL injection vulnerability in LiteLLM allows an unauthenticated attacker...

CVSS Scores:
  - CRITICAL 9.8 [NVD] [CVSS 31] | Network-exploitable, no auth required
  - CRITICAL 9.3 [security-advisories@github.com] [CVSS 4.0] | Network-exploitable, no auth required
  ⚠ Score disagreement: NVD=9.8, vendor=9.3 (delta: 0.5)

Weakness: CWE-89 (SQL Injection)
Affected Versions: litellm: >= 1.81.16, < 1.83.7 (fix in 1.83.7)
References: Patch — https://...
Source: NVD (https://nvd.nist.gov/vuln/detail/CVE-2026-42208)
```

**Why narrative format over raw JSON fields?** Embedding structured JSON produces poor semantic vectors — the model has no context for what `"severity": "CRITICAL"` means relative to the description. Composing a natural-language narrative preserves all structured information while enabling semantic search over human-readable text.

---

## Project Structure

```
argus-sec/
├── app/
│   └── main.py              # FastAPI app — /query, /search, /health
├── services/
│   └── retrieval.py         # Hybrid retrieval module (imported by app + evaluate)
├── scripts/
│   ├── nvd_sync.py          # NVD CVE acquisition (full + incremental)
│   ├── deduplicate.py       # Cross-package deduplication with argus_packages tagging
│   ├── chunk_cves.py        # CVE narrative composition + metadata extraction
│   ├── ingest.py            # PostgreSQL + Qdrant ingestion
│   ├── evaluate.py          # RAGAS evaluation harness
│   ├── generate_golden_questions.py  # GPT-generated eval question set
│   └── patch_golden_questions.py     # Curated edits to golden set
├── data/
│   ├── golden_questions.json  # 25 evaluation questions (committed)
│   ├── eval_summary.txt       # RAGAS results (committed)
│   └── corpus_stats.json      # Corpus statistics (committed)
├── docs/                    # ADRs (see below)
├── docker-compose.yml       # Qdrant v1.17.1 + one-shot ingest container
├── Dockerfile.ingest        # Multi-stage ingest image
├── requirements.txt         # Runtime deps (app + services)
├── requirements.ingest.txt  # Data pipeline deps
├── requirements.eval.txt    # Evaluation deps (RAGAS + langchain)
└── .env.example             # Required environment variables
```

---

## Running Locally

**Prerequisites:** Python 3.11+, Docker, Ollama (for local fallback model)

```bash
# 1. Clone and set up environment
git clone https://github.com/YOUR_USERNAME/argus-sec.git
cd argus-sec
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env: set OPENAI_API_KEY, QDRANT_API_KEY, POSTGRES_URL

# 3. Start Qdrant
docker compose up -d qdrant

# 4. Fetch CVE corpus and ingest
pip install -r requirements.ingest.txt
python -m scripts.nvd_sync          # ~20 min, fetches 912 CVEs
python -m scripts.deduplicate
python -m scripts.chunk_cves
python -m scripts.ingest            # embeds + loads into Qdrant + PostgreSQL

# 5. Start the API
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload

# 6. Query
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Which CVEs in my stack have active exploitation confirmed?", "k": 5}'
```

---

## ADRs

- [ADR-001: Hybrid Retrieval over Dense-Only](docs/ADR-001-hybrid-retrieval.md)
- [ADR-002: CVE Corpus Scope and NVD API Design](docs/ADR-002-corpus-scope.md)
- [ADR-003: Strict Grounding over Augmented Synthesis](docs/ADR-003-strict-grounding.md)
- [ADR-004: CPE Post-Filtering for High-Volume Packages](docs/ADR-004-cpe-filtering.md)

---

## Portfolio Context

This is **Portfolio Project 1** of a five-project GenAI Solution Architect portfolio,
built to demonstrate production-style system design across:

| Project | What it demonstrates |
|---|---|
| **nexus-ai-gateway** (P5) | Governed enterprise AI gateway — EU AI Act risk classification, PII redaction, multi-provider routing, semantic cache |
| **argus-sec** (P1) | RAG depth — hybrid retrieval, cross-encoder rerank, RAGAS evaluation, security domain expertise |
| **atlas-supply** (P3) | Multi-agent with MCP — LangGraph supervisor, OpenAI Agents SDK, FastMCP server |
| **kronos-agent** (P2) | Knowledge work automation — LangGraph + CrewAI, two-layer memory, HITL interrupts |
| **forge-mlops** (P4) | LLMOps — QLoRA fine-tuning, MLflow tracking, DVC versioning, CI eval gate |

---

*Data source: National Vulnerability Database (NVD), NIST. CVE data is public domain. Reuse permitted.*
