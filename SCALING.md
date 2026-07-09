# SCALING.md — argus-sec Production Scaling Analysis

**Target:** 50,000 CVEs · 5,000 concurrent users · AWS ECS/Fargate (eu-west-1)

Current state: 912 CVEs · single-node local deployment · Apple Silicon dev machine

---

## Current Architecture Constraints

| Component | Current | Bottleneck at scale |
|---|---|---|
| Qdrant | Single node, in-memory, 912 points | Memory-bound; single point of failure |
| PostgreSQL | Shared dev_postgres, 912 rows | Connection pool exhaustion under concurrent load |
| FastAPI | Single uvicorn process | No horizontal scaling; CPU-bound on reranking |
| Cross-encoder | MPS (Apple Silicon), ~240ms/query | CPU/MPS serial execution; blocks under concurrency |
| Synthesis | gpt-5.4-nano, ~6s/query | Rate-limited by OpenAI; no request queuing |
| NVD sync | Manual, local Python script | No scheduling; stale corpus if not run regularly |

---

## Scaling Dimensions

### 1. Corpus: 912 → 50,000 CVEs

**Impact on Qdrant:**
50,000 vectors × 384 dimensions × 4 bytes = ~77MB dense storage. Sparse BM25 vectors add ~15-30MB depending on vocabulary size. Total: ~110MB — comfortably in-memory on a single Qdrant node with 4GB RAM. No sharding required at this scale.

**Impact on embedding:**
Re-embedding the full corpus from scratch would take ~15 hours on CPU (50,000 × 1.84s/chunk from current benchmarks). Mitigations:
- Incremental ingestion via lastModStartDate — only embed new/modified CVEs per sync cycle
- GPU-accelerated embedding on an AWS g4dn.xlarge (NVIDIA T4): ~10x speedup reduces full re-embed to ~90 minutes
- Pre-warm the FastEmbed model in the ECS task definition using a startup script

**Impact on PostgreSQL:**
50,000 rows in cve_chunks is trivial for PostgreSQL. GIN indexes on packages[] and cwes[] remain efficient up to ~10M rows. No schema changes required.

**Impact on retrieval latency:**
Qdrant HNSW index lookup is O(log n) — retrieval latency increases from ~96ms to ~120ms at 50,000 points. Acceptable.

---

### 2. Concurrency: 1 → 5,000 concurrent users

**Baseline capacity (current):**
- Single FastAPI process, single cross-encoder thread
- Cross-encoder: ~240ms per query (serial)
- Synthesis: ~6,000ms per query (OpenAI API)
- Effective throughput: ~0.16 queries/second (synthesis-bound)

**Target capacity:**
5,000 concurrent users at realistic query rates (1 query per 30 seconds = ~167 queries/second sustained, ~500 qps peak)

**Scaling strategy — three layers:**

#### Layer 1: FastAPI horizontal scaling (ECS)

```
ALB (Application Load Balancer)
  |
ECS Service — argus-api
  |-- Task 1: FastAPI + cross-encoder (2 vCPU, 4GB RAM)
  |-- Task 2: FastAPI + cross-encoder (2 vCPU, 4GB RAM)
  |-- ...
  +-- Task N: FastAPI + cross-encoder (2 vCPU, 4GB RAM)
```

Each ECS task runs one FastAPI process with the cross-encoder pre-loaded at startup (lifespan hook). ECS auto-scaling trigger: CPU > 70% scale out; < 30% scale in. Target: 10-20 tasks at peak.

**Embedding models shared via EFS:** BAAI/bge-small-en-v1.5 (~130MB) and ms-marco-MiniLM-L-6-v2 (~110MB) mounted from EFS — avoids re-downloading on every task launch, reduces cold start from ~15s to ~3s.

#### Layer 2: Qdrant — dedicated node

Move Qdrant from the ECS task to a dedicated r6g.large instance (8GB RAM, ARM Graviton) or Qdrant Cloud (managed). At 50,000 points, single-node is sufficient — no distributed mode needed.

Qdrant persists to EBS gp3 volume (1,000 IOPS baseline, 16,000 IOPS burst) for durability across restarts.

#### Layer 3: Synthesis — rate limit management

OpenAI rate limits (gpt-5.4-nano): 500 RPM / 200,000 TPM (tier 2). At 167 qps sustained this exceeds the RPM limit.

Mitigations:
1. **Request queue (SQS):** route synthesis requests through SQS FIFO queue — /query returns job_id immediately; client polls /result/{job_id}
2. **Semantic cache:** identical or near-identical queries (cosine > 0.92) served from Redis without an OpenAI call
3. **Tiered synthesis:** exact_cve_lookup queries routed to gemma3:1b local model (no rate limit)
4. **OpenAI Batch API:** non-real-time evaluation runs use Batch API for 50% cost reduction

---

## Failure Modes and Recovery

### Failure 1: Qdrant node unavailable

**Symptom:** POST /query returns 503 after Qdrant healthcheck fails.

**Recovery:**
1. ECS health check detects unhealthy Qdrant — ALB stops routing within 30s
2. Qdrant restarts automatically (ECS restartPolicy)
3. HNSW index reloads from EBS — ~10s at 50,000 points
4. Degraded mode: /search (retrieval only) available; /query returns 503 with Retry-After header

**Mitigation:** Qdrant snapshots to S3 on 6-hour schedule. If EBS lost, restore from S3 (~5 minutes).

---

### Failure 2: OpenAI API unavailable or rate-limited

**Symptom:** synthesis calls return 429 or 503.

**Current behaviour:** fallback to gemma3:1b already implemented with disclaimer.

**Recovery at scale:**
1. SQS queue absorbs burst — synthesis requests queued, not dropped
2. Circuit breaker with exponential backoff before falling back to local model
3. If outage > 30 minutes: all queries temporarily routed to /search; UI banner displayed

**Monitoring:** CloudWatch metric SynthesisP95Latency alarm at > 15s triggers SNS notification.

---

### Failure 3: NVD sync fails mid-run

**Symptom:** incremental sync exits with partial data.

**Current behaviour:** atomic file writes (tmp → rename) — failed sync leaves previously-fetched packages complete. last_run.json only updated on successful completion — failed sync does not advance the sync pointer.

**Recovery:**
Re-run python -m scripts.nvd_sync --incremental — idempotency skips already-fetched packages.

**At scale:** replace manual script with ECS Scheduled Task (cron: 0 2 * * *) — failures trigger CloudWatch Alarm → SNS.

---

### Failure 4: Cross-encoder OOM under concurrent load

**Symptom:** ECS task OOM when multiple concurrent rerank operations overlap.

**Fix:** run cross-encoder in ThreadPoolExecutor with max_workers=2 per task, limiting concurrent rerank operations. 4GB per ECS task is sufficient for one cross-encoder instance + FastAPI overhead + 2 concurrent reranks.

---

## Cost Model

### At 912 CVEs / low traffic (current)

| Component | Monthly cost |
|---|---|
| Qdrant (local Docker) | $0 |
| PostgreSQL (shared dev) | $0 |
| OpenAI synthesis (dev) | ~$2 |
| **Total** | **~$2/month** |

### At 50,000 CVEs / 5,000 concurrent users (production)

| Component | Spec | Monthly cost (est.) |
|---|---|---|
| ECS Fargate (argus-api) | 15 tasks x 2 vCPU x 4GB, ~12hr/day | ~$180 |
| Qdrant node | r6g.large (8GB RAM) | ~$75 |
| RDS PostgreSQL | db.t4g.medium, Multi-AZ | ~$90 |
| ALB | 1 ALB + data processing | ~$25 |
| EBS (Qdrant storage) | 50GB gp3 | ~$4 |
| EFS (model weights) | 500MB | ~$1 |
| S3 (Qdrant snapshots) | 5GB | ~$1 |
| OpenAI synthesis | 167 qps x 0.7 cache miss x $0.001/query x 86,400s/day x 30 days | ~$9,000 |
| SQS + Lambda (queue) | ~167M requests/month | ~$20 |
| CloudWatch + monitoring | Standard | ~$10 |
| **Total** | | **~$9,400/month** |

**Cost driver:** OpenAI synthesis dominates at 92% of total. Semantic cache at 30% hit rate saves ~$4,000/month. Increasing to 60% hit rate (realistic for a large user base) reduces synthesis cost by ~$6,000/month.

---

## Incremental NVD Sync at Scale

At 50,000 CVEs, daily incremental sync handles ~50-200 new/modified CVEs per day.

**Scheduled ECS task (daily, 02:00 UTC):**
```
ECS Scheduled Task
  -> scripts/nvd_sync.py --incremental   (~3 min)
  -> scripts/deduplicate.py              (~1 min)
  -> scripts/chunk_cves.py               (~1 min)
  -> scripts/ingest.py (upsert only)     (~2 min, GPU embed of ~40 CVEs)
```

Total: ~7 minutes daily. Zero downtime — Qdrant upsert is non-blocking.

---

## What Is Not Addressed Here

- **Multi-region deployment** — single eu-west-1 region assumed. Multi-region requires Qdrant replication (Qdrant Cloud) and RDS read replicas.
- **Authentication** — current API has no auth layer. Production adds API key validation consistent with nexus-ai-gateway.
- **GDPR / NIS2 compliance** — if deployed in EU enterprise, CVE data processing under GDPR Article 5 (purpose limitation) and NIS2 vulnerability disclosure obligations apply.
- **Fine-tuned embedding** — BAAI/bge-small-en-v1.5 is general-purpose. A security-domain fine-tuned model would improve Context Precision beyond 0.82.
