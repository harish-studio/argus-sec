# ADR-001: Hybrid Retrieval (BM25 + Dense) over Dense-Only

**Status:** Accepted  
**Date:** 2026-07-07  
**Deciders:** Harry (solo portfolio project)

---

## Context

`argus-sec` retrieves CVE records in response to natural language security queries. The retrieval mechanism determines what context reaches the synthesis LLM — poor retrieval produces hallucinated or incomplete answers regardless of model quality.

Two retrieval approaches were considered:

1. **Dense-only** — embed the query and CVE narratives with `BAAI/bge-small-en-v1.5`, retrieve by cosine similarity.
2. **Hybrid** — combine BM25 sparse retrieval with dense retrieval, fuse scores with Reciprocal Rank Fusion (RRF), rerank with a cross-encoder.

---

## Decision

**Hybrid retrieval (BM25 + dense, RRF fusion, cross-encoder rerank).**

---

## Rationale

CVE queries exhibit two structurally different patterns that require different retrieval mechanisms:

**Pattern 1 — Exact-term queries**

Users frequently search by precise identifiers: `"CVE-2024-3400"`, `"CWE-89"`, `"LangChain 0.3.1"`. Dense embeddings model semantic similarity — they cannot reliably match exact strings that may not appear in training data or that carry no inherent semantic meaning. BM25 term-frequency matching handles these precisely.

**Pattern 2 — Semantic queries**

Users also ask conceptual questions: `"which vulnerabilities allow unauthenticated remote access?"`, `"SQL injection in my AI stack"`. Keyword search misses documents that express the same concept with different vocabulary. Dense retrieval handles these by matching semantic intent.

Neither mechanism alone covers both patterns. Hybrid retrieval with RRF fusion combines them without requiring manual weight tuning — RRF is rank-based and robust to score scale differences between BM25 and cosine similarity.

**Cross-encoder reranking** provides a second-pass relevance signal by scoring (query, chunk) pairs directly — more accurate than vector similarity but too expensive to run over the full corpus. Applied to the top-10 RRF candidates, narrowing to top-5 for synthesis.

---

## Consequences

**Positive:**
- RAGAS Context Precision: 0.8245 — retriever surfaces relevant CVEs at the top of results 82% of the time
- Exact CVE ID queries (`"CVE-2026-42208"`) retrieve correctly via BM25; semantic queries (`"SQL injection without authentication"`) retrieve correctly via dense
- Cross-encoder rerank score provides a reliable confidence signal for the confidence gate

**Negative:**
- Higher latency than dense-only: ~96ms retrieval + ~240ms rerank vs ~50ms dense-only (acceptable given ~6,000ms synthesis dominates total latency)
- Two embedding models at startup (FastEmbed dense + BM25) vs one
- More complex retrieval code than a single `similarity_search()` call

**Trade-off explicitly rejected:**
Dense-only with score-averaging fusion was rejected because it requires manual weight tuning (what weight for BM25 vs dense?) and degrades on exact-term queries. RRF is parameter-free and empirically robust across domains.

---

## Alternatives Considered

| Option | Rejected because |
|---|---|
| Dense-only (BAAI/bge-small-en-v1.5) | Fails on exact CVE ID / CWE code queries |
| BM25-only (rank_bm25) | Misses semantic queries; no conceptual matching |
| Dense + BM25 with score averaging | Requires weight tuning; scale mismatch between BM25 and cosine scores |
| Sparse-only (SPLADE) | Better than BM25 but adds model complexity without clear gain over hybrid |

---

## References

- Qdrant hybrid queries documentation: https://qdrant.tech/documentation/concepts/hybrid-queries/
- RRF original paper: Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods" (SIGIR 2009)
- ms-marco-MiniLM-L-6-v2: https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2
