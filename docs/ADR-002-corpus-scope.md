# ADR-002: CVE Corpus Scope and NVD API Design

**Date:** 2026-07-07  

---

## Context

`argus-sec` answers security questions about a specific technology stack — the 31 packages used across the 5-project GenAI portfolio (`nexus-ai-gateway`, `argus-sec`, `atlas-supply`, `kronos-agent`, `forge-mlops`). The corpus scope decision determines what the system can and cannot answer.

Two dimensions required decisions:
1. **Which packages** to include
2. **What time window** to fetch CVEs for

---

## Decision

**Corpus:** All CVEs from the NVD API 2.0 affecting the 31 packages used across the 5-project portfolio, published or modified in the last 24 months.

**Sync strategy:** `lastModStartDate` / `lastModEndDate` parameters (not `pubStartDate`) for both full and incremental fetches.

---

## Rationale

### Package scope — 31 packages across 5 projects

Restricting the corpus to only one project's stack would produce a useful but narrow demo. Including all 31 packages across the full portfolio enables a stronger narrative: *"this system monitors the security posture of the entire portfolio's dependency surface."*

The 31 packages span infrastructure (Docker, PostgreSQL, Redis), AI orchestration (LangChain, LangGraph, CrewAI, LiteLLM), observability (Prometheus, Grafana), MLOps (MLflow, DVC), and AI SDKs (OpenAI, Anthropic, Ollama) — broad enough to demonstrate cross-domain retrieval, specific enough to have a coherent scope story.

### Time window — 24 months

- **Too short (< 6 months):** misses historically significant CVEs that are still unpatched in many deployments
- **Too long (> 36 months):** includes CVEs for package versions no longer in use; increases corpus noise
- **24 months** covers the active vulnerability surface of packages at their current versions while remaining manageable (~912 CVEs)

### Sync parameter — `lastModStartDate` over `pubStartDate`

NVD retroactively updates CVE records after initial publication — CVSS scores are revised, affected version ranges are added, CNA source scores are appended. Using `pubStartDate` would miss these updates on already-ingested records.

`lastModStartDate` captures both new CVEs and updates to existing ones in a single API call. This is NVD's documented recommendation for synchronisation use cases.

**Empirical finding:** combining `cpeName` (CPE-based exact matching) with date range parameters returns HTTP 404 from the NVD API — these parameters are mutually exclusive in NVD API 2.0. Keyword search (`keywordSearch`) is used for all packages, with CPE post-filtering applied client-side for high-volume packages (Docker, PostgreSQL, Redis) where keyword matching over-counts incidental description mentions.

---

## Consequences

**Positive:**
- 912 CVEs is substantive but not unmanageable — embedding fits in memory, eval runs in ~17 minutes
- 24-month window captures the CVE activity period of all 31 packages at their current versions
- `lastModStartDate` sync keeps the corpus current without re-fetching unchanged records
- Incremental sync can run on a schedule (`python -m scripts.nvd_sync --incremental`) to keep the corpus fresh

**Negative:**
- Keyword-based fetching for 29 of 31 packages introduces some noise (CVEs mentioning a package in context but not affecting it)
- CPE post-filtering for Docker/PostgreSQL/Redis reduces noise but discards some edge cases
- 115 CVEs in the corpus have `UNKNOWN` severity — NVD enrichment is incomplete for recently published CVEs

**Packages with zero CVEs** (HuggingFace PEFT, TRL, BitsAndBytes, Presidio, OpenAI Agents SDK, rank_bm25, cross-encoder): these packages have genuinely low CVE surface — either not network-exposed, very new, or rarely targeted. They are retained in the package list so future incremental syncs will capture CVEs if they emerge.

---

## Alternatives Considered

| Option | Rejected because |
|---|---|
| Legal/regulatory corpus (EU AI Act, GDPR, DSA) | Saturated portfolio pattern — multiple public tutorials build "chat with the EU AI Act" RAG systems |
| Single project stack only (nexus-ai-gateway) | Weaker demo narrative; misses cross-stack vulnerability analysis |
| pubStartDate sync | Misses CVSS score updates and version range additions to existing CVEs |
| cpeName-based fetching | Returns HTTP 404 when combined with date parameters (NVD API 2.0 constraint, confirmed empirically) |

---

## References

- NVD Vulnerability API 2.0: https://nvd.nist.gov/developers/vulnerabilities
- NVD synchronisation guidance: https://nvd.nist.gov/developers/start-here
- CVE data reuse: public domain (Title 17 U.S. Code)
