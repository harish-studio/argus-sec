# ADR-004: CPE Post-Filtering for High-Volume Packages

**Date:** 2026-07-07  

---

## Context

The NVD API's `keywordSearch` parameter matches against CVE description text — not against the formally registered affected products. For most packages (e.g. `LangChain`, `LiteLLM`, `Grafana`), keyword search returns accurate results because these names are distinctive and rarely appear in unrelated CVE descriptions.

For three high-volume packages — **Docker**, **PostgreSQL**, **Redis** — keyword search significantly over-counts:

- `"Docker"` appears in thousands of CVE descriptions as context (`"tested in a Docker container"`, `"deployed via Docker Compose"`) rather than as the affected product
- `"PostgreSQL"` and `"Redis"` similarly appear as infrastructure context in CVEs affecting other products

Initial keyword-based counts before filtering:
- Docker: 294 raw records
- PostgreSQL: 152 raw records  
- Redis: 83 raw records

---

## Decision

**Apply CPE post-filtering client-side** for Docker, PostgreSQL, and Redis after keyword fetch. Keep only records where at least one `cpeMatch` entry with `vulnerable: true` has a `criteria` string matching the expected CPE vendor:product prefix.

CPE prefixes used:
- Docker: `cpe:2.3:a:docker:docker:`
- PostgreSQL: `cpe:2.3:a:postgresql:postgresql:`
- Redis: `cpe:2.3:a:redis:redis:`

All other packages use keyword search without CPE filtering.

---

## Rationale

### Why not use CPE-based API queries instead?

The NVD API 2.0 `cpeName` parameter is the correct tool for CPE-based filtering — but it cannot be combined with date range parameters (`lastModStartDate`/`lastModEndDate`). Combining them returns HTTP 404. This was confirmed empirically during development.

Since the incremental sync strategy requires date parameters, CPE-based API queries are architecturally incompatible with the chosen sync approach. Client-side CPE filtering on keyword-fetched results is the correct workaround.

### Why post-filter rather than accept the noise?

Noise in the corpus degrades retrieval precision in two ways:

1. **Embedding noise** — a Docker CVE that is actually about a third-party application running in Docker will embed with Docker-related semantic content, surfacing incorrectly for Docker security queries
2. **Corpus inflation** — 294 raw Docker records vs ~36 genuine ones means 87% of the Docker corpus is irrelevant, directly harming Context Precision for package-scoped Docker queries

### Why only three packages?

The threshold for applying CPE filtering was set empirically: packages returning > 50 raw keyword results where a significant fraction are plausibly incidental mentions. For packages like `LangChain` (66 records), `Grafana` (136 records), and `MLflow` (74 records), the names are distinctive enough that keyword matches are predominantly genuine. Docker at 294 is the clear outlier; PostgreSQL at 152 and Redis at 83 are borderline but filtered for safety.

### Noise reduction results

After CPE post-filtering:
- Docker: 294 → 36 records (88% noise removed)
- PostgreSQL: 152 → 173 records (post-filter count is *higher* because the PostgreSQL re-fetch corrected an earlier partial-fetch failure)
- Redis: 83 → 46 records (45% noise removed)

---

## Consequences

**Positive:**
- Corpus precision improved for the three highest-volume packages
- CPE `cpeMatch` data is embedded in every NVD record — no additional API calls required
- Filter logic is transparent and inspectable in `scripts/nvd_sync.py`

**Negative:**
- CPE coverage is incomplete for recently published CVEs — NVD sometimes takes weeks to populate `configurations` blocks. A CVE in the enrichment queue may have no `cpeMatch` entries and will be incorrectly excluded by the filter.
- Mitigation: `vulnStatus: "Awaiting Analysis"` CVEs in the corpus (flagged in narrative text) are most likely to have incomplete CPE data. These 115 UNKNOWN-severity records are retained because the keyword description match is still meaningful even without CPE confirmation.

**Packages not filtered:**
All other 28 packages use keyword search without CPE post-filtering. For packages with zero corpus hits (HuggingFace PEFT, TRL, BitsAndBytes, Presidio, OpenAI Agents SDK, rank_bm25, cross-encoder), this reflects genuinely low CVE surface rather than filtering artefacts.

---

## Alternatives Considered

| Option | Rejected because |
|---|---|
| CPE-based API queries | Incompatible with date range parameters (HTTP 404, confirmed empirically) |
| Accept keyword noise for all packages | Degrades retrieval precision for Docker/PostgreSQL/Redis queries |
| Manual curated blocklist | Brittle, not maintainable as corpus grows |
| Raise keyword count threshold to exclude borderline packages | Arbitrary threshold; CPE filtering is more principled |

---

## References

- NVD CPE Dictionary: https://nvd.nist.gov/products/cpe
- NVD Vulnerability API 2.0 parameter reference: https://nvd.nist.gov/developers/vulnerabilities
- CPE 2.3 specification: https://nvlpubs.nist.gov/nistpubs/Legacy/IR/nistir7695.pdf
