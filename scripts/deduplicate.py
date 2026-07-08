"""
argus-sec — CVE corpus deduplication script.

Merges all per-package JSON files from data/raw/cve/ into a single
deduplicated corpus at data/corpus.json.

Each unique CVE appears exactly once, tagged with every package file it
was found in via a `argus_packages` field. This enables retrieval queries
like "vulnerabilities affecting LangChain" to surface CVEs tagged with
["LangChain", "LangGraph"] where both packages share a vulnerability.

Output record structure (extends the original NVD record):
{
    "argus_packages": ["LangChain", "LangGraph"],  # <-- added by this script
    "cve": { ...original NVD fields... }
}

Deduplication key: cve.id (e.g. "CVE-2024-1234")
Conflict resolution: if the same CVE ID appears in multiple package
files with different content (NVD updates CVSS scores retroactively),
the record with the most recent lastModified date wins.

Run:
    python -m scripts.deduplicate

Output:
    data/corpus.json       — deduplicated CVE records with package tags
    data/corpus_stats.json — summary stats for review
"""

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

RAW_DIR     = Path("data/raw/cve")
CORPUS_PATH = Path("data/corpus.json")
STATS_PATH  = Path("data/corpus_stats.json")

# Maps filename stem -> display name, matching PACKAGES in nvd_sync.py
FILENAME_TO_PACKAGE: dict[str, str] = {
    "docker":           "Docker",
    "postgresql":       "PostgreSQL",
    "redis":            "Redis",
    "langchain":        "LangChain",
    "langgraph":        "LangGraph",
    "langsmith":        "LangSmith",
    "litellm":          "LiteLLM",
    "llamaindex":       "LlamaIndex",
    "crewai":           "CrewAI",
    "fastmcp":          "FastMCP",
    "ollama":           "Ollama",
    "ragas":            "RAGAS",
    "huggingface_peft": "HuggingFace PEFT",
    "trl":              "TRL",
    "bitsandbytes":     "BitsAndBytes",
    "mlflow":           "MLflow",
    "dvc":              "DVC",
    "qdrant":           "Qdrant",
    "duckdb":           "DuckDB",
    "fastapi":          "FastAPI",
    "prometheus":       "Prometheus",
    "grafana":          "Grafana",
    "streamlit":        "Streamlit",
    "pydantic":         "Pydantic",
    "presidio":         "Presidio",
    "github_actions":   "GitHub Actions",
    "openai_sdk":       "OpenAI SDK",
    "anthropic_sdk":    "Anthropic SDK",
    "openai_agents_sdk":"OpenAI Agents SDK",
    "rank_bm25":        "rank_bm25",
    "cross_encoder":    "cross-encoder",
}


def parse_nvd_datetime(dt_str: str | None) -> datetime | None:
    """
    Parse NVD's ISO 8601 datetime strings, e.g. "2024-03-15T12:00:00.000".
    Returns None if the string is missing or unparseable.
    """
    if not dt_str:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(dt_str, fmt)
        except ValueError:
            continue
    logger.warning("Could not parse datetime: %s", dt_str)
    return None


def get_last_modified(record: dict) -> datetime | None:
    return parse_nvd_datetime(record.get("cve", {}).get("lastModified"))


def load_package_file(path: Path, pkg_name: str) -> list[tuple[str, str, dict]]:
    """
    Load one package file and return a list of (cve_id, pkg_name, record)
    tuples. Skips malformed or empty files gracefully.
    """
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.warning("Could not read %s: %s — skipping", path.name, exc)
        return []

    if not records:
        return []

    result = []
    for record in records:
        cve_id = record.get("cve", {}).get("id", "")
        if not cve_id:
            logger.warning("Record in %s missing cve.id — skipping", path.name)
            continue
        result.append((cve_id, pkg_name, record))
    return result


def deduplicate(
    entries: list[tuple[str, str, dict]]
) -> list[dict]:
    """
    Merge (cve_id, pkg_name, record) entries into one record per CVE ID.

    Rules:
    - argus_packages: union of all pkg_names that contained this CVE ID,
      sorted alphabetically for deterministic output.
    - Winning record: most recently modified per lastModified field.
      Rationale: NVD retroactively updates CVSS scores and affected
      version ranges; the most recent version is the most accurate.
    - Tie-break: first encountered (deterministic given sorted input).
    """
    # Group by CVE ID
    by_id: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for cve_id, pkg_name, record in entries:
        by_id[cve_id].append((pkg_name, record))

    corpus: list[dict] = []
    for cve_id, pkg_record_pairs in by_id.items():
        packages = sorted({pkg for pkg, _ in pkg_record_pairs})

        # Pick the most recently modified record as the base
        records = [r for _, r in pkg_record_pairs]
        best_record = max(
            records,
            key=lambda r: get_last_modified(r) or datetime.min,
        )

        # Inject argus_packages into a shallow copy — don't mutate the original
        merged = {"argus_packages": packages, **best_record}
        corpus.append(merged)

    return corpus


def build_stats(corpus: list[dict], raw_total: int) -> dict:
    """
    Build a summary stats dict for corpus_stats.json.

    Includes:
    - Total unique CVEs
    - Raw records before dedup (duplicates removed)
    - Per-package CVE counts (how many unique CVEs each package contributed)
    - Multi-package CVEs (CVEs shared across 2+ packages — interview point:
      these are the most architecturally significant vulnerabilities)
    - CVSS severity distribution across the deduplicated corpus
    """
    pkg_counts: dict[str, int] = defaultdict(int)
    multi_package: list[str] = []
    severity_dist: dict[str, int] = defaultdict(int)

    for record in corpus:
        packages = record.get("argus_packages", [])
        for pkg in packages:
            pkg_counts[pkg] += 1
        if len(packages) > 1:
            multi_package.append(record["cve"]["id"])

        # CVSS v3.1 severity (fall back to v3.0, then v2)
        metrics = record.get("cve", {}).get("metrics", {})
        severity = "UNKNOWN"
        for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
            entries = metrics.get(key, [])
            if entries:
                severity = entries[0].get("cvssData", {}).get("baseSeverity", "UNKNOWN")
                break
        severity_dist[severity] += 1

    return {
        "unique_cves":        len(corpus),
        "raw_records_total":  raw_total,
        "duplicates_removed": raw_total - len(corpus),
        "multi_package_cves": len(multi_package),
        "multi_package_ids":  sorted(multi_package),
        "per_package_counts": dict(
            sorted(pkg_counts.items(), key=lambda x: x[1], reverse=True)
        ),
        "severity_distribution": dict(
            sorted(severity_dist.items(), key=lambda x: x[1], reverse=True)
        ),
    }


def main() -> None:
    # Collect all (cve_id, pkg_name, record) entries across all package files
    all_entries: list[tuple[str, str, dict]] = []
    files_found = 0

    for path in sorted(RAW_DIR.glob("*.json")):
        stem = path.stem
        pkg_name = FILENAME_TO_PACKAGE.get(stem)
        if pkg_name is None:
            logger.warning("Unknown file %s — not in FILENAME_TO_PACKAGE, skipping", path.name)
            continue
        entries = load_package_file(path, pkg_name)
        all_entries.extend(entries)
        files_found += 1
        logger.info("%-20s %4d records loaded", path.name, len(entries))

    raw_total = len(all_entries)
    logger.info("Loaded %d raw records from %d files", raw_total, files_found)

    # Deduplicate
    corpus = deduplicate(all_entries)
    stats  = build_stats(corpus, raw_total)

    # Write outputs
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CORPUS_PATH.write_text(
        json.dumps(corpus, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    STATS_PATH.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Summary to console
    logger.info("─" * 60)
    logger.info("Corpus written to %s", CORPUS_PATH)
    logger.info("Unique CVEs:        %d", stats["unique_cves"])
    logger.info("Raw records:        %d", stats["raw_records_total"])
    logger.info("Duplicates removed: %d", stats["duplicates_removed"])
    logger.info("Multi-package CVEs: %d", stats["multi_package_cves"])
    logger.info("Severity breakdown:")
    for sev, count in stats["severity_distribution"].items():
        logger.info("  %-12s %d", sev, count)
    logger.info("Stats written to %s", STATS_PATH)


if __name__ == "__main__":
    main()
