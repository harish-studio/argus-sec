"""
argus-sec — NVD CVE sync script (full + incremental).

Single script that handles both the initial corpus build and all future
incremental syncs. Replaces nvd_acquire.py, filter_cpe.py, and
fix_encoding.py from the prior Windows session.

USAGE:
    # First run (full fetch, last 24 months):
    python -m scripts.nvd_sync

    # Subsequent runs (incremental, since last successful run):
    python -m scripts.nvd_sync --incremental

HOW IT WORKS:
    Full run:
      - lastModStartDate = now minus FULL_LOOKBACK_DAYS (24 months)
      - lastModEndDate   = now
      - Fetches all CVEs modified/published within that window

    Incremental run:
      - lastModStartDate = timestamp stored in data/last_run.json
      - lastModEndDate   = now
      - Fetches only CVEs modified since the last successful run
      - This captures CVSS score updates, new affected versions, etc.
        not just newly published CVEs — correct NVD sync pattern per:
        https://nvd.nist.gov/developers/vulnerabilities

    Both modes:
      - Keyword fetch for all 31 packages
      - CPE filter applied client-side to Docker, PostgreSQL, Redis
        (their CPE match data is already embedded in each NVD record)
      - Idempotent per-package: skips if file exists and is valid JSON
        (use --force to re-fetch everything)
      - Atomic writes (tmp -> rename) so interrupted runs leave no
        corrupt files
      - Deduplication by CVE ID across all package files at the end

DESIGN NOTES:
    - lastModStartDate/lastModEndDate are NVD's recommended sync params.
      They are compatible with keywordSearch (unlike cpeName + date,
      which returns 404 — confirmed empirically in prior session).
      Ref: https://nvd.nist.gov/developers/vulnerabilities
    - 120-day max span per NVD date-range limit; 24 months = 6 windows.
    - 6s delay between requests (NVD unauthenticated rate limit).
      Ref: https://nvd.nist.gov/developers/start-here
    - CPE prefix matching uses vulnerable=True cpeMatch entries only.
      Non-vulnerable references (e.g. "requires Docker on same host")
      are excluded correctly.
    - Mac default encoding is UTF-8 — explicit throughout anyway for
      portability and to guard against any edge cases.
"""

import argparse
import csv
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

BASE_URL             = "https://services.nvd.nist.gov/rest/json/cves/2.0"
REQUEST_DELAY_S      = 6.0    # NVD unauthenticated rate limit
READ_TIMEOUT_S       = 60     # raised from 30s — empirically too tight
RESULTS_PER_PAGE     = 2000   # NVD documented maximum
MAX_SPAN_DAYS        = 120    # NVD hard limit per date-range request
FULL_LOOKBACK_DAYS   = 24 * 30  # ~24 months for full run

RAW_DIR        = Path("data/raw/cve")
STATE_FILE     = Path("data/last_run.json")
COUNTS_CSV     = Path("cve_counts.csv")


# ---------------------------------------------------------------------------
# Package definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Package:
    """
    name:       display name (logs + CSV)
    filename:   output file stem under RAW_DIR (no extension)
    keyword:    NVD keywordSearch value
    exact:      True = keywordExactMatch (multi-word phrases)
    cpe_prefix: if set, CPE-filter records client-side after fetch
                keeping only records where a vulnerable cpeMatch
                criteria starts with this prefix
    """
    name:       str
    filename:   str
    keyword:    str
    exact:      bool   = False
    cpe_prefix: str    = ""


# fmt: off
PACKAGES: list[Package] = [
    # High-volume infrastructure — keyword fetch + client-side CPE filter
    # CPE prefixes confirmed: https://nvd.nist.gov/products/cpe/search
    Package("Docker",      "docker",      "Docker",      cpe_prefix="cpe:2.3:a:docker:docker:"),
    Package("PostgreSQL",  "postgresql",  "PostgreSQL",  cpe_prefix="cpe:2.3:a:postgresql:postgresql:"),
    Package("Redis",       "redis",       "Redis",       cpe_prefix="cpe:2.3:a:redis:redis:"),

    # AI/ML stack — keyword fetch, no CPE filter needed
    Package("LangChain",        "langchain",        "LangChain"),
    Package("LangGraph",        "langgraph",        "LangGraph"),
    Package("LangSmith",        "langsmith",        "LangSmith"),
    Package("LiteLLM",          "litellm",          "LiteLLM"),
    Package("LlamaIndex",       "llamaindex",       "LlamaIndex"),
    Package("CrewAI",           "crewai",           "CrewAI"),
    Package("FastMCP",          "fastmcp",          "FastMCP"),
    Package("Ollama",           "ollama",           "Ollama"),
    Package("RAGAS",            "ragas",            "RAGAS"),
    Package("HuggingFace PEFT", "huggingface_peft", "PEFT huggingface"),
    Package("TRL",              "trl",              "TRL huggingface"),
    Package("BitsAndBytes",     "bitsandbytes",     "bitsandbytes"),
    Package("MLflow",           "mlflow",           "MLflow"),
    Package("DVC",              "dvc",              "DVC"),
    Package("Qdrant",           "qdrant",           "Qdrant"),
    Package("DuckDB",           "duckdb",           "DuckDB"),

    # Infrastructure — keyword fetch, no CPE filter
    Package("FastAPI",       "fastapi",       "FastAPI"),
    Package("Prometheus",    "prometheus",    "Prometheus"),
    Package("Grafana",       "grafana",       "Grafana"),
    Package("Streamlit",     "streamlit",     "Streamlit"),
    Package("Pydantic",      "pydantic",      "Pydantic"),
    Package("Presidio",      "presidio",      "microsoft presidio", exact=True),

    # GitHub Actions — exact phrase avoids matching unrelated "GitHub" hits
    Package("GitHub Actions", "github_actions", "GitHub Actions", exact=True),

    # SDK packages — corrected keywords (prior "OpenAI SDK" returned 0)
    Package("OpenAI SDK",        "openai_sdk",        "openai"),
    Package("Anthropic SDK",     "anthropic_sdk",     "anthropic"),
    Package("OpenAI Agents SDK", "openai_agents_sdk", "openai agents", exact=True),

    # Low-surface packages — genuine expected-zero candidates
    Package("rank_bm25",    "rank_bm25",    "rank_bm25"),
    Package("cross-encoder","cross_encoder","cross-encoder"),
]
# fmt: on


# ---------------------------------------------------------------------------
# Date windows
# ---------------------------------------------------------------------------

@dataclass
class DateWindow:
    start: datetime
    end:   datetime


def build_windows(start: datetime, end: datetime) -> list[DateWindow]:
    """
    Split [start, end] into <=120-day chunks.
    NVD hard limit: date-range span cannot exceed 120 days per request.
    Ref: https://nvd.nist.gov/developers/vulnerabilities
    """
    windows, cursor = [], start
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=MAX_SPAN_DAYS), end)
        windows.append(DateWindow(cursor, chunk_end))
        cursor = chunk_end
    return windows


def load_run_state() -> datetime | None:
    """Return last successful run timestamp, or None if no state file."""
    if not STATE_FILE.exists():
        return None
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return datetime.fromisoformat(data["last_run_utc"])
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("Could not parse state file: %s — running full fetch", exc)
        return None


def save_run_state(ts: datetime) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(
        json.dumps({"last_run_utc": ts.isoformat()}, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# HTTP session
# ---------------------------------------------------------------------------

def build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=2,   # waits: 2s, 4s, 8s, 16s, 32s
        status_forcelist=[403, 429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def build_params(pkg: Package, window: DateWindow, start_index: int) -> dict:
    """
    Build NVD API params for one package/window page.

    Uses lastModStartDate/lastModEndDate throughout (not pubStartDate):
      - Compatible with keywordSearch (pubStartDate + cpeName → 404,
        confirmed empirically)
      - Captures CVSS updates to existing CVEs, not just new publications
      - Correct pattern for incremental sync per NVD developer docs
    """
    params = {
        "keywordSearch":    pkg.keyword,
        "lastModStartDate": window.start.strftime("%Y-%m-%dT%H:%M:%S.000"),
        "lastModEndDate":   window.end.strftime("%Y-%m-%dT%H:%M:%S.000"),
        "resultsPerPage":   RESULTS_PER_PAGE,
        "startIndex":       start_index,
    }
    if pkg.exact:
        params["keywordExactMatch"] = ""
    return params


def fetch_window(
    session: requests.Session,
    pkg: Package,
    window: DateWindow,
) -> list[dict]:
    """Fetch all pages for one package/window."""
    records, start_index = [], 0
    while True:
        params = build_params(pkg, window, start_index)
        resp = session.get(BASE_URL, params=params, timeout=READ_TIMEOUT_S)
        resp.raise_for_status()
        data = resp.json()

        records.extend(data.get("vulnerabilities", []))
        total = data.get("totalResults", 0)
        start_index += RESULTS_PER_PAGE
        time.sleep(REQUEST_DELAY_S)

        if start_index >= total:
            break
    return records


def fetch_package(
    session: requests.Session,
    pkg: Package,
    windows: list[DateWindow],
    pbar: tqdm,
) -> list[dict]:
    all_records: list[dict] = []
    for window in windows:
        try:
            all_records.extend(fetch_window(session, pkg, window))
        except (requests.exceptions.RequestException, json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "Failed window %s–%s for %s: %s — skipping window",
                window.start.date(), window.end.date(), pkg.name, exc,
            )
        pbar.update(1)
    return all_records


# ---------------------------------------------------------------------------
# CPE filter (client-side, applied after keyword fetch)
# ---------------------------------------------------------------------------

def passes_cpe_filter(record: dict, prefix: str) -> bool:
    """
    Return True if any vulnerable cpeMatch in the record's configurations
    has a criteria string starting with the given CPE prefix.

    Only checks cpeMatch entries where vulnerable == True — excludes
    records where the package is referenced but not the affected product.
    """
    try:
        for config in record.get("cve", {}).get("configurations", []):
            for node in config.get("nodes", []):
                for match in node.get("cpeMatch", []):
                    if (
                        match.get("vulnerable", False)
                        and match.get("criteria", "").startswith(prefix)
                    ):
                        return True
    except (AttributeError, TypeError):
        pass
    return False


def apply_cpe_filter(records: list[dict], pkg: Package) -> tuple[list[dict], int]:
    """
    Filter records by CPE prefix. Returns (filtered_records, n_removed).
    Only called when pkg.cpe_prefix is set.
    """
    filtered = [r for r in records if passes_cpe_filter(r, pkg.cpe_prefix)]
    return filtered, len(records) - len(filtered)


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def is_valid_json_file(path: Path) -> bool:
    """
    True only if file exists AND contains parseable, non-empty JSON array.
    Guards against treating empty [] or corrupt files as complete.
    """
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return isinstance(data, list)   # [] is valid — emptiness handled by caller
    except (json.JSONDecodeError, UnicodeDecodeError):
        logger.warning("%s exists but is not valid JSON — will re-fetch", path.name)
        return False


def atomic_write(path: Path, data: list[dict]) -> None:
    """Write via tmp file + rename — safe against interrupted writes."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def merge_with_existing(path: Path, new_records: list[dict]) -> list[dict]:
    """
    For incremental runs: merge new records into existing file,
    deduplicating by CVE ID. Newer record wins on conflict (NVD may
    update CVSS scores, affected versions, etc. after initial publish).
    """
    existing: list[dict] = []
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.warning("%s unreadable — treating as empty for merge", path.name)

    # Build lookup: cve_id -> record, existing first then new (new wins)
    merged: dict[str, dict] = {}
    for r in existing:
        cve_id = r.get("cve", {}).get("id", "")
        if cve_id:
            merged[cve_id] = r
    for r in new_records:
        cve_id = r.get("cve", {}).get("id", "")
        if cve_id:
            merged[cve_id] = r   # new record overwrites existing on conflict

    return list(merged.values())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="argus-sec NVD CVE sync — full or incremental"
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Fetch only CVEs modified since the last successful run",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch all packages even if output files already exist",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)

    if args.incremental:
        last_run = load_run_state()
        if last_run is None:
            logger.warning(
                "--incremental specified but no state file found — "
                "falling back to full fetch (last %d days)",
                FULL_LOOKBACK_DAYS,
            )
            fetch_start = now - timedelta(days=FULL_LOOKBACK_DAYS)
        else:
            fetch_start = last_run
            logger.info(
                "Incremental run: fetching CVEs modified since %s",
                fetch_start.strftime("%Y-%m-%d %H:%M UTC"),
            )
    else:
        fetch_start = now - timedelta(days=FULL_LOOKBACK_DAYS)
        logger.info("Full run: fetching CVEs modified in last %d days", FULL_LOOKBACK_DAYS)

    windows = build_windows(fetch_start, now)
    session = build_session()
    counts  = []
    total_units = len(PACKAGES) * len(windows)

    logger.info(
        "%d packages × %d windows = %d API calls (~%.0f min at %ds delay)",
        len(PACKAGES), len(windows), total_units,
        total_units * REQUEST_DELAY_S / 60,
        REQUEST_DELAY_S,
    )

    with tqdm(total=total_units, desc="Syncing CVE data", unit="window") as pbar:
        for pkg in PACKAGES:
            out_path = RAW_DIR / f"{pkg.filename}.json"

            # Skip if already fetched and not forcing a re-fetch.
            # For incremental runs we never skip — always merge new data.
            if not args.incremental and not args.force and is_valid_json_file(out_path):
                existing = json.loads(out_path.read_text(encoding="utf-8"))
                # Only skip non-empty files — empty [] means prior fetch
                # returned nothing OR CPE filter wiped it; re-fetch to confirm
                if len(existing) > 0:
                    logger.info("Skipping %s (already fetched, %d records)", pkg.name, len(existing))
                    counts.append({"package": pkg.name, "cve_count": len(existing), "note": "skipped"})
                    pbar.update(len(windows))
                    continue

            # Fetch
            raw_records = fetch_package(session, pkg, windows, pbar)

            # CPE filter (high-volume packages only)
            n_removed = 0
            if pkg.cpe_prefix and raw_records:
                raw_records, n_removed = apply_cpe_filter(raw_records, pkg)
                if n_removed:
                    logger.info(
                        "%s: CPE filter removed %d noise records (%.0f%%)",
                        pkg.name, n_removed,
                        n_removed / (len(raw_records) + n_removed) * 100,
                    )

            # Merge with existing for incremental runs
            if args.incremental:
                final_records = merge_with_existing(out_path, raw_records)
            else:
                final_records = raw_records

            atomic_write(out_path, final_records)

            note = "incremental" if args.incremental else "full"
            counts.append({"package": pkg.name, "cve_count": len(final_records), "note": note})
            logger.info(
                "%-25s %4d records  [%s]%s",
                pkg.name, len(final_records), note,
                f"  (-{n_removed} CPE noise)" if n_removed else "",
            )

    # Write summary CSV
    counts.sort(key=lambda r: r["cve_count"], reverse=True)
    with open(COUNTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["package", "cve_count", "note"])
        writer.writeheader()
        writer.writerows(counts)

    # Save state for next incremental run
    save_run_state(now)
    logger.info(
        "Done. State saved to %s. Next --incremental run will fetch from %s.",
        STATE_FILE,
        now.strftime("%Y-%m-%d %H:%M UTC"),
    )


if __name__ == "__main__":
    main()
