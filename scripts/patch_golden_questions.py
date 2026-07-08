"""
argus-sec — Golden questions patch script.

Applies 8 recommended edits to data/golden_questions.json:

  Q4  — Rephrase ambiguous question
  Q5  — Replace duplicate MLflow lookup with CrewAI CVE
  Q8  — Narrow to CRITICAL only (too many expected CVEs)
  Q10 — Relabel type from severity_filtered to exploitation_signal
  Q12 — Narrow to Grafana package (too many expected CVEs)
  Q15 — Narrow to HIGH/CRITICAL network-exploitable OpenAI SDK CVEs
  Q16 — Narrow question to network-exploitable Prometheus CVEs
  Q17 — Verify fix_available=False CVEs and narrow question

Run: python -m scripts.patch_golden_questions
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

INPUT_PATH  = Path("data/golden_questions.json")
OUTPUT_PATH = Path("data/golden_questions.json")
BACKUP_PATH = Path("data/golden_questions.json.bak")


def patch(questions: list[dict]) -> list[dict]:
    """Apply all 8 patches. Returns updated list."""

    by_id = {q["id"]: q for q in questions}

    # ------------------------------------------------------------------
    # Q4 — Rephrase ambiguous phrasing
    # ------------------------------------------------------------------
    by_id[4]["question"] = (
        "What vulnerability is described in CVE-2026-35030 "
        "and which packages does it affect?"
    )
    logger.info("Q4 rephrased")

    # ------------------------------------------------------------------
    # Q5 — Replace duplicate MLflow lookup with CrewAI CVE
    # CVE-2026-2275: CrewAI CRITICAL RCE with PoC — under-represented package
    # ------------------------------------------------------------------
    by_id[5]["question"] = (
        "What vulnerability is described in CVE-2026-2275 "
        "and what is its exploitation status?"
    )
    by_id[5]["expected_cve_ids"] = ["CVE-2026-2275"]
    by_id[5]["filter_hint"] = {"package": "CrewAI"}
    logger.info("Q5 replaced with CrewAI CVE-2026-2275")

    # ------------------------------------------------------------------
    # Q8 — Narrow to CRITICAL severity (was returning 6 CVEs, exceeds k=5)
    # ------------------------------------------------------------------
    by_id[8]["question"] = (
        "Which CRITICAL CVEs affect both LiteLLM and the OpenAI SDK?"
    )
    by_id[8]["expected_cve_ids"] = ["CVE-2026-42208", "CVE-2026-42271", "CVE-2026-49468"]
    if "filter_hint" not in by_id[8]:
        by_id[8]["filter_hint"] = {}
    by_id[8]["filter_hint"]["severity"] = "CRITICAL"
    logger.info("Q8 narrowed to CRITICAL severity")

    # ------------------------------------------------------------------
    # Q10 — Relabel type (GitHub Actions + CISA KEV is exploitation_signal,
    # not severity_filtered)
    # ------------------------------------------------------------------
    by_id[10]["type"] = "exploitation_signal"
    logger.info("Q10 relabelled to exploitation_signal")

    # ------------------------------------------------------------------
    # Q12 — Narrow to Grafana package (was 10 expected CVEs, exceeds k=5)
    # Grafana has confirmed active exploitation CVEs in the corpus
    # ------------------------------------------------------------------
    by_id[12]["question"] = (
        "Which CRITICAL network-exploitable Grafana CVEs "
        "have active exploitation confirmed in the wild?"
    )
    by_id[12]["expected_cve_ids"] = ["CVE-2021-43798", "CVE-2021-39226"]
    if "filter_hint" not in by_id[12]:
        by_id[12]["filter_hint"] = {}
    by_id[12]["filter_hint"]["package"] = "Grafana"
    by_id[12]["filter_hint"]["severity"] = "CRITICAL"
    by_id[12]["filter_hint"]["attack_vector"] = "NETWORK"
    logger.info("Q12 narrowed to Grafana CRITICAL network-exploitable")

    # ------------------------------------------------------------------
    # Q15 — Narrow from 14 expected CVEs to HIGH/CRITICAL network-exploitable
    # ------------------------------------------------------------------
    by_id[15]["question"] = (
        "Which HIGH or CRITICAL OpenAI SDK vulnerabilities "
        "in the corpus are network-exploitable?"
    )
    by_id[15]["expected_cve_ids"] = [
        "CVE-2026-31942",
        "CVE-2026-35030",
        "CVE-2026-42208",
        "CVE-2026-42271",
        "CVE-2026-49468",
    ]
    if "filter_hint" not in by_id[15]:
        by_id[15]["filter_hint"] = {}
    by_id[15]["filter_hint"]["package"]       = "OpenAI SDK"
    by_id[15]["filter_hint"]["attack_vector"] = "NETWORK"
    logger.info("Q15 narrowed to HIGH/CRITICAL network-exploitable")

    # ------------------------------------------------------------------
    # Q16 — Add network-exploitable constraint to Prometheus question
    # ------------------------------------------------------------------
    by_id[16]["question"] = (
        "Which Prometheus vulnerabilities in the corpus "
        "are network-exploitable?"
    )
    if "filter_hint" not in by_id[16]:
        by_id[16]["filter_hint"] = {}
    by_id[16]["filter_hint"]["package"]       = "Prometheus"
    by_id[16]["filter_hint"]["attack_vector"] = "NETWORK"
    logger.info("Q16 narrowed to network-exploitable Prometheus CVEs")

    # ------------------------------------------------------------------
    # Q17 — Narrow to confirmed fix_available=False Grafana CVEs
    # Removed CVEs where fix_available status is uncertain
    # ------------------------------------------------------------------
    by_id[17]["question"] = (
        "Which Grafana vulnerabilities in the corpus "
        "do not have a fix available?"
    )
    by_id[17]["expected_cve_ids"] = ["CVE-2024-51988", "CVE-2025-3454"]
    if "filter_hint" not in by_id[17]:
        by_id[17]["filter_hint"] = {}
    by_id[17]["filter_hint"]["package"]       = "Grafana"
    by_id[17]["filter_hint"]["fix_available"] = False
    logger.info("Q17 narrowed to confirmed fix_available=False Grafana CVEs")

    return list(by_id.values())


def main() -> None:
    questions = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    logger.info("Loaded %d questions", len(questions))

    # Backup before patching
    BACKUP_PATH.write_text(
        json.dumps(questions, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    logger.info("Backup written to %s", BACKUP_PATH)

    patched = patch(questions)

    OUTPUT_PATH.write_text(
        json.dumps(patched, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    logger.info("Patched %d questions written to %s", len(patched), OUTPUT_PATH)

    # Print summary of all 25 questions for final review
    logger.info("─" * 60)
    logger.info("Final question set:")
    for q in sorted(patched, key=lambda x: x["id"]):
        logger.info(
            "  %2d. [%-22s] %s",
            q["id"], q["type"], q["question"][:70]
        )


if __name__ == "__main__":
    main()