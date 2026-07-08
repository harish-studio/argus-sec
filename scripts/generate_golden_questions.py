"""
argus-sec — Golden evaluation set generator.

Uses gpt-5.4-nano to generate 25 diverse, corpus-grounded questions
for RAGAS evaluation. Questions are grounded in real CVE data from
chunks.json to ensure they are answerable by the RAG pipeline.

Output: data/golden_questions.json — review and trim before eval.

Question types generated (across 25 questions):
  1. Exact CVE lookup          — "What does CVE-XXXX-YYYY affect?"
  2. Package-scoped            — "Which [package] CVEs are network-exploitable?"
  3. Severity-filtered         — "Which CRITICAL [package] CVEs have no fix?"
  4. Cross-package             — "Which CVEs affect both X and Y?"
  5. Weakness-type             — "Which CVEs involve SQL injection?"
  6. Exploitation-signal       — "Which CVEs have a PoC exploit?"

Usage:
    python -m scripts.generate_golden_questions

Review output at data/golden_questions.json before running evaluation.
Remove or edit any questions that:
  - Reference CVEs not clearly answerable from the corpus
  - Are too vague (no clear single answer)
  - Duplicate another question's intent
"""

import json
import logging
import os
import random
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHUNKS_PATH   = Path("data/chunks.json")
OUTPUT_PATH   = Path("data/golden_questions.json")
OPENAI_MODEL  = os.environ.get("OPENAI_MODEL", "gpt-5.4-nano")
OPENAI_KEY    = os.environ.get("OPENAI_API_KEY", "")
N_QUESTIONS   = 25
SAMPLE_SIZE   = 50  # chunks shown to GPT for grounding


# ---------------------------------------------------------------------------
# Question type distribution across 25 questions
# ---------------------------------------------------------------------------

QUESTION_TYPES = {
    "exact_cve_lookup":     4,   # What does CVE-X affect?
    "package_scoped":       5,   # Which [package] CVEs are network-exploitable?
    "severity_filtered":    4,   # Which CRITICAL [package] CVEs have no fix?
    "cross_package":        4,   # Which CVEs affect both X and Y?
    "weakness_type":        4,   # Which CVEs involve SQL injection / SSRF / etc.?
    "exploitation_signal":  4,   # Which CVEs have PoC exploit / CISA KEV?
}


SYSTEM_PROMPT = """You are a cybersecurity expert creating an evaluation dataset for a RAG system 
that answers questions about CVE vulnerabilities.

You will be given a sample of CVE records from the corpus. Generate exactly {n} natural language 
questions that:

1. Are DIRECTLY answerable from the provided CVE records
2. Cover these specific types (distribute evenly):
   - exact_cve_lookup: Ask about a specific CVE ID from the records (e.g. "What does CVE-2024-7042 affect?")
   - package_scoped: Ask about vulnerabilities in a specific package (e.g. "Which LangChain CVEs are network-exploitable?")  
   - severity_filtered: Filter by severity AND another attribute (e.g. "Which CRITICAL PostgreSQL CVEs have no fix available?")
   - cross_package: Ask about CVEs affecting multiple packages (e.g. "Which CVEs affect both Anthropic SDK and OpenAI SDK?")
   - weakness_type: Ask about a specific weakness category (e.g. "Which CVEs in the corpus involve SQL injection?")
   - exploitation_signal: Ask about exploitation status (e.g. "Which CVEs have a proof-of-concept exploit available?")

3. Use natural language a security analyst would actually ask
4. Are specific enough to have a clear, verifiable answer
5. Do NOT require knowledge outside the provided CVE records

Return a JSON array of exactly {n} objects, each with:
{{
  "id": 1,
  "question": "natural language question",
  "type": "question_type_from_list_above",
  "expected_cve_ids": ["CVE-XXXX-YYYY"],  // CVE IDs from the provided records that should answer this
  "filter_hint": {{  // optional Qdrant filters that would help retrieve the right CVEs
    "severity": "CRITICAL",  // optional
    "package": "LangChain",  // optional
    "fix_available": false,  // optional
    "attack_vector": "NETWORK"  // optional
  }}
}}

Return ONLY the JSON array, no other text."""


def sample_chunks(chunks: list[dict], n: int) -> list[dict]:
    """
    Sample chunks strategically to give GPT broad coverage:
    - All multi-package CVEs (key for cross-package questions)
    - All CISA KEV CVEs (key for exploitation signal questions)
    - Random sample of the rest
    """
    multi_pkg  = [c for c in chunks if c["metadata"]["multi_package"]]
    kev        = [c for c in chunks if c["metadata"]["cisa_kev"]]
    rest       = [c for c in chunks if not c["metadata"]["multi_package"]
                                    and not c["metadata"]["cisa_kev"]]

    priority   = list({c["cve_id"]: c for c in multi_pkg + kev}.values())
    remaining  = random.sample(rest, min(n - len(priority), len(rest)))
    sample     = priority + remaining

    return sample[:n]


def format_chunk_for_prompt(chunk: dict) -> str:
    m = chunk["metadata"]
    return (
        f"CVE: {chunk['cve_id']} | "
        f"Packages: {', '.join(m['packages'])} | "
        f"Severity: {m['severity']} | "
        f"Attack Vector: {m.get('attack_vector', 'N/A')} | "
        f"Fix Available: {m['fix_available']} | "
        f"CISA KEV: {m['cisa_kev']} | "
        f"SSVC Exploitation: {m.get('exploitation_signal') or 'None'} | "
        f"Multi-package: {m['multi_package']}\n"
        f"Description snippet: {chunk['narrative_text'][chunk['narrative_text'].find('Description:'):chunk['narrative_text'].find('Description:')+200]}\n"
    )


def main() -> None:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"{CHUNKS_PATH} not found — run chunk_cves.py first")

    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    logger.info("Loaded %d chunks", len(chunks))

    sample = sample_chunks(chunks, SAMPLE_SIZE)
    logger.info("Sampled %d chunks for grounding (%d multi-package, %d CISA KEV)",
                len(sample),
                sum(1 for c in sample if c["metadata"]["multi_package"]),
                sum(1 for c in sample if c["metadata"]["cisa_kev"]))

    corpus_summary = "\n".join(format_chunk_for_prompt(c) for c in sample)

    prompt = f"""Here are {len(sample)} CVE records from the corpus:

{corpus_summary}

Generate {N_QUESTIONS} evaluation questions as specified."""

    client = OpenAI(api_key=OPENAI_KEY)

    logger.info("Calling %s to generate %d questions...", OPENAI_MODEL, N_QUESTIONS)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(n=N_QUESTIONS)},
            {"role": "user",   "content": prompt},
        ],
        max_completion_tokens=3000,
        temperature=0.7,   # some variation for diverse question types
    )

    raw = (response.choices[0].message.content or "").strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]

    questions = json.loads(raw)

    # Validate structure
    required_keys = {"id", "question", "type", "expected_cve_ids"}
    valid = []
    for q in questions:
        if required_keys.issubset(q.keys()):
            valid.append(q)
        else:
            logger.warning("Skipping malformed question: %s", q)

    logger.info("Generated %d valid questions (requested %d)", len(valid), N_QUESTIONS)

    # Summary by type
    from collections import Counter
    type_counts = Counter(q["type"] for q in valid)
    for qtype, count in sorted(type_counts.items()):
        logger.info("  %-25s %d", qtype, count)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(valid, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    logger.info("Written to %s — review and trim before running evaluation", OUTPUT_PATH)
    logger.info("Remove questions that: reference unknown CVEs, are too vague, or duplicate intent")


if __name__ == "__main__":
    main()