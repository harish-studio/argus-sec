"""
Schema tests for data/golden_questions.json.

scripts/evaluate.py indexes question["id"], ["type"], ["question"] and
question.get("filter_hint") without guarding. A malformed golden set
therefore fails partway through a paid evaluation run rather than up
front. These tests are the cheap version of that failure.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

GOLDEN_PATH = Path(__file__).resolve().parents[2] / "data" / "golden_questions.json"

# Mirrors the keys evaluate.call_query_endpoint knows how to forward.
KNOWN_HINT_KEYS = {"severity", "package", "fix_available", "attack_vector"}
VALID_SEVERITIES = {"CRITICAL", "HIGH", "MEDIUM", "LOW", "UNKNOWN"}
VALID_VECTORS = {"NETWORK", "ADJACENT", "LOCAL", "PHYSICAL"}


@pytest.fixture(scope="module")
def questions() -> list[dict]:
    if not GOLDEN_PATH.exists():
        pytest.skip(f"{GOLDEN_PATH} not present — run generate_golden_questions.py")
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


class TestGoldenSet:
    def test_is_a_non_empty_list(self, questions):
        assert isinstance(questions, list)
        assert len(questions) > 0

    def test_every_question_has_required_keys(self, questions):
        for q in questions:
            missing = {"id", "type", "question"} - q.keys()
            assert not missing, f"question {q.get('id', '?')} missing {missing}"

    def test_ids_are_unique(self, questions):
        ids = [q["id"] for q in questions]
        assert len(ids) == len(set(ids))

    def test_question_text_is_within_api_limits(self, questions):
        """QueryRequest enforces 3..500 characters."""
        for q in questions:
            assert 3 <= len(q["question"]) <= 500, f"Q{q['id']} out of range"

    def test_filter_hints_use_known_keys(self, questions):
        for q in questions:
            unknown = set(q.get("filter_hint", {})) - KNOWN_HINT_KEYS
            assert not unknown, f"Q{q['id']} has hints evaluate.py ignores: {unknown}"

    def test_severity_hints_are_valid(self, questions):
        for q in questions:
            sev = q.get("filter_hint", {}).get("severity")
            if sev:
                assert sev.upper() in VALID_SEVERITIES, f"Q{q['id']}: {sev}"

    def test_attack_vector_hints_are_valid(self, questions):
        for q in questions:
            av = q.get("filter_hint", {}).get("attack_vector")
            if av:
                assert av.upper() in VALID_VECTORS, f"Q{q['id']}: {av}"

    def test_every_type_has_more_than_one_question(self, questions):
        """
        eval_summary.txt reports a per-type breakdown. A type with a single
        question produces an average of one sample, which reads as a score
        but carries no signal.
        """
        counts: dict[str, int] = {}
        for q in questions:
            counts[q["type"]] = counts.get(q["type"], 0) + 1
        thin = {t: n for t, n in counts.items() if n < 2}
        assert not thin, f"question types with a single sample: {thin}"
