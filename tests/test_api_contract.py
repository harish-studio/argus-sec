"""
API contract tests against a running argus-sec instance.

Skipped automatically when the API is not reachable, so the unit suite
still runs in CI without a stack. Start the API first:

    uvicorn app.main:app --port 8001

Run only these:  pytest -m integration
Skip these:      pytest -m "not integration"

These assert the response *contract* that scripts/evaluate.py depends on
(answer, chunks[].narrative_text, confidence, citations) and the grounding
guarantees the tool claims. They do not assert answer quality — that is
what the RAGAS harness measures.
"""

from __future__ import annotations

import re

import httpx
import pytest

pytestmark = pytest.mark.integration

CVE_PATTERN = re.compile(r"^CVE-\d{4}-\d{4,}$")


@pytest.fixture(scope="module")
def client(api_base):
    try:
        httpx.get(f"{api_base}/health", timeout=5.0).raise_for_status()
    except Exception as exc:
        pytest.skip(f"argus-sec API not reachable at {api_base}: {exc}")
    with httpx.Client(base_url=api_base, timeout=120.0) as c:
        yield c


@pytest.fixture(scope="module")
def query_response(client) -> dict:
    """One real query, reused across assertions to keep API cost down."""
    resp = client.post("/query", json={"query": "arbitrary file write in MLflow", "k": 5})
    resp.raise_for_status()
    return resp.json()


class TestHealth:
    def test_health_returns_ok(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["collection"]


class TestQueryContract:
    def test_required_fields_present(self, query_response):
        """These are exactly the keys scripts/evaluate.py reads."""
        for key in ("answer", "chunks", "confidence", "citations", "model_used", "fallback"):
            assert key in query_response, f"missing {key}"

    def test_confidence_is_a_known_tier(self, query_response):
        assert query_response["confidence"] in {"high", "medium", "low"}

    def test_chunks_carry_narrative_text(self, query_response):
        """evaluate.py builds RAGAS contexts from chunks[].narrative_text."""
        assert query_response["chunks"]
        assert all(c.get("narrative_text") for c in query_response["chunks"])

    def test_k_is_respected(self, client):
        body = client.post("/query", json={"query": "SQL injection", "k": 3}).json()
        assert len(body["chunks"]) <= 3

    def test_citations_come_from_retrieved_chunks(self, query_response):
        """
        Citations are extracted from retrieved CVE IDs, not from LLM output,
        so no citation may reference a CVE that was not retrieved.
        """
        retrieved = {c["cve_id"] for c in query_response["chunks"]}
        assert set(query_response["citations"]) <= retrieved

    def test_citations_are_well_formed_cve_ids(self, query_response):
        for cve in query_response["citations"]:
            assert CVE_PATTERN.match(cve), f"malformed CVE id: {cve}"


class TestGrounding:
    def test_out_of_corpus_query_does_not_invent_an_answer(self, client):
        """
        The corpus holds AI/ML supply-chain CVEs. A query about an unrelated
        product must not produce a confident fabricated CVE.
        """
        body = client.post(
            "/query",
            json={"query": "buffer overflow in the Sega Dreamcast BIOS", "k": 5},
        ).json()
        answer = body["answer"].lower()
        refused = (
            body["confidence"] == "low"
            or "not" in answer
            or "no " in answer
            or "insufficient" in answer
        )
        assert refused, f"model answered an out-of-corpus query: {body['answer'][:200]}"


class TestFilters:
    def test_severity_filter_constrains_results(self, client):
        body = client.post(
            "/query",
            json={"query": "remote code execution", "k": 5, "severity": "CRITICAL"},
        ).json()
        severities = {c["severity"] for c in body["chunks"]}
        assert severities <= {"CRITICAL"}, f"filter leaked: {severities}"

    def test_fix_available_false_is_honoured(self, client):
        """Regression guard for falsy booleans being dropped from filters."""
        body = client.post(
            "/query",
            json={"query": "vulnerabilities", "k": 5, "fix_available": False},
        ).json()
        assert all(c["fix_available"] is False for c in body["chunks"])

    def test_unmatchable_filter_returns_no_chunks(self, client):
        body = client.post(
            "/query",
            json={"query": "anything", "k": 5, "package": "ThisPackageDoesNotExist"},
        ).json()
        assert body["chunks"] == []


class TestSearchEndpoint:
    def test_search_returns_chunks_without_synthesis(self, client):
        body = client.post("/search", json={"query": "directory traversal", "k": 3}).json()
        assert "answer" not in body
        assert "chunks" in body
        assert len(body["chunks"]) <= 3


class TestValidation:
    @pytest.mark.parametrize("payload", [
        {"query": "ab"},
        {"query": "x" * 501},
        {"query": "valid query", "k": 0},
        {"query": "valid query", "k": 21},
    ])
    def test_invalid_requests_rejected_with_422(self, client, payload):
        assert client.post("/query", json=payload).status_code == 422
