"""
Unit tests for request validation and chunk-to-response mapping.

No network, no models — pure Pydantic and dict shuffling.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.main import QueryRequest, _build_context, _chunk_to_out
from services.retrieval import RetrievedChunk


class TestQueryRequest:
    def test_minimal_valid_request(self):
        req = QueryRequest(query="What affects LangChain?")
        assert req.k == 5
        assert req.severity is None

    @pytest.mark.parametrize("query", ["", "ab"])
    def test_query_shorter_than_three_chars_rejected(self, query):
        with pytest.raises(ValidationError):
            QueryRequest(query=query)

    def test_query_longer_than_500_chars_rejected(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="x" * 501)

    @pytest.mark.parametrize("k", [0, -1, 21, 100])
    def test_k_outside_1_to_20_rejected(self, k):
        with pytest.raises(ValidationError):
            QueryRequest(query="valid query", k=k)

    @pytest.mark.parametrize("k", [1, 5, 20])
    def test_k_within_range_accepted(self, k):
        assert QueryRequest(query="valid query", k=k).k == k

    def test_severity_is_not_validated_against_an_enum(self):
        """
        Current behaviour: severity is a free-text str, so a typo passes
        validation and reaches Qdrant as a filter that matches nothing —
        an empty result rather than a 422. Change to an Enum if that
        silent-empty behaviour is not wanted.
        """
        req = QueryRequest(query="valid query", severity="CRTICAL")
        assert req.severity == "CRTICAL"


class TestChunkToOut:
    def _chunk(self, payload: dict) -> RetrievedChunk:
        return RetrievedChunk(
            cve_id=payload["cve_id"],
            narrative_text=payload["narrative_text"],
            dense_score=0.123456,
            rerank_score=0.987654,
            payload=payload,
        )

    def test_maps_payload_fields(self, sample_payload):
        out = _chunk_to_out(self._chunk(sample_payload))
        assert out.cve_id == "CVE-2026-0545"
        assert out.severity == "CRITICAL"
        assert out.packages == ["MLflow"]
        assert out.fix_available is True
        assert out.attack_vector == "NETWORK"

    def test_scores_rounded_to_four_places(self, sample_payload):
        out = _chunk_to_out(self._chunk(sample_payload))
        assert out.rerank_score == 0.9877
        assert out.dense_score == 0.1235

    def test_missing_optional_fields_get_defaults(self):
        payload = {"cve_id": "CVE-2026-1", "narrative_text": "text"}
        out = _chunk_to_out(self._chunk(payload))
        assert out.severity is None
        assert out.packages == []
        assert out.cisa_kev is False
        assert out.nvd_url == ""


class TestBuildContext:
    def test_numbers_records_from_one(self, sample_payload):
        chunks = [
            RetrievedChunk("CVE-1", "first record", 0.9, 0.9, {}),
            RetrievedChunk("CVE-2", "second record", 0.8, 0.8, {}),
        ]
        context = _build_context(chunks)
        assert "--- CVE Record 1 ---" in context
        assert "--- CVE Record 2 ---" in context
        assert context.index("first record") < context.index("second record")

    def test_empty_chunks_produce_empty_context(self):
        assert _build_context([]) == ""
