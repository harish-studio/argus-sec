"""
Unit tests for filter construction.

Covers services.retrieval._build_filter (dict -> Qdrant Filter) and
app.main._build_filters (request fields -> dict).

Filters are applied as Qdrant pre-filters, so a dropped or malformed
condition silently widens the search rather than raising. These tests
exist to catch that class of silent failure.
"""

from __future__ import annotations

import pytest

from services.retrieval import _build_filter


def _keys(f) -> list[str]:
    return [c.key for c in f.must]


def _values(f) -> dict:
    return {c.key: c.match.value for c in f.must}


class TestBuildFilter:
    def test_none_input_returns_none(self):
        assert _build_filter(None) is None

    def test_empty_dict_returns_none(self):
        assert _build_filter({}) is None

    def test_unrecognised_keys_only_returns_none(self):
        assert _build_filter({"colour": "blue", "nonsense": 1}) is None

    def test_severity_is_uppercased(self):
        f = _build_filter({"severity": "critical"})
        assert _values(f) == {"severity": "CRITICAL"}

    def test_attack_vector_is_uppercased(self):
        f = _build_filter({"attack_vector": "network"})
        assert _values(f) == {"attack_vector": "NETWORK"}

    def test_package_is_not_uppercased(self):
        """Package names are case-sensitive in the payload array."""
        f = _build_filter({"package": "LangChain"})
        assert _values(f) == {"packages": "LangChain"}

    @pytest.mark.parametrize("field", ["cisa_kev", "fix_available", "multi_package"])
    def test_false_booleans_are_not_dropped(self, field):
        """
        Regression guard: `if filters.get(x)` would silently discard
        False. "CVEs with no fix available" is a real query, so
        fix_available=False must produce a condition.
        """
        f = _build_filter({field: False})
        assert f is not None, f"{field}=False was dropped"
        assert _values(f) == {field: False}

    @pytest.mark.parametrize("field", ["cisa_kev", "fix_available", "multi_package"])
    def test_true_booleans_are_kept(self, field):
        f = _build_filter({field: True})
        assert _values(f) == {field: True}

    def test_multiple_conditions_combined_with_must(self):
        f = _build_filter({
            "severity": "HIGH",
            "package": "LiteLLM",
            "fix_available": False,
        })
        assert len(f.must) == 3
        assert set(_keys(f)) == {"severity", "packages", "fix_available"}

    def test_empty_string_severity_is_ignored(self):
        assert _build_filter({"severity": ""}) is None


class TestBuildFiltersFromRequest:
    """app.main._build_filters maps request fields onto the dict above."""

    @staticmethod
    def _call(**kwargs):
        from app.main import _build_filters
        defaults = dict(
            severity=None, package=None, cisa_kev=None,
            fix_available=None, attack_vector=None, multi_package=None,
        )
        defaults.update(kwargs)
        return _build_filters(**defaults)

    def test_all_none_returns_none(self):
        assert self._call() is None

    def test_false_boolean_produces_a_filter(self):
        """None means 'unset'; False means 'must be false'."""
        assert self._call(fix_available=False) == {"fix_available": False}

    def test_severity_uppercased(self):
        assert self._call(severity="high") == {"severity": "HIGH"}

    def test_combined_fields(self):
        got = self._call(severity="CRITICAL", package="Grafana", cisa_kev=True)
        assert got == {"severity": "CRITICAL", "package": "Grafana", "cisa_kev": True}
