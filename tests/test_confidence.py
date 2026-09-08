"""
Unit tests for confidence tiering.

The confidence tier drives the synthesis gate in app.main.query_endpoint:
"low" skips the LLM entirely and returns LOW_CONFIDENCE_MSG. Getting the
boundary wrong therefore either suppresses good answers or lets the model
synthesise over poor context — the hallucination path this design exists
to prevent.

Note on scale: retrieve() applies a sigmoid to the raw cross-encoder score
before calling _score_confidence, so the input here is bounded [0, 1].
The docstring on _score_confidence describes a raw [-10, 10] range; these
tests pin the behaviour that is actually in force.
"""

from __future__ import annotations

import math

import pytest

from services.retrieval import (
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    _score_confidence,
)


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


class TestScoreConfidence:
    @pytest.mark.parametrize("score", [0.7, 0.85, 0.999, 1.0])
    def test_high_band(self, score):
        assert _score_confidence(score) == "high"

    @pytest.mark.parametrize("score", [0.4, 0.5, 0.699])
    def test_medium_band(self, score):
        assert _score_confidence(score) == "medium"

    @pytest.mark.parametrize("score", [0.0, 0.1, 0.399])
    def test_low_band(self, score):
        assert _score_confidence(score) == "low"

    def test_boundaries_are_inclusive(self):
        """Thresholds are >=, so the boundary value takes the higher tier."""
        assert _score_confidence(CONFIDENCE_HIGH) == "high"
        assert _score_confidence(CONFIDENCE_MEDIUM) == "medium"

    def test_tiers_are_monotonic(self):
        order = {"low": 0, "medium": 1, "high": 2}
        scores = [0.0, 0.2, 0.4, 0.6, 0.7, 0.9]
        tiers = [order[_score_confidence(s)] for s in scores]
        assert tiers == sorted(tiers)


class TestSigmoidMapping:
    """
    Documents how raw cross-encoder scores map onto the tiers, so a future
    change to normalisation shows up as a test failure rather than a quiet
    shift in how often synthesis is gated.
    """

    def test_zero_raw_score_is_medium_not_low(self):
        """
        A raw score of 0 carries no relevance signal, but sigmoid(0) = 0.5,
        which lands in "medium" and therefore passes the synthesis gate.
        If the gate should reject no-signal matches, this is the line to change.
        """
        assert _score_confidence(sigmoid(0.0)) == "medium"

    def test_strongly_negative_raw_score_is_low(self):
        assert _score_confidence(sigmoid(-5.0)) == "low"

    def test_high_tier_raw_threshold(self):
        """The 0.7 normalised cut-off corresponds to a raw score of ~0.847."""
        raw_cutoff = math.log(CONFIDENCE_HIGH / (1 - CONFIDENCE_HIGH))
        assert _score_confidence(sigmoid(raw_cutoff + 0.01)) == "high"
        assert _score_confidence(sigmoid(raw_cutoff - 0.01)) == "medium"
