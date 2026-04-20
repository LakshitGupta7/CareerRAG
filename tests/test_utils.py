"""
CareerRAG — Unit Tests
Covers: ui_utils, config values, and standalone helpers.

Note: rag_pipeline tests are separated because they require ML dependencies
      (fitz, faiss, sentence-transformers) which may not be in every environment.
"""

from __future__ import annotations

import math
import re

import pytest

# These modules are lightweight — no heavy ML dependencies
from ui_utils import parse_sections, extract_match_score, get_validation_badge
from config import (
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RRF_K,
)


# -------------------------------------------------------
# Inline copies of the pure helpers from rag_pipeline.py
# so tests can run without fitz/faiss/sentence-transformers
# -------------------------------------------------------
_WORD_RE = re.compile(r"\b\w+\b")


def tokenize_for_bm25(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def get_confidence_label(chunks: list[dict]) -> str:
    if not chunks:
        return "Low"
    avg_score = sigmoid(sum(c["score"] for c in chunks) / len(chunks))
    if avg_score >= CONFIDENCE_HIGH:
        return "High"
    elif avg_score >= CONFIDENCE_MEDIUM:
        return "Medium"
    else:
        return "Low"


# =====================================================
# parse_sections
# =====================================================
class TestParseSections:
    def test_basic_sections(self):
        text = "## Section One\nContent A\n## Section Two\nContent B"
        result = parse_sections(text)
        assert "Section One" in result
        assert "Section Two" in result
        assert result["Section One"] == "Content A"
        assert result["Section Two"] == "Content B"

    def test_multiline_content(self):
        text = "## Overview\nLine 1\nLine 2\nLine 3\n## Skills\nPython, JS"
        result = parse_sections(text)
        assert "Line 1" in result["Overview"]
        assert "Line 3" in result["Overview"]
        assert result["Skills"] == "Python, JS"

    def test_empty_string(self):
        assert parse_sections("") == {}

    def test_none(self):
        assert parse_sections(None) == {}

    def test_no_sections(self):
        assert parse_sections("Just plain text without headers") == {}

    def test_whitespace_only(self):
        assert parse_sections("   \n\n  ") == {}


# =====================================================
# extract_match_score
# =====================================================
class TestExtractMatchScore:
    def test_colon_format(self):
        assert extract_match_score("Match Score: 78") == 78

    def test_dash_format(self):
        assert extract_match_score("Match Score - 65") == 65

    def test_no_separator(self):
        assert extract_match_score("Match Score 90") == 90

    def test_clamp_over_100(self):
        assert extract_match_score("Match Score: 150") == 100

    def test_zero(self):
        assert extract_match_score("Match Score: 0") == 0

    def test_no_score(self):
        assert extract_match_score("No score here") is None

    def test_none(self):
        assert extract_match_score(None) is None

    def test_empty(self):
        assert extract_match_score("") is None

    def test_case_insensitive(self):
        assert extract_match_score("match score: 42") == 42


# =====================================================
# get_validation_badge
# =====================================================
class TestGetValidationBadge:
    def test_supported_with_section(self):
        text = "## Validation Status\nSupported\n## Supported Claims\n- claim"
        assert get_validation_badge(text) == "Supported"

    def test_partially_supported(self):
        text = "## Validation Status\nPartially Supported"
        assert get_validation_badge(text) == "Partially Supported"

    def test_unsupported(self):
        text = "## Validation Status\nUnsupported"
        assert get_validation_badge(text) == "Unsupported"

    def test_fallback_no_section(self):
        assert get_validation_badge("The answer is partially supported.") == "Partially Supported"

    def test_empty(self):
        assert get_validation_badge("") == "Unknown"

    def test_none(self):
        assert get_validation_badge(None) == "Unknown"

    def test_no_keywords(self):
        assert get_validation_badge("Some random text") == "Unknown"

    def test_unsupported_not_confused_with_partial(self):
        text = "## Validation Status\nPartially Supported\n## Unsupported Claims\n- one"
        assert get_validation_badge(text) == "Partially Supported"


# =====================================================
# tokenize_for_bm25
# =====================================================
class TestTokenizeForBM25:
    def test_basic(self):
        assert tokenize_for_bm25("Hello World") == ["hello", "world"]

    def test_punctuation_stripped(self):
        tokens = tokenize_for_bm25("Python, Java, and C++.")
        assert "python" in tokens
        assert "java" in tokens
        assert "c" in tokens
        assert "," not in tokens

    def test_empty(self):
        assert tokenize_for_bm25("") == []

    def test_mixed_case(self):
        tokens = tokenize_for_bm25("Machine Learning MODEL")
        assert tokens == ["machine", "learning", "model"]


# =====================================================
# sigmoid
# =====================================================
class TestSigmoid:
    def test_zero(self):
        assert sigmoid(0) == 0.5

    def test_large_positive(self):
        assert sigmoid(100) > 0.99

    def test_large_negative(self):
        assert sigmoid(-100) < 0.01

    def test_known_value(self):
        assert abs(sigmoid(1) - 0.7310585) < 1e-5


# =====================================================
# get_confidence_label
# =====================================================
class TestGetConfidenceLabel:
    def test_empty(self):
        assert get_confidence_label([]) == "Low"

    def test_high_scores(self):
        chunks = [{"score": 5.0}, {"score": 6.0}]  # sigmoid(5.5) ~ 0.996
        assert get_confidence_label(chunks) == "High"

    def test_medium_scores(self):
        chunks = [{"score": -0.1}, {"score": 0.1}]  # sigmoid(0.0) = 0.5
        assert get_confidence_label(chunks) == "Medium"

    def test_low_scores(self):
        chunks = [{"score": -5.0}, {"score": -6.0}]  # sigmoid(-5.5) ~ 0.004
        assert get_confidence_label(chunks) == "Low"


# =====================================================
# config sanity checks
# =====================================================
class TestConfig:
    def test_chunk_size_positive(self):
        assert CHUNK_SIZE > 0

    def test_overlap_less_than_size(self):
        assert CHUNK_OVERLAP < CHUNK_SIZE

    def test_rrf_k_positive(self):
        assert RRF_K > 0

    def test_confidence_thresholds_ordered(self):
        assert CONFIDENCE_HIGH > CONFIDENCE_MEDIUM > 0
