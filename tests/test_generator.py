"""Unit tests for app/generator.py — generators and confidence scoring."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.aggregator import AggregatedData
from app.errors import GenerationError
from app.generator import (
    _build_generation_context,
    _parse_json_response,
    compute_confidence,
    generate_ar,
    generate_en,
)
from app.llm import reset_model_cooldowns


@pytest.fixture(autouse=True)
def _reset_llm_state():
    reset_model_cooldowns()
    yield
    reset_model_cooldowns()


# ---------------------------------------------------------------------------
# Helpers: mock LLM client
# ---------------------------------------------------------------------------


def _make_llm_client(response_text: str):
    """Return a mock llm_client whose chat.completions.create returns *response_text*."""
    choice = SimpleNamespace(message=SimpleNamespace(content=response_text))
    result = SimpleNamespace(choices=[choice])
    client = MagicMock()
    client.chat.completions.create.return_value = result
    return client


def _make_failing_client(exc: Exception):
    client = MagicMock()
    client.chat.completions.create.side_effect = exc
    return client


# ---------------------------------------------------------------------------
# _build_generation_context
# ---------------------------------------------------------------------------


class TestBuildGenerationContext:
    def test_includes_pros_cons_themes_reviews(self):
        agg = AggregatedData(
            ranked_pros=[{"text": "Great battery", "frequency": 10}],
            ranked_cons=[{"text": "Heavy", "frequency": 5}],
            theme_frequencies={"battery": 10, "weight": 5},
            low_confidence=False,
        )
        ctx = _build_generation_context(agg, ["Good product", "Bad weight"])
        assert "Great battery" in ctx
        assert "Heavy" in ctx
        assert "battery" in ctx
        assert "Good product" in ctx

    def test_empty_aggregated_data(self):
        agg = AggregatedData()
        ctx = _build_generation_context(agg, [])
        assert ctx == ""


# ---------------------------------------------------------------------------
# _parse_json_response
# ---------------------------------------------------------------------------


class TestParseJsonResponse:
    def test_plain_json(self):
        data = _parse_json_response('{"key": "value"}')
        assert data == {"key": "value"}

    def test_code_fenced_json(self):
        raw = '```json\n{"key": "value"}\n```'
        data = _parse_json_response(raw)
        assert data == {"key": "value"}

    def test_json_with_extra_text(self):
        raw = 'Sure, here is the JSON:\n{"key": "value"}'
        data = _parse_json_response(raw)
        assert data == {"key": "value"}

    def test_repairs_missing_comma_between_fields(self):
        raw = '{\n  "summary_en": "Good"\n  "verdict_en": "Recommended",\n  "pros": [],\n  "cons": []\n}'
        data = _parse_json_response(raw)
        assert data["summary_en"] == "Good"
        assert data["verdict_en"] == "Recommended"

    def test_repairs_trailing_comma(self):
        data = _parse_json_response('{"key": "value",}')
        assert data == {"key": "value"}

    def test_invalid_json_raises(self):
        with pytest.raises(GenerationError, match="invalid JSON"):
            _parse_json_response("not json at all")

    def test_non_dict_raises(self):
        with pytest.raises(GenerationError, match="not a JSON object"):
            _parse_json_response("[1, 2, 3]")


# ---------------------------------------------------------------------------
# generate_en
# ---------------------------------------------------------------------------


class TestGenerateEn:
    def test_success(self):
        payload = {
            "summary_en": "Great product overall.",
            "verdict_en": "Recommended.",
            "pros": ["battery life"],
            "cons": ["heavy"],
        }
        client = _make_llm_client(json.dumps(payload))
        agg = AggregatedData()
        result = generate_en(agg, ["review 1"], client)
        assert result == payload
        _, kwargs = client.chat.completions.create.call_args
        assert kwargs["response_format"] == {"type": "json_object"}

    def test_missing_keys_raises(self):
        client = _make_llm_client(json.dumps({"summary_en": "hi"}))
        with pytest.raises(GenerationError, match="missing keys"):
            generate_en(AggregatedData(), [], client)

    def test_api_failure_raises(self):
        client = _make_failing_client(RuntimeError("timeout"))
        with pytest.raises(GenerationError, match="generate_en LLM call failed"):
            generate_en(AggregatedData(), [], client)

    def test_empty_response_raises(self):
        client = _make_llm_client("")
        # Empty content after strip → empty string is falsy
        choice = SimpleNamespace(message=SimpleNamespace(content=""))
        client.chat.completions.create.return_value = SimpleNamespace(choices=[choice])
        with pytest.raises(GenerationError, match="empty response"):
            generate_en(AggregatedData(), [], client)

    def test_code_fenced_response(self):
        payload = {
            "summary_en": "Nice.",
            "verdict_en": "Buy it.",
            "pros": [],
            "cons": [],
        }
        fenced = f"```json\n{json.dumps(payload)}\n```"
        client = _make_llm_client(fenced)
        result = generate_en(AggregatedData(), [], client)
        assert result == payload


# ---------------------------------------------------------------------------
# generate_ar
# ---------------------------------------------------------------------------


class TestGenerateAr:
    def test_success(self):
        payload = {"summary_ar": "منتج رائع", "verdict_ar": "موصى به"}
        client = _make_llm_client(json.dumps(payload))
        result = generate_ar(AggregatedData(), ["مراجعة"], client)
        assert result == payload

    def test_missing_keys_raises(self):
        client = _make_llm_client(json.dumps({"summary_ar": "ملخص"}))
        with pytest.raises(GenerationError, match="missing keys"):
            generate_ar(AggregatedData(), [], client)

    def test_api_failure_raises(self):
        client = _make_failing_client(RuntimeError("network"))
        with pytest.raises(GenerationError, match="generate_ar LLM call failed"):
            generate_ar(AggregatedData(), [], client)


# ---------------------------------------------------------------------------
# compute_confidence
# ---------------------------------------------------------------------------


class TestComputeConfidence:
    def test_high_confidence(self):
        score, reason = compute_confidence(
            review_count=20,
            sentiment={"positive": 0.8, "negative": 0.1, "neutral": 0.1},
            theme_count=5,
            low_confidence_flag=False,
        )
        assert 0.5 <= score <= 1.0
        assert reason is None

    def test_low_review_count_caps_below_half(self):
        score, reason = compute_confidence(
            review_count=3,
            sentiment={"positive": 0.9, "negative": 0.05, "neutral": 0.05},
            theme_count=10,
            low_confidence_flag=False,
        )
        assert score < 0.5
        assert reason is not None
        assert "3 reviews" in reason

    def test_low_confidence_flag_caps_below_half(self):
        score, reason = compute_confidence(
            review_count=50,
            sentiment={"positive": 0.9, "negative": 0.05, "neutral": 0.05},
            theme_count=10,
            low_confidence_flag=True,
        )
        assert score < 0.5
        assert reason is not None
        assert "theme diversity" in reason.lower()

    def test_never_1_unless_unanimous_and_enough_themes(self):
        # Even with perfect sentiment but few themes → not 1.0
        score, _ = compute_confidence(
            review_count=100,
            sentiment={"positive": 1.0},
            theme_count=3,
            low_confidence_flag=False,
        )
        assert score < 1.0

    def test_can_reach_1_with_unanimous_and_enough_themes(self):
        score, reason = compute_confidence(
            review_count=100,
            sentiment={"positive": 1.0},
            theme_count=5,
            low_confidence_flag=False,
        )
        assert score == 1.0
        assert reason is None

    def test_zero_reviews(self):
        score, reason = compute_confidence(
            review_count=0,
            sentiment={},
            theme_count=0,
            low_confidence_flag=True,
        )
        assert score < 0.5
        assert reason is not None

    def test_score_always_in_bounds(self):
        score, _ = compute_confidence(
            review_count=1000,
            sentiment={"positive": 0.5, "negative": 0.5},
            theme_count=100,
            low_confidence_flag=False,
        )
        assert 0.0 <= score <= 1.0

    def test_uncertainty_reason_none_when_above_half(self):
        score, reason = compute_confidence(
            review_count=15,
            sentiment={"positive": 0.7, "negative": 0.2, "neutral": 0.1},
            theme_count=4,
            low_confidence_flag=False,
        )
        assert score >= 0.5
        assert reason is None
