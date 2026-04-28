"""Unit tests for app.feature_extractor."""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.errors import GenerationError
from app.feature_extractor import (
    FEATURE_EXTRACTION_PROMPT,
    ExtractedFeatures,
    _build_user_message,
    _parse_llm_response,
    extract_features,
)
from app.llm import reset_model_cooldowns


@pytest.fixture(autouse=True)
def _reset_llm_state():
    reset_model_cooldowns()
    yield
    reset_model_cooldowns()


# ---------------------------------------------------------------------------
# Helpers – fake OpenAI-compatible response objects
# ---------------------------------------------------------------------------

def _make_llm_response(content: str):
    """Build a minimal object that mimics openai ChatCompletion."""
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


VALID_JSON = json.dumps(
    {
        "sentiment": {"positive": 0.6, "negative": 0.3, "neutral": 0.1},
        "themes": [
            {"theme": "battery life", "mentions": ["Great battery life"]}
        ],
        "complaints": [
            {"complaint": "Screen scratches", "sources": ["The screen scratches easily"]}
        ],
        "praises": [
            {"praise": "Excellent camera", "sources": ["Camera is amazing"]}
        ],
    }
)


# ---------------------------------------------------------------------------
# Tests for _build_user_message
# ---------------------------------------------------------------------------

class TestBuildUserMessage:
    def test_formats_reviews_as_numbered_list(self):
        msg = _build_user_message(["Good phone", "Bad battery"])
        assert "1. Good phone" in msg
        assert "2. Bad battery" in msg

    def test_single_review(self):
        msg = _build_user_message(["Only review"])
        assert "1. Only review" in msg


# ---------------------------------------------------------------------------
# Tests for _parse_llm_response
# ---------------------------------------------------------------------------

class TestParseLlmResponse:
    def test_valid_json(self):
        result = _parse_llm_response(VALID_JSON)
        assert isinstance(result, ExtractedFeatures)
        assert result.sentiment["positive"] == 0.6
        assert len(result.themes) == 1
        assert len(result.complaints) == 1
        assert len(result.praises) == 1

    def test_json_wrapped_in_code_fence(self):
        wrapped = f"```json\n{VALID_JSON}\n```"
        result = _parse_llm_response(wrapped)
        assert isinstance(result, ExtractedFeatures)
        assert result.sentiment["positive"] == 0.6

    def test_repairs_missing_comma_between_fields(self):
        raw = """{
  "sentiment": {"positive": 0.6, "negative": 0.3, "neutral": 0.1}
  "themes": [],
  "complaints": [],
  "praises": []
}"""
        result = _parse_llm_response(raw)
        assert isinstance(result, ExtractedFeatures)
        assert result.sentiment["positive"] == 0.6

    def test_invalid_json_raises_generation_error(self):
        with pytest.raises(GenerationError, match="invalid JSON"):
            _parse_llm_response("not json at all")

    def test_non_object_json_raises_generation_error(self):
        with pytest.raises(GenerationError, match="not a JSON object"):
            _parse_llm_response("[1, 2, 3]")

    def test_missing_keys_raises_generation_error(self):
        partial = json.dumps({"sentiment": {"positive": 1.0}})
        with pytest.raises(GenerationError, match="missing keys"):
            _parse_llm_response(partial)


# ---------------------------------------------------------------------------
# Tests for extract_features
# ---------------------------------------------------------------------------

class TestExtractFeatures:
    def test_successful_extraction(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _make_llm_response(VALID_JSON)

        result = extract_features(["Great battery life", "Camera is amazing"], client)

        assert isinstance(result, ExtractedFeatures)
        assert result.sentiment["positive"] == 0.6
        assert result.themes[0]["theme"] == "battery life"
        client.chat.completions.create.assert_called_once()
        _, kwargs = client.chat.completions.create.call_args
        assert kwargs["response_format"] == {"type": "json_object"}

    def test_llm_api_failure_raises_generation_error(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("API down")

        with pytest.raises(GenerationError, match="feature_extractor LLM call failed"):
            extract_features(["review"], client)

    def test_empty_response_raises_generation_error(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _make_llm_response("")

        with pytest.raises(GenerationError, match="empty response"):
            extract_features(["review"], client)

    def test_unparseable_response_raises_generation_error(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _make_llm_response("oops")

        with pytest.raises(GenerationError, match="invalid JSON"):
            extract_features(["review"], client)


# ---------------------------------------------------------------------------
# Tests for prompt content
# ---------------------------------------------------------------------------

class TestPromptContent:
    def test_prompt_requires_source_attribution(self):
        prompt_lower = FEATURE_EXTRACTION_PROMPT.lower()
        assert "source review" in prompt_lower

    def test_prompt_forbids_hallucination(self):
        prompt_lower = FEATURE_EXTRACTION_PROMPT.lower()
        assert "not invent" in prompt_lower or "do not invent" in prompt_lower

    def test_prompt_requires_json_output(self):
        assert "JSON" in FEATURE_EXTRACTION_PROMPT
