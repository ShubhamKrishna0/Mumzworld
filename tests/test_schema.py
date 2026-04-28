"""Tests for validate_and_retry in app/schema.py."""

from __future__ import annotations

import pytest

from app.errors import ValidationError
from app.schema import MomsVerdict, validate_and_retry


def _valid_verdict_dict() -> dict:
    """Return a minimal valid MomsVerdict dict."""
    return {
        "summary_en": "Great product overall.",
        "summary_ar": "منتج رائع بشكل عام.",
        "verdict_en": "Recommended.",
        "verdict_ar": "موصى به.",
        "pros": ["Good quality"],
        "cons": ["Pricey"],
        "confidence": 0.8,
        "uncertainty_reason": None,
    }


class TestValidateAndRetrySuccess:
    """Cases where validation succeeds."""

    def test_succeeds_on_first_attempt(self):
        result = validate_and_retry(lambda: _valid_verdict_dict())
        assert isinstance(result, MomsVerdict)
        assert result.summary_en == "Great product overall."

    def test_succeeds_after_one_retry(self):
        calls = {"count": 0}

        def gen():
            calls["count"] += 1
            if calls["count"] == 1:
                return {"bad": "data"}  # will fail validation
            return _valid_verdict_dict()

        result = validate_and_retry(gen, max_retries=2)
        assert isinstance(result, MomsVerdict)
        assert calls["count"] == 2

    def test_succeeds_on_last_retry(self):
        calls = {"count": 0}

        def gen():
            calls["count"] += 1
            if calls["count"] <= 2:
                return {}  # fail first two attempts
            return _valid_verdict_dict()

        result = validate_and_retry(gen, max_retries=2)
        assert isinstance(result, MomsVerdict)
        assert calls["count"] == 3  # 1 initial + 2 retries


class TestValidateAndRetryFailure:
    """Cases where all retries are exhausted."""

    def test_raises_validation_error_after_retries(self):
        with pytest.raises(ValidationError, match="Schema validation failed"):
            validate_and_retry(lambda: {"bad": "data"}, max_retries=2)

    def test_total_attempts_equals_one_plus_max_retries(self):
        calls = {"count": 0}

        def gen():
            calls["count"] += 1
            return {}

        with pytest.raises(ValidationError):
            validate_and_retry(gen, max_retries=3)

        assert calls["count"] == 4  # 1 + 3

    def test_error_message_includes_last_error_details(self):
        def gen():
            return {"confidence": 2.0}  # out of range

        with pytest.raises(ValidationError) as exc_info:
            validate_and_retry(gen, max_retries=1)

        assert "Last error" in str(exc_info.value)

    def test_zero_retries_means_single_attempt(self):
        calls = {"count": 0}

        def gen():
            calls["count"] += 1
            return {}

        with pytest.raises(ValidationError):
            validate_and_retry(gen, max_retries=0)

        assert calls["count"] == 1

    def test_raises_custom_validation_error_not_pydantic(self):
        """Ensure we raise app.errors.ValidationError, not pydantic's."""
        with pytest.raises(ValidationError):
            validate_and_retry(lambda: {}, max_retries=0)

        # Verify it's our custom class, not pydantic's
        from pydantic import ValidationError as PydanticVE

        with pytest.raises(ValidationError) as exc_info:
            validate_and_retry(lambda: {}, max_retries=0)

        assert not isinstance(exc_info.value, PydanticVE)
