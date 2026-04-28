"""Unit tests for the pipeline orchestrator."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from app.aggregator import AggregatedData
from app.errors import EmptyInputError, GenerationError, ParseError
from app.feature_extractor import ExtractedFeatures
from app.pipeline import _get_llm_client, _parse_input, run_pipeline
from app.preprocessor import PreprocessResult
from app.retriever import RetrieverResult
from app.schema import MomsVerdict


# ---------------------------------------------------------------------------
# _parse_input routing tests
# ---------------------------------------------------------------------------


class TestParseInput:
    def test_json_content_type(self):
        data = json.dumps(["review one two three"]).encode()
        result = _parse_input(data, "application/json")
        assert result == ["review one two three"]

    def test_json_extension(self):
        data = json.dumps(["review one two three"]).encode()
        result = _parse_input(data, ".json")
        assert result == ["review one two three"]

    def test_text_content_type(self):
        data = b"review one two three\nreview four five six"
        result = _parse_input(data, "text/plain")
        assert result == ["review one two three", "review four five six"]

    def test_text_extension(self):
        data = b"review one two three\nreview four five six"
        result = _parse_input(data, ".txt")
        assert result == ["review one two three", "review four five six"]

    def test_unknown_content_type_valid_json(self):
        data = json.dumps(["review one two three"]).encode()
        result = _parse_input(data, "application/octet-stream")
        assert result == ["review one two three"]

    def test_unknown_content_type_falls_back_to_text(self):
        data = b"review one two three\nreview four five six"
        result = _parse_input(data, "application/octet-stream")
        assert result == ["review one two three", "review four five six"]

    def test_empty_json_raises(self):
        data = json.dumps([]).encode()
        with pytest.raises(EmptyInputError):
            _parse_input(data, "application/json")

    def test_invalid_json_raises(self):
        data = b"{not valid json"
        with pytest.raises(ParseError):
            _parse_input(data, "application/json")


# ---------------------------------------------------------------------------
# _get_llm_client tests
# ---------------------------------------------------------------------------


class TestGetLlmClient:
    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key-123"})
    def test_creates_client_with_env_key(self):
        client = _get_llm_client()
        assert client.api_key == "test-key-123"
        assert "openrouter.ai" in str(client.base_url)

    @patch.dict("os.environ", {}, clear=True)
    def test_creates_client_with_empty_key_when_missing(self):
        client = _get_llm_client()
        assert client.api_key == ""


# ---------------------------------------------------------------------------
# run_pipeline integration test (all stages mocked)
# ---------------------------------------------------------------------------


class TestRunPipeline:
    @patch("app.pipeline._get_llm_client")
    @patch("app.pipeline.validate_and_retry")
    @patch("app.pipeline.aggregate")
    @patch("app.pipeline.extract_features")
    @patch("app.pipeline.retrieve")
    @patch("app.pipeline.preprocess")
    def test_full_pipeline_wiring(
        self,
        mock_preprocess,
        mock_retrieve,
        mock_extract,
        mock_aggregate,
        mock_validate,
        mock_get_client,
    ):
        # Setup mocks
        mock_preprocess.return_value = PreprocessResult(
            reviews=["good product with features", "bad quality overall experience"],
            duplicates_removed=0,
            noise_removed=0,
        )
        mock_retrieve.return_value = RetrieverResult(
            selected_reviews=["good product with features", "bad quality overall experience"],
            embeddings=MagicMock(),
        )
        mock_extract.return_value = ExtractedFeatures(
            sentiment={"positive": 0.6, "negative": 0.3, "neutral": 0.1},
            themes=[{"theme": "quality", "mentions": ["good", "bad"]}],
            complaints=[],
            praises=[],
        )
        mock_aggregate.return_value = AggregatedData(
            ranked_pros=[],
            ranked_cons=[],
            theme_frequencies={"quality": 2},
            low_confidence=True,
        )
        expected_verdict = MomsVerdict(
            summary_en="Good product overall.",
            summary_ar="منتج جيد بشكل عام.",
            verdict_en="Recommended.",
            verdict_ar="موصى به.",
            pros=["quality"],
            cons=[],
            confidence=0.4,
            uncertainty_reason="Low theme diversity",
        )
        mock_validate.return_value = expected_verdict
        mock_get_client.return_value = MagicMock()

        raw_data = json.dumps(
            ["good product with features", "bad quality overall experience"]
        ).encode()

        result = run_pipeline(raw_data, "application/json")

        assert result == expected_verdict
        mock_preprocess.assert_called_once()
        mock_retrieve.assert_called_once()
        mock_extract.assert_called_once()
        mock_aggregate.assert_called_once()
        mock_validate.assert_called_once()

    @patch("app.pipeline._get_llm_client")
    @patch("app.pipeline.validate_and_retry")
    @patch("app.pipeline.aggregate")
    @patch("app.pipeline.extract_features")
    @patch("app.pipeline.retrieve")
    @patch("app.pipeline.preprocess")
    def test_generate_fn_passed_to_validate_produces_merged_dict(
        self,
        mock_preprocess,
        mock_retrieve,
        mock_extract,
        mock_aggregate,
        mock_validate,
        mock_get_client,
    ):
        """Verify the generate_fn closure merges EN + AR + confidence correctly."""
        mock_preprocess.return_value = PreprocessResult(
            reviews=["review one two three"],
            duplicates_removed=0,
            noise_removed=0,
        )
        mock_retrieve.return_value = RetrieverResult(
            selected_reviews=["review one two three"],
            embeddings=MagicMock(),
        )
        mock_extract.return_value = ExtractedFeatures(
            sentiment={"positive": 0.8, "negative": 0.1, "neutral": 0.1},
            themes=[],
            complaints=[],
            praises=[],
        )
        mock_aggregate.return_value = AggregatedData(
            ranked_pros=[],
            ranked_cons=[],
            theme_frequencies={},
            low_confidence=True,
        )
        mock_get_client.return_value = MagicMock()

        # Capture the generate_fn passed to validate_and_retry
        captured_fn = None

        def capture_validate(fn, **kwargs):
            nonlocal captured_fn
            captured_fn = fn
            # Call it once to verify it works
            result = fn()
            return MomsVerdict.model_validate(result)

        mock_validate.side_effect = capture_validate

        with patch("app.pipeline.generate_en") as mock_gen_en, \
             patch("app.pipeline.generate_ar") as mock_gen_ar:
            mock_gen_en.return_value = {
                "summary_en": "Summary",
                "verdict_en": "Verdict",
                "pros": ["pro1"],
                "cons": ["con1"],
            }
            mock_gen_ar.return_value = {
                "summary_ar": "ملخص",
                "verdict_ar": "حكم",
            }

            raw_data = json.dumps(["review one two three"]).encode()
            result = run_pipeline(raw_data, "application/json")

            assert result.summary_en == "Summary"
            assert result.summary_ar == "ملخص"
            assert result.verdict_en == "Verdict"
            assert result.verdict_ar == "حكم"
            assert result.pros == ["pro1"]
            assert result.cons == ["con1"]
            assert isinstance(result.confidence, float)

    def test_empty_input_propagates_error(self):
        raw_data = json.dumps([]).encode()
        with pytest.raises(EmptyInputError):
            run_pipeline(raw_data, "application/json")

    def test_invalid_json_propagates_error(self):
        raw_data = b"not valid json at all {"
        with pytest.raises(ParseError):
            run_pipeline(raw_data, "application/json")

    @patch("app.pipeline._get_llm_client")
    @patch("app.pipeline.retrieve")
    @patch("app.pipeline.preprocess")
    def test_feature_extraction_failure_uses_fallback(
        self,
        mock_preprocess,
        mock_retrieve,
        mock_get_client,
    ):
        mock_preprocess.return_value = PreprocessResult(
            reviews=[
                "Great quality and very easy to use",
                "Good quality and worth the price",
            ],
            duplicates_removed=0,
            noise_removed=0,
        )
        mock_retrieve.return_value = RetrieverResult(
            selected_reviews=[
                "Great quality and very easy to use",
                "Good quality and worth the price",
            ],
            embeddings=MagicMock(),
        )
        mock_get_client.return_value = MagicMock()

        with patch("app.pipeline.extract_features") as mock_extract, \
             patch("app.pipeline.generate_en") as mock_gen_en, \
             patch("app.pipeline.generate_ar") as mock_gen_ar:
            mock_extract.side_effect = GenerationError("all models rate limited")
            mock_gen_en.return_value = {
                "summary_en": "Summary",
                "verdict_en": "Verdict",
                "pros": ["quality"],
                "cons": [],
            }
            mock_gen_ar.return_value = {
                "summary_ar": "ملخص",
                "verdict_ar": "حكم",
            }

            raw_data = json.dumps(["review one two three"]).encode()
            result = run_pipeline(raw_data, "application/json")

        assert result.confidence < 0.5
        assert result.uncertainty_reason is not None
        assert "feature extraction unavailable" in result.uncertainty_reason

    @patch("app.pipeline._get_llm_client")
    @patch("app.pipeline.extract_features")
    @patch("app.pipeline.retrieve")
    @patch("app.pipeline.preprocess")
    def test_verdict_generation_failure_uses_fallback(
        self,
        mock_preprocess,
        mock_retrieve,
        mock_extract,
        mock_get_client,
    ):
        mock_preprocess.return_value = PreprocessResult(
            reviews=[
                "Great quality and very easy to use",
                "Good quality and easy setup",
            ],
            duplicates_removed=0,
            noise_removed=0,
        )
        mock_retrieve.return_value = RetrieverResult(
            selected_reviews=[
                "Great quality and very easy to use",
                "Good quality and easy setup",
            ],
            embeddings=MagicMock(),
        )
        mock_extract.return_value = ExtractedFeatures(
            sentiment={"positive": 1.0, "negative": 0.0, "neutral": 0.0},
            themes=[
                {"theme": "quality", "mentions": ["Great quality", "Good quality"]},
                {"theme": "ease of use", "mentions": ["easy to use", "easy setup"]},
            ],
            complaints=[],
            praises=[
                {"praise": "quality", "sources": ["Great quality", "Good quality"]},
                {"praise": "ease of use", "sources": ["easy to use", "easy setup"]},
            ],
        )
        mock_get_client.return_value = MagicMock()

        with patch("app.pipeline.generate_en") as mock_gen_en:
            mock_gen_en.side_effect = GenerationError("all models rate limited")

            raw_data = json.dumps(["review one two three"]).encode()
            result = run_pipeline(raw_data, "application/json")

        assert result.summary_en
        assert result.summary_ar
        assert result.confidence < 0.5
        assert result.uncertainty_reason is not None
        assert "verdict generation unavailable" in result.uncertainty_reason
