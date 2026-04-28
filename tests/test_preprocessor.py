"""Unit tests for the preprocessor module."""

from __future__ import annotations

import unicodedata

import pytest

from app.errors import InsufficientDataError
from app.preprocessor import PreprocessResult, preprocess


class TestPreprocessBasic:
    """Basic preprocessing behavior."""

    def test_strips_whitespace(self):
        result = preprocess(["  hello world today  ", "\thello again friend\n"])
        assert all(r == r.strip() for r in result.reviews)

    def test_normalizes_unicode_nfc(self):
        # é as combining sequence (NFD) should become single codepoint (NFC)
        nfd_e = "caf\u0065\u0301"  # e + combining acute
        nfc_e = unicodedata.normalize("NFC", nfd_e)
        result = preprocess([nfd_e + " is great food"])
        assert result.reviews[0] == nfc_e + " is great food"

    def test_removes_exact_duplicates(self):
        reviews = ["great product overall", "great product overall", "another good review"]
        result = preprocess(reviews)
        assert len(result.reviews) == 2
        assert result.duplicates_removed >= 1

    def test_discards_short_reviews(self):
        reviews = ["ok", "hi", "this is a valid review with enough words"]
        result = preprocess(reviews)
        assert len(result.reviews) == 1
        assert result.noise_removed == 2

    def test_raises_insufficient_data_when_all_discarded(self):
        with pytest.raises(InsufficientDataError):
            preprocess(["ok", "hi", "no"])

    def test_preserves_order_after_dedup(self):
        reviews = [
            "first review is here",
            "second review is here",
            "first review is here",
            "third review is here",
        ]
        result = preprocess(reviews)
        texts = result.reviews
        assert texts[0] == "first review is here"
        assert texts[1] == "second review is here"
        assert texts[2] == "third review is here"

    def test_empty_input_raises_insufficient_data(self):
        with pytest.raises(InsufficientDataError):
            preprocess([])

    def test_all_short_reviews_raises_insufficient_data(self):
        with pytest.raises(InsufficientDataError):
            preprocess(["a b", "c d", "e f"])

    def test_single_valid_review(self):
        result = preprocess(["this is a valid review"])
        assert len(result.reviews) == 1
        assert result.duplicates_removed == 0
        assert result.noise_removed == 0

    def test_counts_are_correct(self):
        reviews = [
            "this is a good product",
            "this is a good product",  # exact dup
            "ok",  # noise (< 3 words)
            "another fine review here",
        ]
        result = preprocess(reviews)
        assert result.duplicates_removed >= 1
        assert result.noise_removed >= 1


class TestPreprocessNearDuplicates:
    """Near-duplicate detection via embeddings."""

    def test_near_duplicates_removed(self):
        reviews = [
            "This product is absolutely amazing and wonderful",
            "This product is absolutely amazing and great",  # near-dup
            "The battery life is terrible and drains fast",
        ]
        result = preprocess(reviews)
        # The near-duplicate should be caught; at most 2 unique reviews
        assert len(result.reviews) <= 2
        assert result.duplicates_removed >= 1

    def test_dissimilar_reviews_kept(self):
        reviews = [
            "The camera quality is outstanding for photos",
            "Battery drains way too fast overnight",
            "Customer service was very helpful and responsive",
        ]
        result = preprocess(reviews)
        assert len(result.reviews) == 3
