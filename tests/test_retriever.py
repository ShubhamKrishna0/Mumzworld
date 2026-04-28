"""Unit tests for the retriever module."""

import numpy as np
import pytest

from app.retriever import RetrieverResult, retrieve, _mmr_select, _cosine_similarity_matrix


class TestRetrieverResult:
    """Tests for the RetrieverResult dataclass."""

    def test_dataclass_fields(self):
        result = RetrieverResult(
            selected_reviews=["review 1", "review 2"],
            embeddings=np.array([[1.0, 2.0], [3.0, 4.0]]),
        )
        assert result.selected_reviews == ["review 1", "review 2"]
        assert result.embeddings.shape == (2, 2)


class TestCosineSimilarityMatrix:
    """Tests for the cosine similarity matrix computation."""

    def test_identical_vectors(self):
        emb = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        sim = _cosine_similarity_matrix(emb)
        assert sim.shape == (2, 2)
        np.testing.assert_almost_equal(sim[0, 1], 1.0)

    def test_orthogonal_vectors(self):
        emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        sim = _cosine_similarity_matrix(emb)
        np.testing.assert_almost_equal(sim[0, 1], 0.0)

    def test_diagonal_is_one(self):
        emb = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        sim = _cosine_similarity_matrix(emb)
        np.testing.assert_almost_equal(sim[0, 0], 1.0)
        np.testing.assert_almost_equal(sim[1, 1], 1.0)


class TestMMRSelect:
    """Tests for the MMR selection algorithm."""

    def test_returns_all_when_under_max_k(self):
        emb = np.random.randn(5, 10).astype(np.float32)
        indices = _mmr_select(emb, max_k=10)
        assert sorted(indices) == list(range(5))

    def test_selects_exactly_max_k(self):
        emb = np.random.randn(20, 10).astype(np.float32)
        indices = _mmr_select(emb, max_k=5)
        assert len(indices) == 5
        assert len(set(indices)) == 5  # all unique

    def test_first_selected_is_zero(self):
        emb = np.random.randn(10, 10).astype(np.float32)
        indices = _mmr_select(emb, max_k=3)
        assert indices[0] == 0

    def test_prefers_diverse_vectors(self):
        """MMR should prefer vectors that are dissimilar to already-selected ones."""
        # Create embeddings: one cluster of similar vectors + one outlier
        cluster = np.array([[1.0, 0.0]] * 5, dtype=np.float32)
        cluster += np.random.randn(5, 2).astype(np.float32) * 0.01
        outlier = np.array([[-1.0, 0.0]], dtype=np.float32)
        emb = np.vstack([cluster, outlier])  # outlier is index 5

        indices = _mmr_select(emb, max_k=2)
        # The outlier (index 5) should be selected as the 2nd pick for diversity
        assert 5 in indices


class TestRetrieve:
    """Tests for the retrieve function."""

    def test_small_input_returns_all(self):
        reviews = [
            "This product is great for kids",
            "Battery life is terrible and short",
            "Good value for the money spent",
        ]
        result = retrieve(reviews, max_k=50)
        assert len(result.selected_reviews) == 3
        assert result.selected_reviews == reviews
        assert result.embeddings.shape[0] == 3

    def test_large_input_selects_max_k(self):
        # Generate enough distinct reviews to exceed max_k
        reviews = [f"Review number {i} about product feature {i % 7}" for i in range(60)]
        result = retrieve(reviews, max_k=10)
        assert len(result.selected_reviews) == 10
        assert result.embeddings.shape[0] == 10
        # All selected reviews should be from the original list
        for r in result.selected_reviews:
            assert r in reviews

    def test_exact_max_k_returns_all(self):
        reviews = [f"Review about topic {i} with details" for i in range(50)]
        result = retrieve(reviews, max_k=50)
        assert len(result.selected_reviews) == 50

    def test_embeddings_are_float32(self):
        reviews = ["Great product for toddlers", "Terrible quality overall"]
        result = retrieve(reviews, max_k=50)
        assert result.embeddings.dtype == np.float32

    def test_result_type(self):
        reviews = ["Nice product with good features"]
        result = retrieve(reviews, max_k=50)
        assert isinstance(result, RetrieverResult)
        assert isinstance(result.selected_reviews, list)
        assert isinstance(result.embeddings, np.ndarray)
