"""Retriever: embed reviews and select a diverse subset via MMR sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RetrieverResult:
    """Result of the retrieval stage."""

    selected_reviews: list[str]
    embeddings: np.ndarray


_model = None


def _load_embedding_model():
    """Lazy-load the sentence-transformers model (singleton)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix for a set of embeddings."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1e-10, norms)
    normalized = embeddings / norms
    return normalized @ normalized.T


def _mmr_select(
    embeddings: np.ndarray,
    max_k: int,
) -> list[int]:
    """Select up to max_k indices via Maximal Marginal Relevance (MMR).

    MMR diversity sampling: iteratively pick the candidate that has the
    *lowest* maximum similarity to any already-selected embedding.
    This maximises diversity in the selected set.

    Returns a list of selected indices into the embeddings array.
    """
    n = len(embeddings)
    if n <= max_k:
        return list(range(n))

    sim_matrix = _cosine_similarity_matrix(embeddings)

    # Start with the first review (index 0) as the seed
    selected: list[int] = [0]
    remaining = set(range(1, n))

    while len(selected) < max_k and remaining:
        best_idx = -1
        best_score = float("inf")  # We want to minimise max-similarity

        for candidate in remaining:
            # Max similarity between candidate and any already-selected review
            max_sim = max(sim_matrix[candidate, s] for s in selected)
            if max_sim < best_score:
                best_score = max_sim
                best_idx = candidate

        selected.append(best_idx)
        remaining.discard(best_idx)

    return selected


def retrieve(reviews: list[str], max_k: int = 50) -> RetrieverResult:
    """Embed reviews and select a diverse subset.

    If len(reviews) <= max_k, all reviews are returned with their embeddings.
    Otherwise, all reviews are embedded, a FAISS IndexFlatL2 is built, and
    up to max_k reviews are selected via MMR diversity sampling.

    Parameters
    ----------
    reviews : list[str]
        Preprocessed review strings.
    max_k : int
        Maximum number of reviews to select (default 50).

    Returns
    -------
    RetrieverResult
        Selected reviews and their embedding vectors.
    """
    model = _load_embedding_model()
    embeddings = model.encode(reviews)
    embeddings = np.array(embeddings, dtype=np.float32)

    if len(reviews) <= max_k:
        return RetrieverResult(selected_reviews=list(reviews), embeddings=embeddings)

    # Build FAISS index (required by spec even though MMR uses the sim matrix)
    import faiss

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Select diverse subset via MMR
    selected_indices = _mmr_select(embeddings, max_k)

    selected_reviews = [reviews[i] for i in selected_indices]
    selected_embeddings = embeddings[selected_indices]

    return RetrieverResult(
        selected_reviews=selected_reviews,
        embeddings=selected_embeddings,
    )
