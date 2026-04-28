"""Review preprocessing: dedup, normalize, and filter noisy reviews."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from app.errors import InsufficientDataError


@dataclass
class PreprocessResult:
    """Result of preprocessing raw reviews."""

    reviews: list[str]
    duplicates_removed: int
    noise_removed: int


def _load_embedding_model():
    """Lazy-load the sentence-transformers model for near-duplicate detection."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


def _cosine_similarity(a, b) -> float:
    """Compute cosine similarity between two vectors."""
    import numpy as np

    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def preprocess(raw_reviews: list[str]) -> PreprocessResult:
    """Clean, deduplicate, and filter a list of raw review strings.

    Steps:
    1. Strip leading/trailing whitespace
    2. Normalize Unicode to NFC form
    3. Remove exact duplicates (preserve order)
    4. Remove near-duplicates (cosine similarity >= 0.9)
    5. Discard reviews with fewer than 3 words

    Raises
    ------
    InsufficientDataError
        If 0 reviews remain after all filtering.
    """
    # Step 1 & 2: Strip whitespace and normalize Unicode
    cleaned = []
    for review in raw_reviews:
        stripped = review.strip()
        normalized = unicodedata.normalize("NFC", stripped)
        cleaned.append(normalized)

    # Step 3: Remove exact duplicates (preserve order)
    seen: set[str] = set()
    deduped: list[str] = []
    exact_dupes = 0
    for review in cleaned:
        if review in seen:
            exact_dupes += 1
        else:
            seen.add(review)
            deduped.append(review)

    # Step 4: Remove near-duplicates using sentence-transformers embeddings
    near_dupes = 0
    if len(deduped) > 1:
        # Filter out reviews that would be discarded as noise before embedding
        # to avoid embedding very short strings, but we need indices to track
        candidates = deduped  # We embed all, then filter noise after near-dedup

        model = _load_embedding_model()
        embeddings = model.encode(candidates)

        accepted: list[int] = [0]  # First review is always accepted
        for i in range(1, len(candidates)):
            is_near_dup = False
            for j in accepted:
                sim = _cosine_similarity(embeddings[i], embeddings[j])
                if sim >= 0.9:
                    is_near_dup = True
                    break
            if is_near_dup:
                near_dupes += 1
            else:
                accepted.append(i)

        deduped = [candidates[i] for i in accepted]

    # Step 5: Discard reviews with fewer than 3 words
    noise_removed = 0
    final: list[str] = []
    for review in deduped:
        if len(review.split()) < 3:
            noise_removed += 1
        else:
            final.append(review)

    duplicates_removed = exact_dupes + near_dupes

    # Step 6: Raise if nothing remains
    if len(final) == 0:
        raise InsufficientDataError("All reviews were discarded during preprocessing")

    return PreprocessResult(
        reviews=final,
        duplicates_removed=duplicates_removed,
        noise_removed=noise_removed,
    )
