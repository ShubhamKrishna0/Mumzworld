"""English and Arabic verdict generators with deterministic confidence scoring."""

from __future__ import annotations

from app.aggregator import AggregatedData
from app.errors import GenerationError
from app.json_utils import parse_llm_json_object
from app.llm import complete_json
from app.prompts import SYSTEM_PROMPT_AR, SYSTEM_PROMPT_EN

# Fallback model list — if one is rate-limited, try the next
# ---------------------------------------------------------------------------
# Helper: build the user-message context for generation LLM calls
# ---------------------------------------------------------------------------


def _build_generation_context(aggregated: AggregatedData, reviews: list[str]) -> str:
    """Format aggregated data and reviews into a user message for the LLM."""
    parts: list[str] = []

    # Ranked pros
    if aggregated.ranked_pros:
        pros_lines = "\n".join(
            f"  - {p['text']} (mentioned {p['frequency']} times)"
            for p in aggregated.ranked_pros
        )
        parts.append(f"Ranked Pros:\n{pros_lines}")

    # Ranked cons
    if aggregated.ranked_cons:
        cons_lines = "\n".join(
            f"  - {c['text']} (mentioned {c['frequency']} times)"
            for c in aggregated.ranked_cons
        )
        parts.append(f"Ranked Cons:\n{cons_lines}")

    # Theme frequencies
    if aggregated.theme_frequencies:
        theme_lines = "\n".join(
            f"  - {theme}: {freq} mentions"
            for theme, freq in sorted(
                aggregated.theme_frequencies.items(), key=lambda x: x[1], reverse=True
            )
        )
        parts.append(f"Theme Frequencies:\n{theme_lines}")

    # Selected reviews
    if reviews:
        review_lines = "\n".join(f"  {i + 1}. {r}" for i, r in enumerate(reviews))
        parts.append(f"Selected Customer Reviews:\n{review_lines}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Helper: parse LLM JSON response (handles code fences)
# ---------------------------------------------------------------------------


def _parse_json_response(content: str) -> dict:
    """Parse a JSON string from the LLM, stripping markdown code fences if present.

    Raises GenerationError if the content is not valid JSON or not a dict.
    """
    return parse_llm_json_object(content)


# ---------------------------------------------------------------------------
# English generator
# ---------------------------------------------------------------------------


def generate_en(aggregated: AggregatedData, reviews: list[str], llm_client) -> dict:
    """Make an independent English LLM call and return structured EN output.

    Returns
    -------
    dict with keys: summary_en, verdict_en, pros, cons

    Raises
    ------
    GenerationError
        If the LLM call fails or the response cannot be parsed.
    """
    user_message = _build_generation_context(aggregated, reviews)

    content = complete_json(
        llm_client,
        stage="generate_en",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_EN},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
        max_tokens=2048,
    )
    data = _parse_json_response(content)

    required_keys = {"summary_en", "verdict_en", "pros", "cons"}
    missing = required_keys - data.keys()
    if missing:
        raise GenerationError(f"English LLM response missing keys: {missing}")

    return {
        "summary_en": data["summary_en"],
        "verdict_en": data["verdict_en"],
        "pros": data["pros"],
        "cons": data["cons"],
    }


# ---------------------------------------------------------------------------
# Arabic generator
# ---------------------------------------------------------------------------


def generate_ar(aggregated: AggregatedData, reviews: list[str], llm_client) -> dict:
    """Make an independent Arabic LLM call and return structured AR output.

    Returns
    -------
    dict with keys: summary_ar, verdict_ar

    Raises
    ------
    GenerationError
        If the LLM call fails or the response cannot be parsed.
    """
    user_message = _build_generation_context(aggregated, reviews)

    content = complete_json(
        llm_client,
        stage="generate_ar",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_AR},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
        max_tokens=2048,
    )
    data = _parse_json_response(content)

    required_keys = {"summary_ar", "verdict_ar"}
    missing = required_keys - data.keys()
    if missing:
        raise GenerationError(f"Arabic LLM response missing keys: {missing}")

    return {
        "summary_ar": data["summary_ar"],
        "verdict_ar": data["verdict_ar"],
    }


# ---------------------------------------------------------------------------
# Deterministic confidence scoring
# ---------------------------------------------------------------------------


def compute_confidence(
    review_count: int,
    sentiment: dict[str, float],
    theme_count: int,
    low_confidence_flag: bool,
) -> tuple[float, str | None]:
    """Compute a deterministic confidence score with optional uncertainty reason.

    Scoring formula:
        base        = min(review_count / 20, 0.5)
        sent_bonus  = max(sentiment.values()) * 0.3
        theme_bonus = min(theme_count / 5, 0.2)
        total       = clamp(base + sent_bonus + theme_bonus, 0.0, 1.0)

    Caps & overrides:
        - MUST cap below 0.5 if review_count < 5 OR low_confidence_flag is True
        - MUST never return 1.0 unless max(sentiment) == 1.0 AND theme_count >= 5

    Returns
    -------
    (confidence, uncertainty_reason | None)
        uncertainty_reason is a descriptive string when confidence < 0.5, else None.
    """
    # --- base components ---
    base = min(review_count / 20, 0.5)
    sentiment_values = list(sentiment.values()) if sentiment else [0.0]
    max_sentiment = max(sentiment_values)
    sent_bonus = max_sentiment * 0.3
    theme_bonus = min(theme_count / 5, 0.2)

    total = base + sent_bonus + theme_bonus
    total = max(0.0, min(total, 1.0))  # clamp [0, 1]

    # --- cap below 0.5 when data is insufficient ---
    reasons: list[str] = []
    if review_count < 5:
        reasons.append(f"Only {review_count} review{'s' if review_count != 1 else ''} available")
    if low_confidence_flag:
        reasons.append("Insufficient theme diversity")

    if review_count < 5 or low_confidence_flag:
        total = min(total, 0.4999)

    # --- never 1.0 unless truly unanimous + enough themes ---
    if not (max_sentiment == 1.0 and theme_count >= 5):
        total = min(total, 0.9999)

    # Final clamp
    total = max(0.0, min(total, 1.0))

    # --- build uncertainty reason ---
    if total < 0.5:
        if not reasons:
            reasons.append("Low overall confidence from review data")
        return (total, "; ".join(reasons))

    return (total, None)
