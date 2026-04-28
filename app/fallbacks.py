"""Deterministic fallbacks used when LLM providers are unavailable."""

from __future__ import annotations

from app.aggregator import AggregatedData
from app.feature_extractor import ExtractedFeatures

_POSITIVE_WORDS = {
    "amazing",
    "comfortable",
    "durable",
    "easy",
    "excellent",
    "good",
    "great",
    "helpful",
    "love",
    "loved",
    "nice",
    "perfect",
    "quality",
    "recommend",
    "safe",
    "sturdy",
    "worth",
}

_NEGATIVE_WORDS = {
    "bad",
    "broken",
    "cheap",
    "difficult",
    "disappointed",
    "expensive",
    "flimsy",
    "hard",
    "heavy",
    "issue",
    "poor",
    "problem",
    "return",
    "small",
    "terrible",
    "uncomfortable",
    "unsafe",
}

_THEME_KEYWORDS = {
    "quality": {"quality", "sturdy", "durable", "flimsy", "cheap", "broken"},
    "ease of use": {"easy", "simple", "difficult", "hard", "install", "setup", "assemble"},
    "comfort": {"comfortable", "comfort", "uncomfortable", "soft"},
    "safety": {"safe", "safety", "secure", "unsafe"},
    "size and fit": {"size", "small", "large", "fit", "fits", "heavy", "light"},
    "value": {"price", "expensive", "cheap", "worth", "value"},
}


def _contains_any(text: str, words: set[str]) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in words)


def _sentiment_label(review: str) -> str:
    positive = _contains_any(review, _POSITIVE_WORDS)
    negative = _contains_any(review, _NEGATIVE_WORDS)
    if positive and not negative:
        return "positive"
    if negative and not positive:
        return "negative"
    return "neutral"


def build_fallback_features(reviews: list[str]) -> ExtractedFeatures:
    """Build conservative, source-grounded features without an LLM."""
    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
    theme_mentions: dict[str, list[str]] = {}
    praise_sources: dict[str, list[str]] = {}
    complaint_sources: dict[str, list[str]] = {}

    for review in reviews:
        label = _sentiment_label(review)
        sentiment_counts[label] += 1

        matched_themes = [
            theme
            for theme, keywords in _THEME_KEYWORDS.items()
            if _contains_any(review, keywords)
        ]
        for theme in matched_themes:
            theme_mentions.setdefault(theme, []).append(review)

        if label == "positive":
            target_themes = matched_themes or ["general positive feedback"]
            for theme in target_themes:
                praise_sources.setdefault(theme, []).append(review)
        elif label == "negative":
            target_themes = matched_themes or ["general negative feedback"]
            for theme in target_themes:
                complaint_sources.setdefault(theme, []).append(review)

    total = max(sum(sentiment_counts.values()), 1)
    sentiment = {
        key: round(value / total, 4)
        for key, value in sentiment_counts.items()
    }

    return ExtractedFeatures(
        sentiment=sentiment,
        themes=[
            {"theme": theme, "mentions": mentions}
            for theme, mentions in theme_mentions.items()
        ],
        complaints=[
            {"complaint": complaint, "sources": sources}
            for complaint, sources in complaint_sources.items()
        ],
        praises=[
            {"praise": praise, "sources": sources}
            for praise, sources in praise_sources.items()
        ],
    )


def build_fallback_generation(aggregated: AggregatedData, reviews: list[str]) -> dict:
    """Build a valid low-risk verdict when generation LLMs fail."""
    pros = [item["text"] for item in aggregated.ranked_pros[:5] if item.get("text")]
    cons = [item["text"] for item in aggregated.ranked_cons[:5] if item.get("text")]

    if pros and cons:
        summary_en = (
            f"Reviews most often mention {', '.join(pros)} as positives, "
            f"with {', '.join(cons)} as recurring concerns."
        )
        verdict_en = "Mixed feedback; consider the trade-offs before buying."
    elif pros:
        summary_en = f"Reviews are mostly positive, especially around {', '.join(pros)}."
        verdict_en = "Generally positive based on the available reviews."
    elif cons:
        summary_en = f"Reviews raise recurring concerns around {', '.join(cons)}."
        verdict_en = "Approach with caution based on the available reviews."
    else:
        summary_en = (
            f"{len(reviews)} review{'s' if len(reviews) != 1 else ''} were available, "
            "but they did not provide enough repeated evidence for strong pros or cons."
        )
        verdict_en = "Insufficient data to form a strong verdict."

    return {
        "summary_en": summary_en,
        "summary_ar": "تعذر توليد ملخص عربي موثوق عبر النموذج، لذلك تم استخدام ملخص احتياطي منخفض الثقة.",
        "verdict_en": verdict_en,
        "verdict_ar": "البيانات غير كافية لتكوين حكم قوي.",
        "pros": pros,
        "cons": cons,
    }
