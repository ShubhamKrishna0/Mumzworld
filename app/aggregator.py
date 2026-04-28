"""Aggregator: ranks pros/cons by frequency and flags low-confidence data."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.feature_extractor import ExtractedFeatures

_MIN_MENTIONS = 2  # Minimum source attributions to keep a theme/pro/con


@dataclass
class AggregatedData:
    """Ranked pros, cons, theme frequencies, and confidence flag."""

    ranked_pros: list[dict] = field(default_factory=list)
    ranked_cons: list[dict] = field(default_factory=list)
    theme_frequencies: dict[str, int] = field(default_factory=dict)
    low_confidence: bool = False


def aggregate(features: ExtractedFeatures) -> AggregatedData:
    """Aggregate extracted features into ranked pros/cons with frequencies.

    1. Count the frequency of each theme (len of its "mentions" list).
    2. Rank praises by descending frequency (len of "sources").
    3. Rank complaints by descending frequency (len of "sources").
    4. Filter out themes/pros/cons with fewer than 2 mentions.
    5. Set low_confidence=True if fewer than 2 distinct themes remain.
    """

    # --- Theme frequencies ---------------------------------------------------
    raw_theme_freq: dict[str, int] = {}
    for t in features.themes:
        label = t.get("theme", "")
        mentions = t.get("mentions", [])
        raw_theme_freq[label] = raw_theme_freq.get(label, 0) + len(mentions)

    # Filter themes with < 2 mentions
    theme_frequencies = {
        k: v for k, v in raw_theme_freq.items() if v >= _MIN_MENTIONS
    }

    # --- Pros (praises) ranked by frequency ----------------------------------
    pro_freq: dict[str, int] = {}
    for p in features.praises:
        text = p.get("praise", "")
        sources = p.get("sources", [])
        pro_freq[text] = pro_freq.get(text, 0) + len(sources)

    ranked_pros = [
        {"text": text, "frequency": freq}
        for text, freq in sorted(pro_freq.items(), key=lambda x: x[1], reverse=True)
        if freq >= _MIN_MENTIONS
    ]

    # --- Cons (complaints) ranked by frequency -------------------------------
    con_freq: dict[str, int] = {}
    for c in features.complaints:
        text = c.get("complaint", "")
        sources = c.get("sources", [])
        con_freq[text] = con_freq.get(text, 0) + len(sources)

    ranked_cons = [
        {"text": text, "frequency": freq}
        for text, freq in sorted(con_freq.items(), key=lambda x: x[1], reverse=True)
        if freq >= _MIN_MENTIONS
    ]

    # --- Low-confidence flag -------------------------------------------------
    low_confidence = len(theme_frequencies) < 2

    return AggregatedData(
        ranked_pros=ranked_pros,
        ranked_cons=ranked_cons,
        theme_frequencies=theme_frequencies,
        low_confidence=low_confidence,
    )
