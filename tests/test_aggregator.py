"""Unit tests for the aggregator module."""

from app.aggregator import AggregatedData, aggregate
from app.feature_extractor import ExtractedFeatures


def _make_features(
    themes=None, praises=None, complaints=None, sentiment=None
) -> ExtractedFeatures:
    return ExtractedFeatures(
        sentiment=sentiment or {"positive": 0.5, "negative": 0.3, "neutral": 0.2},
        themes=themes or [],
        complaints=complaints or [],
        praises=praises or [],
    )


# ---------- Theme frequency counting ----------


def test_theme_frequencies_counted_by_mentions():
    features = _make_features(
        themes=[
            {"theme": "battery", "mentions": ["a", "b", "c"]},
            {"theme": "camera", "mentions": ["x", "y"]},
        ]
    )
    result = aggregate(features)
    assert result.theme_frequencies == {"battery": 3, "camera": 2}


def test_duplicate_theme_labels_are_summed():
    features = _make_features(
        themes=[
            {"theme": "battery", "mentions": ["a"]},
            {"theme": "battery", "mentions": ["b", "c"]},
        ]
    )
    result = aggregate(features)
    assert result.theme_frequencies == {"battery": 3}


# ---------- Pros ranking ----------


def test_pros_ranked_descending_by_frequency():
    features = _make_features(
        praises=[
            {"praise": "Great camera", "sources": ["r1", "r2"]},
            {"praise": "Long battery", "sources": ["r1", "r2", "r3", "r4"]},
            {"praise": "Nice design", "sources": ["r1", "r2", "r3"]},
        ]
    )
    result = aggregate(features)
    freqs = [p["frequency"] for p in result.ranked_pros]
    assert freqs == sorted(freqs, reverse=True)
    assert result.ranked_pros[0]["text"] == "Long battery"


# ---------- Cons ranking ----------


def test_cons_ranked_descending_by_frequency():
    features = _make_features(
        complaints=[
            {"complaint": "Overheats", "sources": ["r1", "r2", "r3"]},
            {"complaint": "Scratches", "sources": ["r1", "r2", "r3", "r4", "r5"]},
            {"complaint": "Slow", "sources": ["r1", "r2"]},
        ]
    )
    result = aggregate(features)
    freqs = [c["frequency"] for c in result.ranked_cons]
    assert freqs == sorted(freqs, reverse=True)
    assert result.ranked_cons[0]["text"] == "Scratches"


# ---------- Minimum mention threshold ----------


def test_themes_below_threshold_filtered():
    features = _make_features(
        themes=[
            {"theme": "battery", "mentions": ["a"]},  # 1 mention → filtered
            {"theme": "camera", "mentions": ["x", "y"]},  # 2 mentions → kept
        ]
    )
    result = aggregate(features)
    assert "battery" not in result.theme_frequencies
    assert result.theme_frequencies == {"camera": 2}


def test_pros_below_threshold_filtered():
    features = _make_features(
        praises=[
            {"praise": "Good screen", "sources": ["r1"]},  # 1 → filtered
            {"praise": "Fast charging", "sources": ["r1", "r2"]},  # 2 → kept
        ]
    )
    result = aggregate(features)
    assert len(result.ranked_pros) == 1
    assert result.ranked_pros[0]["text"] == "Fast charging"


def test_cons_below_threshold_filtered():
    features = _make_features(
        complaints=[
            {"complaint": "Heavy", "sources": ["r1"]},  # 1 → filtered
            {"complaint": "Loud fan", "sources": ["r1", "r2", "r3"]},  # 3 → kept
        ]
    )
    result = aggregate(features)
    assert len(result.ranked_cons) == 1
    assert result.ranked_cons[0]["text"] == "Loud fan"


# ---------- Low confidence flag ----------


def test_low_confidence_true_when_fewer_than_2_themes():
    features = _make_features(
        themes=[{"theme": "battery", "mentions": ["a", "b"]}]
    )
    result = aggregate(features)
    assert result.low_confidence is True


def test_low_confidence_true_when_no_themes():
    result = aggregate(_make_features())
    assert result.low_confidence is True


def test_low_confidence_false_when_2_or_more_themes():
    features = _make_features(
        themes=[
            {"theme": "battery", "mentions": ["a", "b"]},
            {"theme": "camera", "mentions": ["x", "y"]},
        ]
    )
    result = aggregate(features)
    assert result.low_confidence is False


def test_low_confidence_considers_filtered_themes():
    """If themes drop below 2 after filtering, low_confidence should be True."""
    features = _make_features(
        themes=[
            {"theme": "battery", "mentions": ["a", "b"]},  # 2 → kept
            {"theme": "camera", "mentions": ["x"]},  # 1 → filtered
        ]
    )
    result = aggregate(features)
    assert result.low_confidence is True


# ---------- Empty input ----------


def test_empty_features_returns_defaults():
    result = aggregate(_make_features())
    assert result.ranked_pros == []
    assert result.ranked_cons == []
    assert result.theme_frequencies == {}
    assert result.low_confidence is True
