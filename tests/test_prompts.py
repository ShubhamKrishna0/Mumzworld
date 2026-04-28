"""Unit tests for app.prompts — validates prompt template content."""

from __future__ import annotations

import pytest

from app.prompts import (
    FEATURE_EXTRACTION_PROMPT,
    SYSTEM_PROMPT_AR,
    SYSTEM_PROMPT_EN,
)


# ---------------------------------------------------------------------------
# English generation prompt (SYSTEM_PROMPT_EN)
# ---------------------------------------------------------------------------

class TestSystemPromptEN:
    def test_requires_grounding_in_reviews(self):
        lower = SYSTEM_PROMPT_EN.lower()
        assert "grounded" in lower or "traceable" in lower or "ground" in lower

    def test_forbids_hallucination(self):
        lower = SYSTEM_PROMPT_EN.lower()
        assert "hallucinate" in lower or "do not" in lower and "invent" in lower

    def test_requires_json_output(self):
        assert "JSON" in SYSTEM_PROMPT_EN

    def test_includes_summary_en_field(self):
        assert "summary_en" in SYSTEM_PROMPT_EN

    def test_includes_verdict_en_field(self):
        assert "verdict_en" in SYSTEM_PROMPT_EN

    def test_includes_pros_field(self):
        assert "pros" in SYSTEM_PROMPT_EN.lower()

    def test_includes_cons_field(self):
        assert "cons" in SYSTEM_PROMPT_EN.lower()

    def test_encourages_uncertainty_reporting(self):
        lower = SYSTEM_PROMPT_EN.lower()
        assert "uncertainty" in lower or "sparse" in lower or "insufficient" in lower

    def test_anti_hallucination_instruction(self):
        lower = SYSTEM_PROMPT_EN.lower()
        assert "not present in" in lower or "not in the" in lower or "not supported" in lower


# ---------------------------------------------------------------------------
# Arabic generation prompt (SYSTEM_PROMPT_AR)
# ---------------------------------------------------------------------------

class TestSystemPromptAR:
    def test_requires_native_arabic_phrasing(self):
        # Check for Arabic text about native phrasing
        assert "عربية أصيلة" in SYSTEM_PROMPT_AR or "أصيلة" in SYSTEM_PROMPT_AR

    def test_forbids_literal_translation(self):
        # Check for Arabic instruction forbidding literal translation
        assert "لا تترجم حرفياً" in SYSTEM_PROMPT_AR or "الترجمة الحرفية" in SYSTEM_PROMPT_AR

    def test_forbids_hallucination(self):
        # Check for Arabic anti-hallucination instruction
        assert "لا تختلق" in SYSTEM_PROMPT_AR or "لا تولّد" in SYSTEM_PROMPT_AR

    def test_requires_grounding_in_reviews(self):
        # Check for Arabic grounding instruction
        assert "المراجعات" in SYSTEM_PROMPT_AR

    def test_requires_json_output(self):
        assert "JSON" in SYSTEM_PROMPT_AR

    def test_includes_summary_ar_field(self):
        assert "summary_ar" in SYSTEM_PROMPT_AR

    def test_includes_verdict_ar_field(self):
        assert "verdict_ar" in SYSTEM_PROMPT_AR

    def test_written_in_arabic(self):
        # Verify the prompt contains substantial Arabic text (Arabic Unicode range)
        arabic_chars = sum(1 for c in SYSTEM_PROMPT_AR if "\u0600" <= c <= "\u06FF")
        assert arabic_chars > 50, "Prompt should contain substantial Arabic text"

    def test_encourages_uncertainty_reporting(self):
        # Check for Arabic uncertainty reporting instruction
        assert "عدم اليقين" in SYSTEM_PROMPT_AR or "غير كافية" in SYSTEM_PROMPT_AR


# ---------------------------------------------------------------------------
# Feature extraction prompt (moved from feature_extractor.py)
# ---------------------------------------------------------------------------

class TestFeatureExtractionPrompt:
    def test_requires_source_attribution(self):
        lower = FEATURE_EXTRACTION_PROMPT.lower()
        assert "source review" in lower

    def test_forbids_hallucination(self):
        lower = FEATURE_EXTRACTION_PROMPT.lower()
        assert "not invent" in lower or "do not invent" in lower

    def test_requires_json_output(self):
        assert "JSON" in FEATURE_EXTRACTION_PROMPT

    def test_includes_sentiment_structure(self):
        assert "sentiment" in FEATURE_EXTRACTION_PROMPT

    def test_includes_themes_structure(self):
        assert "themes" in FEATURE_EXTRACTION_PROMPT

    def test_includes_complaints_structure(self):
        assert "complaints" in FEATURE_EXTRACTION_PROMPT

    def test_includes_praises_structure(self):
        assert "praises" in FEATURE_EXTRACTION_PROMPT
