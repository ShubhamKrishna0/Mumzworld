"""Pipeline orchestrator: wires all stages together to produce a MomsVerdict."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # Load .env file from project root

from app.aggregator import aggregate
from app.errors import GenerationError, ParseError
from app.fallbacks import build_fallback_features, build_fallback_generation
from app.feature_extractor import extract_features
from app.generator import compute_confidence, generate_ar, generate_en
from app.parsers import parse_json, parse_text
from app.preprocessor import preprocess
from app.retriever import retrieve
from app.schema import MomsVerdict, validate_and_retry


def _get_llm_client() -> OpenAI:
    """Create an OpenAI client configured for OpenRouter."""
    return OpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        base_url="https://openrouter.ai/api/v1",
    )


def _parse_input(raw_data: bytes, content_type: str) -> list[str]:
    """Route to the correct parser based on content_type."""
    if content_type in ("application/json", ".json"):
        return parse_json(raw_data)
    if content_type in ("text/plain", ".txt"):
        return parse_text(raw_data)
    # Unknown content_type — try JSON first, fall back to text
    try:
        return parse_json(raw_data)
    except (ParseError, Exception):
        return parse_text(raw_data)


def run_pipeline(raw_data: bytes, content_type: str) -> MomsVerdict:
    """Execute the full verdict pipeline end-to-end.

    Stages:
        1. Parse input
        2. Preprocess (dedup, normalize, filter)
        3. Retrieve diverse subset
        4. Extract features via LLM
        5. Aggregate themes, pros, cons
        6. Generate EN + AR verdicts via LLM
        7. Compute confidence
        8. Merge results
        9. Validate against schema (with retry)

    All pipeline errors propagate as-is so the API layer can map them
    to the appropriate HTTP status codes.
    """
    # 1. Parse
    raw_reviews = _parse_input(raw_data, content_type)

    # 2. Preprocess
    prep = preprocess(raw_reviews)

    # 3. Retrieve
    ret = retrieve(prep.reviews)

    # 4. Extract features
    llm_client = _get_llm_client()
    llm_fallback_reason: str | None = None
    try:
        features = extract_features(ret.selected_reviews, llm_client)
    except GenerationError as exc:
        llm_fallback_reason = f"LLM feature extraction unavailable: {exc}"
        print(f"[pipeline] {llm_fallback_reason}; using deterministic fallback features.")
        features = build_fallback_features(ret.selected_reviews)

    # 5. Aggregate
    aggregated = aggregate(features)

    # 6-9. Generate, merge, validate (with retry on validation failure)
    review_count = len(prep.reviews)
    sentiment = features.sentiment
    theme_count = len(aggregated.theme_frequencies)
    low_confidence_flag = aggregated.low_confidence
    selected_reviews = ret.selected_reviews

    def _generate() -> dict:
        nonlocal llm_fallback_reason
        try:
            en_output = generate_en(aggregated, selected_reviews, llm_client)
            ar_output = generate_ar(aggregated, selected_reviews, llm_client)
            generated_output = {**en_output, **ar_output}
        except GenerationError as exc:
            llm_fallback_reason = f"LLM verdict generation unavailable: {exc}"
            print(f"[pipeline] {llm_fallback_reason}; using deterministic fallback verdict.")
            generated_output = build_fallback_generation(aggregated, selected_reviews)

        confidence, uncertainty_reason = compute_confidence(
            review_count,
            sentiment,
            theme_count,
            low_confidence_flag or llm_fallback_reason is not None,
        )
        if llm_fallback_reason is not None:
            uncertainty_reason = (
                f"{uncertainty_reason}; {llm_fallback_reason}"
                if uncertainty_reason
                else llm_fallback_reason
            )
        return {
            **generated_output,
            "confidence": confidence,
            "uncertainty_reason": uncertainty_reason,
        }

    return validate_and_retry(_generate)
