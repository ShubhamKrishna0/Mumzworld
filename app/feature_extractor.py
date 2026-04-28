"""Feature extraction via a single LLM call (OpenRouter / OpenAI-compatible)."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.errors import GenerationError
from app.json_utils import parse_llm_json_object
from app.llm import complete_json
from app.prompts import FEATURE_EXTRACTION_PROMPT

# Fallback model list — if one is rate-limited, try the next
@dataclass
class ExtractedFeatures:
    """Structured features extracted from customer reviews."""

    sentiment: dict[str, float] = field(default_factory=dict)
    themes: list[dict] = field(default_factory=list)
    complaints: list[dict] = field(default_factory=list)
    praises: list[dict] = field(default_factory=list)


def _build_user_message(reviews: list[str]) -> str:
    """Format the reviews into a numbered list for the LLM."""
    numbered = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(reviews))
    return f"Here are the customer reviews:\n\n{numbered}"


def _parse_llm_response(content: str) -> ExtractedFeatures:
    """Parse the raw LLM response text into an ExtractedFeatures instance.

    Raises GenerationError if the response is not valid JSON or is missing
    required keys.
    """
    data = parse_llm_json_object(content)

    required_keys = {"sentiment", "themes", "complaints", "praises"}
    missing = required_keys - data.keys()
    if missing:
        raise GenerationError(f"LLM response missing keys: {missing}")

    return ExtractedFeatures(
        sentiment=data["sentiment"],
        themes=data["themes"],
        complaints=data["complaints"],
        praises=data["praises"],
    )


def extract_features(reviews: list[str], llm_client) -> ExtractedFeatures:
    """Extract sentiment, themes, complaints, and praises from reviews.

    Makes a single LLM call via an OpenAI-compatible client (e.g. OpenRouter).

    Parameters
    ----------
    reviews : list[str]
        Selected review strings to analyse.
    llm_client
        An ``openai.OpenAI`` client instance configured for OpenRouter.

    Returns
    -------
    ExtractedFeatures

    Raises
    ------
    GenerationError
        If the LLM call fails or the response cannot be parsed.
    """
    content = complete_json(
        llm_client,
        stage="feature_extractor",
        messages=[
            {"role": "system", "content": FEATURE_EXTRACTION_PROMPT},
            {"role": "user", "content": _build_user_message(reviews)},
        ],
        temperature=0.2,
        max_tokens=2048,
    )
    return _parse_llm_response(content)
