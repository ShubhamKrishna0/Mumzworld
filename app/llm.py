"""Shared LLM routing helpers."""

from __future__ import annotations

import os
import time
from collections.abc import Sequence

from app.errors import GenerationError

_DEFAULT_MODELS = (
    "openai/gpt-oss-20b:free",
    "nvidia/nemotron-nano-9b-v2:free",
    "google/gemma-4-26b-a4b-it:free",
)

_UNAVAILABLE_UNTIL: dict[str, float] = {}


def reset_model_cooldowns() -> None:
    """Clear model cooldown state. Intended for tests and local debugging."""
    _UNAVAILABLE_UNTIL.clear()


def get_model_candidates() -> tuple[str, ...]:
    """Return models in preferred order.

    Set OPENROUTER_MODELS to a comma-separated list to override without code
    changes.
    """
    configured = os.environ.get("OPENROUTER_MODELS", "")
    models = tuple(model.strip() for model in configured.split(",") if model.strip())
    return models or _DEFAULT_MODELS


def _is_rate_limit_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    message = str(exc).lower()
    return status_code == 429 or "429" in message or "rate-limit" in message or "rate limited" in message


def _cooldown_seconds(exc: Exception) -> int:
    if _is_rate_limit_error(exc):
        return 300
    if isinstance(exc, GenerationError):
        return 120
    return 60


def _mark_unavailable(model: str, exc: Exception) -> None:
    _UNAVAILABLE_UNTIL[model] = time.monotonic() + _cooldown_seconds(exc)


def _mark_available(model: str) -> None:
    _UNAVAILABLE_UNTIL.pop(model, None)


def _is_available(model: str) -> bool:
    unavailable_until = _UNAVAILABLE_UNTIL.get(model)
    if unavailable_until is None:
        return True
    if time.monotonic() >= unavailable_until:
        _UNAVAILABLE_UNTIL.pop(model, None)
        return True
    return False


def _short_error(exc: Exception) -> str:
    message = str(exc).replace("\n", " ").strip()
    if len(message) <= 220:
        return message
    return f"{message[:217]}..."


def _create_completion(
    llm_client,
    *,
    model: str,
    messages: Sequence[dict],
    temperature: float,
    max_tokens: int,
):
    """Create a chat completion, requesting provider-side JSON mode when available."""
    kwargs = {
        "model": model,
        "messages": list(messages),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    try:
        return llm_client.chat.completions.create(**kwargs)
    except TypeError:
        kwargs.pop("response_format")
        return llm_client.chat.completions.create(**kwargs)


def complete_json(
    llm_client,
    *,
    stage: str,
    messages: Sequence[dict],
    temperature: float,
    max_tokens: int,
) -> str:
    """Return JSON text from the first currently healthy model."""
    last_error: Exception | None = None
    attempted = 0

    for model in get_model_candidates():
        if not _is_available(model):
            continue

        attempted += 1
        try:
            response = _create_completion(
                llm_client,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            if not content:
                raise GenerationError("LLM returned an empty response")
            _mark_available(model)
            return content
        except Exception as exc:
            last_error = exc
            _mark_unavailable(model, exc)
            print(f"[{stage}] Model {model} unavailable: {_short_error(exc)}")

    if attempted == 0:
        raise GenerationError("No LLM models available; all are cooling down from recent failures")

    raise GenerationError(f"{stage} LLM call failed (all models exhausted): {last_error}") from last_error
