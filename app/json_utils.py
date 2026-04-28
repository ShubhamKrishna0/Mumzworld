"""Utilities for handling JSON returned by LLMs."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping

from app.errors import GenerationError


def _strip_code_fence(text: str) -> str:
    """Remove a surrounding markdown code fence if the model added one."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_json_object(text: str) -> str:
    """Return the first balanced JSON object from *text*.

    Some models still wrap JSON with a sentence despite being instructed not to.
    This keeps the parser tolerant without accepting partial or ambiguous data.
    """
    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_string:
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    return text[start:]


def _repair_common_json_errors(text: str) -> str:
    """Repair common LLM JSON mistakes while avoiding broad, lossy rewrites."""
    repaired = text.strip()
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    repaired = re.sub(
        r'([}\]"\d])(\s*\n\s*)("[-A-Za-z0-9_ ]+"\s*:)',
        r"\1,\2\3",
        repaired,
    )
    return repaired


def parse_llm_json_object(content: str) -> dict:
    """Parse an LLM response into a JSON object.

    The parser accepts clean JSON, fenced JSON, JSON surrounded by incidental
    text, and a few common formatting mistakes such as trailing commas or a
    missing comma before the next object key.
    """
    text = _extract_json_object(_strip_code_fence(content))

    errors: list[json.JSONDecodeError] = []
    for candidate in (text, _repair_common_json_errors(text)):
        try:
            data = json.loads(candidate)
            break
        except json.JSONDecodeError as exc:
            errors.append(exc)
    else:
        raise GenerationError(f"LLM returned invalid JSON: {errors[-1]}") from errors[-1]

    if not isinstance(data, Mapping):
        raise GenerationError("LLM response is not a JSON object")

    return dict(data)
