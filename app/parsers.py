"""Input parsers for JSON and plain-text review files."""

from __future__ import annotations

import json

from app.errors import EmptyInputError, ParseError


def parse_json(data: bytes) -> list[str]:
    """Parse *data* as a JSON array of strings.

    Raises
    ------
    ParseError
        If *data* is not valid JSON or not an array of strings.
    EmptyInputError
        If the parsed array is empty (0 reviews).
    """
    try:
        parsed = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ParseError(f"Invalid JSON: {exc}") from exc

    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ParseError("Expected a JSON array of strings")

    if len(parsed) == 0:
        raise EmptyInputError("Input contains no reviews")

    return parsed


def parse_text(data: bytes) -> list[str]:
    """Parse *data* as UTF-8 text with one review per line.

    Empty lines are stripped.

    Raises
    ------
    ParseError
        If *data* cannot be decoded as UTF-8 text.
    EmptyInputError
        If no non-empty lines are found.
    """
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ParseError(f"Cannot decode input as text: {exc}") from exc

    reviews = [line for line in text.splitlines() if line.strip()]

    if len(reviews) == 0:
        raise EmptyInputError("Input contains no reviews")

    return reviews
