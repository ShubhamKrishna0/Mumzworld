"""Unit tests for app.parsers."""

from __future__ import annotations

import json

import pytest

from app.errors import EmptyInputError, ParseError
from app.parsers import parse_json, parse_text


# ── parse_json ──────────────────────────────────────────────────────────────


class TestParseJson:
    def test_valid_array(self):
        data = json.dumps(["review one", "review two"]).encode()
        assert parse_json(data) == ["review one", "review two"]

    def test_single_review(self):
        data = json.dumps(["only review"]).encode()
        assert parse_json(data) == ["only review"]

    def test_200_reviews(self):
        reviews = [f"review {i}" for i in range(200)]
        data = json.dumps(reviews).encode()
        assert parse_json(data) == reviews

    def test_empty_array_raises_empty_input(self):
        data = json.dumps([]).encode()
        with pytest.raises(EmptyInputError):
            parse_json(data)

    def test_not_json_raises_parse_error(self):
        with pytest.raises(ParseError):
            parse_json(b"not json at all")

    def test_json_object_raises_parse_error(self):
        with pytest.raises(ParseError):
            parse_json(json.dumps({"key": "value"}).encode())

    def test_array_of_ints_raises_parse_error(self):
        with pytest.raises(ParseError):
            parse_json(json.dumps([1, 2, 3]).encode())

    def test_mixed_types_raises_parse_error(self):
        with pytest.raises(ParseError):
            parse_json(json.dumps(["ok", 42]).encode())


# ── parse_text ──────────────────────────────────────────────────────────────


class TestParseText:
    def test_valid_lines(self):
        data = b"review one\nreview two\n"
        assert parse_text(data) == ["review one", "review two"]

    def test_strips_empty_lines(self):
        data = b"\n\nreview one\n\nreview two\n\n"
        assert parse_text(data) == ["review one", "review two"]

    def test_single_line(self):
        data = b"only review"
        assert parse_text(data) == ["only review"]

    def test_all_empty_lines_raises_empty_input(self):
        data = b"\n\n\n"
        with pytest.raises(EmptyInputError):
            parse_text(data)

    def test_empty_bytes_raises_empty_input(self):
        with pytest.raises(EmptyInputError):
            parse_text(b"")

    def test_invalid_utf8_raises_parse_error(self):
        with pytest.raises(ParseError):
            parse_text(b"\x80\x81\x82")
