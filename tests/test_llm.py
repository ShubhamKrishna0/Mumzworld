"""Tests for shared LLM routing."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.llm import complete_json, reset_model_cooldowns


def _make_response(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


@pytest.fixture(autouse=True)
def _reset_llm_state():
    reset_model_cooldowns()
    yield
    reset_model_cooldowns()


def test_skips_recently_failed_model_on_next_call():
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        RuntimeError("429 rate limited"),
        _make_response('{"ok": true}'),
        _make_response('{"ok": true}'),
    ]

    with patch.dict(
        "os.environ",
        {"OPENROUTER_MODELS": "bad-model,good-model"},
    ):
        first = complete_json(
            client,
            stage="test_stage",
            messages=[],
            temperature=0,
            max_tokens=10,
        )
        second = complete_json(
            client,
            stage="test_stage",
            messages=[],
            temperature=0,
            max_tokens=10,
        )

    assert first == '{"ok": true}'
    assert second == '{"ok": true}'
    called_models = [
        call.kwargs["model"]
        for call in client.chat.completions.create.call_args_list
    ]
    assert called_models == ["bad-model", "good-model", "good-model"]


def test_requests_json_response_format():
    client = MagicMock()
    client.chat.completions.create.return_value = _make_response('{"ok": true}')

    complete_json(
        client,
        stage="test_stage",
        messages=[],
        temperature=0,
        max_tokens=10,
    )

    _, kwargs = client.chat.completions.create.call_args
    assert kwargs["response_format"] == {"type": "json_object"}
