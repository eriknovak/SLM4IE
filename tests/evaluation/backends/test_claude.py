"""Tests for slm4ie.evaluation.backends.claude."""

import json
import subprocess

import pytest

from slm4ie.evaluation.backends import claude as claude_module
from slm4ie.evaluation.backends.claude import ClaudeBackend

# ===============================
# Test Data
# ===============================

SCHEMA = {"type": "object"}
MESSAGES = [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}]
USAGE = {"input_tokens": 2, "cache_creation_input_tokens": 500, "cache_read_input_tokens": 70, "output_tokens": 31}
COST = 0.0123
MENTIONS = {"mentions": [{"text": "Maribor", "label": "LOC"}]}


def fake_run(captured, stdout, returncode=0, stderr=""):
    """Build a `subprocess.run` stand-in that records the call."""

    def run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr=stderr)

    return run


# ===============================
# ClaudeBackend
# ===============================


def test_chat_passes_prompt_schema_and_model(monkeypatch):
    """Chat passes prompt schema and model."""
    captured = {}
    reply = json.dumps({"result": "", "structured_output": MENTIONS, "usage": USAGE})
    monkeypatch.setattr(claude_module.subprocess, "run", fake_run(captured, reply))

    ClaudeBackend("haiku").chat(MESSAGES, SCHEMA)

    command = captured["command"]
    assert command[command.index("--model") + 1] == "haiku"
    assert command[command.index("--system-prompt") + 1] == "sys"
    assert json.loads(command[command.index("--json-schema") + 1]) == SCHEMA
    assert captured["kwargs"]["input"] == "usr"


def test_chat_turns_off_schema_retries(monkeypatch):
    """The CLI gets one try per window, as D3 allows every model."""
    captured = {}
    reply = json.dumps({"result": "", "structured_output": MENTIONS, "usage": USAGE})
    monkeypatch.setattr(claude_module.subprocess, "run", fake_run(captured, reply))

    ClaudeBackend("haiku").chat(MESSAGES, SCHEMA)

    assert captured["kwargs"]["env"]["MAX_STRUCTURED_OUTPUT_RETRIES"] == "0"


def test_chat_runs_on_the_subscription_even_with_an_exported_api_key(monkeypatch):
    """An exported ANTHROPIC_API_KEY never reaches the CLI, so no window is billed to the API."""
    captured = {}
    reply = json.dumps({"result": "", "structured_output": MENTIONS, "usage": USAGE})
    monkeypatch.setattr(claude_module.subprocess, "run", fake_run(captured, reply))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-key")

    ClaudeBackend("haiku").chat(MESSAGES, SCHEMA)

    assert "ANTHROPIC_API_KEY" not in captured["kwargs"]["env"]


def test_chat_returns_structured_output_and_split_usage(monkeypatch):
    """Chat returns structured output and keeps fresh, cache-write and cache-read tokens apart."""
    reply = json.dumps({"result": "", "structured_output": MENTIONS, "usage": USAGE, "total_cost_usd": COST})
    monkeypatch.setattr(claude_module.subprocess, "run", fake_run({}, reply))

    result = ClaudeBackend("haiku").chat(MESSAGES, SCHEMA)

    assert json.loads(result.text) == MENTIONS
    assert (result.input_tokens, result.cache_write_tokens, result.cache_read_tokens) == (2, 500, 70)
    assert (result.output_tokens, result.cost_usd) == (31, COST)


def test_chat_falls_back_to_result_text(monkeypatch):
    """Chat falls back to result text."""
    reply = json.dumps({"result": "<think>a</think>[]", "usage": USAGE})
    monkeypatch.setattr(claude_module.subprocess, "run", fake_run({}, reply))

    assert ClaudeBackend("haiku").chat(MESSAGES, SCHEMA).text == "[]"


def test_chat_raises_on_a_reported_error(monkeypatch):
    """Chat raises on a reported error."""
    reply = json.dumps({"is_error": True, "result": "rate limited", "usage": USAGE})
    monkeypatch.setattr(claude_module.subprocess, "run", fake_run({}, reply))

    with pytest.raises(RuntimeError, match="rate limited"):
        ClaudeBackend("haiku").chat(MESSAGES, SCHEMA)


def test_chat_returns_empty_text_when_the_schema_retries_run_out(monkeypatch):
    """Chat returns empty text when the schema retries run out."""
    reply = json.dumps({"is_error": True, "subtype": "error_max_structured_output_retries", "usage": USAGE})
    monkeypatch.setattr(claude_module.subprocess, "run", fake_run({}, reply))

    result = ClaudeBackend("haiku").chat(MESSAGES, SCHEMA)

    assert result.text == ""
    assert result.output_tokens == 31


def test_off_schema_answer_is_a_miss_even_when_the_cli_exits_non_zero(monkeypatch):
    """An off-schema answer counts as a miss whatever exit code the CLI gives it."""
    reply = json.dumps({"is_error": True, "subtype": "error_max_structured_output_retries", "usage": USAGE})
    monkeypatch.setattr(claude_module.subprocess, "run", fake_run({}, reply, returncode=1))

    assert ClaudeBackend("haiku").chat(MESSAGES, SCHEMA).text == ""


def test_chat_raises_on_a_non_zero_exit_with_another_result(monkeypatch):
    """A non-zero exit with any other result still stops the run."""
    reply = json.dumps({"is_error": True, "subtype": "error_during_execution", "usage": USAGE})
    monkeypatch.setattr(claude_module.subprocess, "run", fake_run({}, reply, returncode=1, stderr="boom"))

    with pytest.raises(RuntimeError, match="exited 1"):
        ClaudeBackend("haiku").chat(MESSAGES, SCHEMA)


def test_chat_raises_on_a_failed_process(monkeypatch):
    """Chat raises on a failed process."""
    monkeypatch.setattr(claude_module.subprocess, "run", fake_run({}, "", returncode=1, stderr="not logged in"))

    with pytest.raises(RuntimeError, match="not logged in"):
        ClaudeBackend("haiku").chat(MESSAGES, SCHEMA)


def test_chat_rejects_messages_without_a_system_turn():
    """Chat rejects messages without a system turn."""
    with pytest.raises(ValueError, match="roles"):
        ClaudeBackend("haiku").chat(MESSAGES[1:], SCHEMA)
