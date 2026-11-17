"""Tests for slm4ie.evaluation.backends.vllm."""

import io
import json
import urllib.error

import pytest

from slm4ie.evaluation.backends import vllm as vllm_module
from slm4ie.evaluation.backends.vllm import VllmBackend

# ===============================
# Test Data
# ===============================

SCHEMA = {"type": "object"}
MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
REPLY = {
    "choices": [{"message": {"content": "<think>x</think>" + '{"mentions": []}'}}],
    "usage": {"prompt_tokens": 120, "completion_tokens": 9},
}


def fake_urlopen(captured, reply=REPLY):
    """Build a `urlopen` stand-in that records the request."""

    def urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data)
        return io.StringIO(json.dumps(reply))

    return urlopen


# ===============================
# VllmBackend
# ===============================


def test_chat_posts_greedy_schema_constrained_request(monkeypatch):
    """Chat posts greedy schema constrained request."""
    captured = {}
    monkeypatch.setattr(vllm_module.urllib.request, "urlopen", fake_urlopen(captured))

    VllmBackend("gemma4-12b").chat(MESSAGES, SCHEMA)

    body = captured["body"]
    assert captured["url"] == "http://localhost:8090/v1/chat/completions"
    assert body["model"] == "gemma4-12b"
    assert body["temperature"] == 0
    assert body["response_format"]["json_schema"]["schema"] == SCHEMA
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_chat_returns_stripped_text_and_usage(monkeypatch):
    """Chat returns stripped text and usage."""
    monkeypatch.setattr(vllm_module.urllib.request, "urlopen", fake_urlopen({}))

    result = VllmBackend("m").chat(MESSAGES, SCHEMA)

    assert result.text == '{"mentions": []}'
    assert (result.input_tokens, result.output_tokens) == (120, 9)


def test_chat_sends_the_fixed_seed(monkeypatch):
    """Chat sends the fixed seed."""
    captured = {}
    monkeypatch.setattr(vllm_module.urllib.request, "urlopen", fake_urlopen(captured))

    VllmBackend("m", seed=7).chat(MESSAGES, SCHEMA)

    assert captured["body"]["seed"] == 7


def test_peak_gpu_memory_is_not_reported():
    """Peak gpu memory is not reported."""
    assert VllmBackend("m").peak_gpu_memory_bytes() is None


def _failing_then_ok(errors):
    """Build a `urlopen` stand-in that raises each of `errors` in turn, then replies."""
    calls = {"n": 0}

    def urlopen(request, timeout):
        calls["n"] += 1
        if calls["n"] <= len(errors):
            raise errors[calls["n"] - 1]
        return io.StringIO(json.dumps(REPLY))

    return urlopen, calls


def _http_error(code):
    return urllib.error.HTTPError("http://x", code, "error", {}, None)


def test_a_server_error_is_retried_before_the_answer_counts(monkeypatch):
    """A 502 while llama-swap swaps models is retried, and the run keeps the answer."""
    urlopen, calls = _failing_then_ok([_http_error(502), urllib.error.URLError("refused")])
    monkeypatch.setattr(vllm_module.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(vllm_module.time, "sleep", lambda seconds: None)

    result = VllmBackend("gemma4-12b").chat(MESSAGES, SCHEMA)

    assert calls["n"] == 3
    assert result.text == '{"mentions": []}'


def test_a_server_error_on_every_try_is_raised(monkeypatch):
    """After the last try the error reaches the run."""
    urlopen, calls = _failing_then_ok([_http_error(502)] * 3)
    monkeypatch.setattr(vllm_module.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(vllm_module.time, "sleep", lambda seconds: None)

    with pytest.raises(urllib.error.HTTPError):
        VllmBackend("gemma4-12b").chat(MESSAGES, SCHEMA)
    assert calls["n"] == 3


def test_a_client_error_is_not_retried(monkeypatch):
    """A 400, such as a prompt over the context, fails at once."""
    urlopen, calls = _failing_then_ok([_http_error(400)])
    monkeypatch.setattr(vllm_module.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(vllm_module.time, "sleep", lambda seconds: None)

    with pytest.raises(urllib.error.HTTPError):
        VllmBackend("gemma4-12b").chat(MESSAGES, SCHEMA)
    assert calls["n"] == 1
