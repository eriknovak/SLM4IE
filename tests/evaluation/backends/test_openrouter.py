"""Tests for slm4ie.evaluation.backends.openrouter."""

import io
import json
import urllib.error

import pytest

from slm4ie.evaluation.backends import get_backend
from slm4ie.evaluation.backends import openrouter as openrouter_module

# ===============================
# Test Data
# ===============================

SCHEMA = {"type": "object"}
MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
REPLY = {
    "choices": [{"message": {"content": '{"mentions": []}'}}],
    "usage": {
        "prompt_tokens": 120,
        "completion_tokens": 9,
        "prompt_tokens_details": {"cached_tokens": 100, "cache_write_tokens": 5},
        "cost": 0.0042,
    },
}


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    """Give every test a key and no pause between retries."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(openrouter_module.time, "sleep", lambda seconds: None)


def fake_urlopen(captured, reply=REPLY):
    """Build a `urlopen` stand-in that records the request."""

    def urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["auth"] = request.get_header("Authorization")
        captured["body"] = json.loads(request.data)
        return io.StringIO(json.dumps(reply))

    return urlopen


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


# ===============================
# OpenRouterBackend
# ===============================


def test_chat_posts_schema_seed_reasoning_and_provider_block(monkeypatch):
    """The request carries the model, the strict schema, the seed, the effort and require_parameters."""
    captured = {}
    monkeypatch.setattr(openrouter_module.urllib.request, "urlopen", fake_urlopen(captured))

    get_backend("openrouter", "google/gemini-2.5-flash", seed=7).chat(MESSAGES, SCHEMA)

    body = captured["body"]
    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert captured["auth"] == "Bearer test-key"
    assert body["model"] == "google/gemini-2.5-flash"
    assert body["seed"] == 7
    assert body["temperature"] == 0
    assert body["response_format"]["json_schema"] == {"name": "answer", "strict": True, "schema": SCHEMA}
    assert body["reasoning"] == {"effort": "low"}
    assert body["provider"] == {"require_parameters": True}


def test_seed_and_temperature_left_out_when_none(monkeypatch):
    """A model that accepts no seed or temperature is sent neither, so require_parameters keeps its providers."""
    captured = {}
    monkeypatch.setattr(openrouter_module.urllib.request, "urlopen", fake_urlopen(captured))

    get_backend("openrouter", "anthropic/claude-haiku-5.5", seed=None, temperature=None).chat(MESSAGES, SCHEMA)

    assert "seed" not in captured["body"]
    assert "temperature" not in captured["body"]


def test_provider_order_is_sent_without_fallbacks(monkeypatch):
    """A provider order pins the providers and turns the fallback off."""
    captured = {}
    monkeypatch.setattr(openrouter_module.urllib.request, "urlopen", fake_urlopen(captured))

    get_backend("openrouter", "m", provider_order=["google-vertex"], reasoning_effort="high").chat(MESSAGES, SCHEMA)

    assert captured["body"]["provider"] == {
        "require_parameters": True,
        "order": ["google-vertex"],
        "allow_fallbacks": False,
    }
    assert captured["body"]["reasoning"] == {"effort": "high"}


def test_chat_returns_text_token_counts_and_reported_cost(monkeypatch):
    """Cached tokens leave the fresh input count; the cost is the one the reply reports."""
    monkeypatch.setattr(openrouter_module.urllib.request, "urlopen", fake_urlopen({}))

    result = get_backend("openrouter", "m").chat(MESSAGES, SCHEMA)

    assert result.text == '{"mentions": []}'
    assert (result.input_tokens, result.output_tokens) == (20, 9)
    assert (result.cache_read_tokens, result.cache_write_tokens) == (100, 5)
    assert result.cost_usd == 0.0042


def test_cost_is_none_when_the_reply_reports_none(monkeypatch):
    """A reply without a cost gives no cost, not zero."""
    reply = {"choices": REPLY["choices"], "usage": {"prompt_tokens": 12, "completion_tokens": 3}}
    monkeypatch.setattr(openrouter_module.urllib.request, "urlopen", fake_urlopen({}, reply))

    result = get_backend("openrouter", "m").chat(MESSAGES, SCHEMA)

    assert result.cost_usd is None
    assert (result.input_tokens, result.cache_read_tokens) == (12, 0)


def test_missing_key_raises_naming_the_variable(monkeypatch):
    """Building the backend without a key fails and names the variable."""
    monkeypatch.delenv("OPENROUTER_API_KEY")

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        get_backend("openrouter", "m")


def test_reply_with_an_error_object_raises_naming_model_and_message(monkeypatch):
    """An error object in a 200 reply reaches the run with the model and the provider's message."""
    reply = {"error": {"code": 500, "message": "provider fell over"}}
    monkeypatch.setattr(openrouter_module.urllib.request, "urlopen", fake_urlopen({}, reply))

    with pytest.raises(RuntimeError, match=r"'m'.*provider fell over"):
        get_backend("openrouter", "m").chat(MESSAGES, SCHEMA)


def test_insufficient_credits_is_raised_without_a_retry(monkeypatch):
    """A 402 fails at once."""
    urlopen, calls = _failing_then_ok([_http_error(402)])
    monkeypatch.setattr(openrouter_module.urllib.request, "urlopen", urlopen)

    with pytest.raises(urllib.error.HTTPError):
        get_backend("openrouter", "m").chat(MESSAGES, SCHEMA)
    assert calls["n"] == 1


def test_a_rate_limit_is_retried_before_the_answer_counts(monkeypatch):
    """A 429 followed by a 503 is retried, and the run keeps the answer."""
    urlopen, calls = _failing_then_ok([_http_error(429), _http_error(503)])
    monkeypatch.setattr(openrouter_module.urllib.request, "urlopen", urlopen)

    result = get_backend("openrouter", "m").chat(MESSAGES, SCHEMA)

    assert calls["n"] == 3
    assert result.text == '{"mentions": []}'
