"""Chat backend for OpenRouter's OpenAI-compatible `/chat/completions` endpoint."""

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from slm4ie.evaluation.backends.base import DEFAULT_SEED, MAX_NEW_TOKENS, ChatBackend, ChatResult, register_backend

# ===============================
# Defaults
# ===============================

API_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY_ENV = "OPENROUTER_API_KEY"
DEFAULT_REASONING_EFFORT = "low"
DEFAULT_TEMPERATURE = 0.0

REQUEST_TIMEOUT_SECONDS = 600
# 429 rate limited, 502 provider error, 503 no provider: a pause often clears them
RETRYABLE_STATUS_CODES = frozenset({429, 502, 503})
RETRY_ATTEMPTS = 4
RETRY_PAUSE_SECONDS = 10

# ===============================
# Backend class
# ===============================


@register_backend("openrouter")
class OpenRouterBackend(ChatBackend):
    """Chat through OpenRouter, with schema-constrained decoding and the reported cost.

    The request fails rather than fall back to a provider that drops the JSON schema.

    Args:
        model: The OpenRouter model id (e.g. `google/gemini-2.5-flash`).
        seed: Sent as the request's `seed`; `None` leaves it out, for a model that accepts no seed.
        fold_system: Fold the system turn into the first user turn, for a model that rejects a system role.
        reasoning_effort: Sent as `reasoning.effort` (`max`, `xhigh`, `high`, `medium`, `low`, `minimal`, `none`).
        max_tokens: Cap on the tokens the model writes, reasoning included.
        provider_order: Providers to try in order, with no fallback to others; `None` lets OpenRouter route.
        temperature: Sent as the request's `temperature`; `None` leaves it out, for a model that accepts none.

    Raises:
        RuntimeError: If the environment variable `OPENROUTER_API_KEY` is missing or empty.
    """

    def __init__(
        self,
        model: str,
        seed: int | None = DEFAULT_SEED,
        fold_system: bool = False,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        max_tokens: int = MAX_NEW_TOKENS,
        provider_order: list[str] | None = None,
        temperature: float | None = DEFAULT_TEMPERATURE,
    ):
        """Read the API key from the environment and store the request settings."""
        super().__init__(model, seed, fold_system)
        api_key = os.environ.get(API_KEY_ENV, "")
        if not api_key:
            raise RuntimeError(f"environment variable {API_KEY_ENV} is missing or empty")
        self._api_key = api_key
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.provider_order = provider_order
        self.temperature = temperature

    def _complete(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> ChatResult:
        """Post the chat and read the answer, usage and cost from the reply."""
        provider: dict[str, Any] = {"require_parameters": True}
        if self.provider_order:
            provider.update(order=self.provider_order, allow_fallbacks=False)
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "answer", "strict": True, "schema": schema},
            },
            "reasoning": {"effort": self.reasoning_effort},
            "provider": provider,
        }
        # require_parameters drops every provider of a model that rejects a sent setting
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        request = urllib.request.Request(
            API_URL,
            data=json.dumps(payload).encode("utf8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._api_key}"},
        )
        reply = _post(request)

        if reply.get("error") or not reply.get("choices"):
            message = (reply.get("error") or {}).get("message", "no choices in the reply")
            raise RuntimeError(f"openrouter model {self.model!r} returned an error: {message}")

        usage = reply["usage"]
        prompt_details = usage.get("prompt_tokens_details") or {}
        cache_read = prompt_details.get("cached_tokens") or 0
        text = reply["choices"][0]["message"]["content"] or ""
        return ChatResult(
            text,
            input_tokens=usage["prompt_tokens"] - cache_read,
            output_tokens=usage["completion_tokens"],
            seconds=0.0,
            cache_write_tokens=prompt_details.get("cache_write_tokens") or 0,
            cache_read_tokens=cache_read,
            cost_usd=usage.get("cost"),
        )


# ===============================
# Helper functions
# ===============================


def _post(request: urllib.request.Request) -> dict[str, Any]:
    """Send a request and decode its JSON reply, retrying a rate-limit or provider error.

    Any other HTTP error (400 bad request, 401 bad key, 402 no credits) is raised at once:
    retrying the same request cannot fix it.

    Raises:
        urllib.error.HTTPError: On a non-retryable status, or if the last of `RETRY_ATTEMPTS` tries still fails.
    """
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code not in RETRYABLE_STATUS_CODES or attempt == RETRY_ATTEMPTS:
                raise
            time.sleep(RETRY_PAUSE_SECONDS)
    raise AssertionError("unreachable: the last attempt returns or raises")
