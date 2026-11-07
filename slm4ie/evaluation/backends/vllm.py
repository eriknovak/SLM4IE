"""Chat backend for the local vllm-service (llama-swap + vllm, OpenAI-compatible)."""

import json
import time
import urllib.error
import urllib.request
from typing import Any

from slm4ie.evaluation.backends.base import DEFAULT_SEED, MAX_NEW_TOKENS, ChatBackend, ChatResult, register_backend

# ===============================
# Defaults
# ===============================

DEFAULT_URL = "http://localhost:8090"

# the first request starts the model container and may download its weights
REQUEST_TIMEOUT_SECONDS = 3_600
# a model swap in llama-swap can answer 502 for a moment; retry before failing the run
RETRY_ATTEMPTS = 3
RETRY_PAUSE_SECONDS = 30
SERVER_ERROR = 500

# ===============================
# Backend class
# ===============================


@register_backend("vllm")
class VllmBackend(ChatBackend):
    """Chat through the vllm-service's `/v1/chat/completions`, with schema-constrained decoding.

    Peak GPU memory is not reported: the service preallocates most of the GPU for every model,
    so the memory in use says nothing about the model.

    Args:
        model: The served model name from the service's `config.yaml` (e.g. `gemma4-12b`).
        url: The service's base URL.
        enable_thinking: Passed to the chat template; `False` turns a reasoning model's thinking off.
        seed: Sent as the request's `seed`.
        fold_system: Fold the system turn into the first user turn, for a template that rejects a system role.
    """

    def __init__(
        self,
        model: str,
        url: str = DEFAULT_URL,
        enable_thinking: bool = False,
        seed: int = DEFAULT_SEED,
        fold_system: bool = False,
    ):
        """Store the endpoint, the template switch, the seed and the fold switch."""
        super().__init__(model, seed, fold_system)
        self.url = url.rstrip("/")
        self.enable_thinking = enable_thinking

    def _complete(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> ChatResult:
        """Post the chat and read the answer and usage from the reply."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "seed": self.seed,
            "max_tokens": MAX_NEW_TOKENS,
            "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}},
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        request = urllib.request.Request(
            f"{self.url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf8"),
            headers={"Content-Type": "application/json"},
        )
        reply = _post(request)

        usage = reply["usage"]
        text = reply["choices"][0]["message"]["content"] or ""
        return ChatResult(text, usage["prompt_tokens"], usage["completion_tokens"], 0.0)


# ===============================
# Helper functions
# ===============================


def _post(request: urllib.request.Request) -> dict[str, Any]:
    """Send a request and decode its JSON reply, retrying a server or connection error.

    A client error (4xx) is raised at once: retrying the same request cannot fix it.

    Raises:
        urllib.error.URLError: If the last of `RETRY_ATTEMPTS` tries still fails.
    """
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                return json.load(response)
        except urllib.error.URLError as exc:
            is_client_error = isinstance(exc, urllib.error.HTTPError) and exc.code < SERVER_ERROR
            if is_client_error or attempt == RETRY_ATTEMPTS:
                raise
            time.sleep(RETRY_PAUSE_SECONDS)
    raise AssertionError("unreachable: the last attempt returns or raises")
