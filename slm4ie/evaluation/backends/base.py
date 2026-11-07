"""The chat-backend interface, its registry and the reasoning-block stripper.

A backend answers one chat per call and reports what step 7 logs: the answer text, the
input and output token counts and the wall time. The transformers backend also reports the
peak GPU memory; vllm does not, since vllm-service reserves most of the GPU up front. Every
backend decodes greedily where it allows it and, where it can, constrains the answer to the
JSON schema from `slm4ie.evaluation.prompts.answer_schema`.
"""

import time
from dataclasses import dataclass, replace
from abc import ABC, abstractmethod
from typing import Any

# ===============================
# Defaults
# ===============================

# room for a window's mention list; the same cap for every backend that takes one
MAX_NEW_TOKENS = 2_048

# fixed seed every backend that takes one uses
DEFAULT_SEED = 0

# what joins a folded system turn to the first user turn, as Gemma 3's chat template joins them
FOLD_SEPARATOR = "\n\n"

THINK_CLOSE = "</think>"
THINK_OPEN = "<think>"

# ===============================
# Result and helpers
# ===============================


@dataclass(frozen=True)
class ChatResult:
    """One answered chat.

    Attributes:
        text: The answer with any reasoning block removed, ready for `parse_answer`.
        input_tokens: Fresh (uncached) tokens the model read.
        output_tokens: Tokens the model wrote, reasoning included.
        seconds: Wall time of the call.
        cache_write_tokens: Input tokens written to the prompt cache (0 where there is none).
        cache_read_tokens: Input tokens read from the prompt cache (0 where there is none).
        cost_usd: List-price cost the backend reports, or `None` where it reports none.
    """

    text: str
    input_tokens: int
    output_tokens: int
    seconds: float
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float | None = None


def strip_think(text: str) -> str:
    """Remove a `<think>...</think>` reasoning block from an answer.

    A template that opens the block itself leaves only the closing tag, so the text after
    the last closing tag is the answer. A block that never closes (the answer was cut off
    while reasoning) leaves no answer.

    Args:
        text: The raw text the model returned.

    Returns:
        The answer, stripped of surrounding whitespace.
    """
    if THINK_CLOSE in text:
        text = text.rsplit(THINK_CLOSE, 1)[1]
    elif text.lstrip().startswith(THINK_OPEN):
        text = ""
    return text.strip()


def fold_system_turn(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Move a leading system turn into the first user turn, for a chat template that rejects a system role.

    The system text opens the user turn, followed by `FOLD_SEPARATOR`, as Gemma 3's own template folds it.

    Args:
        messages: The chat turns.

    Returns:
        The turns without a system turn; unchanged when the first turn is not a system turn.
    """
    if len(messages) < 2 or messages[0]["role"] != "system" or messages[1]["role"] != "user":
        return list(messages)
    folded = {"role": "user", "content": messages[0]["content"] + FOLD_SEPARATOR + messages[1]["content"]}
    return [folded, *messages[2:]]


# ===============================
# Interface
# ===============================


class ChatBackend(ABC):
    """A model behind one chat call, decoded greedily where the backend allows it.

    Subclasses implement `_complete`; `chat` times it and strips the reasoning block.

    Args:
        model: The backend's name for the model (served name, HuggingFace id or Claude alias).
        seed: The sampling seed. vllm and transformers use it; the Claude CLI has no seed option and ignores it.
        fold_system: Fold the system turn into the first user turn, for a model whose template rejects it.
    """

    def __init__(self, model: str, seed: int = DEFAULT_SEED, fold_system: bool = False):
        """Store the model name, the seed and the fold switch."""
        self.model = model
        self.seed = seed
        self.fold_system = fold_system

    def chat(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> ChatResult:
        """Answer one chat.

        Args:
            messages: The chat turns, as `build_messages` returns them.
            schema: The JSON schema the answer must satisfy, as `answer_schema` returns it.

        Returns:
            The answer text, token counts and wall time.
        """
        if self.fold_system:
            messages = fold_system_turn(messages)
        start = time.perf_counter()
        raw = self._complete(messages, schema)
        return replace(raw, text=strip_think(raw.text), seconds=time.perf_counter() - start)

    def peak_gpu_memory_bytes(self) -> int | None:
        """Return the peak GPU memory of this process's model so far, or `None` where it cannot be measured."""
        return None

    def reset_peak_gpu_memory(self) -> None:
        """Start a new peak-memory measurement; a no-op where memory is not measured."""

    @abstractmethod
    def _complete(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> ChatResult:
        """Return the raw answer and its usage as a `ChatResult` with `seconds` 0 and the text not yet stripped."""


# ===============================
# Registry
# ===============================

_BACKEND_REGISTRY: dict[str, type[ChatBackend]] = {}


def register_backend(name: str):
    """Return a decorator registering a `ChatBackend` subclass by name.

    Args:
        name: Registry key (e.g. `"vllm"`).

    Returns:
        A class decorator that registers the class and returns it unchanged.
    """

    def decorator(cls: type[ChatBackend]) -> type[ChatBackend]:
        """Register `cls` under the captured name."""
        _BACKEND_REGISTRY[name] = cls
        return cls

    return decorator


def get_backend(name: str, model: str, **options: Any) -> ChatBackend:
    """Build the backend registered under `name`.

    Args:
        name: Registry key, one of `available_backends()`.
        model: The model, in the backend's own naming.
        **options: Backend-specific settings (endpoint URL, quantisation, ...).

    Returns:
        The backend, ready to chat.

    Raises:
        KeyError: If no backend is registered under `name`.
    """
    if name not in _BACKEND_REGISTRY:
        raise KeyError(f"unknown chat backend {name!r}; available: {available_backends()}")
    return _BACKEND_REGISTRY[name](model, **options)


def available_backends() -> list[str]:
    """Return the registered backend names, sorted."""
    return sorted(_BACKEND_REGISTRY)
