"""Chat backends for the zero-shot NER benchmark.

Importing this package registers every backend with the registry in `base`; build one with
`get_backend(name, model, **options)`. The heavy libraries (`torch`, `transformers`, `xgrammar`) load
only when a `transformers` backend is built.
"""

from slm4ie.evaluation.backends import claude, huggingface, vllm  # noqa: F401
from slm4ie.evaluation.backends.base import (
    MAX_NEW_TOKENS,
    ChatBackend,
    ChatResult,
    available_backends,
    fold_system_turn,
    get_backend,
    register_backend,
    strip_think,
)

__all__ = [
    "MAX_NEW_TOKENS",
    "ChatBackend",
    "ChatResult",
    "available_backends",
    "fold_system_turn",
    "get_backend",
    "register_backend",
    "strip_think",
]
