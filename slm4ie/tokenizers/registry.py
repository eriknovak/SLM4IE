"""Tokenizer registry for config-driven selection."""

from typing import Dict, Type

_TOKENIZER_REGISTRY: Dict[str, Type] = {}


def register_tokenizer(name: str):
    """Decorator to register a tokenizer class by name.

    Args:
        name (str): Registry key for the tokenizer.
    """
    def decorator(cls):
        _TOKENIZER_REGISTRY[name] = cls
        return cls
    return decorator


def get_tokenizer(name: str):
    """Retrieve a registered tokenizer class by name.

    Args:
        name (str): Registry key.

    Returns:
        The tokenizer class.

    Raises:
        KeyError: If the tokenizer name is not registered.
    """
    if name not in _TOKENIZER_REGISTRY:
        raise KeyError(
            f"Tokenizer '{name}' not found. "
            f"Available: {list(_TOKENIZER_REGISTRY.keys())}"
        )
    return _TOKENIZER_REGISTRY[name]
