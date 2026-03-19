"""Model registry for config-driven selection."""

from typing import Dict, Type

_MODEL_REGISTRY: Dict[str, Type] = {}


def register_model(name: str):
    """Decorator to register a model class by name.

    Args:
        name (str): Registry key for the model.
    """
    def decorator(cls):
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model(name: str):
    """Retrieve a registered model class by name.

    Args:
        name (str): Registry key.

    Returns:
        The model class.

    Raises:
        KeyError: If the model name is not registered.
    """
    if name not in _MODEL_REGISTRY:
        raise KeyError(
            f"Model '{name}' not found. "
            f"Available: {list(_MODEL_REGISTRY.keys())}"
        )
    return _MODEL_REGISTRY[name]
