"""YAML config loading with the gitignored `*.local.yaml` overlay.

Every registry under `configs/` may have a sibling `<name>.local.yaml`
holding secrets or machine-local values (presigned URLs, absolute paths).
`load_yaml` deep-merges that overlay over the committed file so no caller
has to know the overlay exists.
"""

from pathlib import Path
from typing import Any, Dict

import yaml


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge `override` into `base` without mutating either.

    Mappings are merged key by key; for matching keys that are both
    mappings the merge recurses, otherwise the `override` value wins.
    This lets a local overlay patch individual dataset fields (e.g. add
    a `urls` list) without restating the whole entry.

    Args:
        base: Base mapping (lower precedence).
        override: Overlay mapping whose values take precedence.

    Returns:
        New merged mapping.
    """
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def load_yaml(config_path: Path) -> Dict[str, Any]:
    """Load a YAML config and deep-merge its `*.local.yaml` sibling over it.

    Args:
        config_path: Path to the committed YAML file.

    Returns:
        The merged mapping; an empty file yields an empty dict.

    Raises:
        FileNotFoundError: If `config_path` does not exist.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    local_path = config_path.with_suffix(".local.yaml")
    if local_path.exists():
        local_raw = yaml.safe_load(local_path.read_text(encoding="utf-8")) or {}
        raw = deep_merge(raw, local_raw)
    return raw
