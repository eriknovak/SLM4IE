"""The tokenization registry `configs/data/tokenization.yaml` loaded into `TokenizationConfig`."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml

from slm4ie.utils.io import resolve_project_path


@dataclass
class TokenizationConfig:
    """Resolved contents of `configs/data/tokenization.yaml`.

    Attributes:
        input_dir: Root holding the per-dataset raw download directories.
        output_dir: Directory the `<key>.jsonl.gz` outputs are written to.
        datasets: Dataset keys declared for conversion.
    """

    input_dir: Path
    output_dir: Path
    datasets: List[str] = field(default_factory=list)


def load_tokenization_config(config_path: Path) -> TokenizationConfig:
    """Load and validate the tokenization YAML config.

    Args:
        config_path: Path to `configs/data/tokenization.yaml`.

    Returns:
        The parsed config with project-relative directories resolved.

    Raises:
        FileNotFoundError: If `config_path` does not exist.
        ValueError: If required fields are missing or malformed.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Tokenization config not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    input_dir = raw.get("input_dir")
    output_dir = raw.get("output_dir")
    datasets = raw.get("datasets") or []

    missing: List[str] = []
    if not input_dir:
        missing.append("input_dir")
    if not output_dir:
        missing.append("output_dir")
    if missing:
        raise ValueError(f"Tokenization config {config_path} is missing required fields: {', '.join(missing)}")
    if not isinstance(input_dir, str) or not isinstance(output_dir, str):
        raise ValueError(f"Tokenization config {config_path}: `input_dir` and `output_dir` must be strings.")
    if not isinstance(datasets, list) or not all(isinstance(k, str) for k in datasets):
        raise ValueError(f"Tokenization config {config_path}: `datasets` must be a list of strings.")

    return TokenizationConfig(
        input_dir=resolve_project_path(input_dir),
        output_dir=resolve_project_path(output_dir),
        datasets=list(datasets),
    )
