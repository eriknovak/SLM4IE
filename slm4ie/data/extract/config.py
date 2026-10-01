"""The extraction registry `configs/data/extract.yaml` loaded into `ExtractionConfig`."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class ExtractionConfig:
    """Configuration for dataset extraction pipeline.

    Attributes:
        input_dir: Base directory for raw datasets.
        output_dir: Base directory for processed output.
        datasets: Dict mapping dataset key to config dict with 'extractor' and 'domain' keys.
        mlflow: MLflow tracking settings (`enabled`, `experiment`,
            `tracking_uri`) for post-hoc extraction-build logging.
    """

    input_dir: str
    output_dir: str
    datasets: Dict[str, Dict] = field(default_factory=dict)
    mlflow: Dict[str, Any] = field(default_factory=dict)


def load_extraction_config(config_path: Path) -> ExtractionConfig:
    """Load extraction config from YAML file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        ExtractionConfig: Parsed config.

    Raises:
        FileNotFoundError: If config file does not exist.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    return ExtractionConfig(
        input_dir=raw.get("input_dir", "data/raw"),
        output_dir=raw.get("output_dir", "data/processed"),
        datasets=raw.get("datasets", {}),
        mlflow=raw.get("mlflow") or {},
    )
