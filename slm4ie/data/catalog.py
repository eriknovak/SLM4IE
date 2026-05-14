"""Dataset registry loaded from YAML. Decoupled from the downloader."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


class ConfigError(Exception):
    """Raised when dataset registry configuration is invalid.

    Carries an ordered list of human-readable problem descriptions
    so callers can present all issues at once instead of forcing
    iterative fix-then-retry rounds.

    Attributes:
        problems: One short description per problem.
    """

    def __init__(self, problems: List[str]):
        """Build a ConfigError listing all problems.

        Args:
            problems: One short description per problem.
        """
        self.problems = list(problems)
        summary = "\n  - ".join(self.problems)
        super().__init__(
            f"Invalid dataset configuration ({len(self.problems)} problem(s)):\n  - {summary}"
        )


@dataclass
class DatasetConfig:
    """Configuration for a single dataset.

    Attributes:
        key: Config key identifier (e.g., 'classla_web_sl').
        name: Human-readable dataset name.
        enabled: Whether to include in default download.
        source: Source type ('clarin' or 'huggingface').
        urls: Download URLs (CLARIN datasets).
        output_dir: Subdirectory name under base output dir.
        manual: Whether this requires manual download.
        repo_id: HuggingFace repository ID.
        configs: HuggingFace dataset config names.
        note: Informational note.
        benchmark: True for evaluation benchmark datasets, False for
            pretraining corpora. Used by downstream scripts to filter
            which datasets to materialize.
        tasks: Supported NLP tasks (e.g., POS, NER, SA, NLI). Empty
            list for pretraining corpora.
        provider: Optional source/host provider name for attribution
            (e.g., 'clarin.si'). Purely descriptive; carries no
            dispatch behaviour.
    """

    key: str
    name: str
    enabled: bool = True
    source: str = ""
    urls: List[str] = field(default_factory=list)
    output_dir: str = ""
    manual: bool = False
    repo_id: Optional[str] = None
    configs: Optional[List[str]] = None
    note: Optional[str] = None
    benchmark: bool = False
    tasks: List[str] = field(default_factory=list)
    provider: Optional[str] = None

    @classmethod
    def from_dict(cls, key: str, data: Dict) -> "DatasetConfig":
        """Create a DatasetConfig from a config dictionary.

        Args:
            key: The dataset key identifier.
            data: Dictionary of config values.

        Returns:
            DatasetConfig: Populated config instance.

        Raises:
            ConfigError: If the entry is enabled and not manual but is
                missing or has an empty `output_dir`.
        """
        enabled = data.get("enabled", True)
        manual = data.get("manual", False)
        output_dir = data.get("output_dir", "")
        if enabled and not manual and not output_dir:
            raise ConfigError([f"{key}: missing or empty 'output_dir'"])

        return cls(
            key=key,
            name=data.get("name", key),
            enabled=enabled,
            source=data.get("source", ""),
            urls=data.get("urls", []),
            output_dir=output_dir,
            manual=manual,
            repo_id=data.get("repo_id"),
            configs=data.get("configs"),
            note=data.get("note"),
            benchmark=data.get("benchmark", False),
            tasks=data.get("tasks", []),
            provider=data.get("provider"),
        )


def load_config(
    config_path: Path,
) -> Tuple[str, Dict[str, DatasetConfig]]:
    """Load dataset download configuration from YAML file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Tuple of (output_dir, dict of dataset key to DatasetConfig).

    Raises:
        FileNotFoundError: If config file does not exist.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    output_dir = raw.get("output_dir", "data/raw")
    datasets: Dict[str, DatasetConfig] = {}

    for key, data in raw.get("datasets", {}).items():
        datasets[key] = DatasetConfig.from_dict(key, data)

    return output_dir, datasets
