"""The dataset catalog `configs/data/download.yaml` loaded into `DatasetConfig` entries."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from slm4ie.utils.config import load_yaml

#: Allowed `role` values in the download catalog. Unrelated to the `role`
#: field in configs/data/tasks.yaml (`finetune_and_eval` | `held_out`).
DATASET_ROLES = frozenset({"pretrain", "benchmark", "lexicon"})


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
        super().__init__(f"Invalid dataset configuration ({len(self.problems)} problem(s)):\n  - {summary}")


@dataclass
class DatasetConfig:
    """Configuration for a single dataset.

    Attributes:
        key: Config key identifier (e.g., 'classla_web_sl').
        name: Human-readable dataset name.
        enabled: Whether to include in default download.
        source: Source backend ('http' or 'huggingface').
        urls: Download URLs (CLARIN datasets).
        output_dir: Subdirectory name under base output dir.
        manual: Whether this requires manual download.
        repo_id: HuggingFace repository ID.
        configs: HuggingFace dataset config names.
        note: Informational note.
        role: What the dataset is for. One of `pretrain` (pretraining
            corpus, the default), `benchmark` (evaluation benchmark),
            or `lexicon` (tokenizer/morphology lexicon; never enters
            the pretraining corpus). Used by downstream scripts to
            filter which datasets to materialize. Distinct from the
            `role` field in configs/data/tasks.yaml, which governs
            task-split isolation.
        tasks: Supported NLP tasks (e.g., POS, NER, SA, NLI). Empty
            list for pretraining corpora.
        publisher: Optional host the dataset is published on, for attribution
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
    role: str = "pretrain"
    tasks: List[str] = field(default_factory=list)
    publisher: Optional[str] = None

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
                missing or has an empty `output_dir`, or if `role` is
                not one of the allowed values.
        """
        enabled = data.get("enabled", True)
        manual = data.get("manual", False)
        output_dir = data.get("output_dir", "")
        if enabled and not manual and not output_dir:
            raise ConfigError([f"{key}: missing or empty 'output_dir'"])

        role = data.get("role", "pretrain")
        if role not in DATASET_ROLES:
            raise ConfigError([f"{key}: unknown role '{role}' (expected one of: {', '.join(sorted(DATASET_ROLES))})"])

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
            role=role,
            tasks=data.get("tasks", []),
            publisher=data.get("publisher"),
        )


@dataclass
class DownloadConfig:
    """Resolved contents of `configs/data/download.yaml`.

    Attributes:
        output_dir: Root the per-dataset raw subdirectories are created under.
        datasets: Every declared dataset keyed by its dataset key.
    """

    output_dir: str
    datasets: Dict[str, DatasetConfig] = field(default_factory=dict)


def load_download_config(config_path: Path) -> DownloadConfig:
    """Load the dataset catalog from YAML.

    If a sibling `*.local.yaml` overlay exists (e.g. `download.local.yaml`
    next to `download.yaml`), it is deep-merged over the base config. The
    overlay is gitignored and holds secrets or ephemeral values (such as
    presigned download URLs) that must not be committed.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        The catalog with every dataset parsed into a `DatasetConfig`.

    Raises:
        FileNotFoundError: If config file does not exist.
    """
    raw = load_yaml(config_path)

    output_dir = raw.get("output_dir", "data/raw")
    datasets: Dict[str, DatasetConfig] = {}

    for key, data in raw.get("datasets", {}).items():
        datasets[key] = DatasetConfig.from_dict(key, data)

    return DownloadConfig(output_dir=output_dir, datasets=datasets)
