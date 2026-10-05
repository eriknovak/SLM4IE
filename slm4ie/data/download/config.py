"""The dataset catalog `configs/data/download.yaml` loaded into `DatasetConfig` entries."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from slm4ie.utils.config import load_yaml

#: Allowed `role` values in the download catalog. Unrelated to the `role`
#: field in configs/data/tasks.yaml (`finetune_and_eval` | `held_out`).
DATASET_ROLES = frozenset({"pretrain", "benchmark", "lexicon"})

#: Entry fields that name other registry keys: full and partial inclusion.
RELATIONS = ("contains", "overlaps")


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
        contains: Registry keys whose documents this entry fully includes.
            Transitive: the pretraining selection keeps only the outermost
            enabled entry of a containment group.
        overlaps: Registry keys this entry shares part of its documents
            with (the same upstream crawl or source). Symmetric and never
            transitive; it only warns, dedup removes the shared text.
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
    contains: List[str] = field(default_factory=list)
    overlaps: List[str] = field(default_factory=list)

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
                missing or has an empty `output_dir`, if `role` is not one
                of the allowed values, or if `contains` / `overlaps` is not
                a list of keys.
        """
        enabled = data.get("enabled", True)
        manual = data.get("manual", False)
        output_dir = data.get("output_dir", "")
        if enabled and not manual and not output_dir:
            raise ConfigError([f"{key}: missing or empty 'output_dir'"])

        role = data.get("role", "pretrain")
        if role not in DATASET_ROLES:
            raise ConfigError([f"{key}: unknown role '{role}' (expected one of: {', '.join(sorted(DATASET_ROLES))})"])

        relations = {name: data.get(name) or [] for name in RELATIONS}
        for name, keys in relations.items():
            if not isinstance(keys, list) or not all(isinstance(k, str) and k for k in keys):
                raise ConfigError([f"{key}: '{name}' must be a list of dataset keys"])

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
            contains=relations["contains"],
            overlaps=relations["overlaps"],
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
        ConfigError: If an entry is malformed, or a `contains` / `overlaps`
            relation names an unknown key, the entry itself, or closes a
            `contains` cycle.
    """
    raw = load_yaml(config_path)

    output_dir = raw.get("output_dir", "data/raw")
    datasets: Dict[str, DatasetConfig] = {}

    for key, data in raw.get("datasets", {}).items():
        datasets[key] = DatasetConfig.from_dict(key, data)
    validate_relations(datasets)

    return DownloadConfig(output_dir=output_dir, datasets=datasets)


def validate_relations(datasets: Dict[str, DatasetConfig]) -> None:
    """Check every `contains` / `overlaps` relation against the registry.

    Args:
        datasets: The parsed catalog entries.

    Raises:
        ConfigError: Listing every relation that names an unknown key or the
            entry itself, and every `contains` cycle.
    """
    problems: List[str] = []
    for key, spec in datasets.items():
        for name in RELATIONS:
            for other in getattr(spec, name):
                if other == key:
                    problems.append(f"{key}: '{name}' lists the entry itself")
                elif other not in datasets:
                    problems.append(f"{key}: '{name}' names unknown dataset '{other}'")
    if problems:
        raise ConfigError(problems)
    for key in datasets:
        if key in contained_in(datasets, key):
            problems.append(f"{key}: 'contains' cycle through {key}")
    if problems:
        raise ConfigError(problems)


def contained_in(datasets: Dict[str, DatasetConfig], key: str) -> Set[str]:
    """Return every key *key* contains, directly or through another entry.

    Args:
        datasets: The parsed catalog entries.
        key: The containing entry.

    Returns:
        The transitive closure of `key`'s `contains` relation; holds `key`
        itself only when a cycle runs back to it.
    """
    seen: Set[str] = set()
    stack = list(datasets[key].contains)
    while stack:
        other = stack.pop()
        if other in seen or other not in datasets:
            continue
        seen.add(other)
        stack.extend(datasets[other].contains)
    return seen


#: Status of a catalog entry the pretraining selection keeps.
SELECTED = "selected"


@dataclass
class Selection:
    """The pretraining selection resolved from the catalog's containment relations.

    Attributes:
        status: Every catalog key, in declaration order, mapped to `selected`
            or `skipped: <reason>` (`contained in <key>`, `disabled`, or
            `role <role>`).
        overlaps: Selected pairs declared as `overlaps`, each once, in
            declaration order; dedup is expected to remove what they share.
    """

    status: Dict[str, str] = field(default_factory=dict)
    overlaps: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def selected(self) -> List[str]:
        """Keys the selection keeps, in declaration order."""
        return [key for key, state in self.status.items() if state == SELECTED]

    def report(self) -> List[str]:
        """Render the selection as the lines `status` and `run` print.

        Returns:
            One line per catalog entry, then one warning line per overlap.
        """
        width = max((len(key) for key in self.status), default=0)
        lines = [f"{key:<{width}}  {state}" for key, state in self.status.items()]
        lines += [
            f"warning: {a} overlaps {b}; exact and sentence dedup are expected to remove the shared documents"
            for a, b in self.overlaps
        ]
        return lines


def resolve_selection(datasets: Dict[str, DatasetConfig]) -> Selection:
    """Pick one representative per containment group among the enabled pretraining entries.

    An enabled `role: pretrain` entry is selected unless another such entry
    contains it, directly or transitively; it is then skipped in favour of the
    outermost one, which no eligible entry contains. `overlaps` never changes
    the selection: a selected pair is only reported for a warning.

    Args:
        datasets: Catalog entries whose relations passed `validate_relations`.

    Returns:
        The status of every entry and the selected overlapping pairs.
    """
    eligible = [key for key, spec in datasets.items() if spec.enabled and spec.role == "pretrain"]
    closure = {key: contained_in(datasets, key) for key in eligible}
    outermost = [key for key in eligible if not any(key in closure[other] for other in eligible)]
    container = {member: key for key in reversed(outermost) for member in closure[key]}
    selection = Selection()
    for key, spec in datasets.items():
        if spec.role != "pretrain":
            selection.status[key] = f"skipped: role {spec.role}"
        elif not spec.enabled:
            selection.status[key] = "skipped: disabled"
        elif key in container:
            selection.status[key] = f"skipped: contained in {container[key]}"
        else:
            selection.status[key] = SELECTED
    order = {key: index for index, key in enumerate(datasets)}
    chosen = set(selection.selected)
    pairs: Set[Tuple[str, str]] = {
        (key, other) if order[key] < order[other] else (other, key)
        for key in chosen
        for other in datasets[key].overlaps
        if other in chosen
    }
    selection.overlaps = sorted(pairs, key=lambda pair: (order[pair[0]], order[pair[1]]))
    return selection
