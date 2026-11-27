"""Loader for the shared fine-tuning config, `configs/evaluation/finetune.yaml`.

The shared config holds one block of training settings under `defaults:` and
per-family overrides under `families:`, keyed by the task family (the first
segment of an entry key, such as `ner`). An experiment's dial is deep-merged
over the shared config, so it names its encoders and overrides only the keys
it changes. Every family's settings expand into one `FinetuneSettings` per
learning rate and seed, and `planned_runs` crosses those with encoders and
entries into the grid a fine-tuning script runs.

An optional `search:` block describes a hyperparameter search instead: the
sampler (`grid` for an exhaustive grid, `tpe` for Optuna's Tree-structured
Parzen Estimator), the number of trials, the sampler seed, whether to prune,
and the space of settings to search. `slm4ie.finetune.search` runs it.
"""

import itertools
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from slm4ie.data.tasks.config import TaskEntry
from slm4ie.finetune.heads import available_heads
from slm4ie.utils.config import deep_merge, load_yaml
from slm4ie.utils.io import resolve_project_path

# ===============================
# Constants
# ===============================

TOP_LEVEL_KEYS = {"tasks_config", "encoders", "defaults", "families", "search"}
POSITIVE_INT_KEYS = ("epochs", "batch_size", "eval_batch_size", "gradient_accumulation", "max_length")
PRECISIONS = {"fp32", "bf16"}
SETTING_KEYS = {
    *POSITIVE_INT_KEYS,
    "learning_rates",
    "seeds",
    "weight_decay",
    "warmup_ratio",
    "max_grad_norm",
    "precision",
}

# the samplers a search may name, and the settings its space may vary
SAMPLERS = {"grid", "tpe"}
SEARCH_KEYS = {"sampler", "trials", "seed", "pruning", "space"}
SEARCHABLE_KEYS = {"learning_rate", *POSITIVE_INT_KEYS, "weight_decay", "warmup_ratio", "max_grad_norm"} - {
    "eval_batch_size"
}
RANGE_KEYS = {"low", "high", "log"}

# ===============================
# Config classes
# ===============================


@dataclass(frozen=True)
class FinetuneSettings:
    """The settings of one fine-tuning run: one learning rate and one seed.

    Attributes:
        learning_rate: Peak learning rate of AdamW.
        seed: Seed of every random generator the run uses.
        epochs: Passes over the train split; validation runs after each.
        batch_size: Examples per forward pass during training.
        eval_batch_size: Examples per forward pass on val and test.
        gradient_accumulation: Forward passes summed into one optimizer step.
        max_length: Most tokens per input; longer token-tagging inputs are cut
            into windows, longer classification inputs are truncated.
        weight_decay: AdamW weight decay on every weight matrix.
        warmup_ratio: Share of optimizer steps spent warming the learning rate up.
        max_grad_norm: The gradient norm is clipped to this value before each optimizer step.
        precision: `bf16` trains and infers under bfloat16 autocast on a GPU;
            `fp32` does not. A CPU run is always fp32.
    """

    learning_rate: float
    seed: int
    epochs: int
    batch_size: int
    eval_batch_size: int
    gradient_accumulation: int
    max_length: int
    weight_decay: float
    warmup_ratio: float
    max_grad_norm: float
    precision: str


@dataclass(frozen=True)
class FamilySettings:
    """The settings of one task family, with its learning-rate grid and seed list.

    Attributes:
        learning_rates: The learning-rate grid.
        seeds: The seed list.
        settings: Every other setting, keyed by its `FinetuneSettings` field.
    """

    learning_rates: tuple[float, ...]
    seeds: tuple[int, ...]
    settings: dict[str, Any] = field(default_factory=dict)

    def grid(self) -> list[FinetuneSettings]:
        """Return one run's settings per learning rate and seed, learning rate first.

        Returns:
            The settings, in grid order.
        """
        return [
            FinetuneSettings(learning_rate=lr, seed=seed, **self.settings)
            for lr, seed in itertools.product(self.learning_rates, self.seeds)
        ]


@dataclass(frozen=True)
class SearchRange:
    """A range a TPE search draws one setting from.

    Attributes:
        low: The smallest value.
        high: The largest value.
        log: Draw on a log scale, as suits a learning rate.
    """

    low: float
    high: float
    log: bool


@dataclass(frozen=True)
class SearchConfig:
    """A hyperparameter search over the settings of one encoder on one entry.

    Attributes:
        sampler: `grid` runs every combination of the space's lists; `tpe` draws
            `trials` settings with Optuna's Tree-structured Parzen Estimator.
        trials: The number of trials of a TPE search; the grid size for a grid.
        seed: Seed of the sampler, so a search draws the same settings again.
        pruning: Stop a TPE trial whose val score after an epoch falls below the
            median of earlier trials at that epoch; a grid never prunes.
        space: Per searched setting, its values (a grid axis, or a TPE choice)
            or the range a TPE search draws it from.
    """

    sampler: str
    trials: int
    seed: int
    pruning: bool
    space: dict[str, tuple[Any, ...] | SearchRange]


@dataclass(frozen=True)
class FinetuneConfig:
    """The shared fine-tuning config with an experiment's dial merged over it.

    Attributes:
        tasks_config: The task registry the entries come from.
        encoders: Encoders by HuggingFace name or local path; empty until a dial names them.
        defaults: The settings of a family with no override.
        families: The settings of every family that overrides the defaults.
        search: The hyperparameter search, or `None` when the config describes none.
    """

    tasks_config: Path
    encoders: tuple[str, ...]
    defaults: FamilySettings
    families: dict[str, FamilySettings]
    search: SearchConfig | None = None

    def settings_for(self, task: str) -> FamilySettings:
        """Return the settings of a task family.

        Args:
            task: The task family, the first segment of an entry key.

        Returns:
            The family's settings, or the defaults when it has no override.
        """
        return self.families.get(task, self.defaults)


@dataclass(frozen=True)
class PlannedRun:
    """One combination of encoder, entry, learning rate and seed.

    Attributes:
        encoder: The encoder, by HuggingFace name or local path.
        entry: The task registry entry.
        settings: The run's settings.
    """

    encoder: str
    entry: TaskEntry
    settings: FinetuneSettings


# ===============================
# Loader and planner
# ===============================


def load_finetune_config(config_path: Path, dial_path: Path | None = None) -> FinetuneConfig:
    """Load the shared fine-tuning config and merge an experiment's dial over it.

    Args:
        config_path: The shared config, `configs/evaluation/finetune.yaml`.
        dial_path: An experiment's dial, deep-merged over the shared config.

    Returns:
        The parsed config.

    Raises:
        FileNotFoundError: If either file does not exist.
        ValueError: If a key is unknown, missing or holds a value of the wrong
            type or range; the message names the key, where it is and the value.
            Also if `families` names a family with no registered task head,
            or the `search` block is malformed.
    """
    # merge the dial over the shared config
    raw = load_yaml(config_path)
    if dial_path is not None:
        raw = deep_merge(raw, load_yaml(dial_path))
    unknown = sorted(set(raw) - TOP_LEVEL_KEYS)
    if unknown:
        raise ValueError(f"fine-tuning config has unknown keys {unknown}; expected {sorted(TOP_LEVEL_KEYS)}")

    # read the task registry path and the encoders
    tasks_config = raw.get("tasks_config")
    if not isinstance(tasks_config, str) or not tasks_config:
        raise ValueError(f"fine-tuning config 'tasks_config' must be a path; got {tasks_config!r}")
    encoders = raw.get("encoders") or []
    if not isinstance(encoders, list) or not all(isinstance(e, str) and e for e in encoders):
        raise ValueError(f"fine-tuning config 'encoders' must be a list of names or paths; got {encoders!r}")

    # parse the defaults, then each family merged over them
    defaults_raw = raw.get("defaults")
    if not isinstance(defaults_raw, dict):
        raise ValueError(f"fine-tuning config 'defaults' must be a mapping; got {defaults_raw!r}")
    families_raw = raw.get("families") or {}
    if not isinstance(families_raw, dict):
        raise ValueError(f"fine-tuning config 'families' must be a mapping; got {families_raw!r}")
    unknown_families = sorted(set(families_raw) - set(available_heads()))
    if unknown_families:
        raise ValueError(
            f"fine-tuning config 'families' names {unknown_families}, which have no task head; "
            f"known families: {available_heads()}"
        )
    families = {}
    for family, override in families_raw.items():
        if not isinstance(override, dict):
            raise ValueError(f"fine-tuning config 'families.{family}' must be a mapping; got {override!r}")
        families[family] = _parse_family({**defaults_raw, **override}, f"families.{family}")

    return FinetuneConfig(
        tasks_config=resolve_project_path(tasks_config),
        encoders=tuple(encoders),
        defaults=_parse_family(defaults_raw, "defaults"),
        families=families,
        search=_parse_search(raw["search"]) if raw.get("search") is not None else None,
    )


def planned_runs(config: FinetuneConfig, encoders: list[str], entries: list[TaskEntry]) -> list[PlannedRun]:
    """Cross encoders, entries and each entry family's learning rates and seeds.

    Args:
        config: The fine-tuning config.
        encoders: Encoders by HuggingFace name or local path.
        entries: Task registry entries.

    Returns:
        One planned run per combination, encoder first, then entry, learning rate and seed.
    """
    return [
        PlannedRun(encoder=encoder, entry=entry, settings=settings)
        for encoder in encoders
        for entry in entries
        for settings in config.settings_for(entry.task).grid()
    ]


# ===============================
# Helpers
# ===============================


def _parse_family(raw: dict[str, Any], where: str) -> FamilySettings:
    """Validate one block of settings into a `FamilySettings`.

    Args:
        raw: The block, defaults already merged under a family override.
        where: The block's place in the config, for error messages.

    Returns:
        The parsed settings.

    Raises:
        ValueError: If a key is unknown, missing or out of range.
    """
    unknown, missing = sorted(set(raw) - SETTING_KEYS), sorted(SETTING_KEYS - set(raw))
    if unknown or missing:
        raise ValueError(f"fine-tuning config '{where}' has unknown keys {unknown} and lacks keys {missing}")

    # check the integer counts, the optimizer settings and the grid lists
    for key in (*POSITIVE_INT_KEYS, "weight_decay", "warmup_ratio", "max_grad_norm"):
        _check_setting(key, raw[key], f"{where}.{key}")
    rates, seeds = raw["learning_rates"], raw["seeds"]
    if not isinstance(rates, list) or not rates or not all(_is_number(r) and r > 0 for r in rates):
        raise ValueError(f"fine-tuning config '{where}.learning_rates' must be positive numbers; got {rates!r}")
    if not isinstance(seeds, list) or not seeds or not all(_is_int(s) for s in seeds):
        raise ValueError(f"fine-tuning config '{where}.seeds' must be a non-empty list of integers; got {seeds!r}")

    # check the precision
    if raw["precision"] not in PRECISIONS:
        raise ValueError(
            f"fine-tuning config '{where}.precision' must be one of {sorted(PRECISIONS)}; got {raw['precision']!r}"
        )

    settings = {key: raw[key] for key in POSITIVE_INT_KEYS}
    settings.update(
        weight_decay=float(raw["weight_decay"]),
        warmup_ratio=float(raw["warmup_ratio"]),
        max_grad_norm=float(raw["max_grad_norm"]),
        precision=raw["precision"],
    )
    return FamilySettings(learning_rates=tuple(float(r) for r in rates), seeds=tuple(seeds), settings=settings)


def _parse_search(raw: Any) -> SearchConfig:
    """Validate the `search` block into a `SearchConfig`.

    Args:
        raw: The block as read from YAML.

    Returns:
        The parsed search.

    Raises:
        ValueError: If a key is unknown or missing, the sampler is unknown, a
            searched setting is not searchable or holds an out-of-range value, a
            grid holds a range or a trial count, or a TPE search names no trials.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"fine-tuning config 'search' must be a mapping; got {raw!r}")
    unknown = sorted(set(raw) - SEARCH_KEYS)
    if unknown or "sampler" not in raw or "space" not in raw:
        raise ValueError(f"fine-tuning config 'search' has unknown keys {unknown}; it needs 'sampler' and 'space'")
    sampler = raw["sampler"]
    if sampler not in SAMPLERS:
        raise ValueError(f"fine-tuning config 'search.sampler' must be one of {sorted(SAMPLERS)}; got {sampler!r}")

    # parse the space: a list of values or, for a TPE search, a range
    space_raw = raw["space"]
    if not isinstance(space_raw, dict) or not space_raw:
        raise ValueError(f"fine-tuning config 'search.space' must be a non-empty mapping; got {space_raw!r}")
    space: dict[str, tuple[Any, ...] | SearchRange] = {}
    for key, values in space_raw.items():
        where = f"search.space.{key}"
        if key not in SEARCHABLE_KEYS:
            raise ValueError(f"fine-tuning config '{where}' is not searchable; searchable: {sorted(SEARCHABLE_KEYS)}")
        if isinstance(values, list) and values:
            for value in values:
                _check_setting(key, value, where)
            space[key] = tuple(values)
        elif isinstance(values, dict) and sampler == "tpe":
            space[key] = _parse_range(key, values, where)
        else:
            shape = "a non-empty list" if sampler == "grid" else "a non-empty list or a {low, high, log} range"
            raise ValueError(f"fine-tuning config '{where}' must be {shape} for a {sampler} search; got {values!r}")

    # a grid runs every combination; a TPE search runs a set number of trials
    trials, seed, pruning = raw.get("trials"), raw.get("seed", 0), raw.get("pruning", False)
    if not _is_int(seed):
        raise ValueError(f"fine-tuning config 'search.seed' must be an integer; got {seed!r}")
    if not isinstance(pruning, bool):
        raise ValueError(f"fine-tuning config 'search.pruning' must be true or false; got {pruning!r}")
    if sampler == "grid":
        if trials is not None or pruning:
            raise ValueError("fine-tuning config 'search' of a grid takes no 'trials' and no 'pruning'")
        trials = math.prod(len(values) for values in space.values())
    elif not _is_int(trials) or trials < 1:
        raise ValueError(f"fine-tuning config 'search.trials' must be a positive integer; got {trials!r}")
    return SearchConfig(sampler=sampler, trials=trials, seed=seed, pruning=pruning, space=space)


def _parse_range(key: str, raw: dict[str, Any], where: str) -> SearchRange:
    """Validate one `{low, high, log}` range of a TPE search."""
    unknown = sorted(set(raw) - RANGE_KEYS)
    if unknown or "low" not in raw or "high" not in raw:
        raise ValueError(f"fine-tuning config '{where}' has unknown keys {unknown}; a range needs 'low' and 'high'")
    low, high, log = raw["low"], raw["high"], raw.get("log", False)
    _check_setting(key, low, f"{where}.low")
    _check_setting(key, high, f"{where}.high")
    if low >= high:
        raise ValueError(f"fine-tuning config '{where}' needs low below high; got low={low!r} high={high!r}")
    if not isinstance(log, bool) or (log and low <= 0):
        raise ValueError(f"fine-tuning config '{where}.log' must be true or false, and true needs low above 0")
    return SearchRange(low=low, high=high, log=log)


def _check_setting(key: str, value: Any, where: str) -> None:
    """Raise a ValueError naming `where` when a setting's value is out of its range."""
    if key in POSITIVE_INT_KEYS:
        if not _is_int(value) or value < 1:
            raise ValueError(f"fine-tuning config '{where}' must be a positive integer; got {value!r}")
    elif key == "weight_decay":
        if not _is_number(value) or value < 0:
            raise ValueError(f"fine-tuning config '{where}' must be at least 0; got {value!r}")
    elif key == "warmup_ratio":
        if not _is_number(value) or not 0 <= value < 1:
            raise ValueError(f"fine-tuning config '{where}' must be in [0, 1); got {value!r}")
    elif not _is_number(value) or value <= 0:
        raise ValueError(f"fine-tuning config '{where}' must be positive; got {value!r}")


def _is_int(value: Any) -> bool:
    """Tell whether a YAML value is an integer and not a boolean."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    """Tell whether a YAML value is a number and not a boolean or a string such as `1e-5`."""
    return isinstance(value, int | float) and not isinstance(value, bool)
