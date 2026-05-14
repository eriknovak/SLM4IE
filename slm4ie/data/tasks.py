"""Loader and resolver for the SLM4IE task registry.

The task registry (`configs/data/tasks.yaml`) declares every downstream
fine-tuning or evaluation target under a `<task>/<dataset>` key. This
module loads the YAML, validates its shape, resolves the converter for
each entry, and provides helpers for mapping entries to source/output
paths on disk.
"""

import dataclasses
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml

_KEY_PATTERN = re.compile(r"^[a-z]+/[a-z0-9_]+$")
_VALID_ROLES = {"finetune_and_eval", "held_out"}
_VALID_KINDS = {"extracted", "raw"}


@dataclasses.dataclass
class TasksRoots:
    """Filesystem roots referenced by the task registry.

    Attributes:
        extracted: Root of the extracted dataset tree.
        raw: Root of the raw dataset tree.
        tasks: Root under which `<task>/<dataset>/` output is written.
    """

    extracted: Path
    raw: Path
    tasks: Path


@dataclasses.dataclass
class TaskSource:
    """Source descriptor for a single task entry.

    Attributes:
        kind: Either ``"extracted"`` (resolved against
            ``TasksRoots.extracted`` as a `<key>.jsonl` file) or
            ``"raw"`` (resolved against ``TasksRoots.raw`` as a
            subdirectory).
        keys: Non-empty list of source keys.
    """

    kind: Literal["extracted", "raw"]
    keys: List[str]


@dataclasses.dataclass
class TaskEntry:
    """A single `<task>/<dataset>` entry from the task registry.

    Attributes:
        task: First segment of the entry key (e.g. ``"ner"``).
        dataset: Second segment of the entry key (e.g. ``"ssj500k"``).
        role: ``"finetune_and_eval"`` or ``"held_out"``.
        source: Source descriptor.
        splits: Mapping from split name (``"train"`` / ``"val"`` /
            ``"test"``) to the output filename for that split.
        labels: Optional list of label values for the task. ``None``
            when the task family does not need a closed label set.
        suite: Optional benchmark suite name (e.g.
            ``"superglue_sl"``).
        language: ISO language code (e.g. ``"sl"``).
        license: License identifier for the dataset.
        converter: Resolved converter module name (e.g.
            ``"to_spans"``). Defaults from the registry's
            ``converters:`` map and may be overridden per entry.
    """

    task: str
    dataset: str
    role: Literal["finetune_and_eval", "held_out"]
    source: TaskSource
    splits: Dict[str, str]
    labels: Optional[List[Any]]
    suite: Optional[str]
    language: str
    license: str
    converter: str


@dataclasses.dataclass
class TasksConfig:
    """Fully parsed task registry.

    Attributes:
        roots: Filesystem roots.
        converter_defaults: Default converter name per task family.
        entries: All task entries in declaration order.
    """

    roots: TasksRoots
    converter_defaults: Dict[str, str]
    entries: List[TaskEntry]


def load_tasks(yaml_path: Path) -> TasksConfig:
    """Loads and validates ``configs/data/tasks.yaml``.

    Args:
        yaml_path: Path to the task registry YAML.

    Returns:
        A fully parsed and validated ``TasksConfig``.

    Raises:
        FileNotFoundError: If ``yaml_path`` does not exist.
        ValueError: If the YAML structure violates any registry
            invariant (missing fields, malformed keys, unknown role
            or source kind, empty source keys, etc.).
        KeyError: If an entry's task has no default converter in the
            ``converters:`` map and no per-entry ``converter:``
            override.
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"Task registry not found: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(
            f"Task registry must be a mapping at the top level; got {type(raw).__name__}"
        )

    roots = _parse_roots(raw.get("roots"))
    converter_defaults = _parse_converter_defaults(raw.get("converters"))
    entries_raw = raw.get("entries")
    if not isinstance(entries_raw, dict) or not entries_raw:
        raise ValueError("Task registry `entries:` must be a non-empty mapping.")

    entries: List[TaskEntry] = []
    for key, value in entries_raw.items():
        entries.append(_parse_entry(key, value, converter_defaults))

    return TasksConfig(
        roots=roots,
        converter_defaults=converter_defaults,
        entries=entries,
    )


def filter_for_converter(
    config: TasksConfig, converter_name: str
) -> List[TaskEntry]:
    """Returns entries whose resolved converter matches ``converter_name``.

    Args:
        config: Parsed task registry.
        converter_name: Converter module name (e.g. ``"to_spans"``).

    Returns:
        Entries from ``config.entries`` whose ``converter`` field
        equals ``converter_name``, in declaration order.
    """
    return [e for e in config.entries if e.converter == converter_name]


def resolve_source_paths(entry: TaskEntry, roots: TasksRoots) -> List[Path]:
    """Resolves source paths for a task entry.

    For ``kind="extracted"``, each key resolves to
    ``roots.extracted / f"{key}.jsonl"``. For ``kind="raw"``, each
    key resolves to ``roots.raw / key`` (a directory).

    Args:
        entry: A task entry.
        roots: Filesystem roots.

    Returns:
        One path per source key in declaration order.

    Raises:
        ValueError: If ``entry.source.kind`` is not recognized.
    """
    if entry.source.kind == "extracted":
        return [roots.extracted / f"{k}.jsonl" for k in entry.source.keys]
    if entry.source.kind == "raw":
        return [roots.raw / k for k in entry.source.keys]
    raise ValueError(
        f"Unknown source kind {entry.source.kind!r} for {entry.task}/{entry.dataset}"
    )


def resolve_output_dir(entry: TaskEntry, roots: TasksRoots) -> Path:
    """Returns the output directory for a task entry.

    Args:
        entry: A task entry.
        roots: Filesystem roots.

    Returns:
        ``roots.tasks / entry.task / entry.dataset``.
    """
    return roots.tasks / entry.task / entry.dataset


def _parse_roots(raw: Any) -> TasksRoots:
    """Parses and validates the ``roots:`` block.

    Args:
        raw: Raw value of the ``roots:`` key from the YAML.

    Returns:
        Parsed ``TasksRoots``.

    Raises:
        ValueError: If any required root is missing or not a string.
    """
    if not isinstance(raw, dict):
        raise ValueError("Task registry `roots:` must be a mapping.")
    for required in ("extracted", "raw", "tasks"):
        if required not in raw:
            raise ValueError(f"Task registry `roots.{required}` is required.")
        if not isinstance(raw[required], str) or not raw[required]:
            raise ValueError(
                f"Task registry `roots.{required}` must be a non-empty string."
            )
    return TasksRoots(
        extracted=Path(raw["extracted"]),
        raw=Path(raw["raw"]),
        tasks=Path(raw["tasks"]),
    )


def _parse_converter_defaults(raw: Any) -> Dict[str, str]:
    """Parses and validates the ``converters:`` defaults map.

    Args:
        raw: Raw value of the ``converters:`` key from the YAML.

    Returns:
        Mapping from task family to converter name.

    Raises:
        ValueError: If the map is missing, empty, or contains
            non-string values.
    """
    if not isinstance(raw, dict) or not raw:
        raise ValueError(
            "Task registry `converters:` must be a non-empty mapping."
        )
    out: Dict[str, str] = {}
    for task, converter in raw.items():
        if not isinstance(task, str) or not task:
            raise ValueError(
                f"Converter map keys must be non-empty strings; got {task!r}"
            )
        if not isinstance(converter, str) or not converter:
            raise ValueError(
                f"Converter for task {task!r} must be a non-empty string."
            )
        out[task] = converter
    return out


def _parse_entry(
    key: Any, value: Any, converter_defaults: Dict[str, str]
) -> TaskEntry:
    """Parses and validates a single entry under ``entries:``.

    Args:
        key: Entry key from the YAML (expected ``<task>/<dataset>``).
        value: Entry body mapping.
        converter_defaults: Default converter map from the registry.

    Returns:
        Parsed ``TaskEntry``.

    Raises:
        ValueError: On any structural violation.
        KeyError: If no converter can be resolved for the entry.
    """
    if not isinstance(key, str) or not _KEY_PATTERN.match(key):
        raise ValueError(
            f"Entry key {key!r} must match '<task>/<dataset>' "
            f"(lowercase task, lowercase/digit/underscore dataset)."
        )
    if not isinstance(value, dict):
        raise ValueError(f"Entry {key!r} body must be a mapping.")

    task, dataset = key.split("/", 1)

    role = value.get("role")
    if role not in _VALID_ROLES:
        raise ValueError(
            f"Entry {key!r} role must be one of {sorted(_VALID_ROLES)}; got {role!r}"
        )

    source = _parse_source(key, value.get("source"))
    splits = _parse_splits(key, value.get("splits"))
    labels = _parse_labels(key, value.get("labels"))

    if "suite" not in value:
        raise ValueError(f"Entry {key!r} must declare `suite` (may be null).")
    suite = value["suite"]
    if suite is not None and not isinstance(suite, str):
        raise ValueError(f"Entry {key!r} `suite` must be a string or null.")

    language = value.get("language")
    if not isinstance(language, str) or not language:
        raise ValueError(
            f"Entry {key!r} `language` must be a non-empty string."
        )

    license_ = value.get("license")
    if not isinstance(license_, str) or not license_:
        raise ValueError(
            f"Entry {key!r} `license` must be a non-empty string."
        )

    converter_override = value.get("converter")
    if converter_override is not None:
        if not isinstance(converter_override, str) or not converter_override:
            raise ValueError(
                f"Entry {key!r} `converter` override must be a non-empty string."
            )
        converter = converter_override
    elif task in converter_defaults:
        converter = converter_defaults[task]
    else:
        raise KeyError(
            f"No converter resolved for entry {key!r}: task {task!r} has no "
            f"entry in `converters:` and no per-entry override."
        )

    return TaskEntry(
        task=task,
        dataset=dataset,
        role=role,
        source=source,
        splits=splits,
        labels=labels,
        suite=suite,
        language=language,
        license=license_,
        converter=converter,
    )


def _parse_source(key: str, raw: Any) -> TaskSource:
    """Parses and validates an entry's ``source:`` block.

    Args:
        key: Owning entry key, used for error messages.
        raw: Raw ``source`` mapping.

    Returns:
        Parsed ``TaskSource``.

    Raises:
        ValueError: If kind is unknown or keys are missing/invalid.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"Entry {key!r} `source` must be a mapping.")
    kind = raw.get("kind")
    if kind not in _VALID_KINDS:
        raise ValueError(
            f"Entry {key!r} source.kind must be one of {sorted(_VALID_KINDS)}; "
            f"got {kind!r}"
        )
    keys = raw.get("keys")
    if not isinstance(keys, list) or not keys:
        raise ValueError(
            f"Entry {key!r} `source.keys` must be a non-empty list."
        )
    for k in keys:
        if not isinstance(k, str) or not k:
            raise ValueError(
                f"Entry {key!r} `source.keys` entries must be non-empty strings."
            )
    return TaskSource(kind=kind, keys=list(keys))


def _parse_splits(key: str, raw: Any) -> Dict[str, str]:
    """Parses and validates an entry's ``splits:`` block.

    Args:
        key: Owning entry key, used for error messages.
        raw: Raw ``splits`` mapping.

    Returns:
        Mapping from split name to output filename.

    Raises:
        ValueError: If splits is empty or values are not strings.
    """
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Entry {key!r} `splits` must be a non-empty mapping.")
    out: Dict[str, str] = {}
    for split_name, filename in raw.items():
        if not isinstance(split_name, str) or not split_name:
            raise ValueError(
                f"Entry {key!r} split names must be non-empty strings."
            )
        if not isinstance(filename, str) or not filename:
            raise ValueError(
                f"Entry {key!r} split {split_name!r} filename must be a "
                f"non-empty string."
            )
        out[split_name] = filename
    return out


def _parse_labels(key: str, raw: Any) -> Optional[List[Any]]:
    """Parses and validates an entry's ``labels`` field.

    Args:
        key: Owning entry key, used for error messages.
        raw: Raw ``labels`` value (list or null).

    Returns:
        The list of labels, or ``None`` if absent / null.

    Raises:
        ValueError: If ``labels`` is present but not a list.
    """
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError(f"Entry {key!r} `labels` must be a list or null.")
    return list(raw)
