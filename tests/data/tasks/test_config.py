"""Validation tests for the real `configs/data/tasks.yaml` registry."""

from pathlib import Path
from typing import Any, Dict, Optional

import pytest
import yaml

from slm4ie.data.tasks.config import (
    TasksConfig,
    load_tasks_config,
    resolve_output_dir,
    resolve_source_paths,
)


REPO_ROOT: Path = Path(__file__).resolve().parents[3]
TASKS_YAML: Path = REPO_ROOT / "configs" / "data" / "tasks.yaml"


@pytest.fixture(scope="module")
def tasks_config() -> TasksConfig:
    """Load the project's task registry once per test module.

    Returns:
        Parsed `TasksConfig`.
    """
    return load_tasks_config(TASKS_YAML)


def test_loads_successfully(tasks_config: TasksConfig) -> None:
    """`load_tasks_config` accepts the shipped registry without raising."""
    assert isinstance(tasks_config, TasksConfig)


def test_has_eleven_entries(tasks_config: TasksConfig) -> None:
    """The registry currently declares 11 task entries."""
    assert len(tasks_config.entries) == 11


def test_every_entry_has_required_fields(tasks_config: TasksConfig) -> None:
    """Every entry exposes role / source / splits / language / license."""
    for entry in tasks_config.entries:
        assert entry.role in {"finetune_and_eval", "held_out"}
        assert entry.source is not None
        assert entry.source.kind in {"extracted", "raw"}
        assert entry.source.keys, f"{entry.task}/{entry.dataset} has empty source keys"
        assert entry.splits, f"{entry.task}/{entry.dataset} has empty splits"
        assert entry.language
        assert entry.license


def test_every_entry_has_a_converter(tasks_config: TasksConfig) -> None:
    """Every entry resolves to a converter module."""
    for entry in tasks_config.entries:
        assert entry.converter, f"{entry.task}/{entry.dataset} has no resolved converter"


def test_converter_resolves_via_defaults_or_override(
    tasks_config: TasksConfig,
) -> None:
    """Each entry's task has either a default converter or a per-entry override.

    The registry's invariant: ``entry.converter`` must be a value in
    ``converter_defaults`` for ``entry.task`` *or* a non-empty string
    set explicitly on the entry. We can't distinguish here, but both
    paths converge on ``entry.converter`` being non-empty.
    """
    for entry in tasks_config.entries:
        from_defaults = tasks_config.converter_defaults.get(entry.task)
        assert entry.converter == from_defaults or isinstance(entry.converter, str)
        assert entry.converter


def test_resolve_output_dir_under_tasks_root(
    tasks_config: TasksConfig,
) -> None:
    """`resolve_output_dir` always returns a subpath of `roots.tasks`."""
    for entry in tasks_config.entries:
        out = resolve_output_dir(entry, tasks_config.roots)
        assert tasks_config.roots.tasks in out.parents or out == tasks_config.roots.tasks
        assert out.parts[-2:] == (entry.task, entry.dataset)


def test_resolve_source_paths_uses_correct_root(
    tasks_config: TasksConfig,
) -> None:
    """Source paths are anchored at `extracted` or `raw` per the entry's kind."""
    for entry in tasks_config.entries:
        paths = resolve_source_paths(entry, tasks_config.roots)
        assert len(paths) == len(entry.source.keys)
        anchor = tasks_config.roots.extracted if entry.source.kind == "extracted" else tasks_config.roots.raw
        for path in paths:
            assert str(path).startswith(str(anchor))


def _ner_entry(role: str, keys: list, exclude: Optional[list] = None) -> dict:
    """Return an NER task entry body reading *keys*, optionally excluding documents of *exclude*."""
    source: Dict[str, Any] = {"kind": "extracted", "keys": keys}
    if exclude is not None:
        source["exclude"] = exclude
    splits = {"val": "val.jsonl.gz", "test": "test.jsonl.gz"}
    if role == "finetune_and_eval":
        splits = {"train": "train.jsonl.gz", **splits}
    return {
        "role": role,
        "source": source,
        "splits": splits,
        "labels": None,
        "suite": None,
        "language": "sl",
        "license": "cc-by-4.0",
    }


def _write_registry(tmp_path: Path, entries: dict, catalog: Optional[dict] = None) -> Path:
    """Write a task registry, plus a sibling download catalog when *catalog* is given."""
    if catalog is not None:
        datasets = {key: {"source": "http", "output_dir": key, **fields} for key, fields in catalog.items()}
        (tmp_path / "download.yaml").write_text(yaml.safe_dump({"datasets": datasets}))
    path = tmp_path / "tasks.yaml"
    body = {
        "roots": {"extracted": "x", "raw": "r", "tasks": "t"},
        "converters": {"ner": "spans"},
        "entries": entries,
    }
    path.write_text(yaml.safe_dump(body))
    return path


_CATALOG = {
    "big": {"role": "benchmark", "contains": ["small"]},
    "small": {"role": "benchmark"},
    "other": {"role": "benchmark"},
}


class TestSourceIsolation:
    """Held-out entries must not share documents with a same-task training entry."""

    def test_containing_pair_is_rejected(self, tmp_path: Path) -> None:
        """A held-out source that contains the training source fails at load."""
        path = _write_registry(
            tmp_path,
            {"ner/small": _ner_entry("finetune_and_eval", ["small"]), "ner/big": _ner_entry("held_out", ["big"])},
            _CATALOG,
        )
        with pytest.raises(ValueError, match=r"ner/small.*ner/big.*big contains small"):
            load_tasks_config(path, tmp_path / "download.yaml")

    def test_equal_sources_are_rejected(self, tmp_path: Path) -> None:
        """Training and held-out entries reading the same source fail at load."""
        path = _write_registry(
            tmp_path,
            {"ner/a": _ner_entry("finetune_and_eval", ["other"]), "ner/b": _ner_entry("held_out", ["other"])},
            _CATALOG,
        )
        with pytest.raises(ValueError, match="share source other"):
            load_tasks_config(path, tmp_path / "download.yaml")

    def test_disjoint_pair_is_accepted(self, tmp_path: Path) -> None:
        """Unrelated sources load."""
        path = _write_registry(
            tmp_path,
            {"ner/small": _ner_entry("finetune_and_eval", ["small"]), "ner/o": _ner_entry("held_out", ["other"])},
            _CATALOG,
        )
        assert len(load_tasks_config(path, tmp_path / "download.yaml").entries) == 2

    def test_excluding_the_contained_source_is_accepted(self, tmp_path: Path) -> None:
        """A held-out superset that drops the training source's documents loads."""
        path = _write_registry(
            tmp_path,
            {
                "ner/small": _ner_entry("finetune_and_eval", ["small"]),
                "ner/big": _ner_entry("held_out", ["big"], exclude=["small"]),
            },
            _CATALOG,
        )
        by_dataset = {e.dataset: e for e in load_tasks_config(path, tmp_path / "download.yaml").entries}
        assert by_dataset["big"].source.exclude == ["small"]

    def test_other_tasks_are_not_compared(self, tmp_path: Path) -> None:
        """Isolation only binds entries of the same task."""
        entries = {"ner/small": _ner_entry("finetune_and_eval", ["small"])}
        held = _ner_entry("held_out", ["big"])
        path = _write_registry(tmp_path, {**entries, "nli/big": held}, _CATALOG)
        body = yaml.safe_load(path.read_text())
        body["converters"]["nli"] = "superglue"
        path.write_text(yaml.safe_dump(body))
        assert len(load_tasks_config(path, tmp_path / "download.yaml").entries) == 2

    def test_shipped_registry_is_isolated(self) -> None:
        """The committed registry passes the check against the committed catalog."""
        load_tasks_config(TASKS_YAML)
