"""Validation tests for the real `configs/data/tasks.yaml` registry."""

from pathlib import Path

import pytest

from slm4ie.data.tasks import (
    TasksConfig,
    load_tasks,
    resolve_output_dir,
    resolve_source_paths,
)


REPO_ROOT: Path = Path(__file__).resolve().parents[2]
TASKS_YAML: Path = REPO_ROOT / "configs" / "data" / "tasks.yaml"


@pytest.fixture(scope="module")
def tasks_config() -> TasksConfig:
    """Load the project's task registry once per test module.

    Returns:
        Parsed `TasksConfig`.
    """
    return load_tasks(TASKS_YAML)


def test_loads_successfully(tasks_config: TasksConfig) -> None:
    """`load_tasks` accepts the shipped registry without raising."""
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
        assert entry.converter, (
            f"{entry.task}/{entry.dataset} has no resolved converter"
        )


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
        anchor = (
            tasks_config.roots.extracted
            if entry.source.kind == "extracted"
            else tasks_config.roots.raw
        )
        for path in paths:
            assert str(path).startswith(str(anchor))
