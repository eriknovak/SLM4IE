"""Fixtures shared by the task-conversion tests."""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _logs_in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Send `convert_tasks` per-entry logs to `tmp_path`, not the repo's `logs/`."""
    monkeypatch.setattr("slm4ie.data.tasks.run.stamped_log_dir", lambda name: tmp_path / "logs" / name)
