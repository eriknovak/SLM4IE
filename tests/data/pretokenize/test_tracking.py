"""Tests for slm4ie/data/pretokenize/tracking.py (MLflow lineage of finished outputs)."""

from pathlib import Path

import pytest

from slm4ie.data.pretokenize.config import PretokenizeConfig
from slm4ie.data.pretokenize.run import convert_entry, run_pretokenize
from slm4ie.data.pretokenize.tracking import DEFAULT_EXPERIMENT, log_pretokenized
from tests.data.pretokenize.conftest import make_entry, write_corpus


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Point MLflow at a SQLite store and artifact root under `tmp_path`.

    Args:
        tmp_path: Test directory.
        monkeypatch: The pytest monkeypatch fixture.

    Returns:
        The tracking URI.
    """
    pytest.importorskip("mlflow")
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.chdir(tmp_path)
    return uri


def _finished(tmp_path: Path, tokenizer_dir: Path):
    """Write the corpus and convert it.

    Args:
        tmp_path: Test directory.
        tokenizer_dir: The exported tokenizer folder.

    Returns:
        The entry.
    """
    write_corpus(tmp_path / "corpus")
    entry = make_entry(tmp_path, tokenizer_dir)
    convert_entry(entry)
    return entry


def _runs(uri: str) -> list:
    """Return the runs of the pretokenize experiment.

    Args:
        uri: The tracking URI.

    Returns:
        The MLflow runs.
    """
    import mlflow

    client = mlflow.MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name(DEFAULT_EXPERIMENT)
    return client.search_runs([experiment.experiment_id])


def test_logs_counts_settings_and_manifest(store: str, tmp_path: Path, tokenizer_dir: Path):
    """A finished output is logged with its token counts, settings, language tag and manifest artifact."""
    import mlflow

    entry = _finished(tmp_path, tokenizer_dir)
    digest = log_pretokenized(entry, tracking_uri=store)
    (run,) = _runs(store)
    artifacts = [item.path for item in mlflow.MlflowClient(tracking_uri=store).list_artifacts(run.info.run_id)]
    assert run.data.tags["output_digest"] == digest and run.data.tags["language"] == "sl"
    assert run.data.params["max_seq_len"] == "16"
    assert run.data.metrics["tokens__train"] > 0
    assert "manifest.json" in artifacts


def test_same_manifest_is_logged_once_unless_forced(store: str, tmp_path: Path, tokenizer_dir: Path):
    """Logging the same output twice keeps one run; `force` replaces it."""
    entry = _finished(tmp_path, tokenizer_dir)
    log_pretokenized(entry, tracking_uri=store)
    log_pretokenized(entry, tracking_uri=store)
    first = _runs(store)
    log_pretokenized(entry, tracking_uri=store, force=True)
    second = _runs(store)
    assert len(first) == len(second) == 1
    assert first[0].info.run_id != second[0].info.run_id


def test_forced_rebuild_replaces_its_run(store: str, tmp_path: Path, tokenizer_dir: Path):
    """A forced rebuild of the same output keeps one run, though its timings differ."""
    write_corpus(tmp_path / "corpus")
    entry = make_entry(tmp_path, tokenizer_dir)
    config = PretokenizeConfig(entries={entry.key: entry}, mlflow={"enabled": True, "tracking_uri": store})
    run_pretokenize(config, [entry.key])
    first = _runs(store)
    run_pretokenize(config, [entry.key], force=True)
    second = _runs(store)
    assert len(first) == len(second) == 1
    assert first[0].data.tags["output_digest"] == second[0].data.tags["output_digest"]
    assert first[0].info.run_id != second[0].info.run_id


def test_output_without_manifest_is_not_logged(tmp_path: Path, tokenizer_dir: Path):
    """An output with no manifest logs nothing."""
    assert log_pretokenized(make_entry(tmp_path, tokenizer_dir)) is None


def test_run_pretokenize_logs_when_the_registry_enables_it(store: str, tmp_path: Path, tokenizer_dir: Path):
    """A run with `mlflow.enabled` in the registry logs the written output."""
    write_corpus(tmp_path / "corpus")
    entry = make_entry(tmp_path, tokenizer_dir)
    config = PretokenizeConfig(entries={entry.key: entry}, mlflow={"enabled": True, "tracking_uri": store})
    run_pretokenize(config, [entry.key])
    assert len(_runs(store)) == 1
