"""Integration tests for the curation stage runner.

Asserts the sentinel-skip + cascade-invalidate contract by stubbing
out the actual executor builds. The real builders are tested in
test_curate_pipeline.py.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Set, Tuple

import pytest

pytest.importorskip("datatrove")

import scripts.curate_pretraining_corpus as curate_cli  # noqa: E402
import slm4ie.data.curate.runner as curate_runner  # noqa: E402
from slm4ie.data.curate import (  # noqa: E402
    STAGE_DIRS,
    STAGE_NAMES,
    read_sentinel,
)

#: Single dataset key the stubbed roster exposes. Scoped stages now run
#: per-dataset, so the runner needs at least one key to do any scoped work.
_DATASET: str = "d1"


def _setup_output(tmp_path: Path) -> Path:
    """Build a minimal <output_dir>/ for a test run.

    Args:
        tmp_path: pytest's tmp_path fixture.

    Returns:
        Path to the freshly created output directory.
    """
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    return output_dir


#: Fixed (records_in, records_out) the stub runner reports for every stage.
_STUB_COUNTS: Tuple[int, int] = (100, 50)

#: Fixed (records_in, records_out) the stubbed shard counter reports for
#: every dataset of a scoped stage.
_STUB_PER_KEY_COUNTS: Tuple[int, int] = (80, 40)


def _stub_runner(monkeypatch: pytest.MonkeyPatch, ran: List[str]) -> None:
    """Replace `_stage_runner` so each stage just records its name.

    Args:
        monkeypatch: pytest's monkeypatch fixture.
        ran: A list the stub appends each invoked stage's name to. The
            test reads this back to assert which stages actually ran.
    """

    def fake_runner(
        stage: str,
        paths: Any,
        cfg: Dict[str, Any],
        workers: int,
        stopwords: Set[str],
        spam_assets: Any,
        dataset_keys: List[str],
        input_view: Any = None,
        log_dir: Any = None,
        tasks: Any = None,
    ):
        def run() -> Tuple[int, int]:
            ran.append(stage)
            return _STUB_COUNTS

        return run

    monkeypatch.setattr(curate_runner, "_stage_runner", fake_runner)
    # Scoped stages materialize a symlink view of their upstream output.
    # With the real executors stubbed out there are no shards to mirror,
    # so stub the view builder to a harmless throwaway directory and treat
    # every dataset as having upstream output (the no-output guard reads
    # the real filesystem, which the stub runner never writes to).
    monkeypatch.setattr(
        curate_runner,
        "_filter_stage_subset",
        lambda _stage_dir, _keys, holder=None: holder,
    )
    monkeypatch.setattr(curate_runner, "_input_fingerprint", lambda _view: (1, "stub"))
    monkeypatch.setattr(
        curate_runner,
        "_has_stage_output",
        lambda _stage_dir, _key: True,
    )
    # A scoped stage takes its per-dataset counts off the on-disk shards,
    # which the stubbed executors never write.
    monkeypatch.setattr(
        curate_runner,
        "per_key_stage_counts",
        lambda _stage, _paths, keys: {key: _STUB_PER_KEY_COUNTS for key in keys},
    )


def _common_cfg(input_dir: Path, output_dir: Path) -> Dict[str, Any]:
    """Return a minimal `pretrain.yaml`-equivalent dict for stub-driven tests.

    Args:
        input_dir: Extraction input directory.
        output_dir: Curation output directory.

    Returns:
        Mapping with every top-level key the runner reads.
    """
    return {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "convert": {"text_field": "text"},
        "language": {"targets": ["sl"]},
        "spam": {"languages": ["sl"]},
        "quality": {"min_doc_words": 50},
        "repetition": {},
        "exact_dedup": {"precision": 64, "hash_fc": "xxhash"},
        "sentence_dedup": {"n_sentences": 3},
        "statistics": {"top_k_words": 5000},
    }


def _run_cli(
    monkeypatch: pytest.MonkeyPatch,
    cfg: Dict[str, Any],
    args: List[str],
    project_root: Path,
) -> None:
    """Drive `curate_cli.main()` with a stubbed YAML loader and key list.

    Args:
        monkeypatch: pytest's monkeypatch fixture.
        cfg: pretrain.yaml-equivalent dict.
        args: CLI argv tail (excluding the script name).
        project_root: Stubbed return value for `_find_project_root`.
    """
    monkeypatch.setattr(curate_runner, "_load_yaml", lambda _p: cfg)
    monkeypatch.setattr(curate_runner, "_load_stopwords", lambda _cfg: (set(), b""))
    monkeypatch.setattr(
        curate_runner,
        "_load_spam_assets",
        lambda _cfg: SimpleNamespace(adult_words={}, spam_words={}, domains=set(), raw_bytes=b""),
    )
    monkeypatch.setattr(curate_runner, "_find_project_root", lambda: project_root)
    monkeypatch.setattr(curate_runner, "_list_datasets", lambda _p: [_DATASET])
    # _load_yaml is stubbed, so the path only has to satisfy the required flag.
    monkeypatch.setattr(
        curate_cli.sys,
        "argv",
        ["curate_pretraining_corpus.py", "run", "--config", str(project_root / "curation.yaml"), *args],
    )
    curate_cli.main()


def test_first_run_executes_every_stage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A clean output_dir causes every stage to run."""
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = _setup_output(tmp_path)
    cfg = _common_cfg(input_dir, output_dir)
    ran: List[str] = []
    _stub_runner(monkeypatch, ran)
    _run_cli(monkeypatch, cfg, ["--all"], tmp_path)
    assert ran == list(STAGE_NAMES)


def test_unchanged_rerun_skips_every_stage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Running twice with identical config runs everything once then nothing."""
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = _setup_output(tmp_path)
    cfg = _common_cfg(input_dir, output_dir)
    ran: List[str] = []
    _stub_runner(monkeypatch, ran)
    _run_cli(monkeypatch, cfg, ["--all"], tmp_path)
    ran.clear()
    _run_cli(monkeypatch, cfg, ["--all"], tmp_path)
    assert ran == []


def test_quality_config_change_cascades(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Editing quality config invalidates quality + downstream, not upstream stages."""
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = _setup_output(tmp_path)
    cfg = _common_cfg(input_dir, output_dir)
    ran: List[str] = []
    _stub_runner(monkeypatch, ran)
    _run_cli(monkeypatch, cfg, ["--all"], tmp_path)
    ran.clear()
    cfg["quality"]["min_doc_words"] = 100
    _run_cli(monkeypatch, cfg, ["--all"], tmp_path)
    assert ran == ["quality", "repetition", "exact_dedup", "sentence_dedup", "statistics"]


def test_statistics_config_change_only_reruns_statistics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Editing statistics config invalidates only statistics, leaves dedup alone."""
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = _setup_output(tmp_path)
    cfg = _common_cfg(input_dir, output_dir)
    ran: List[str] = []
    _stub_runner(monkeypatch, ran)
    _run_cli(monkeypatch, cfg, ["--all"], tmp_path)
    ran.clear()
    cfg["statistics"]["top_k_words"] = 9999
    _run_cli(monkeypatch, cfg, ["--all"], tmp_path)
    assert ran == ["statistics"]


def test_force_stage_invalidates_downstream(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """--force --stage exact_dedup runs only that stage; downstream reruns on next --all.

    `--force --stage X` drops the sentinels for X and all downstream stages but
    only executes X itself (because `--stage X` limits the requested set).  A
    subsequent `--all` run will then pick up the stale downstream stages.
    """
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = _setup_output(tmp_path)
    cfg = _common_cfg(input_dir, output_dir)
    ran: List[str] = []
    _stub_runner(monkeypatch, ran)
    # First full run — all stages execute.
    _run_cli(monkeypatch, cfg, ["--all"], tmp_path)
    ran.clear()
    # Force-rebuild exact_dedup only; downstream sentinels are dropped.
    _run_cli(monkeypatch, cfg, ["--all", "--force", "--stage", "exact_dedup"], tmp_path)
    assert ran == ["exact_dedup"]
    ran.clear()
    # Second full --all run: exact_dedup sentinel is now current, but downstream
    # sentinels were deleted so sentence_dedup and statistics must re-run.
    _run_cli(monkeypatch, cfg, ["--all"], tmp_path)
    assert ran == ["sentence_dedup", "statistics"]


def test_run_only_one_stage_skips_others(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """--stage quality runs only quality (assuming it was stale)."""
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = _setup_output(tmp_path)
    cfg = _common_cfg(input_dir, output_dir)
    ran: List[str] = []
    _stub_runner(monkeypatch, ran)
    _run_cli(monkeypatch, cfg, ["--all", "--stage", "quality"], tmp_path)
    assert ran == ["quality"]


def test_sentinel_records_config_slice(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The sentinel JSON stores the actual config slice the stage saw."""
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = _setup_output(tmp_path)
    cfg = _common_cfg(input_dir, output_dir)
    ran: List[str] = []
    _stub_runner(monkeypatch, ran)
    _run_cli(monkeypatch, cfg, ["--all", "--stage", "quality"], tmp_path)
    # Scoped stages write a per-dataset sentinel under <stage>/<dataset>/.
    sentinel = read_sentinel(output_dir / STAGE_DIRS["quality"] / _DATASET)
    assert sentinel is not None
    assert sentinel.config_slice == {"min_doc_words": 50}
    # The hash includes the dataset_keys_bytes payload, so it does NOT equal
    # config_hash(slice_) alone. Just assert the hash was recorded and is a
    # sha256 hex prefixed string.
    assert sentinel.config_hash.startswith("sha256:")


def test_scoped_sentinel_records_per_dataset_counts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A scoped sentinel stores that dataset's own shard counts.

    One executor covers every dataset in a config-hash bucket, so its
    aggregate return would give all bucket-mates identical counts. The
    per-dataset counts come from the shards instead.
    """
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = _setup_output(tmp_path)
    cfg = _common_cfg(input_dir, output_dir)
    ran: List[str] = []
    _stub_runner(monkeypatch, ran)
    _run_cli(monkeypatch, cfg, ["--all", "--stage", "quality"], tmp_path)
    # Scoped stages write a per-dataset sentinel under <stage>/<dataset>/.
    sentinel = read_sentinel(output_dir / STAGE_DIRS["quality"] / _DATASET)
    assert sentinel is not None
    records_in, records_out = _STUB_PER_KEY_COUNTS
    assert sentinel.records_in == records_in
    assert sentinel.records_out == records_out


def test_corpus_sentinel_records_counts_from_runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A corpus sentinel stores the record counts the stage runner returned."""
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = _setup_output(tmp_path)
    cfg = _common_cfg(input_dir, output_dir)
    ran: List[str] = []
    _stub_runner(monkeypatch, ran)
    _run_cli(monkeypatch, cfg, ["--all", "--stage", "statistics"], tmp_path)
    # Corpus stages write one stage-level sentinel for the whole corpus.
    sentinel = read_sentinel(output_dir / STAGE_DIRS["statistics"])
    assert sentinel is not None
    records_in, records_out = _STUB_COUNTS
    assert sentinel.records_in == records_in
    assert sentinel.records_out == records_out


def _write_stage_shards(stage_dir: Path, key: str, n_shards: int) -> None:
    """Write *n_shards* placeholder shards for *key* under *stage_dir*.

    Args:
        stage_dir: Stage output folder.
        key: Dataset key (subfolder name).
        n_shards: Number of `<rank>.jsonl.gz` files to create.
    """
    (stage_dir / key).mkdir(parents=True, exist_ok=True)
    for rank in range(n_shards):
        (stage_dir / key / f"{rank:05d}.jsonl.gz").write_bytes(b"\x1f\x8b")


def test_corpus_stage_reads_only_roster_datasets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A corpus stage skips upstream folders of keys outside the roster and runs one task per shard."""
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = _setup_output(tmp_path)
    repetition_dir = output_dir / STAGE_DIRS["repetition"]
    _write_stage_shards(repetition_dir, _DATASET, 3)
    _write_stage_shards(repetition_dir, "benchmark", 2)
    seen: Dict[str, Any] = {}

    def fake_runner(stage: str, paths: Any, cfg: Any, workers: int, *args: Any, **kwargs: Any):
        def run() -> Tuple[int, int]:
            view = kwargs["input_view"]
            seen.update(
                keys=kwargs["dataset_keys"],
                tasks=kwargs["tasks"],
                view_keys=sorted(child.name for child in view.iterdir()),
            )
            return _STUB_COUNTS

        return run

    monkeypatch.setattr(curate_runner, "_stage_runner", fake_runner)
    _run_cli(monkeypatch, _common_cfg(input_dir, output_dir), ["--all", "--stage", "exact_dedup"], tmp_path)
    assert seen == {"keys": [_DATASET], "tasks": 3, "view_keys": [_DATASET]}
    assert not (output_dir / STAGE_DIRS["exact_dedup"] / curate_runner.PROGRESS_NAME).exists()


class TestPrepareCorpusStage:
    """`_prepare_corpus_stage` resumes a matching unfinished run and clears anything else."""

    def _paths(self, tmp_path: Path) -> Any:
        """Return curate paths with a stale shard, completion marker and dedup scratch in place."""
        paths = curate_runner.CuratePaths(input_folder=tmp_path / "in", output_dir=tmp_path / "out")
        _write_stage_shards(paths.stage_dir("sentence_dedup"), "d1", 1)
        (paths.logs_dir("sentence_dedup") / "1_sig" / "completions").mkdir(parents=True)
        (paths.logs_dir("sentence_dedup") / "1_sig" / "completions" / "00000").touch()
        (paths.dedup_state_dir / "sent_sigs").mkdir(parents=True)
        return paths

    def test_fresh_start_clears_stage(self, tmp_path: Path) -> None:
        """Without a progress file, output, logs and dedup scratch are removed."""
        paths = self._paths(tmp_path)
        resumed = curate_runner._prepare_corpus_stage(paths, "sentence_dedup", "h", 4, "i")
        assert resumed is False
        assert list(paths.stage_dir("sentence_dedup").glob("*/*.jsonl.gz")) == []
        assert not paths.logs_dir("sentence_dedup").exists()
        assert not (paths.dedup_state_dir / "sent_sigs").exists()
        assert (paths.stage_dir("sentence_dedup") / curate_runner.PROGRESS_NAME).is_file()

    def test_matching_progress_resumes(self, tmp_path: Path) -> None:
        """A progress file with the same hash, tasks and inputs keeps finished work."""
        paths = self._paths(tmp_path)
        curate_runner._prepare_corpus_stage(paths, "sentence_dedup", "h", 4, "i")
        _write_stage_shards(paths.stage_dir("sentence_dedup"), "d1", 1)
        (paths.logs_dir("sentence_dedup") / "1_sig" / "completions").mkdir(parents=True)
        assert curate_runner._prepare_corpus_stage(paths, "sentence_dedup", "h", 4, "i") is True
        assert (paths.stage_dir("sentence_dedup") / "d1" / "00000.jsonl.gz").exists()
        assert (paths.logs_dir("sentence_dedup") / "1_sig" / "completions").exists()

    @pytest.mark.parametrize("changed", [("h2", 4, "i"), ("h", 5, "i"), ("h", 4, "i2")])
    def test_changed_settings_start_fresh(self, tmp_path: Path, changed: Tuple[str, int, str]) -> None:
        """A different config hash, task count or input fingerprint clears the stage."""
        paths = self._paths(tmp_path)
        curate_runner._prepare_corpus_stage(paths, "sentence_dedup", "h", 4, "i")
        _write_stage_shards(paths.stage_dir("sentence_dedup"), "d1", 1)
        assert curate_runner._prepare_corpus_stage(paths, "sentence_dedup", *changed) is False
        assert list(paths.stage_dir("sentence_dedup").glob("*/*.jsonl.gz")) == []


def test_input_fingerprint_tracks_shard_rewrites(tmp_path: Path) -> None:
    """The fingerprint counts shards and changes when a shard is rewritten."""
    view = tmp_path / "view"
    _write_stage_shards(view, "d1", 2)
    count, digest = curate_runner._input_fingerprint(view)
    assert count == 2
    assert curate_runner._input_fingerprint(view) == (2, digest)
    (view / "d1" / "00001.jsonl.gz").write_bytes(b"\x1f\x8b\x08")
    assert curate_runner._input_fingerprint(view)[1] != digest
