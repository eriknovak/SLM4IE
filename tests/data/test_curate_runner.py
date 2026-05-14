"""Integration tests for the stage runner in scripts/data/to_pretrain.py.

Asserts the sentinel-skip + cascade-invalidate contract by stubbing
out the actual executor builds. The real builders are tested in
test_curate_pipeline.py.
"""

from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pytest

import scripts.data.to_pretrain as curate_cli
from slm4ie.data.curate import (
    STAGE_DIRS,
    STAGE_NAMES,
    read_sentinel,
)


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


def _stub_runner(
    monkeypatch: pytest.MonkeyPatch, ran: List[str]
) -> None:
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
        dataset_keys: List[str],
        convert_view: Any = None,
        log_dir: Any = None,
    ):
        def run() -> Tuple[int, int]:
            ran.append(stage)
            return _STUB_COUNTS

        return run

    monkeypatch.setattr(curate_cli, "_stage_runner", fake_runner)


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
        "quality": {"min_doc_words": 50},
        "repetition": {},
        "exact_dedup": {"precision": 64, "hash_fc": "xxhash"},
        "sentence_dedup": {"n_sentences": 3},
        "stats": {"top_k_words": 5000},
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
        project_root: Project root used for stopword resolution.
    """
    monkeypatch.setattr(curate_cli, "_load_yaml", lambda _p: cfg)
    monkeypatch.setattr(
        curate_cli, "_load_stopwords", lambda _root, _cfg: (set(), b"")
    )
    monkeypatch.setattr(curate_cli, "_find_project_root", lambda: project_root)
    monkeypatch.setattr(curate_cli, "_list_datasets", lambda _p: [])
    monkeypatch.setattr(curate_cli.sys, "argv", ["to_pretrain.py", *args])
    curate_cli.main()


def test_first_run_executes_every_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A clean output_dir causes every stage to run."""
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = _setup_output(tmp_path)
    cfg = _common_cfg(input_dir, output_dir)
    ran: List[str] = []
    _stub_runner(monkeypatch, ran)
    _run_cli(monkeypatch, cfg, ["--all"], tmp_path)
    assert ran == list(STAGE_NAMES)


def test_unchanged_rerun_skips_every_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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


def test_quality_config_change_cascades(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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
    assert ran == ["quality", "repetition", "exact_dedup", "sentence_dedup", "stats"]


def test_stats_config_change_only_reruns_stats(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Editing stats config invalidates only stats, leaves dedup alone."""
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = _setup_output(tmp_path)
    cfg = _common_cfg(input_dir, output_dir)
    ran: List[str] = []
    _stub_runner(monkeypatch, ran)
    _run_cli(monkeypatch, cfg, ["--all"], tmp_path)
    ran.clear()
    cfg["stats"]["top_k_words"] = 9999
    _run_cli(monkeypatch, cfg, ["--all"], tmp_path)
    assert ran == ["stats"]


def test_force_stage_invalidates_downstream(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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
    # sentinels were deleted so sentence_dedup and stats must re-run.
    _run_cli(monkeypatch, cfg, ["--all"], tmp_path)
    assert ran == ["sentence_dedup", "stats"]


def test_run_only_one_stage_skips_others(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--stage quality runs only quality (assuming it was stale)."""
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = _setup_output(tmp_path)
    cfg = _common_cfg(input_dir, output_dir)
    ran: List[str] = []
    _stub_runner(monkeypatch, ran)
    _run_cli(monkeypatch, cfg, ["--all", "--stage", "quality"], tmp_path)
    assert ran == ["quality"]


def test_sentinel_records_config_slice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The sentinel JSON stores the actual config slice the stage saw."""
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = _setup_output(tmp_path)
    cfg = _common_cfg(input_dir, output_dir)
    ran: List[str] = []
    _stub_runner(monkeypatch, ran)
    _run_cli(monkeypatch, cfg, ["--all", "--stage", "quality"], tmp_path)
    sentinel = read_sentinel(output_dir / STAGE_DIRS["quality"])
    assert sentinel is not None
    assert sentinel.config_slice == {"min_doc_words": 50}
    # The hash includes the dataset_keys_bytes payload, so it does NOT equal
    # config_hash(slice_) alone. Just assert the hash was recorded and is a
    # sha256 hex prefixed string.
    assert sentinel.config_hash.startswith("sha256:")


def test_sentinel_records_counts_from_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The sentinel stores the record counts the stage runner returned."""
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = _setup_output(tmp_path)
    cfg = _common_cfg(input_dir, output_dir)
    ran: List[str] = []
    _stub_runner(monkeypatch, ran)
    _run_cli(monkeypatch, cfg, ["--all", "--stage", "quality"], tmp_path)
    sentinel = read_sentinel(output_dir / STAGE_DIRS["quality"])
    assert sentinel is not None
    records_in, records_out = _STUB_COUNTS
    assert sentinel.records_in == records_in
    assert sentinel.records_out == records_out
