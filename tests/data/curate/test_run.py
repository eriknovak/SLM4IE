"""Integration tests for the curation stage runner.

The stage executors are replaced by a stub that writes real shards — each
stage appends a tag derived from its config slice to every document — so the
sentinel lineage, the document digests, the atomic swap and the integrity
check all run for real. The real builders are tested in test_curate_pipeline.py.
"""

import gzip
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import pytest

pytest.importorskip("datatrove")

import scripts.curate_pretraining_corpus as curate_cli  # noqa: E402
import slm4ie.data.curate.config as curate_config  # noqa: E402
import slm4ie.data.curate.paths as curate_paths  # noqa: E402
import slm4ie.data.curate.stages as curate_stages  # noqa: E402
import slm4ie.data.curate.status as curate_status  # noqa: E402
import slm4ie.data.curate.run as curate_runner  # noqa: E402
from slm4ie.data.curate.stages.spam import SpamFilter, load_spam_assets  # noqa: E402
from slm4ie.utils.versioning import read_lock, write_lock  # noqa: E402
from slm4ie.data.curate import (
    STAGE_DIRS,
    STAGE_NAMES,
)
from slm4ie.data.curate.lineage import read_sentinel
from slm4ie.data.curate.paths import read_bucket_index
from slm4ie.data.curate.lineage import (
    CONFIG_CHANGED,
    NOT_BUILT,
    INPUT_CHANGED,
    LEGACY,
    OUTPUT_CHANGED,
    SENTINEL_NAME,
    STAGE_VERSION_CHANGED,
    UNVERSIONED,
)

#: Single dataset key the default roster exposes.
_DATASET: str = "d1"

#: Fixed (records_in, records_out) the stub reports for the statistics stage.
_STUB_STATS_COUNTS: Tuple[int, int] = (100, 0)

#: Documents per dataset in the extracted fixture.
_DOCS_PER_DATASET: int = 4


def _write_extracted(input_dir: Path, keys: List[str]) -> None:
    """Write an extracted `<key>.jsonl` with a few documents per key.

    Args:
        input_dir: Extraction-tier folder.
        keys: Dataset keys to write.
    """
    input_dir.mkdir(parents=True, exist_ok=True)
    for key in keys:
        lines = [json.dumps({"uid": f"{key}:{i}", "text": f"{key} dokument {i}"}) for i in range(_DOCS_PER_DATASET)]
        (input_dir / f"{key}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_docs(folder: Path) -> List[Dict[str, Any]]:
    """Read every document from the shards under *folder*, in shard order."""
    docs = []
    for shard in sorted(folder.glob("*.jsonl.gz")):
        with gzip.open(shard, "rt", encoding="utf-8") as fh:
            docs.extend(json.loads(line) for line in fh if line.strip())
    return docs


def _write_docs(folder: Path, docs: List[Dict[str, Any]], shards: int) -> None:
    """Write *docs* round-robin into *shards* rank-named shards, like datatrove does."""
    folder.mkdir(parents=True, exist_ok=True)
    for rank in range(shards):
        part = docs[rank::shards]
        if part:
            with gzip.open(folder / f"{rank:05d}.jsonl.gz", "wt", encoding="utf-8") as fh:
                fh.writelines(json.dumps(doc) + "\n" for doc in part)


class _Stub:
    """Stands in for `run_stage`, recording each run and writing real shards.

    Attributes:
        ran: `(stage, dataset keys)` per executed stage run.
        log_dirs: The executor log folder of each run, in order.
        duplicate: Stage whose next run writes one document twice.
        drop: Stage whose next run keeps no documents.
    """

    def __init__(self) -> None:
        """Start with no recorded runs."""
        self.ran: List[Tuple[str, Tuple[str, ...]]] = []
        self.log_dirs: List[Path] = []
        self.duplicate: Optional[str] = None
        self.drop: Optional[str] = None

    @property
    def stages(self) -> List[str]:
        """Return the stage name of each recorded run."""
        return [stage for stage, _ in self.ran]

    def __call__(self, stage: str, job: Any) -> Tuple[int, int]:
        """Run the stubbed stage on *job*."""
        self.ran.append((stage, tuple(job.dataset_keys)))
        self.log_dirs.append(job.paths.logs_dir(stage))
        if stage == "statistics":
            job.output_folder.mkdir(parents=True, exist_ok=True)
            (job.output_folder / "aggregate.json").write_text("{}", encoding="utf-8")
            return _STUB_STATS_COUNTS
        tag = json.dumps(job.config, sort_keys=True)
        total = 0
        for key in job.dataset_keys:
            if stage == "convert":
                lines = (job.paths.input_folder / f"{key}.jsonl").read_text(encoding="utf-8").splitlines()
                records = [json.loads(line) for line in lines if line]
                docs = [{"text": r["text"], "id": r["uid"], "metadata": {"dataset": key}} for r in records]
            else:
                docs = _read_docs(job.input_view / key)
            docs = [{**d, "text": f"{d['text']}|{stage}:{tag}"} for d in docs]
            if self.duplicate == stage:
                docs.append(docs[0])
            if self.drop == stage:
                docs = []
            _write_docs(job.output_folder / key, docs, shards=job.workers)
            total += len(docs)
        return total, total


def _run_cli(
    monkeypatch: pytest.MonkeyPatch,
    cfg: Dict[str, Any],
    args: List[str],
    project_root: Path,
    roster: Optional[List[str]] = None,
    command: str = "run",
) -> None:
    """Drive `curate_cli.main()` with a stubbed YAML loader and key list.

    Args:
        monkeypatch: pytest's monkeypatch fixture.
        cfg: curate.yaml-equivalent dict.
        args: CLI argv tail after the subcommand and `--config`.
        project_root: Stubbed return value for `_find_project_root`.
        roster: Dataset keys `extract.yaml` declares.
        command: Subcommand to run.
    """
    monkeypatch.setattr(curate_config, "_load_yaml", lambda _p: cfg)
    monkeypatch.setattr(curate_config, "_load_stopwords", lambda _cfg: (set(), b""))
    monkeypatch.setattr(
        curate_config,
        "_load_spam_assets",
        lambda _cfg: SimpleNamespace(adult_words={}, spam_words={}, domains=set(), raw_bytes=b""),
    )
    monkeypatch.setattr(curate_config, "_find_project_root", lambda: project_root)
    monkeypatch.setattr(curate_config, "_list_datasets", lambda _p: list(roster or [_DATASET]))
    # _load_yaml is stubbed, so the path only has to satisfy the required flag.
    monkeypatch.setattr(
        curate_cli.sys,
        "argv",
        ["curate_pretraining_corpus.py", command, "--config", str(project_root / "curation.yaml"), *args],
    )
    curate_cli.main()


class _Env:
    """A stubbed curation environment rooted at a pytest tmp_path.

    Attributes:
        root: The tmp folder, also the project root the lock file lands in.
        output_dir: Curation output root.
        cfg: The curate.yaml-equivalent config.
        roster: Dataset keys in the roster.
        stub: The installed stage stub.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch, root: Path, roster: Optional[List[str]] = None) -> None:
        """Write the extracted fixture and install the stage stub."""
        self.monkeypatch = monkeypatch
        self.root = root
        self.roster = roster or [_DATASET]
        input_dir = root / "in"
        self.output_dir = root / "out"
        _write_extracted(input_dir, self.roster)
        self.cfg: Dict[str, Any] = {
            "input_dir": str(input_dir),
            "output_dir": str(self.output_dir),
            "convert": {"text_field": "text"},
            "language": {"targets": ["sl"]},
            "spam": {"languages": ["sl"]},
            "quality": {"min_doc_words": 50},
            "repetition": {},
            "exact_dedup": {"precision": 64, "hash_fc": "xxhash"},
            "sentence_dedup": {"n_sentences": 3},
            "statistics": {"top_k_words": 5000},
        }
        self.stub = _Stub()
        monkeypatch.setattr(curate_runner, "run_stage", self.stub)

    def run(self, *args: str) -> List[str]:
        """Run the CLI's `run` subcommand and return the stages that executed."""
        self.stub.ran.clear()
        self.stub.log_dirs.clear()
        _run_cli(self.monkeypatch, self.cfg, list(args), self.root, self.roster)
        return self.stub.stages

    def status(self, *args: str) -> Tuple[int, Dict[Tuple[str, str], Tuple[str, str]]]:
        """Run the `status` subcommand; return its exit code and each unit's state and reason."""
        self.stub.ran.clear()
        results: List[Any] = []
        real_status = curate_status.status

        def capture(**kwargs: Any) -> Any:
            results.extend(real_status(**kwargs))
            return results

        self.monkeypatch.setattr(curate_cli, "status", capture)
        code = 0
        try:
            _run_cli(self.monkeypatch, self.cfg, list(args), self.root, self.roster, command="status")
        except SystemExit as exc:
            code = int(exc.code or 0)
        return code, {(u.stage, u.dataset or "-"): (u.state, u.reason or "") for u in results}

    def unit(self, stage: str, key: Optional[str] = _DATASET) -> Path:
        """Return a unit's output folder."""
        folder = self.output_dir / STAGE_DIRS[stage]
        return folder / key if key and stage in curate_stages.SCOPED_STAGES else folder


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _Env:
    """Return a stubbed single-dataset curation environment."""
    return _Env(monkeypatch, tmp_path)


def test_first_run_executes_every_stage(env: _Env) -> None:
    """A clean output_dir causes every stage to run."""
    assert env.run("--all") == list(STAGE_NAMES)


def test_unchanged_rerun_runs_no_stage(env: _Env) -> None:
    """Running twice with identical config runs everything once then nothing."""
    env.run("--all")
    assert env.run("--all") == []


def test_worker_count_change_runs_no_stage(env: _Env) -> None:
    """The worker count is a runtime knob: changing it rebuilds nothing."""
    env.run("--all", "--max-workers", "1")
    assert env.run("--all", "--max-workers", "3") == []


def test_touched_and_copied_files_run_no_stage(env: _Env) -> None:
    """New mtimes on identical bytes — shards and the extracted input — rebuild nothing."""
    env.run("--all")
    for path in list(env.output_dir.rglob("*")) + list((env.root / "in").glob("*")):
        if path.is_file():
            copy = path.with_name(path.name + ".copy")
            shutil.copy(path, copy)
            os.replace(copy, path)
            os.utime(path, ns=(4_000_000_000_000_000_000, 4_000_000_000_000_000_000))
    assert env.run("--all") == []


def test_stage_version_bump_with_same_documents_stops_at_that_stage(env: _Env, monkeypatch: pytest.MonkeyPatch) -> None:
    """A version bump reruns its stage; identical documents keep downstream current."""
    env.run("--all")
    monkeypatch.setitem(curate_stages.STAGE_VERSIONS, "quality", "sha256:edited")
    assert env.run("--all") == ["quality"]
    assert env.run("--all") == []


def test_quality_config_change_cascades(env: _Env) -> None:
    """Editing quality config reruns quality and downstream, not upstream stages."""
    env.run("--all")
    env.cfg["quality"]["min_doc_words"] = 100
    assert env.run("--all") == ["quality", "repetition", "exact_dedup", "sentence_dedup", "statistics"]


def test_statistics_config_change_only_reruns_statistics(env: _Env) -> None:
    """Editing statistics config reruns only statistics."""
    env.run("--all")
    env.cfg["statistics"]["top_k_words"] = 9999
    assert env.run("--all") == ["statistics"]


def test_override_rebuilds_only_its_dataset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A per-dataset override rebuilds that dataset downstream; other datasets stay current."""
    env = _Env(monkeypatch, tmp_path, roster=["d1", "d2"])
    env.run("--all")
    env.cfg["overrides"] = {"d2": {"quality": {"min_doc_words": 10}}}
    env.run("--all")
    assert env.stub.ran == [
        ("quality", ("d2",)),
        ("repetition", ("d2",)),
        ("exact_dedup", ("d1", "d2")),
        ("sentence_dedup", ("d1", "d2")),
        ("statistics", ("d1", "d2")),
    ]


def test_config_buckets_log_into_their_own_folders(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Each bucket's executor logs under the bucket's hash, and the stage index names every bucket."""
    env = _Env(monkeypatch, tmp_path, roster=["d1", "d2"])
    env.cfg["overrides"] = {"d2": {"quality": {"min_doc_words": 10}}}
    env.run("--all")
    stage_logs = env.output_dir / "_logs" / "quality"
    quality_dirs = [d for (stage, _), d in zip(env.stub.ran, env.stub.log_dirs) if stage == "quality"]
    assert len(quality_dirs) == 2 and len(set(quality_dirs)) == 2
    assert all(d.parent == stage_logs for d in quality_dirs)
    index = read_bucket_index(stage_logs)
    assert sorted(entry["datasets"] for entry in index.values()) == [["d1"], ["d2"]]
    assert {stage_logs / folder for folder in index} == set(quality_dirs)
    corpus_dirs = [d for (stage, _), d in zip(env.stub.ran, env.stub.log_dirs) if stage == "exact_dedup"]
    assert corpus_dirs == [env.output_dir / "_logs" / "exact_dedup"]


def test_partial_rerun_of_a_bucket_keeps_the_earlier_logs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Rebuilding one dataset of a bucket logs into a new folder; the index moves only that dataset."""
    env = _Env(monkeypatch, tmp_path, roster=["d1", "d2"])
    quality_dirs: List[Path] = []
    for args in (["--all"], ["--force", "d1"]):
        env.run(*args)
        quality_dirs += [d for (stage, _), d in zip(env.stub.ran, env.stub.log_dirs) if stage == "quality"]
    assert len(quality_dirs) == 2 and quality_dirs[0] != quality_dirs[1]
    index = read_bucket_index(env.output_dir / "_logs" / "quality")
    assert {folder: entry["datasets"] for folder, entry in index.items()} == {
        quality_dirs[0].name: ["d2"],
        quality_dirs[1].name: ["d1"],
    }


def test_stale_shards_cannot_survive_a_rebuild(env: _Env) -> None:
    """Higher-rank shards from a wider run, or stray files, are gone after a rebuild."""
    env.run("--all", "--max-workers", "3")
    quality = env.unit("quality")
    assert sorted(p.name for p in quality.glob("*.jsonl.gz")) == ["00000.jsonl.gz", "00001.jsonl.gz", "00002.jsonl.gz"]
    env.cfg["quality"]["min_doc_words"] = 100
    env.run("--all", "--max-workers", "1")
    assert sorted(p.name for p in quality.glob("*.jsonl.gz")) == ["00000.jsonl.gz"]
    assert len(_read_docs(quality)) == _DOCS_PER_DATASET

    shutil.copy(quality / "00000.jsonl.gz", quality / "00007.jsonl.gz")
    assert env.status()[1][("quality", _DATASET)] == ("stale", OUTPUT_CHANGED)
    assert env.run("--all")[0] == "quality"
    assert sorted(p.name for p in quality.glob("*.jsonl.gz")) == ["00000.jsonl.gz"]


@pytest.mark.parametrize("stage", ["quality", "exact_dedup"])
def test_integrity_failure_keeps_previous_output(env: _Env, stage: str) -> None:
    """A stage that writes an id more often than it read it raises and promotes nothing."""
    env.run("--all")
    unit = env.unit(stage)
    before = {p.relative_to(unit): p.read_bytes() for p in unit.rglob("*") if p.is_file()}
    env.cfg[stage]["tweak"] = True
    env.stub.duplicate = stage
    with pytest.raises(RuntimeError, match="integrity check failed"):
        env.run("--all")
    assert {p.relative_to(unit): p.read_bytes() for p in unit.rglob("*") if p.is_file()} == before
    env.stub.duplicate = None
    assert env.run("--all")[0] == stage


def test_sentinel_records_lineage_and_counts(env: _Env) -> None:
    """A scoped sentinel stores its config slice, version, digests and per-dataset counts."""
    env.run("--all")
    sentinel = read_sentinel(env.unit("quality"))
    assert sentinel is not None
    assert sentinel.config_slice == {"min_doc_words": 50}
    assert sentinel.config_hash.startswith("sha256:")
    assert sentinel.stage_version == curate_stages.STAGE_VERSIONS["quality"]
    assert (sentinel.records_in, sentinel.records_out) == (_DOCS_PER_DATASET, _DOCS_PER_DATASET)
    assert sentinel.document_digest and sentinel.document_digest.startswith("sum256:")
    assert sentinel.input_digest == read_sentinel(env.unit("spam")).document_digest  # type: ignore[union-attr]
    assert sentinel.shards == {"00000.jsonl.gz": (env.unit("quality") / "00000.jsonl.gz").stat().st_size}
    convert = read_sentinel(env.unit("convert"))
    assert convert is not None and convert.input_files and f"{_DATASET}.jsonl" in convert.input_files


def test_corpus_sentinel_records_counts_from_runner(env: _Env) -> None:
    """A corpus sentinel stores the record counts the stage runner returned."""
    env.run("--all")
    sentinel = read_sentinel(env.unit("statistics"))
    assert sentinel is not None
    assert (sentinel.records_in, sentinel.records_out) == _STUB_STATS_COUNTS
    assert sentinel.document_digest is None


def test_run_writes_lock_file(env: _Env) -> None:
    """A run records every unit's lineage in the lock file beside the config."""
    env.run("--all")
    lock = read_lock(env.root / "curation.lock.yaml")
    assert set(lock) == set(STAGE_NAMES)
    entry = lock["quality"][_DATASET]
    assert entry["document_digest"] == read_sentinel(env.unit("quality")).document_digest  # type: ignore[union-attr]
    assert "completed_at" not in entry


def test_status_reports_reasons_and_exit_code(env: _Env, monkeypatch: pytest.MonkeyPatch) -> None:
    """`status` is 0 when everything is current, else 1 with each unit's reason."""
    env.run("--all")
    code, units = env.status()
    assert code == 0
    assert {state for state, _ in units.values()} == {"current"}
    assert len(units) == len(STAGE_NAMES)

    env.cfg["quality"]["min_doc_words"] = 100
    monkeypatch.setitem(curate_stages.STAGE_VERSIONS, "spam", "sha256:edited")
    code, units = env.status()
    assert code == 1
    assert units[("spam", _DATASET)] == ("stale", STAGE_VERSION_CHANGED)
    assert units[("quality", _DATASET)] == ("stale", CONFIG_CHANGED)
    assert units[("convert", _DATASET)] == ("current", "")
    assert env.stub.ran == []


def test_status_reports_changed_input_and_lock_drift(env: _Env) -> None:
    """A rewritten upstream unit shows downstream as input-changed; an edited lock shows drift."""
    env.run("--all")
    lock_path = env.root / "curation.lock.yaml"
    lock = read_lock(lock_path)
    lock["repetition"][_DATASET]["document_digest"] = "sum256:other"
    write_lock(lock_path, lock)
    sentinel_path = env.unit("quality") / SENTINEL_NAME
    payload = json.loads(sentinel_path.read_text())
    payload["document_digest"] = "sum256:changed"
    sentinel_path.write_text(json.dumps(payload))
    code, units = env.status()
    assert code == 1
    assert units[("repetition", _DATASET)] == ("stale", INPUT_CHANGED)
    assert units[("language", _DATASET)] == ("current", "")
    lock = read_lock(lock_path)
    lock["language"][_DATASET]["records_out"] = 0
    write_lock(lock_path, lock)
    assert env.status()[1][("language", _DATASET)] == ("stale", curate_status.LOCK_DIFFERS)


def _make_legacy(output_dir: Path) -> None:
    """Strip every sentinel under *output_dir* to the pre-lineage fields, and drop the lock beside it."""
    (output_dir.parent / "curation.lock.yaml").unlink(missing_ok=True)
    for path in output_dir.rglob(SENTINEL_NAME):
        payload = json.loads(path.read_text())
        keep = ("completed_at", "config_hash", "config_slice", "records_in", "records_out")
        path.write_text(json.dumps({k: payload[k] for k in keep}))


def test_legacy_sentinels_are_adopted_without_rebuild(env: _Env) -> None:
    """A legacy corpus is refused by `run`, adopted by `status --adopt`, then fully current."""
    env.run("--all")
    _make_legacy(env.output_dir)
    with pytest.raises(RuntimeError, match="status --adopt"):
        env.run("--all")
    assert env.status()[1][("quality", _DATASET)] == ("stale", LEGACY)
    code, units = env.status("--adopt", "--max-workers", "2")
    assert code == 0, units
    assert env.stub.ran == []
    assert env.run("--all") == []


def test_adoption_flags_stray_shards_for_rebuild(env: _Env) -> None:
    """A legacy unit holding a stale duplicate shard fails adoption and is the only one rebuilt."""
    env.run("--all")
    language = env.unit("language")
    shutil.copy(language / "00000.jsonl.gz", language / "00003.jsonl.gz")
    _make_legacy(env.output_dir)
    # Downstream stages were built from the duplicated shards, so their own
    # outputs agree with their inputs; only the language unit breaks the rule.
    code, units = env.status("--adopt")
    assert code == 1
    stale = {unit for unit, (state, _) in units.items() if state == "stale"}
    assert stale == {("language", _DATASET)}
    assert units[("language", _DATASET)][1].startswith("integrity failure")
    assert env.run("--all")[0] == "language"
    assert env.run("--all") == []


def test_corpus_stage_reads_only_roster_datasets(env: _Env) -> None:
    """A corpus stage skips upstream folders of keys outside the roster."""
    env.run("--all")
    stray = env.output_dir / STAGE_DIRS["repetition"] / "benchmark"
    shutil.copytree(env.unit("repetition"), stray)
    env.cfg["exact_dedup"]["precision"] = 32
    env.run("--all", "--stage", "exact_dedup")
    assert env.stub.ran == [("exact_dedup", (_DATASET,))]
    assert not (env.unit("exact_dedup") / "benchmark").exists()
    assert not (env.unit("exact_dedup") / curate_runner.PROGRESS_NAME).exists()


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


class TestPrepareCorpusStage:
    """`_prepare_corpus_stage` resumes a matching unfinished run and clears anything else."""

    def _paths(self, tmp_path: Path) -> Any:
        """Return curate paths with a stale staged shard, completion marker and dedup scratch in place."""
        paths = curate_paths.CuratePaths(input_folder=tmp_path / "in", output_dir=tmp_path / "out")
        _write_stage_shards(paths.staging_dir("sentence_dedup"), "d1", 1)
        (paths.logs_dir("sentence_dedup") / "1_sig" / "completions").mkdir(parents=True)
        (paths.logs_dir("sentence_dedup") / "1_sig" / "completions" / "00000").touch()
        (paths.scratch_dir("sentence_dedup") / "sigs").mkdir(parents=True)
        _write_stage_shards(paths.stage_dir("sentence_dedup"), "d1", 1)
        return paths

    def test_fresh_start_clears_staging_only(self, tmp_path: Path) -> None:
        """Without a progress file, staging, logs and dedup scratch go; the promoted output stays."""
        paths = self._paths(tmp_path)
        staging = paths.staging_dir("sentence_dedup")
        resumed = curate_runner._prepare_corpus_stage(paths, "sentence_dedup", "h", 4, "i")
        assert resumed is False
        assert list(staging.glob("*/*.jsonl.gz")) == []
        assert not paths.logs_dir("sentence_dedup").exists()
        assert not paths.scratch_dir("sentence_dedup").exists()
        assert (staging / curate_runner.PROGRESS_NAME).is_file()
        assert (paths.stage_dir("sentence_dedup") / "d1" / "00000.jsonl.gz").exists()

    def test_matching_progress_resumes(self, tmp_path: Path) -> None:
        """A progress file with the same hash, tasks and inputs keeps finished work."""
        paths = self._paths(tmp_path)
        staging = paths.staging_dir("sentence_dedup")
        curate_runner._prepare_corpus_stage(paths, "sentence_dedup", "h", 4, "i")
        _write_stage_shards(staging, "d1", 1)
        (paths.logs_dir("sentence_dedup") / "1_sig" / "completions").mkdir(parents=True)
        assert curate_runner._prepare_corpus_stage(paths, "sentence_dedup", "h", 4, "i") is True
        assert (staging / "d1" / "00000.jsonl.gz").exists()
        assert (paths.logs_dir("sentence_dedup") / "1_sig" / "completions").exists()

    @pytest.mark.parametrize("changed", [("h2", 4, "i"), ("h", 5, "i"), ("h", 4, "i2")])
    def test_changed_settings_start_fresh(self, tmp_path: Path, changed: Tuple[str, int, str]) -> None:
        """A different config hash, task count or inputs clears the staging folder."""
        paths = self._paths(tmp_path)
        staging = paths.staging_dir("sentence_dedup")
        curate_runner._prepare_corpus_stage(paths, "sentence_dedup", "h", 4, "i")
        _write_stage_shards(staging, "d1", 1)
        assert curate_runner._prepare_corpus_stage(paths, "sentence_dedup", *changed) is False
        assert list(staging.glob("*/*.jsonl.gz")) == []


def test_shard_layout_tracks_shard_layout_not_mtime(tmp_path: Path) -> None:
    """The resume fingerprint counts shards and follows their sizes, not their mtimes."""
    view = tmp_path / "view"
    _write_stage_shards(view, "d1", 2)
    count, digest = curate_paths.shard_layout(view)
    assert count == 2
    os.utime(view / "d1" / "00001.jsonl.gz", ns=(1, 1))
    assert curate_paths.shard_layout(view) == (2, digest)
    (view / "d1" / "00001.jsonl.gz").write_bytes(b"\x1f\x8b\x08")
    assert curate_paths.shard_layout(view)[1] != digest


def test_fully_filtered_dataset_empties_downstream(env: _Env) -> None:
    """A dataset an upstream stage now drops entirely leaves no old documents downstream."""
    env.run("--all")
    env.cfg["quality"]["min_doc_words"] = 100
    env.stub.drop = "quality"
    env.run("--all")
    assert _read_docs(env.unit("repetition")) == []
    assert read_sentinel(env.unit("repetition")).records_out == 0  # type: ignore[union-attr]
    assert not (env.unit("exact_dedup") / _DATASET).exists()
    env.stub.drop = None
    assert env.run("--all") == []


def test_legacy_sentinel_with_changed_config_is_rebuilt(env: _Env) -> None:
    """A legacy unit whose config changed is rebuilt rather than blocking the run."""
    env.run("--all")
    _make_legacy(env.output_dir)
    env.status("--adopt")
    for path in env.unit("quality").glob(SENTINEL_NAME):
        payload = json.loads(path.read_text())
        path.write_text(json.dumps({k: payload[k] for k in ("completed_at", "config_hash", "config_slice")}))
    env.cfg["quality"]["min_doc_words"] = 100
    assert env.run("--all")[0] == "quality"


def test_interrupted_swap_is_finished_not_rebuilt(env: _Env) -> None:
    """A checked unit left in staging by a crash between the swap's renames is promoted."""
    env.run("--all")
    staged = env.output_dir / "_partial" / STAGE_DIRS["quality"] / _DATASET
    staged.parent.mkdir(parents=True)
    os.rename(env.unit("quality"), staged)
    assert env.run("--all") == []
    assert (env.unit("quality") / SENTINEL_NAME).is_file()


def test_status_keeps_unit_whose_input_is_gone(env: _Env) -> None:
    """A built unit whose extracted file vanished is reported, not stale, matching what run does."""
    env.run("--all")
    (env.root / "in" / f"{_DATASET}.jsonl").unlink()
    code, units = env.status()
    assert code == 0
    assert units[("convert", _DATASET)] == ("missing", curate_status.NO_INPUT)
    assert env.run("--all") == []
    assert NOT_BUILT


def test_partial_rebuild_updates_lock_file(env: _Env) -> None:
    """A subset run that rebuilds units records them; one that rebuilds nothing leaves the file as is."""
    lock_path = env.root / "curation.lock.yaml"
    env.run("--all")
    before = lock_path.read_bytes()
    mtime = lock_path.stat().st_mtime_ns
    env.run(_DATASET)
    assert lock_path.stat().st_mtime_ns == mtime
    env.cfg["spam"]["min_spam_hits"] = 5
    env.run(_DATASET, "--stage", "spam")
    lock = read_lock(lock_path)
    assert lock_path.read_bytes() != before
    assert lock["spam"][_DATASET]["config_hash"] == read_sentinel(env.unit("spam")).config_hash  # type: ignore[union-attr]


def test_units_without_a_code_version_are_stamped_not_rebuilt(env: _Env) -> None:
    """Units recorded before code versions block `run` until adoption stamps the current versions."""
    env.run("--all")
    for path in env.output_dir.rglob(SENTINEL_NAME):
        payload = json.loads(path.read_text())
        payload["stage_version"] = "1"
        path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="status --adopt"):
        env.run("--all")
    assert env.status()[1][("quality", _DATASET)] == ("stale", UNVERSIONED)
    code, units = env.status("--adopt")
    assert code == 0, units
    assert env.stub.ran == []
    assert read_sentinel(env.unit("quality")).stage_version == curate_runner.STAGE_VERSIONS["quality"]  # type: ignore[union-attr]
    assert env.run("--all") == []


def _spam_run(monkeypatch: pytest.MonkeyPatch, root: Path, *, exclusion: bool) -> Path:
    """Run every stage, the spam stage for real, over a fixture with two adult docs.

    Args:
        monkeypatch: pytest's monkeypatch fixture.
        root: tmp folder for this environment.
        exclusion: Keep the stage's exclusion writer; when False it is removed.

    Returns:
        The spam unit's output folder.
    """
    env = _Env(monkeypatch, root)
    texts = ["porno in seks", "lep dan", "pornič in seksi oglas", "seks, seksa"]
    lines = [json.dumps({"uid": f"{_DATASET}:{i}", "text": t}) for i, t in enumerate(texts)]
    (Path(env.cfg["input_dir"]) / f"{_DATASET}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    assets = load_spam_assets(["sl"], url_blocklist=False)
    stub = env.stub

    def run_stage(stage: str, job: Any) -> Tuple[int, int]:
        if stage != "spam":
            return stub(stage, job)
        return curate_stages.run_stage(stage, replace(job, spam_assets=assets))

    monkeypatch.setattr(curate_runner, "run_stage", run_stage)
    if not exclusion:
        init = SpamFilter.__init__
        monkeypatch.setattr(SpamFilter, "__init__", lambda self, **kw: init(self, **{**kw, "exclusion_writer": None}))
    env.run("--all")
    return env.unit("spam")


def test_spam_writes_dropped_docs_without_touching_lineage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Dropped docs land in the unit's `removed/` folder; digest and counts match a run without them."""
    unit = _spam_run(monkeypatch, tmp_path / "with", exclusion=True)
    removed = _read_docs(unit / "removed")
    assert sorted(d["id"] for d in removed) == [f"{_DATASET}:0", f"{_DATASET}:2"]
    assert all(d["metadata"]["spam_reason"] == "adult_lexicon" and d["metadata"]["spam_terms"] for d in removed)
    with_writer = read_sentinel(unit)
    quality = read_sentinel(unit.parent.parent / STAGE_DIRS["quality"] / _DATASET)
    assert quality is not None and quality.records_in == 2

    bare = _spam_run(monkeypatch, tmp_path / "without", exclusion=False)
    assert not (bare / "removed").exists()
    without_writer = read_sentinel(bare)
    assert with_writer is not None and without_writer is not None
    assert with_writer.document_digest == without_writer.document_digest
    assert (with_writer.records_in, with_writer.records_out) == (4, 2)
    assert without_writer.records_out == with_writer.records_out
