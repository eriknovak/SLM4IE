"""Tests for the per-stage curate pipeline builders.

Structural assertions only; the heavy end-to-end smoke test lives at
the bottom of this file (marked `@pytest.mark.slow`).
"""

import importlib.metadata  # noqa: F401  (datatrove workaround)
import importlib.util  # noqa: F401  (datatrove workaround)
from pathlib import Path

import pytest

pytest.importorskip("datatrove")

from datatrove.pipeline.dedup import (  # noqa: E402
    ExactDedupFilter,
    ExactDedupSignature,
    ExactFindDedups,
    SentenceDedupFilter,
    SentenceDedupSignature,
    SentenceFindDedups,
)
from datatrove.pipeline.filters import (  # noqa: E402
    GopherQualityFilter,
    GopherRepetitionFilter,
)
from datatrove.pipeline.readers import JsonlReader  # noqa: E402
from datatrove.pipeline.writers.jsonl import JsonlWriter  # noqa: E402
from datatrove.utils.typeshelper import Languages  # noqa: E402

from slm4ie.data.curate.language import LinguaLanguageFilter  # noqa: E402
from slm4ie.data.curate.pipeline import (  # noqa: E402
    CuratePaths,
    QualityConfig,
    build_exact_dedup_executors,
    build_language_executors,
    build_quality_executors,
    build_repetition_executors,
    build_sentence_dedup_executors,
    build_spam_executors,
    build_stats_executors,
)
from slm4ie.data.curate.spam import SpamConfig, SpamFilter  # noqa: E402
from slm4ie.data.curate.stats import CorpusStats, CorpusStatsReduce  # noqa: E402


def _paths(tmp_path: Path) -> CuratePaths:
    """Build a CuratePaths anchored under *tmp_path* for structural tests."""
    return CuratePaths(
        input_folder=tmp_path / "datatrove",
        output_dir=tmp_path / "curated",
    )


class TestLanguageStage:
    """The language stage is a single parallel executor."""

    def test_returns_one_executor(self, tmp_path: Path) -> None:
        """The language stage runs as a single executor."""
        execs = build_language_executors(_paths(tmp_path))
        assert len(execs) == 1
        assert execs[0].depends is None

    def test_pipeline_contains_lingua_and_writer(self, tmp_path: Path) -> None:
        """The pipeline reads input, applies lingua, writes to 01_language/."""
        execs = build_language_executors(_paths(tmp_path))
        types_ = [type(s) for s in execs[0].pipeline]
        assert JsonlReader in types_
        assert LinguaLanguageFilter in types_
        assert JsonlWriter in types_

    def test_writes_to_language_folder(self, tmp_path: Path) -> None:
        """The writer's output_folder is `<output_dir>/01_language`."""
        paths = _paths(tmp_path)
        execs = build_language_executors(paths)
        writer = next(s for s in execs[0].pipeline if isinstance(s, JsonlWriter))
        assert str(paths.stage_dir("language")) in writer.output_folder.path

    def test_lang_minimum_relative_distance_is_threaded(self, tmp_path: Path) -> None:
        """`minimum_relative_distance` reaches the LinguaLanguageFilter."""
        execs = build_language_executors(
            _paths(tmp_path), lang_minimum_relative_distance=0.15
        )
        lang = next(s for s in execs[0].pipeline if isinstance(s, LinguaLanguageFilter))
        assert lang.minimum_relative_distance == 0.15


class TestSpamStage:
    """The spam stage reads 01_language/ and writes 02_spam/."""

    def test_returns_one_executor(self, tmp_path: Path) -> None:
        """The spam stage runs as a single executor."""
        execs = build_spam_executors(_paths(tmp_path))
        assert len(execs) == 1

    def test_pipeline_contains_spam_filter(self, tmp_path: Path) -> None:
        """The pipeline reads input, applies SpamFilter, writes shards."""
        execs = build_spam_executors(_paths(tmp_path))
        types_ = [type(s) for s in execs[0].pipeline]
        assert JsonlReader in types_
        assert SpamFilter in types_
        assert JsonlWriter in types_

    def test_writes_to_spam_folder(self, tmp_path: Path) -> None:
        """The writer's output_folder is `<output_dir>/02_spam`."""
        paths = _paths(tmp_path)
        execs = build_spam_executors(paths)
        writer = next(s for s in execs[0].pipeline if isinstance(s, JsonlWriter))
        assert str(paths.stage_dir("spam")) in writer.output_folder.path

    def test_config_and_assets_threaded(self, tmp_path: Path) -> None:
        """SpamConfig and lexicon assets reach the underlying SpamFilter."""
        cfg = SpamConfig(min_adult_hits=5, use_ldnoobw=False)
        execs = build_spam_executors(
            _paths(tmp_path),
            spam_config=cfg,
            adult_words={"sl": {"porno"}},
            spam_words={"sl": {"viagra"}},
            domains={"pornhub.com"},
        )
        spam = next(s for s in execs[0].pipeline if isinstance(s, SpamFilter))
        assert spam.config.min_adult_hits == 5
        assert "pornhub.com" in spam.domains


class TestQualityStage:
    """The quality stage reads 02_spam/ and writes 03_quality/."""

    def test_returns_one_executor(self, tmp_path: Path) -> None:
        """The quality stage runs as a single executor."""
        execs = build_quality_executors(_paths(tmp_path))
        assert len(execs) == 1

    def test_pipeline_contains_gopher_quality(self, tmp_path: Path) -> None:
        """The pipeline runs GopherQualityFilter but NOT the repetition filter."""
        execs = build_quality_executors(_paths(tmp_path))
        types_ = [type(s) for s in execs[0].pipeline]
        assert GopherQualityFilter in types_
        assert GopherRepetitionFilter not in types_

    def test_quality_config_threaded(self, tmp_path: Path) -> None:
        """QualityConfig overrides reach the underlying GopherQualityFilter."""
        cfg = QualityConfig(min_doc_words=10, max_doc_words=200, min_stop_words=0)
        execs = build_quality_executors(_paths(tmp_path), quality_config=cfg)
        quality = next(s for s in execs[0].pipeline if isinstance(s, GopherQualityFilter))
        assert quality.min_doc_words == 10
        assert quality.max_doc_words == 200
        assert quality.min_stop_words == 0

    def test_stopwords_become_gopher_stop_words(self, tmp_path: Path) -> None:
        """Stopwords are wired into GopherQualityFilter."""
        execs = build_quality_executors(_paths(tmp_path), stopwords={"in", "je", "na"})
        quality = next(s for s in execs[0].pipeline if isinstance(s, GopherQualityFilter))
        assert {"in", "je", "na"}.issubset(quality.stop_words)


class TestRepetitionStage:
    """The repetition stage reads 02_quality/ and writes 03_repetition/."""

    def test_returns_one_executor(self, tmp_path: Path) -> None:
        """The repetition stage runs as a single executor."""
        execs = build_repetition_executors(_paths(tmp_path))
        assert len(execs) == 1

    def test_pipeline_contains_repetition_filter(self, tmp_path: Path) -> None:
        """The pipeline runs GopherRepetitionFilter but NOT the quality filter."""
        execs = build_repetition_executors(_paths(tmp_path))
        types_ = [type(s) for s in execs[0].pipeline]
        assert GopherRepetitionFilter in types_
        assert GopherQualityFilter not in types_


class TestExactDedupStage:
    """Exact dedup is three internal executors: sig -> find -> filter+write."""

    def test_returns_three_executors_chained(self, tmp_path: Path) -> None:
        """The stage returns three executors chained via `depends`."""
        execs = build_exact_dedup_executors(_paths(tmp_path))
        assert len(execs) == 3
        assert execs[0].depends is None
        assert execs[1].depends is execs[0]
        assert execs[2].depends is execs[1]

    def test_executor_blocks(self, tmp_path: Path) -> None:
        """Each internal executor carries the right datatrove block."""
        execs = build_exact_dedup_executors(_paths(tmp_path))
        types_ = [[type(s) for s in ex.pipeline] for ex in execs]
        assert ExactDedupSignature in types_[0]
        assert ExactFindDedups in types_[1]
        assert ExactDedupFilter in types_[2]
        assert JsonlWriter in types_[2]
        assert SentenceDedupSignature not in types_[0] + types_[1] + types_[2]
        assert SentenceDedupFilter not in types_[0] + types_[1] + types_[2]

    def test_finder_workers_propagates(self, tmp_path: Path) -> None:
        """`finder_workers` reaches the signature stage and the find executor."""
        execs = build_exact_dedup_executors(_paths(tmp_path), finder_workers=4)
        sig = next(s for s in execs[0].pipeline if isinstance(s, ExactDedupSignature))
        assert sig.finder_workers == 4
        assert execs[1].tasks == 4


class TestSentenceDedupStage:
    """Sentence dedup is three internal executors: sig -> find -> filter+write."""

    def test_returns_three_executors_chained(self, tmp_path: Path) -> None:
        """The stage returns three executors chained via `depends`."""
        execs = build_sentence_dedup_executors(_paths(tmp_path))
        assert len(execs) == 3
        assert execs[0].depends is None
        assert execs[1].depends is execs[0]
        assert execs[2].depends is execs[1]

    def test_executor_blocks(self, tmp_path: Path) -> None:
        """Each internal executor carries the right datatrove block."""
        execs = build_sentence_dedup_executors(_paths(tmp_path))
        types_ = [[type(s) for s in ex.pipeline] for ex in execs]
        assert SentenceDedupSignature in types_[0]
        assert SentenceFindDedups in types_[1]
        assert SentenceDedupFilter in types_[2]
        assert JsonlWriter in types_[2]
        assert ExactDedupSignature not in types_[0] + types_[1] + types_[2]
        assert ExactDedupFilter not in types_[0] + types_[1] + types_[2]

    def test_sentence_blocks_run_in_slovenian(self, tmp_path: Path) -> None:
        """Sentence sig/filter use Languages.slovenian by default."""
        execs = build_sentence_dedup_executors(_paths(tmp_path))
        sent_sig = next(s for s in execs[0].pipeline if isinstance(s, SentenceDedupSignature))
        sent_filter = next(s for s in execs[2].pipeline if isinstance(s, SentenceDedupFilter))
        assert sent_sig.language == Languages.slovenian
        assert sent_filter.language == Languages.slovenian

    def test_sentence_config_threaded(self, tmp_path: Path) -> None:
        """SentDedupConfig overrides reach the sig stage."""
        cfg = SentDedupConfig(
            n_sentences=4, min_doc_words=10, min_num_sentences=1, split_sentences=True
        )
        execs = build_sentence_dedup_executors(_paths(tmp_path), sentence_config=cfg)
        sig = next(s for s in execs[0].pipeline if isinstance(s, SentenceDedupSignature))
        assert sig.config.n_sentences == 4


class TestStatsStage:
    """The stats stage runs as map (N workers) → reduce (1 worker)."""

    def test_returns_map_and_reduce_executors(self, tmp_path: Path) -> None:
        """Stats returns two executors with the reduce depending on the map."""
        execs = build_stats_executors(_paths(tmp_path), tasks=4)
        assert len(execs) == 2
        map_exec, reduce_exec = execs
        assert map_exec.tasks == 4
        assert map_exec.workers == 4
        assert reduce_exec.tasks == 1
        assert reduce_exec.workers == 1
        assert reduce_exec.depends is map_exec

    def test_map_pipeline_contains_reader_and_corpus_stats(self, tmp_path: Path) -> None:
        """The map pipeline reads JSONL and runs CorpusStats in partials mode."""
        execs = build_stats_executors(_paths(tmp_path))
        map_types = [type(s) for s in execs[0].pipeline]
        assert JsonlReader in map_types
        assert CorpusStats in map_types
        stats_step = next(s for s in execs[0].pipeline if isinstance(s, CorpusStats))
        assert stats_step.partials_dir is not None

    def test_reduce_pipeline_contains_corpus_stats_reduce(self, tmp_path: Path) -> None:
        """The reduce pipeline is a single CorpusStatsReduce step."""
        execs = build_stats_executors(_paths(tmp_path))
        reduce_types = [type(s) for s in execs[1].pipeline]
        assert reduce_types == [CorpusStatsReduce]

    def test_stats_reads_from_final_corpus_folder(self, tmp_path: Path) -> None:
        """The reader's data_folder is `<output_dir>/04_2_dedup`."""
        paths = _paths(tmp_path)
        execs = build_stats_executors(paths)
        reader = next(s for s in execs[0].pipeline if isinstance(s, JsonlReader))
        assert str(paths.stage_dir("sentence_dedup")) in reader.data_folder.path


# --- End-to-end smoke test ----------------------------------------------------
# Builds two synthetic shards with one cross-shard verbatim duplicate and one
# shared 3-sentence span, runs `build_curate_executors(...)` against a tempdir,
# and asserts the final corpus contains the expected survivors plus a
# `statistics/` folder with both an aggregate JSON and per-dataset breakdowns.

import gzip  # noqa: E402
import json  # noqa: E402
from typing import List  # noqa: E402

from datatrove.pipeline.dedup import SentDedupConfig  # noqa: E402

pytest.importorskip("lingua")


SHARED_DOC = (
    "Slovenščina je uradni jezik Republike Slovenije. "
    "Govori jo približno dva milijona ljudi. "
    "V Evropski uniji je eden od uradnih jezikov. "
    "Spada v skupino južnoslovanskih jezikov. "
    "Razvijala se je iz praslovanščine pred več stoletji. "
    "Standardni jezik se uporablja v javnem govoru, šolstvu in medijih. "
    "Pogovorni jezik ima številne narečne različice po vsej državi. "
    "Pisni jezik temelji na latinici z dodatki za posebne glasove."
)
SHARED_SPAN = (
    "Akademske raziskave preučujejo družbene vzorce. "
    "Sociologi analizirajo gibanja prebivalstva. "
    "Lingvisti dokumentirajo razvoj jezika skozi čas."
)
A2_TEXT = (
    SHARED_SPAN
    + " Filozofi razmišljajo o naravi spoznanja. "
      "Zgodovinarji raziskujejo arhive in primarne vire. "
      "Znanstveniki sodelujejo v interdisciplinarnih projektih po vsem svetu. "
      "Rezultati se objavljajo v recenziranih revijah."
)
B2_TEXT = (
    "Pravna pravila urejajo razmerja med posamezniki in državo. "
    + SHARED_SPAN
    + " Sodišča razlagajo zakone v posameznih primerih. "
      "Ustavni sodniki varujejo temeljne pravice državljanov. "
      "Mednarodno pravo ureja odnose med državami v globalnem sistemu."
)


def _write_shard(path: Path, dataset: str, domain: str, docs: List[dict]) -> None:
    """Gzip-write a list of (id, text) docs as datatrove JSONL.

    *path* must point at a `<dataset>/<NNNNN>.jsonl.gz` file inside the
    new sharded layout; the parent folder is created if missing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for d in docs:
            line = {
                "text": d["text"],
                "id": d["id"],
                "dataset": dataset,
                "domain": domain,
            }
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")


@pytest.mark.slow
def test_final_corpus_drops_cross_dataset_duplicates(tmp_path: Path) -> None:
    """Two shards with one full-doc dup and one shared span produce 3 survivors.

    Drives every per-stage builder in sequence against a synthetic input
    and asserts dedup invariants on the `04_2_dedup/` output plus the
    statistics bundle.
    """
    output_dir = tmp_path / "curated"
    # Drop synthetic shards directly into the convert stage's output
    # folder; the language stage reads from `<output_dir>/00_convert/`,
    # so the in-tree convert step is effectively pre-populated here.
    convert_folder = output_dir / "00_convert"
    _write_shard(
        convert_folder / "alpha" / "00000.jsonl.gz",
        dataset="alpha",
        domain="scientific",
        docs=[
            {"id": "alpha:1", "text": SHARED_DOC},
            {"id": "alpha:2", "text": A2_TEXT},
            {"id": "alpha:3", "text": "Solnce sveti nad gorami in dolinami slovenskih krajev. " * 8},
        ],
    )
    _write_shard(
        convert_folder / "beta" / "00000.jsonl.gz",
        dataset="beta",
        domain="legal",
        docs=[
            {"id": "beta:1", "text": SHARED_DOC},
            {"id": "beta:2", "text": B2_TEXT},
            {"id": "beta:3", "text": "Pravna doktrina se razvija s časom in družbenimi spremembami. " * 8},
        ],
    )

    paths = CuratePaths(input_folder=tmp_path / "extracted", output_dir=output_dir)

    loose_quality = QualityConfig(
        min_doc_words=5,
        min_stop_words=0,
        max_non_alpha_words_ratio=0.6,
        max_avg_word_length=15,
    )
    loose_sentence = SentDedupConfig(
        n_sentences=3,
        min_doc_words=5,
        min_num_sentences=1,
        split_sentences=True,
    )

    build_language_executors(paths, tasks=1)[-1].run()
    build_quality_executors(
        paths, tasks=1, quality_config=loose_quality, stopwords=set()
    )[-1].run()
    build_repetition_executors(paths, tasks=1)[-1].run()
    build_exact_dedup_executors(paths, tasks=1)[-1].run()
    build_sentence_dedup_executors(paths, tasks=1, sentence_config=loose_sentence)[-1].run()
    build_stats_executors(paths, stopwords=set())[-1].run()

    final_folder = paths.stage_dir("sentence_dedup")
    survivors: List[str] = []
    survivor_dirs: set = set()
    for shard in sorted(final_folder.glob("**/*.jsonl.gz")):
        survivor_dirs.add(shard.parent.name)
        with gzip.open(shard, "rt", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                survivors.append(rec["id"])

    # 6 input docs:
    #   - alpha:1 / beta:1 are exact duplicates of each other (SHARED_DOC) → 1 survives
    #   - alpha:2 / beta:2 share a 3-sentence span; sentence dedup trims one window
    #     but both docs survive (loose floors)
    #   - alpha:3 / beta:3 are heavy n-gram repetition → both killed by repetition filter
    # Expected: 3 survivors total.
    assert len(survivors) == 3
    assert "alpha:1" in survivors
    assert "beta:1" not in survivors
    assert "alpha:3" not in survivors
    assert "beta:3" not in survivors
    assert survivor_dirs == {"alpha", "beta"}

    stats_folder = paths.stage_dir("stats")
    bundle = json.loads((stats_folder / "aggregate.json").read_text(encoding="utf-8"))
    assert bundle["total_docs"] == 3
    assert "alpha" in bundle["by_dataset"]
    assert "beta" in bundle["by_dataset"]
    assert bundle["by_dataset"]["alpha"]["doc_count"] == 2
    assert bundle["by_dataset"]["beta"]["doc_count"] == 1

    per_dataset_dir = stats_folder / "per_dataset"
    assert (per_dataset_dir / "alpha.json").exists()
    assert (per_dataset_dir / "beta.json").exists()


@pytest.mark.slow
def test_pipeline_io_counts_reports_reader_and_writer_totals(tmp_path: Path) -> None:
    """`pipeline_io_counts` reads records_in from the reader, records_out from the writer.

    Runs the quality stage over three docs — two long enough to clear
    `min_doc_words` and one too short — so the reader sees 3 documents
    and the writer sees 2, and asserts the helper reports `(3, 2)`.
    """
    from slm4ie.data.curate.pipeline import pipeline_io_counts

    paths = CuratePaths(
        input_folder=tmp_path / "extracted", output_dir=tmp_path / "curated"
    )
    _write_shard(
        paths.stage_dir("language") / "alpha" / "00000.jsonl.gz",
        dataset="alpha",
        domain="scientific",
        docs=[
            {"id": "alpha:1", "text": "beseda " * 40},
            {"id": "alpha:2", "text": "beseda " * 40},
            {"id": "alpha:3", "text": "beseda " * 3},
        ],
    )
    quality = QualityConfig(
        min_doc_words=20,
        min_stop_words=0,
        max_non_alpha_words_ratio=0.6,
        max_avg_word_length=15,
    )
    stats = build_quality_executors(
        paths, tasks=1, quality_config=quality, stopwords=set()
    )[-1].run()

    assert pipeline_io_counts(stats) == (3, 2)


# --- CLI: multi-key dataset selection ----------------------------------------
# `scripts/data/to_pretrain.py` accepts either one or more positional dataset
# keys, or `--all`. The CLI lives outside `slm4ie/` but is importable as
# `scripts.data.to_pretrain` because both `scripts/` and `scripts/data/` ship
# empty `__init__.py` files.

from scripts.data import to_pretrain as curate_cli  # noqa: E402


class TestCurateCLISelection:
    """parse_args contract for the new --stage CLI."""

    def test_parse_args_accepts_single_key(self) -> None:
        """A single positional resolves into `args.datasets`."""
        args = curate_cli.parse_args(["kzb"])
        assert args.datasets == ["kzb"]
        assert args.all is False

    def test_parse_args_accepts_multiple_keys(self) -> None:
        """Multiple positionals are gathered into `args.datasets`."""
        args = curate_cli.parse_args(["kzb", "solar"])
        assert args.datasets == ["kzb", "solar"]
        assert args.all is False

    def test_parse_args_accepts_all_flag(self) -> None:
        """`--all` parses without any positional keys."""
        args = curate_cli.parse_args(["--all"])
        assert args.all is True
        assert args.datasets == []

    def test_parse_args_errors_when_nothing_selected(self) -> None:
        """Bare invocation fails: must pass datasets or `--all`."""
        with pytest.raises(SystemExit):
            curate_cli.parse_args([])

    def test_parse_args_default_stage_is_all(self) -> None:
        """No `--stage` flag means run every stage in order."""
        args = curate_cli.parse_args(["--all"])
        assert args.stage == "all"

    def test_parse_args_accepts_each_stage_name(self) -> None:
        """Every documented `--stage` value parses."""
        for name in (
            "convert", "language", "quality", "repetition",
            "exact_dedup", "sentence_dedup", "stats", "all",
        ):
            args = curate_cli.parse_args(["--all", "--stage", name])
            assert args.stage == name

    def test_parse_args_rejects_unknown_stage(self) -> None:
        """An unknown `--stage` value is rejected by argparse."""
        with pytest.raises(SystemExit):
            curate_cli.parse_args(["--all", "--stage", "lang"])

    def test_parse_args_default_workers_is_serial(self) -> None:
        """The default worker count is 1 (serial)."""
        args = curate_cli.parse_args(["--all"])
        assert args.workers == 1

    def test_parse_args_max_workers_zero(self) -> None:
        """`--max-workers 0` is the cpu_default sentinel."""
        args = curate_cli.parse_args(["--all", "--max-workers", "0"])
        assert args.workers == 0

    def test_parse_args_tasks_alias(self) -> None:
        """`--tasks N` is accepted as a back-compat alias for `--max-workers N`."""
        args = curate_cli.parse_args(["--all", "--tasks", "4"])
        assert args.workers == 4

    def test_filter_stage_subset_mirrors_multiple_datasets(self, tmp_path: Path) -> None:
        """`_filter_stage_subset` builds symlinks for every requested key."""
        convert_dir = tmp_path / "00_convert"
        for key in ("kzb", "solar"):
            shard_dir = convert_dir / key
            shard_dir.mkdir(parents=True)
            (shard_dir / "00000.jsonl.gz").write_bytes(b"\x1f\x8b")

        holder = curate_cli._filter_stage_subset(convert_dir, ["kzb", "solar"])
        try:
            assert (holder / "kzb" / "00000.jsonl.gz").is_symlink()
            assert (holder / "solar" / "00000.jsonl.gz").is_symlink()
        finally:
            import shutil

            shutil.rmtree(holder, ignore_errors=True)

    def test_filter_stage_subset_lists_all_missing_keys(self, tmp_path: Path) -> None:
        """Missing shard folders are reported together in one error."""
        convert_dir = tmp_path / "00_convert"
        (convert_dir / "kzb").mkdir(parents=True)
        (convert_dir / "kzb" / "00000.jsonl.gz").write_bytes(b"\x1f\x8b")

        with pytest.raises(FileNotFoundError) as excinfo:
            curate_cli._filter_stage_subset(
                convert_dir, ["kzb", "missing1", "missing2"]
            )
        msg = str(excinfo.value)
        assert "missing1" in msg
        assert "missing2" in msg
        assert "'kzb'" not in msg


def test_quality_executor_honors_input_override(tmp_path: Path) -> None:
    """build_quality_executors reads from input_override when provided."""
    from slm4ie.data.curate.pipeline import CuratePaths, build_quality_executors

    paths = CuratePaths(input_folder=tmp_path / "in", output_dir=tmp_path / "out")
    override = tmp_path / "view"
    override.mkdir()
    execs = build_quality_executors(paths, tasks=1, input_override=override)
    reader = execs[0].pipeline[0]
    assert str(override) in reader.data_folder.path


def test_repetition_executor_honors_input_override(tmp_path: Path) -> None:
    """build_repetition_executors reads from input_override when provided."""
    from slm4ie.data.curate.pipeline import CuratePaths, build_repetition_executors

    paths = CuratePaths(input_folder=tmp_path / "in", output_dir=tmp_path / "out")
    override = tmp_path / "view"
    override.mkdir()
    execs = build_repetition_executors(paths, tasks=1, input_override=override)
    reader = execs[0].pipeline[0]
    assert str(override) in reader.data_folder.path
