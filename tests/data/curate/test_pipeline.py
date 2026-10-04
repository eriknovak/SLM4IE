"""Tests for the per-stage curate pipeline builders.

Structural assertions only; the heavy end-to-end smoke test lives at
the bottom of this file (marked `@pytest.mark.slow`).
"""

import importlib.metadata  # noqa: F401  (datatrove workaround)
import importlib.util  # noqa: F401  (datatrove workaround)
from dataclasses import replace
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

from slm4ie.data.curate.stages.dedup import CompactSentenceDedupSignature  # noqa: E402
from slm4ie.data.curate.stages.language import LinguaLanguageFilter  # noqa: E402
from slm4ie.data.curate.paths import CuratePaths, bucket_log_scope, read_bucket_index, record_bucket
from slm4ie.data.curate.stages.quality import (
    QualityConfig,
    build_quality_executors,
)
from slm4ie.data.curate.stages.dedup import (
    build_exact_dedup_executors,
    build_sentence_dedup_executors,
)
from slm4ie.data.curate.stages.language import build_language_executors
from slm4ie.data.curate.stages.repetition import build_repetition_executors
from slm4ie.data.curate.stages.spam import build_spam_executors
from slm4ie.data.curate.stages.statistics import build_statistics_executors
from slm4ie.data.curate.stages.common import stage_io_counts
from slm4ie.data.curate.stages.spam import (  # noqa: E402
    SpamConfig,
    SpamFilter,
)
from slm4ie.data.curate.stages.statistics import (  # noqa: E402
    CorpusStats,
    CorpusStatsReduce,
)


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
        assert any(issubclass(t, JsonlReader) for t in types_)
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
        execs = build_language_executors(_paths(tmp_path), lang_minimum_relative_distance=0.15)
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
        assert any(issubclass(t, JsonlReader) for t in types_)
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
        assert CompactSentenceDedupSignature in types_[0]
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
        cfg = SentDedupConfig(n_sentences=4, min_doc_words=10, min_num_sentences=1, split_sentences=True)
        execs = build_sentence_dedup_executors(_paths(tmp_path), sentence_config=cfg)
        sig = next(s for s in execs[0].pipeline if isinstance(s, SentenceDedupSignature))
        assert sig.config.n_sentences == 4


class TestStatisticsStage:
    """The statistics stage runs as map (N workers) → reduce (1 worker)."""

    def test_returns_map_and_reduce_executors(self, tmp_path: Path) -> None:
        """Stats returns two executors with the reduce depending on the map."""
        execs = build_statistics_executors(_paths(tmp_path), tasks=4)
        assert len(execs) == 2
        map_exec, reduce_exec = execs
        assert map_exec.tasks == 4
        assert map_exec.workers == 4
        assert reduce_exec.tasks == 1
        assert reduce_exec.workers == 1
        assert reduce_exec.depends is map_exec

    def test_map_pipeline_contains_reader_and_corpus_stats(self, tmp_path: Path) -> None:
        """The map pipeline reads JSONL and runs CorpusStats in partials mode."""
        execs = build_statistics_executors(_paths(tmp_path))
        map_types = [type(s) for s in execs[0].pipeline]
        assert any(issubclass(t, JsonlReader) for t in map_types)
        assert CorpusStats in map_types
        stats_step = next(s for s in execs[0].pipeline if isinstance(s, CorpusStats))
        assert stats_step.partials_dir is not None

    def test_reduce_pipeline_contains_corpus_stats_reduce(self, tmp_path: Path) -> None:
        """The reduce pipeline is a single CorpusStatsReduce step."""
        execs = build_statistics_executors(_paths(tmp_path))
        reduce_types = [type(s) for s in execs[1].pipeline]
        assert reduce_types == [CorpusStatsReduce]

    def test_statistics_reads_from_final_corpus_folder(self, tmp_path: Path) -> None:
        """The reader's data_folder is `<output_dir>/04_2_dedup`."""
        paths = _paths(tmp_path)
        execs = build_statistics_executors(paths)
        reader = next(s for s in execs[0].pipeline if isinstance(s, JsonlReader))
        assert str(paths.stage_dir("sentence_dedup")) in reader.data_folder.path


# --- End-to-end smoke test ----------------------------------------------------
# Builds two synthetic shards with one cross-shard verbatim duplicate and one
# shared 3-sentence span, runs `build_curate_executors(...)` against a tempdir,
# and asserts the final corpus contains the expected survivors plus a
# `statistics/` folder with both an aggregate JSON and per-dataset breakdowns.

import gzip  # noqa: E402
import json  # noqa: E402
from typing import Any, List, Tuple  # noqa: E402

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
    SHARED_SPAN + " Filozofi razmišljajo o naravi spoznanja. "
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
    build_quality_executors(paths, tasks=1, quality_config=loose_quality, stopwords=set())[-1].run()
    build_repetition_executors(paths, tasks=1)[-1].run()
    build_exact_dedup_executors(paths, tasks=1)[-1].run()
    build_sentence_dedup_executors(paths, tasks=1, sentence_config=loose_sentence)[-1].run()
    build_statistics_executors(paths, stopwords=set())[-1].run()

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

    stats_folder = paths.stage_dir("statistics")
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
    from slm4ie.data.curate.stages.common import pipeline_io_counts

    paths = CuratePaths(input_folder=tmp_path / "extracted", output_dir=tmp_path / "curated")
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
    stats = build_quality_executors(paths, tasks=1, quality_config=quality, stopwords=set())[-1].run()

    assert pipeline_io_counts(stats) == (3, 2)


from slm4ie.data.curate import paths as curate_paths  # noqa: E402


class TestStageSubsetFiltering:
    """`filter_stage_subset` mirrors the requested keys into a scratch view."""

    def test_filter_stage_subset_mirrors_multiple_datasets(self, tmp_path: Path) -> None:
        """`filter_stage_subset` builds symlinks for every requested key."""
        convert_dir = tmp_path / "00_convert"
        for key in ("kzb", "solar"):
            shard_dir = convert_dir / key
            shard_dir.mkdir(parents=True)
            (shard_dir / "00000.jsonl.gz").write_bytes(b"\x1f\x8b")

        holder = curate_paths.filter_stage_subset(convert_dir, ["kzb", "solar"])
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
            curate_paths.filter_stage_subset(convert_dir, ["kzb", "missing1", "missing2"])
        msg = str(excinfo.value)
        assert "missing1" in msg
        assert "missing2" in msg
        assert "'kzb'" not in msg


def test_quality_executor_honors_input_override(tmp_path: Path) -> None:
    """build_quality_executors reads from input_override when provided."""
    from slm4ie.data.curate.paths import CuratePaths
    from slm4ie.data.curate.stages.quality import build_quality_executors

    paths = CuratePaths(input_folder=tmp_path / "in", output_dir=tmp_path / "out")
    override = tmp_path / "view"
    override.mkdir()
    execs = build_quality_executors(paths, tasks=1, input_override=override)
    reader = execs[0].pipeline[0]
    assert str(override) in reader.data_folder.path


def test_repetition_executor_honors_input_override(tmp_path: Path) -> None:
    """build_repetition_executors reads from input_override when provided."""
    from slm4ie.data.curate.paths import CuratePaths
    from slm4ie.data.curate.stages.repetition import build_repetition_executors

    paths = CuratePaths(input_folder=tmp_path / "in", output_dir=tmp_path / "out")
    override = tmp_path / "view"
    override.mkdir()
    execs = build_repetition_executors(paths, tasks=1, input_override=override)
    reader = execs[0].pipeline[0]
    assert str(override) in reader.data_folder.path


@pytest.mark.parametrize(
    "builder",
    [build_exact_dedup_executors, build_sentence_dedup_executors, build_statistics_executors],
)
class TestCorpusStageExecutors:
    """Corpus stages resume per task, read a roster view, and decouple tasks from workers."""

    def test_tasks_and_workers_are_independent(self, tmp_path: Path, builder: Any) -> None:
        """Parallel executors take `tasks` and `workers` separately."""
        execs = builder(_paths(tmp_path), tasks=10, workers=3)
        parallel = [ex for ex in execs if any(isinstance(s, JsonlReader) for s in ex.pipeline)]
        assert [(ex.tasks, ex.workers) for ex in parallel] == [(10, 3)] * len(parallel)

    def test_skip_completed_tasks(self, tmp_path: Path, builder: Any) -> None:
        """Every executor skips tasks datatrove already marked complete."""
        assert all(ex.skip_completed for ex in builder(_paths(tmp_path)))

    def test_honors_input_override(self, tmp_path: Path, builder: Any) -> None:
        """Readers use the roster view when one is given."""
        override = tmp_path / "view"
        override.mkdir()
        execs = builder(_paths(tmp_path), input_override=override)
        readers = [s for ex in execs for s in ex.pipeline if isinstance(s, JsonlReader)]
        assert readers
        assert all(str(override) in r.data_folder.path for r in readers)


def test_scoped_stage_executors_do_not_skip_completed(tmp_path: Path) -> None:
    """Scoped buckets share a logging folder, so their executors never skip tasks."""
    execs = build_quality_executors(_paths(tmp_path)) + build_repetition_executors(_paths(tmp_path))
    assert not any(ex.skip_completed for ex in execs)


def _write_task_stats(stats_dir: Path, counts: List[Tuple[int, int]]) -> None:
    """Write one datatrove stats file per `(read, written)` pair, ranked in order."""
    from datatrove.utils.stats import PipelineStats, Stats

    stats_dir.mkdir(parents=True)
    for rank, (read, written) in enumerate(counts):
        reader, writer = Stats("reader"), Stats("writer")
        reader["documents"].update(read)
        writer["total"].update(written)
        with (stats_dir / f"{rank:05d}.json").open("w") as fh:
            PipelineStats([reader, writer]).save_to_disk(fh)


def test_stage_io_counts_sums_finished_tasks(tmp_path: Path) -> None:
    """Counts are summed over every per-task stats file, not only the last run's tasks."""
    _write_task_stats(tmp_path / "logs" / "stats", [(5, 3), (7, 4)])
    assert stage_io_counts(tmp_path / "logs") == (12, 7)
    assert stage_io_counts(tmp_path / "missing") == (0, 0)


def test_stage_io_counts_skips_ranks_beyond_the_executor(tmp_path: Path) -> None:
    """Stats files left by a larger executor that used the folder earlier are ignored."""
    _write_task_stats(tmp_path / "logs" / "stats", [(5, 3), (7, 4), (100, 100)])
    (tmp_path / "logs" / "executor.json").write_text(json.dumps({"tasks": 2}), encoding="utf-8")
    assert stage_io_counts(tmp_path / "logs") == (12, 7)


def test_logs_dir_is_scoped_by_config_bucket(tmp_path: Path) -> None:
    """Scoped paths put each bucket's executor under its own folder; unscoped paths keep the stage folder."""
    paths = _paths(tmp_path)
    assert paths.logs_dir("quality") == paths.output_dir / "_logs" / "quality"
    scoped = replace(paths, log_scope="abc123")
    assert scoped.logs_dir("quality") == paths.output_dir / "_logs" / "quality" / "abc123"
    assert scoped.stage_dir("quality") == paths.stage_dir("quality")


def test_bucket_log_scope_names_config_and_dataset_set(tmp_path: Path) -> None:
    """The folder changes with the config hash or the dataset set and ignores key order."""
    same = bucket_log_scope("sha256:abc", ["d2", "d1"])
    assert same == bucket_log_scope("sha256:abc", ["d1", "d2"])
    assert same.startswith("abc-")
    assert same != bucket_log_scope("sha256:abc", ["d1"])
    assert same != bucket_log_scope("sha256:abd", ["d1", "d2"])


def test_bucket_index_lists_each_dataset_under_its_last_executor(tmp_path: Path) -> None:
    """A dataset moves to the executor that ran it last; an executor left with nothing is dropped."""
    (tmp_path / "f1").mkdir()
    record_bucket(tmp_path, "f1", "sha256:h1", ["d1", "d2"])
    record_bucket(tmp_path, "f2", "sha256:h2", ["d3"])
    record_bucket(tmp_path, "f3", "sha256:h1", ["d1"])
    assert read_bucket_index(tmp_path) == {
        "f1": {"config_hash": "sha256:h1", "datasets": ["d2"]},
        "f2": {"config_hash": "sha256:h2", "datasets": ["d3"]},
        "f3": {"config_hash": "sha256:h1", "datasets": ["d1"]},
    }
    record_bucket(tmp_path, "f4", "sha256:h3", ["d2"])
    assert "f1" not in read_bucket_index(tmp_path) and not (tmp_path / "f1").exists()
    assert read_bucket_index(tmp_path / "missing") == {}
