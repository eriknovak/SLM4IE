"""Tests for the datatrove plumbing shared by the stage modules.

Covers the stage I/O counts, the executor settings every stage builder
follows, and one end-to-end smoke test (marked `@pytest.mark.slow`) of the
shard layout that carries dataset provenance through every stage.
"""

import gzip
import importlib.metadata  # noqa: F401  (datatrove workaround)
import importlib.util  # noqa: F401  (datatrove workaround)
import json
from pathlib import Path
from typing import Any, List, Tuple

import pytest

pytest.importorskip("datatrove")
pytest.importorskip("lingua")

from datatrove.pipeline.dedup import SentDedupConfig  # noqa: E402
from datatrove.pipeline.readers import JsonlReader  # noqa: E402

from slm4ie.data.curate.paths import CuratePaths  # noqa: E402
from slm4ie.data.curate.stages.common import stage_io_counts  # noqa: E402
from slm4ie.data.curate.stages.dedup import (  # noqa: E402
    build_exact_dedup_executors,
    build_sentence_dedup_executors,
)
from slm4ie.data.curate.stages.language import build_language_executors  # noqa: E402
from slm4ie.data.curate.stages.quality import (  # noqa: E402
    QualityConfig,
    build_quality_executors,
)
from slm4ie.data.curate.stages.repetition import build_repetition_executors  # noqa: E402
from slm4ie.data.curate.stages.statistics import build_statistics_executors  # noqa: E402


def _paths(tmp_path: Path) -> CuratePaths:
    """Build a CuratePaths anchored under *tmp_path* for structural tests."""
    return CuratePaths(
        input_folder=tmp_path / "datatrove",
        output_dir=tmp_path / "curated",
    )


# --- End-to-end smoke test ----------------------------------------------------
# Builds two synthetic shards with one cross-shard verbatim duplicate and one
# shared 3-sentence span, runs `build_curate_executors(...)` against a tempdir,
# and asserts the final corpus contains the expected survivors plus a
# `statistics/` folder with both an aggregate JSON and per-dataset breakdowns.


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
