"""Tests for the merged datatrove curation pipeline.

These tests don't actually run the dedup pipeline (which is slow and
filesystem-heavy). They verify the structural contract of
`build_curate_executors`: six executors in the correct order, each
chained via `depends=` to the previous, the language argument
threaded through to the sentence-dedup blocks, and `finder_workers`
propagated everywhere it matters.
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

from slm4ie.data.curate.dedup import default_exact_config, doc_text  # noqa: E402
from slm4ie.data.curate.language import LinguaLanguageFilter  # noqa: E402
from slm4ie.data.curate.pipeline import (  # noqa: E402
    CuratePaths,
    QualityConfig,
    build_curate_executors,
)
from slm4ie.data.curate.stats import CorpusStats  # noqa: E402


def _paths(tmp_path: Path) -> CuratePaths:
    """Build a CuratePaths anchored under *tmp_path* for structural tests."""
    return CuratePaths(
        input_folder=tmp_path / "datatrove",
        final_folder=tmp_path / "final",
        statistics_folder=tmp_path / "final" / "statistics",
        scratch_folder=tmp_path / "scratch",
    )


class TestExactDedupHelpers:
    """Tiny helpers that pipeline.py composes."""

    def test_doc_text_returns_text_payload(self) -> None:
        """`doc_text` extracts the text body for hashing."""
        from datatrove.data import Document

        d = Document(text="hello", id="1", metadata={})
        assert doc_text(d) == "hello"

    def test_default_exact_config_uses_doc_text(self) -> None:
        """The default ExactDedupConfig hashes `doc.text`, not metadata."""
        cfg = default_exact_config()
        assert cfg.content_getter is doc_text


class TestBuildCurateExecutors:
    """Structure of the six-executor ladder."""

    def test_six_executors_chained_via_depends(self, tmp_path: Path) -> None:
        """We get exactly six executors and each one depends on the prior."""
        execs = build_curate_executors(_paths(tmp_path))
        assert len(execs) == 6
        assert execs[0].depends is None
        for i in range(1, 6):
            assert execs[i].depends is execs[i - 1]

    def test_each_executor_carries_the_right_blocks(self, tmp_path: Path) -> None:
        """Every executor's pipeline contains the datatrove blocks the plan promised."""
        execs = build_curate_executors(_paths(tmp_path))
        types_per_executor = [[type(step) for step in ex.pipeline] for ex in execs]

        # Executor 1: lang + quality + repetition + exact-sig.
        assert LinguaLanguageFilter in types_per_executor[0]
        assert GopherQualityFilter in types_per_executor[0]
        assert GopherRepetitionFilter in types_per_executor[0]
        assert ExactDedupSignature in types_per_executor[0]
        # Executor 2: exact-find.
        assert ExactFindDedups in types_per_executor[1]
        # Executor 3: exact-filter + sentence-sig.
        assert ExactDedupFilter in types_per_executor[2]
        assert SentenceDedupSignature in types_per_executor[2]
        # Executor 4: sentence-find.
        assert SentenceFindDedups in types_per_executor[3]
        # Executor 5: sentence-filter + write final corpus.
        assert SentenceDedupFilter in types_per_executor[4]
        assert JsonlWriter in types_per_executor[4]
        assert CorpusStats not in types_per_executor[4]
        # Executor 6: read final corpus + stats (single-process).
        assert JsonlReader in types_per_executor[5]
        assert CorpusStats in types_per_executor[5]
        assert execs[5].tasks == 1

    def test_sentence_blocks_run_in_slovenian(self, tmp_path: Path) -> None:
        """The sentence dedup blocks must use Languages.slovenian."""
        execs = build_curate_executors(_paths(tmp_path))
        sent_sig = next(
            step for step in execs[2].pipeline if isinstance(step, SentenceDedupSignature)
        )
        sent_filter = next(
            step for step in execs[4].pipeline if isinstance(step, SentenceDedupFilter)
        )
        assert sent_sig.language == Languages.slovenian
        assert sent_filter.language == Languages.slovenian

    def test_finder_workers_propagates_to_sig_and_find(self, tmp_path: Path) -> None:
        """`finder_workers` reaches every signature and find executor."""
        execs = build_curate_executors(_paths(tmp_path), finder_workers=4)
        exact_sig = next(
            step for step in execs[0].pipeline if isinstance(step, ExactDedupSignature)
        )
        sent_sig = next(
            step for step in execs[2].pipeline if isinstance(step, SentenceDedupSignature)
        )
        assert exact_sig.finder_workers == 4
        assert sent_sig.finder_workers == 4
        assert execs[1].tasks == 4  # exact-find
        assert execs[3].tasks == 4  # sentence-find

    def test_debug_mode_wires_exclusion_writers(self, tmp_path: Path) -> None:
        """When debug=True, every drop stage gets an exclusion_writer."""
        paths = CuratePaths(
            input_folder=tmp_path / "datatrove",
            final_folder=tmp_path / "final",
            statistics_folder=tmp_path / "final" / "statistics",
            scratch_folder=tmp_path / "scratch",
            debug=True,
        )
        execs = build_curate_executors(paths)
        lang = next(
            step for step in execs[0].pipeline if isinstance(step, LinguaLanguageFilter)
        )
        quality = next(
            step for step in execs[0].pipeline if isinstance(step, GopherQualityFilter)
        )
        repetition = next(
            step for step in execs[0].pipeline if isinstance(step, GopherRepetitionFilter)
        )
        exact_filter = next(
            step for step in execs[2].pipeline if isinstance(step, ExactDedupFilter)
        )
        sent_filter = next(
            step for step in execs[4].pipeline if isinstance(step, SentenceDedupFilter)
        )
        assert lang.exclusion_writer is not None
        assert quality.exclusion_writer is not None
        assert repetition.exclusion_writer is not None
        assert exact_filter.exclusion_writer is not None
        assert sent_filter.exclusion_writer is not None

    def test_default_mode_skips_exclusion_writers(self, tmp_path: Path) -> None:
        """Without --debug we don't materialize exclusion writers."""
        execs = build_curate_executors(_paths(tmp_path))
        lang = next(
            step for step in execs[0].pipeline if isinstance(step, LinguaLanguageFilter)
        )
        quality = next(
            step for step in execs[0].pipeline if isinstance(step, GopherQualityFilter)
        )
        repetition = next(
            step for step in execs[0].pipeline if isinstance(step, GopherRepetitionFilter)
        )
        exact_filter = next(
            step for step in execs[2].pipeline if isinstance(step, ExactDedupFilter)
        )
        sent_filter = next(
            step for step in execs[4].pipeline if isinstance(step, SentenceDedupFilter)
        )
        assert lang.exclusion_writer is None
        assert quality.exclusion_writer is None
        assert repetition.exclusion_writer is None
        assert exact_filter.exclusion_writer is None
        assert sent_filter.exclusion_writer is None

    def test_quality_config_is_threaded_to_gopher_filter(self, tmp_path: Path) -> None:
        """`QualityConfig` overrides reach the underlying GopherQualityFilter."""
        cfg = QualityConfig(min_doc_words=10, max_doc_words=200, min_stop_words=0)
        execs = build_curate_executors(_paths(tmp_path), quality_config=cfg)
        quality = next(
            step for step in execs[0].pipeline if isinstance(step, GopherQualityFilter)
        )
        assert quality.min_doc_words == 10
        assert quality.max_doc_words == 200
        assert quality.min_stop_words == 0

    def test_stopwords_become_gopher_stop_words(self, tmp_path: Path) -> None:
        """The Slovenian stopword set is wired into GopherQualityFilter."""
        execs = build_curate_executors(_paths(tmp_path), stopwords={"in", "je", "na"})
        quality = next(
            step for step in execs[0].pipeline if isinstance(step, GopherQualityFilter)
        )
        assert {"in", "je", "na"}.issubset(quality.stop_words)

    def test_lang_minimum_relative_distance_is_threaded(self, tmp_path: Path) -> None:
        """The minimum_relative_distance reaches LinguaLanguageFilter."""
        execs = build_curate_executors(
            _paths(tmp_path), lang_minimum_relative_distance=0.15
        )
        lang = next(
            step for step in execs[0].pipeline if isinstance(step, LinguaLanguageFilter)
        )
        assert lang.minimum_relative_distance == 0.15


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
    """Two shards with one full-doc dup and one shared span produce 5 survivors."""
    input_folder = tmp_path / "datatrove"
    _write_shard(
        input_folder / "alpha" / "00000.jsonl.gz",
        dataset="alpha",
        domain="scientific",
        docs=[
            {"id": "alpha:1", "text": SHARED_DOC},
            {"id": "alpha:2", "text": A2_TEXT},
            {"id": "alpha:3", "text": "Solnce sveti nad gorami in dolinami slovenskih krajev. " * 8},
        ],
    )
    _write_shard(
        input_folder / "beta" / "00000.jsonl.gz",
        dataset="beta",
        domain="legal",
        docs=[
            {"id": "beta:1", "text": SHARED_DOC},  # whole-doc dup of alpha:1
            {"id": "beta:2", "text": B2_TEXT},     # shares 3-sentence span with alpha:2
            {"id": "beta:3", "text": "Pravna doktrina se razvija s časom in družbenimi spremembami. " * 8},
        ],
    )

    final_folder = tmp_path / "final"
    paths = CuratePaths(
        input_folder=input_folder,
        final_folder=final_folder,
        statistics_folder=final_folder / "statistics",
        scratch_folder=tmp_path / "scratch",
    )
    executors = build_curate_executors(
        paths,
        # Skip classla on the smoke test — it would download a model.
        compute_keywords=False,
        # Lower min_doc_words so our small fixtures aren't filtered out
        # after sentence dedup. Real corpus runs use the default 50.
        sentence_config=SentDedupConfig(
            n_sentences=3,
            min_doc_words=5,
            min_num_sentences=1,
            split_sentences=True,
        ),
        # Loosen Gopher quality so the short, repetitive fixtures
        # aren't dropped by quality heuristics. `min_stop_words=0`
        # avoids needing a real Slovenian stopword list for the smoke
        # test (we pass an empty stopwords set below).
        # `max_non_alpha_words_ratio` is misleadingly named in
        # datatrove: it is actually the *minimum* fraction of words
        # that must contain at least one alphabetic char. Short fixtures
        # have proportionally more punctuation tokens (each `.` counts
        # as its own word), so the alpha fraction sits ~0.85 instead
        # of the ~0.9+ that real-corpus docs hit. Drop to 0.6 here.
        quality_config=QualityConfig(
            min_doc_words=5,
            min_stop_words=0,
            max_non_alpha_words_ratio=0.6,
            max_avg_word_length=15,
        ),
        stopwords=set(),
    )
    executors[-1].run()

    survivors: List[str] = []
    survivor_dirs: set = set()
    for shard in sorted(final_folder.glob("**/*.jsonl.gz")):
        survivor_dirs.add(shard.parent.name)
        with gzip.open(shard, "rt", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                survivors.append(rec["id"])

    # 6 input docs minus 1 whole-document duplicate (exact dedup)
    # minus 2 docs killed by GopherRepetitionFilter (each is a
    # 55-char sentence repeated 8x: heavy top-2-gram density) = 3
    # survivors. The two "×8" docs are an intentional rep-filter
    # fixture; the cross-dataset exact dup is the dedup fixture.
    assert len(survivors) == 3
    assert "alpha:1" in survivors  # SHARED_DOC, kept on alpha rank
    assert "beta:1" not in survivors  # exact dup of alpha:1
    assert "alpha:3" not in survivors  # killed by repetition filter
    assert "beta:3" not in survivors   # killed by repetition filter
    # Final corpus is sharded as <final>/<dataset>/<rank>.jsonl.gz —
    # both datasets must show up as their own subfolders.
    assert survivor_dirs == {"alpha", "beta"}

    bundle = json.loads((final_folder / "statistics" / "aggregate.json").read_text(encoding="utf-8"))
    assert bundle["total_docs"] == 3
    assert "alpha" in bundle["by_dataset"]
    assert "beta" in bundle["by_dataset"]
    assert bundle["by_dataset"]["alpha"]["doc_count"] == 2
    assert bundle["by_dataset"]["beta"]["doc_count"] == 1

    per_dataset_dir = final_folder / "statistics" / "per_dataset"
    assert (per_dataset_dir / "alpha.json").exists()
    assert (per_dataset_dir / "beta.json").exists()
    alpha_slim = json.loads((per_dataset_dir / "alpha.json").read_text(encoding="utf-8"))
    assert alpha_slim["dataset"] == "alpha"
    assert alpha_slim["domain"] == "scientific"
    assert alpha_slim["doc_count"] == 2


# --- CLI: multi-key dataset selection ----------------------------------------
# `scripts/data/curate.py` accepts either one or more positional dataset keys,
# or `--all`. The CLI lives outside `slm4ie/` but is importable as
# `scripts.data.curate` because both `scripts/` and `scripts/data/` ship empty
# `__init__.py` files.

from scripts.data import curate as curate_cli  # noqa: E402


class TestCurateCLISelection:
    """`parse_args` and `_filter_input_keys` accept multiple dataset keys."""

    def test_parse_args_accepts_single_key(self) -> None:
        """A single positional still resolves into `args.datasets`."""
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

    def test_filter_input_keys_mirrors_multiple_datasets(self, tmp_path: Path) -> None:
        """`_filter_input_keys` builds symlinks for every requested key."""
        input_dir = tmp_path / "datatrove"
        for key in ("kzb", "solar"):
            shard_dir = input_dir / key
            shard_dir.mkdir(parents=True)
            (shard_dir / "00000.jsonl.gz").write_bytes(b"")

        holder = curate_cli._filter_input_keys(input_dir, ["kzb", "solar"])
        try:
            assert (holder / "kzb" / "00000.jsonl.gz").is_symlink()
            assert (holder / "solar" / "00000.jsonl.gz").is_symlink()
        finally:
            import shutil

            shutil.rmtree(holder, ignore_errors=True)

    def test_filter_input_keys_lists_all_missing_keys(self, tmp_path: Path) -> None:
        """Missing shard folders are reported together in one error."""
        input_dir = tmp_path / "datatrove"
        (input_dir / "kzb").mkdir(parents=True)
        (input_dir / "kzb" / "00000.jsonl.gz").write_bytes(b"")

        with pytest.raises(FileNotFoundError) as excinfo:
            curate_cli._filter_input_keys(input_dir, ["kzb", "missing1", "missing2"])
        msg = str(excinfo.value)
        assert "missing1" in msg
        assert "missing2" in msg
        assert "'kzb'" not in msg
