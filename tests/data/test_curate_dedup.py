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
from datatrove.pipeline.readers import JsonlReader  # noqa: E402
from datatrove.pipeline.writers.jsonl import JsonlWriter  # noqa: E402
from datatrove.utils.typeshelper import Languages  # noqa: E402

from slm4ie.data.curate.dedup import default_exact_config, doc_text  # noqa: E402
from slm4ie.data.curate.language import LinguaLanguageFilter  # noqa: E402
from slm4ie.data.curate.pipeline import CuratePaths, build_curate_executors  # noqa: E402
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

        # Executor 1: lang + exact-sig.
        assert LinguaLanguageFilter in types_per_executor[0]
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
        """The sentence-dedup blocks must use Languages.slovenian, not English."""
        execs = build_curate_executors(_paths(tmp_path))
        sent_sig = next(step for step in execs[2].pipeline if isinstance(step, SentenceDedupSignature))
        sent_filter = next(step for step in execs[4].pipeline if isinstance(step, SentenceDedupFilter))
        assert sent_sig.language == Languages.slovenian
        assert sent_filter.language == Languages.slovenian

    def test_finder_workers_propagates_to_sig_and_find(self, tmp_path: Path) -> None:
        """`finder_workers` reaches both signature and find executors."""
        execs = build_curate_executors(_paths(tmp_path), finder_workers=4)
        exact_sig = next(step for step in execs[0].pipeline if isinstance(step, ExactDedupSignature))
        sent_sig = next(step for step in execs[2].pipeline if isinstance(step, SentenceDedupSignature))
        assert exact_sig.finder_workers == 4
        assert sent_sig.finder_workers == 4
        assert execs[1].tasks == 4  # exact-find
        assert execs[3].tasks == 4  # sentence-find

    def test_debug_mode_wires_exclusion_writers(self, tmp_path: Path) -> None:
        """When debug=True both filter stages get an exclusion_writer."""
        paths = CuratePaths(
            input_folder=tmp_path / "datatrove",
            final_folder=tmp_path / "final",
            statistics_folder=tmp_path / "final" / "statistics",
            scratch_folder=tmp_path / "scratch",
            debug=True,
        )
        execs = build_curate_executors(paths)
        exact_filter = next(step for step in execs[2].pipeline if isinstance(step, ExactDedupFilter))
        sent_filter = next(step for step in execs[4].pipeline if isinstance(step, SentenceDedupFilter))
        assert exact_filter.exclusion_writer is not None
        assert sent_filter.exclusion_writer is not None

    def test_default_mode_skips_exclusion_writers(self, tmp_path: Path) -> None:
        """Without --debug we don't materialize exclusion writers."""
        execs = build_curate_executors(_paths(tmp_path))
        exact_filter = next(step for step in execs[2].pipeline if isinstance(step, ExactDedupFilter))
        sent_filter = next(step for step in execs[4].pipeline if isinstance(step, SentenceDedupFilter))
        assert exact_filter.exclusion_writer is None
        assert sent_filter.exclusion_writer is None
