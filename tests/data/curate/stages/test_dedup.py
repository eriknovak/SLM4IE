"""Tests for the dedup stages in `slm4ie.data.curate.stages.dedup`.

Covers the small content getters and config factories, then the structure of
the exact and sentence dedup executor ladders.
"""

import importlib.metadata  # noqa: F401  (datatrove workaround)
import importlib.util  # noqa: F401  (datatrove workaround)
from pathlib import Path
from typing import Dict

import pytest

pytest.importorskip("datatrove")

from datatrove.data import Document  # noqa: E402

from datatrove.pipeline.dedup import (  # noqa: E402
    ExactDedupFilter,
    ExactDedupSignature,
    ExactFindDedups,
    SentDedupConfig,
    SentenceDedupFilter,
    SentenceDedupSignature,
    SentenceFindDedups,
)
from datatrove.pipeline.writers.jsonl import JsonlWriter  # noqa: E402
from datatrove.utils.typeshelper import Languages  # noqa: E402

import slm4ie.data.curate.stages.dedup as dedup_module  # noqa: E402
from slm4ie.data.curate.paths import CuratePaths  # noqa: E402
from slm4ie.data.curate.stages.dedup import (  # noqa: E402
    CompactSentenceDedupSignature,
    build_exact_dedup_executors,
    build_sentence_dedup_executors,
    default_exact_config,
    doc_text,
    make_exact_config,
)


class TestExactDedupHelpers:
    """Helpers used by the exact-dedup signature stage."""

    def test_doc_text_returns_text_payload(self) -> None:
        """`doc_text` extracts the text body for hashing."""
        d = Document(text="hello", id="1", metadata={})
        assert doc_text(d) == "hello"

    def test_default_exact_config_uses_doc_text(self) -> None:
        """The default ExactDedupConfig hashes `doc.text`, not metadata."""
        cfg = default_exact_config()
        assert cfg.content_getter is doc_text

    def test_make_exact_config_defaults_are_pinned(self) -> None:
        """Defaults are 64-bit xxhash, only_dedup_in_index=True, doc_text getter."""
        cfg = make_exact_config()
        assert cfg.content_getter is doc_text
        assert cfg.hash_config.precision == 64
        assert cfg.hash_config.hash_fc == "xxhash"
        assert cfg.only_dedup_in_index is True

    def test_make_exact_config_overrides_precision(self) -> None:
        """make_exact_config threads the precision arg into HashConfig."""
        cfg = make_exact_config(precision=32)
        assert cfg.hash_config.precision == 32

    def test_make_exact_config_overrides_hash_fc(self) -> None:
        """make_exact_config threads hash_fc into HashConfig."""
        cfg = make_exact_config(hash_fc="sha1")
        assert cfg.hash_config.hash_fc == "sha1"

    def test_make_exact_config_overrides_only_dedup_in_index(self) -> None:
        """make_exact_config threads the only_dedup_in_index flag."""
        cfg = make_exact_config(only_dedup_in_index=False)
        assert cfg.only_dedup_in_index is False


def _signature_files(folder: Path) -> Dict[str, bytes]:
    """Map each signature file under *folder* (relative path) to its bytes."""
    return {str(f.relative_to(folder)): f.read_bytes() for f in sorted(folder.rglob("*")) if f.is_file()}


class TestCompactSentenceDedupSignature:
    """The compact signature step is a byte-for-byte drop-in for datatrove's."""

    def test_writes_identical_signature_files(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Both steps write the same files for the same documents, across flush boundaries."""
        monkeypatch.setattr(dedup_module, "SIGNATURE_FLUSH_EVERY", 3)
        docs = [
            Document(
                text=" ".join(f"Stavek {i} v dokumentu {d} govori o temi {i % 3}." for i in range(6)),
                id=str(d),
                metadata={},
            )
            for d in range(5)
        ]
        cfg = SentDedupConfig(n_sentences=2, split_sentences=True)
        kwargs = {"config": cfg, "finder_workers": 2, "language": Languages.slovenian}
        SentenceDedupSignature(output_folder=str(tmp_path / "upstream"), **kwargs).run(iter(docs), rank=1)
        CompactSentenceDedupSignature(output_folder=str(tmp_path / "compact"), **kwargs).run(iter(docs), rank=1)

        upstream = _signature_files(tmp_path / "upstream")
        assert upstream
        assert _signature_files(tmp_path / "compact") == upstream


def _paths(tmp_path: Path) -> CuratePaths:
    """Build a CuratePaths anchored under *tmp_path* for structural tests."""
    return CuratePaths(
        input_folder=tmp_path / "datatrove",
        output_dir=tmp_path / "curated",
    )


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
