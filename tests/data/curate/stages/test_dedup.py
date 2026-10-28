"""Tests for the dedup stages in `slm4ie.data.curate.stages.dedup`.

Covers the small content getters and config factories, then the structure of
the exact and sentence dedup executor ladders.
"""

import importlib.metadata  # noqa: F401  (datatrove workaround)
import importlib.util  # noqa: F401  (datatrove workaround)
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
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
from slm4ie.data.curate.stages import StageRun  # noqa: E402
from slm4ie.data.curate.stages.dedup import (  # noqa: E402
    DROP_BELOW_KEPT_FRACTION,
    DROP_BELOW_MIN_DOC_WORDS,
    DROP_BELOW_MIN_NUM_SENTENCES,
    DROP_EMPTY,
    CompactSentenceDedupSignature,
    KeptFractionSentenceDedupFilter,
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

    def test_skips_windows_past_the_16_bit_sentence_limit(self, tmp_path: Path) -> None:
        """A document with more sentences than a 16-bit index holds is hashed up to the limit."""
        lines = dedup_module.MAX_SENTENCE_INDEX + 5
        doc = Document(text="\n".join(f"vrstica {i}" for i in range(lines)), id="0", metadata={})
        cfg = SentDedupConfig(n_sentences=1, split_sentences=False)
        step = CompactSentenceDedupSignature(output_folder=str(tmp_path), config=cfg, finder_workers=1)

        step.run(iter([doc]), rank=0)

        dtype = np.dtype([("hash", "<u8"), ("doc", "<u4"), ("sent", "<u2")])
        sents = np.concatenate([np.frombuffer(b, dtype=dtype)["sent"] for b in _signature_files(tmp_path).values()])
        assert len(sents) == dedup_module.MAX_SENTENCE_INDEX + 1


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
        assert KeptFractionSentenceDedupFilter in types_[2]
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

    def test_kept_fraction_reaches_the_filter_step(self, tmp_path: Path) -> None:
        """The kept-fraction floor is set on the filter step the builder returns."""
        execs = build_sentence_dedup_executors(_paths(tmp_path), min_kept_fraction=0.3)
        filt = next(s for s in execs[2].pipeline if isinstance(s, KeptFractionSentenceDedupFilter))
        assert filt.min_kept_fraction == 0.3


class TestSentenceDedupRunner:
    """`run_sentence_dedup` hands the stage's config section to the builder."""

    @staticmethod
    def _captured_builder_kwargs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: Dict) -> Dict:
        captured: Dict = {}

        class _Done:
            def run(self) -> None:
                return None

        def fake_builder(paths: CuratePaths, **kwargs: object) -> List[object]:
            captured.update(kwargs)
            return [_Done()]

        monkeypatch.setattr(dedup_module, "build_sentence_dedup_executors", fake_builder)
        monkeypatch.setattr(dedup_module, "stage_io_counts", lambda _dir: (0, 0))
        job = StageRun(paths=_paths(tmp_path), config=config, workers=1, dataset_keys=[], output_folder=tmp_path)
        dedup_module.run_sentence_dedup(job)
        return captured

    def test_reads_the_kept_fraction_from_the_stage_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A `min_kept_fraction` in the config section reaches the builder."""
        kwargs = self._captured_builder_kwargs(tmp_path, monkeypatch, {"min_kept_fraction": 0.25})
        assert kwargs["min_kept_fraction"] == 0.25

    def test_defaults_the_kept_fraction_to_half(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """An unset `min_kept_fraction` means the half floor."""
        kwargs = self._captured_builder_kwargs(tmp_path, monkeypatch, {})
        assert kwargs["min_kept_fraction"] == dedup_module.DEFAULT_MIN_KEPT_FRACTION == 0.5


#: One line of five words; a document is a few of these joined by newlines.
_LINE = "ena dva tri štiri pet"


def _run_sentence_filter(
    tmp_path: Path,
    docs: List[Document],
    dups: List[Tuple[int, int]],
    *,
    min_kept_fraction: float,
    min_doc_words: int = 0,
    min_num_sentences: int = 0,
) -> Tuple[List[Document], KeptFractionSentenceDedupFilter]:
    """Run the filter step from the builder on `docs` with a prepared duplicates index.

    Windows are one line wide (`n_sentences=1`, line mode), so `(doc, sent)` in
    `dups` removes line `sent` of document `doc`.
    """
    cfg = SentDedupConfig(
        n_sentences=1, min_doc_words=min_doc_words, min_num_sentences=min_num_sentences, split_sentences=False
    )
    paths = _paths(tmp_path)
    execs = build_sentence_dedup_executors(paths, sentence_config=cfg, min_kept_fraction=min_kept_fraction)
    filt = next(s for s in execs[2].pipeline if isinstance(s, KeptFractionSentenceDedupFilter))
    dups_dir = paths.scratch_dir("sentence_dedup") / "dups" / "0000"
    dups_dir.mkdir(parents=True)
    np.array(dups, dtype=[("doc", "<u4"), ("sent", "<u2")]).tofile(dups_dir / "00000.c4_dup")
    return list(filt.run(iter(docs), rank=0)), filt


def _doc(id_: str, lines: int) -> Document:
    return Document(text="\n".join([_LINE] * lines), id=id_, metadata={})


class TestKeptFractionSentenceDedupFilter:
    """The filter step drops a changed document that keeps too little of its text."""

    def test_drops_a_changed_document_below_the_fraction(self, tmp_path: Path) -> None:
        """Three of four lines removed leaves a quarter: under the half floor, dropped and counted."""
        out, filt = _run_sentence_filter(tmp_path, [_doc("0", 4)], [(0, 0), (0, 1), (0, 2)], min_kept_fraction=0.5)
        assert out == []
        assert filt.stats[DROP_BELOW_KEPT_FRACTION].total == 1

    def test_keeps_a_changed_document_above_the_fraction(self, tmp_path: Path) -> None:
        """One of four lines removed leaves three quarters: kept, trimmed, nothing counted."""
        out, filt = _run_sentence_filter(tmp_path, [_doc("0", 4)], [(0, 0)], min_kept_fraction=0.5)
        assert [d.text for d in out] == ["\n".join([_LINE] * 3)]
        assert filt.stats[DROP_BELOW_KEPT_FRACTION].total == 0

    def test_keeps_an_untouched_short_document(self, tmp_path: Path) -> None:
        """A document no window touched is kept whatever its length."""
        out, _ = _run_sentence_filter(tmp_path, [_doc("0", 1)], [], min_kept_fraction=0.5, min_doc_words=50)
        assert [d.id for d in out] == ["0"]

    def test_zero_disables_the_fraction_floor(self, tmp_path: Path) -> None:
        """With the floor at zero a quarter of the text survives."""
        out, filt = _run_sentence_filter(tmp_path, [_doc("0", 4)], [(0, 0), (0, 1), (0, 2)], min_kept_fraction=0)
        assert [d.text for d in out] == [_LINE]
        assert filt.stats[DROP_BELOW_KEPT_FRACTION].total == 0

    def test_counts_each_floor_separately(self, tmp_path: Path) -> None:
        """Every dropped document lands under the first floor it fails; a document clearing all three is kept."""
        sentences = "Prvi stavek je dolg.\nDrugi stavek je dolg.\nTretji stavek je dolg."
        docs = [
            _doc("fraction", 4),
            _doc("words", 2),
            Document(text="\n".join([" ".join([_LINE] * 2)] * 2), id="sentences", metadata={}),
            Document(text=sentences, id="kept", metadata={}),
        ]
        dups = [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (3, 0)]
        out, filt = _run_sentence_filter(
            tmp_path, docs, dups, min_kept_fraction=0.5, min_doc_words=8, min_num_sentences=2
        )
        assert [d.id for d in out] == ["kept"]
        assert filt.stats[DROP_BELOW_KEPT_FRACTION].total == 1
        assert filt.stats[DROP_BELOW_MIN_DOC_WORDS].total == 1
        assert filt.stats[DROP_BELOW_MIN_NUM_SENTENCES].total == 1

    def test_counts_a_document_with_nothing_left_as_empty(self, tmp_path: Path) -> None:
        """Every line removed leaves nothing: dropped under the empty stat, not a floor."""
        out, filt = _run_sentence_filter(tmp_path, [_doc("0", 2)], [(0, 0), (0, 1)], min_kept_fraction=0.5)
        assert out == []
        assert filt.stats[DROP_EMPTY].total == 1
        assert filt.stats[DROP_BELOW_KEPT_FRACTION].total == 0

    def test_rejects_a_fraction_out_of_range(self, tmp_path: Path) -> None:
        """The step refuses a floor outside [0, 1] at construction."""
        with pytest.raises(ValueError, match="min_kept_fraction"):
            KeptFractionSentenceDedupFilter(data_folder=str(tmp_path), min_kept_fraction=1.5)
