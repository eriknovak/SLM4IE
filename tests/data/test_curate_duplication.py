"""Tests for the dedup assessment: does a dropped document survive as a twin."""

import gzip
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from slm4ie.data.curate.duplication import (
    assess_dedup,
    dropped_documents,
    sentences,
    summarise,
    window_hashes,
)

_DATASET = "demo"

#: Five sentences, so three sliding windows of three.
_LONG = "Alpha one. Beta two. Gamma three. Delta four. Epsilon five."

#: Shares its first four sentences with `_LONG`: two of its three windows.
_NEAR = "Alpha one. Beta two. Gamma three. Delta four. Zeta other."


def _write(path: Path, documents: List[Dict[str, Any]]) -> None:
    """Write documents as one gzipped JSONL shard.

    Args:
        path: Shard to write.
        documents: Documents carrying `id` and `text`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document) + "\n")


def _doc(doc_id: str, text: str) -> Dict[str, Any]:
    """Build a pipeline-shaped document.

    Args:
        doc_id: The document id.
        text: The document text.

    Returns:
        The document dict.
    """
    return {"id": doc_id, "text": text, "metadata": {"dataset": _DATASET}}


def _sample_row(doc_id: str, stage: str) -> Dict[str, Any]:
    """Build one sample row recording a dedup drop.

    Args:
        doc_id: The dropped document's id.
        stage: The dedup stage that dropped it.

    Returns:
        A row shaped like the stratified sample's.
    """
    return {
        "id": doc_id,
        "dataset": _DATASET,
        "cells": [{"stage": stage, "decision": "dropped", "shard": "00000.jsonl.gz"}],
        "text": "trimmed",
    }


class TestHashing:
    """Sentence windows are what sentence-level twins are matched on."""

    def test_sentences_split_on_end_punctuation_and_newlines(self) -> None:
        """Sentence ends and line breaks both split, and blanks are dropped."""
        assert sentences("One. Two!\n\nThree") == ["One.", "Two!", "Three"]

    def test_long_document_yields_one_hash_per_window(self) -> None:
        """Five sentences give three windows of three."""
        assert len(window_hashes(_LONG)) == 3

    def test_short_document_still_yields_one_hash(self) -> None:
        """A document shorter than a window stays matchable."""
        assert len(window_hashes("Only one. And two.")) == 1
        assert window_hashes("") == set()

    def test_shared_sentences_share_windows(self) -> None:
        """Two documents agreeing on four sentences share two windows."""
        assert len(window_hashes(_LONG) & window_hashes(_NEAR)) == 2


class TestDroppedDocuments:
    """Dropped documents are read back in full from the stage's input."""

    def test_repeated_ids_yield_one_document(self, tmp_path: Path) -> None:
        """An id the source repeats is assessed once, as its first copy."""
        _write(
            tmp_path / "04_repetition" / _DATASET / "00000.jsonl.gz",
            [_doc("demo:1", "first copy"), _doc("demo:1", "second copy"), _doc("demo:2", "other")],
        )
        sample = tmp_path / "sample.jsonl"
        sample.write_text(json.dumps(_sample_row("demo:1", "exact_dedup")) + "\n", encoding="utf-8")

        found = dropped_documents(sample, "exact_dedup", tmp_path)

        assert found == [{"id": "demo:1", "dataset": _DATASET, "text": "first copy"}]


class TestAssessDedup:
    """A drop counts as matched only when its twin reaches the finished corpus."""

    @pytest.fixture
    def corpus(self, tmp_path: Path) -> Path:
        """Build a corpus with one drop of every kind.

        - `exact-kept`: exact drop whose twin survives to the end.
        - `exact-lost`: exact drop whose twin sentence dedup removed later.
        - `sent-kept`: sentence drop with a surviving twin holding 2 of 3 windows.
        - `sent-lost`: sentence drop sharing no window with anything.

        Args:
            tmp_path: pytest's temporary directory.

        Returns:
            The curation output root.
        """
        _write(
            tmp_path / "04_repetition" / _DATASET / "00000.jsonl.gz",
            [_doc("demo:exact-kept", "Same text A."), _doc("demo:exact-lost", "Same text B.")],
        )
        _write(
            tmp_path / "05_exact_dedup" / _DATASET / "00000.jsonl.gz",
            [
                _doc("demo:twin-a", "Same text A."),
                _doc("demo:twin-b", "Same text B."),
                _doc("demo:twin-near", _NEAR),
                _doc("demo:sent-kept", _LONG),
                _doc("demo:sent-lost", "Pi a. Rho b. Sigma c. Tau d."),
            ],
        )
        _write(
            tmp_path / "06_sentence_dedup" / _DATASET / "00000.jsonl.gz",
            [_doc("demo:twin-a", "Same text A."), _doc("demo:twin-near", _NEAR)],
        )
        sample = tmp_path / "sample.jsonl"
        rows = [
            _sample_row("demo:exact-kept", "exact_dedup"),
            _sample_row("demo:exact-lost", "exact_dedup"),
            _sample_row("demo:sent-kept", "sentence_dedup"),
            _sample_row("demo:sent-lost", "sentence_dedup"),
        ]
        sample.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return tmp_path

    def test_summary_counts_only_surviving_twins(self, corpus: Path) -> None:
        """Each stage matches exactly the drop whose twin survives."""
        rows, twins = assess_dedup(corpus / "sample.jsonl", corpus, workers=1)
        by_stage = {row["stage"]: row for row in rows}

        assert by_stage["exact_dedup"]["exact_twin"] == 2
        assert by_stage["exact_dedup"]["twin_survives_to_corpus"] == 1
        assert by_stage["exact_dedup"]["twin_match_rate"] == 0.5
        assert by_stage["sentence_dedup"]["shares_any_window"] == 1
        assert by_stage["sentence_dedup"]["twin_match_rate"] == 0.5
        assert twins["demo:sent-kept"]["best_twin"] == "demo:twin-near"
        assert twins["demo:sent-kept"]["window_coverage"] == pytest.approx(2 / 3)

    def test_unmatched_drops_are_written_for_judging(self, corpus: Path) -> None:
        """The two drops that left nothing behind go to the unmatched file."""
        out = corpus / "unmatched.jsonl"
        assess_dedup(corpus / "sample.jsonl", corpus, workers=1, unmatched_path=out)

        unmatched = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        assert sorted(row["id"] for row in unmatched) == ["demo:exact-lost", "demo:sent-lost"]
        assert all(row["text"] for row in unmatched)


class TestSummarise:
    """The coverage floor decides which sentence-dedup drops count as matched."""

    def test_coverage_below_floor_is_unmatched(self) -> None:
        """A surviving twin holding under half the windows does not match."""
        twins = {
            "a": {"exact_twin": None, "sharers": 1, "window_coverage": 0.4, "twin_survives": True},
            "b": {"exact_twin": None, "sharers": 1, "window_coverage": 0.6, "twin_survives": True},
        }
        row = summarise("sentence_dedup", twins, coverage_floor=0.5)
        assert row["covered_at_floor"] == 1
        assert row["unmatched"] == 1
