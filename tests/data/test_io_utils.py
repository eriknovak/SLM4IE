"""Tests for slm4ie.data.io_utils helpers."""

import gzip
import json
import sys
from pathlib import Path

import pytest

from slm4ie.data.io_utils import (
    find_dataset_files,
    iter_joined_records,
    open_output,
)


def _write_jsonl(path: Path, records) -> None:
    """Write *records* as one JSON object per line to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False))
            fh.write("\n")


def _write_gz_jsonl(path: Path, records) -> None:
    """Write *records* as one JSON object per line, gzipped."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False))
            fh.write("\n")


class TestFindDatasetFiles:
    """Tests for find_dataset_files."""

    def test_returns_text_only_when_no_annotations(
        self, tmp_path: Path
    ) -> None:
        """Without an annotations sidecar, the second tuple slot is None."""
        text = tmp_path / "k.jsonl"
        text.write_text("")
        result = find_dataset_files(tmp_path, "k")
        assert result == (text, None)

    def test_returns_both_when_annotations_present(
        self, tmp_path: Path
    ) -> None:
        """Annotations sidecar is returned alongside the text file."""
        text = tmp_path / "k.jsonl"
        ann = tmp_path / "k.annotations.jsonl.gz"
        text.write_text("")
        ann.write_text("")
        result = find_dataset_files(tmp_path, "k")
        assert result == (text, ann)

    def test_returns_none_when_text_missing(self, tmp_path: Path) -> None:
        """No <key>.jsonl → None (caller decides what to do)."""
        result = find_dataset_files(tmp_path, "ghost")
        assert result is None


class TestIterJoinedRecords:
    """Tests for iter_joined_records."""

    def test_merges_text_and_annotations(self, tmp_path: Path) -> None:
        """Annotations are attached as the 'annotations' field."""
        text = tmp_path / "k.jsonl"
        ann = tmp_path / "k.annotations.jsonl.gz"
        _write_jsonl(text, [
            {"text": "Lepa beseda.", "source": "k", "domain": "x",
             "doc_id": "s1"},
        ])
        _write_gz_jsonl(ann, [
            {"doc_id": "s1", "forms": ["Lepa", "beseda", "."],
             "lemmas": ["lep", "beseda", "."],
             "upos": ["ADJ", "NOUN", "PUNCT"], "feats": [None, None, None],
             "sentences": [[0, 2]]},
        ])

        records = list(iter_joined_records(text, ann))
        assert len(records) == 1
        rec = records[0]
        assert rec["text"] == "Lepa beseda."
        assert rec["annotations"]["forms"] == ["Lepa", "beseda", "."]
        assert "doc_id" not in rec["annotations"]

    def test_passthrough_when_annotations_missing(
        self, tmp_path: Path
    ) -> None:
        """Without annotations the text records stream through unchanged."""
        text = tmp_path / "k.jsonl"
        _write_jsonl(text, [
            {"text": "a", "source": "k", "domain": "x"},
            {"text": "b", "source": "k", "domain": "x"},
        ])

        records = list(iter_joined_records(text, None))
        assert [r["text"] for r in records] == ["a", "b"]
        assert all("annotations" not in r for r in records)

    def test_doc_id_mismatch_raises(self, tmp_path: Path) -> None:
        """A doc_id mismatch is treated as a hard error."""
        text = tmp_path / "k.jsonl"
        ann = tmp_path / "k.annotations.jsonl.gz"
        _write_jsonl(text, [
            {"text": "x", "source": "k", "domain": "x", "doc_id": "a"},
        ])
        _write_gz_jsonl(ann, [
            {"doc_id": "b", "forms": ["x"], "lemmas": [None],
             "upos": [None], "feats": [None], "sentences": [[0, 0]]},
        ])

        with pytest.raises(ValueError, match="doc_id mismatch"):
            list(iter_joined_records(text, ann))

    def test_length_mismatch_raises(self, tmp_path: Path) -> None:
        """If line counts differ the iterator aborts with a clear error."""
        text = tmp_path / "k.jsonl"
        ann = tmp_path / "k.annotations.jsonl.gz"
        _write_jsonl(text, [
            {"text": "a", "source": "k", "domain": "x", "doc_id": "1"},
            {"text": "b", "source": "k", "domain": "x", "doc_id": "2"},
        ])
        _write_gz_jsonl(ann, [
            {"doc_id": "1", "forms": ["a"], "lemmas": [None],
             "upos": [None], "feats": [None], "sentences": [[0, 0]]},
        ])

        with pytest.raises(ValueError, match="Line counts differ"):
            list(iter_joined_records(text, ann))


class TestOpenOutput:
    """Tests for open_output."""

    def test_gzip_suffix_writes_gzipped(self, tmp_path: Path) -> None:
        """An output path ending in .gz produces a gzip file."""
        out_path = tmp_path / "out.jsonl.gz"
        with open_output(out_path) as fh:
            fh.write("hello\n")
        with gzip.open(out_path, "rt", encoding="utf-8") as fh:
            assert fh.read() == "hello\n"

    def test_plain_path_writes_plain(self, tmp_path: Path) -> None:
        """Without .gz the output is a plain text file."""
        out_path = tmp_path / "out.jsonl"
        with open_output(out_path) as fh:
            fh.write("hi\n")
        assert out_path.read_text(encoding="utf-8") == "hi\n"

    def test_none_means_stdout(self) -> None:
        """Passing None yields sys.stdout (not closed afterwards)."""
        with open_output(None) as fh:
            assert fh is sys.stdout
