"""Tests for scripts/data/to_datatrove.py."""

import gzip
import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2] / "scripts" / "data"),
)
import to_datatrove  # noqa: E402


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


class TestConvertRecord:
    """Unit tests for to_datatrove.convert_record."""

    def test_annotated_record(self) -> None:
        """Annotated record → datatrove shape with annotations preserved."""
        record = {
            "text": "Lepa beseda.",
            "source": "kzb",
            "domain": "scientific",
            "doc_id": "s1",
            "uid": "kzb:s1",
            "metadata": {"sent_id": "s1"},
            "annotations": {
                "forms": ["Lepa", "beseda", "."],
                "lemmas": ["lep", "beseda", "."],
                "upos": ["ADJ", "NOUN", "PUNCT"],
                "feats": [None, None, None],
                "sentences": [[0, 2]],
            },
        }
        out = to_datatrove.convert_record(record, 0)

        assert out["text"] == "Lepa beseda."
        assert out["id"] == "kzb:s1"
        assert out["dataset"] == "kzb"
        assert out["domain"] == "scientific"
        assert out["doc_id"] == "s1"
        assert out["sent_id"] == "s1"          # flattened from metadata
        assert "source" not in out             # renamed to dataset
        assert "uid" not in out                # renamed to id
        assert "metadata" not in out           # flattened
        assert out["annotations"]["forms"] == ["Lepa", "beseda", "."]

    def test_unannotated_record(self) -> None:
        """Record without annotations → no annotations key on output."""
        record = {
            "text": "Hello.",
            "source": "macocu_sl",
            "domain": "web",
            "doc_id": "d-1",
            "uid": "macocu_sl:d-1",
            "metadata": {"url": "https://x", "lm_score": 0.9},
        }
        out = to_datatrove.convert_record(record, 0)

        assert out["id"] == "macocu_sl:d-1"
        assert out["dataset"] == "macocu_sl"
        assert out["url"] == "https://x"
        assert out["lm_score"] == 0.9
        assert "annotations" not in out

    def test_fallback_id_when_uid_missing(self) -> None:
        """Without uid, id falls back to '<source>:idx-<index>'."""
        record = {
            "text": "no uid here",
            "source": "ds",
            "domain": "web",
        }
        out = to_datatrove.convert_record(record, 7)

        assert out["id"] == "ds:idx-00000000000007"

    def test_metadata_key_collision_renamed(self) -> None:
        """Metadata keys clashing with reserved fields get a meta_ prefix."""
        record = {
            "text": "x",
            "source": "ds",
            "domain": "web",
            "doc_id": "1",
            "uid": "ds:1",
            "metadata": {
                "dataset": "wrong",
                "domain": "wrong",
                "annotations": "wrong",
                "url": "ok",
            },
        }
        out = to_datatrove.convert_record(record, 0)

        assert out["dataset"] == "ds"
        assert out["domain"] == "web"
        assert "annotations" not in out
        assert out["meta_dataset"] == "wrong"
        assert out["meta_domain"] == "wrong"
        assert out["meta_annotations"] == "wrong"
        assert out["url"] == "ok"


class TestConvertStream:
    """Tests for to_datatrove.convert_stream."""

    def test_streams_records(self) -> None:
        """Each input record is converted and emitted in order."""
        records = [
            {"text": "a", "source": "kzb", "domain": "sci",
             "doc_id": "1", "uid": "kzb:1"},
            {"text": "b", "source": "kzb", "domain": "sci",
             "doc_id": "2", "uid": "kzb:2"},
        ]

        out = io.StringIO()
        n = to_datatrove.convert_stream(records, out)

        assert n == 2
        lines = out.getvalue().splitlines()
        rec0 = json.loads(lines[0])
        rec1 = json.loads(lines[1])
        assert rec0["id"] == "kzb:1"
        assert rec1["id"] == "kzb:2"
        assert rec0["dataset"] == "kzb"


class TestConvertDataset:
    """Tests for end-to-end per-dataset conversion."""

    def test_joins_text_and_annotations_on_the_fly(
        self, tmp_path: Path
    ) -> None:
        """Text + annotations are joined and converted to datatrove shape."""
        processed = tmp_path / "processed"
        out_dir = tmp_path / "out"
        _write_jsonl(processed / "kzb.jsonl", [
            {"text": "Lepa beseda.", "source": "kzb", "domain": "sci",
             "doc_id": "s1", "uid": "kzb:s1"},
        ])
        _write_gz_jsonl(processed / "kzb.annotations.jsonl.gz", [
            {"doc_id": "s1", "uid": "kzb:s1",
             "forms": ["Lepa", "beseda", "."],
             "lemmas": ["lep", "beseda", "."],
             "upos": ["ADJ", "NOUN", "PUNCT"],
             "feats": [None, None, None],
             "sentences": [[0, 2]]},
        ])

        n = to_datatrove.convert_dataset("kzb", processed, out_dir)
        assert n == 1

        out_path = out_dir / "kzb.jsonl.gz"
        with gzip.open(out_path, "rt", encoding="utf-8") as fh:
            rec = json.loads(fh.readline())
        assert rec["id"] == "kzb:s1"
        assert rec["annotations"]["forms"] == ["Lepa", "beseda", "."]

    def test_unannotated_dataset_passes_through(self, tmp_path: Path) -> None:
        """Without annotations sidecar, text records flow through unchanged."""
        processed = tmp_path / "processed"
        out_dir = tmp_path / "out"
        _write_jsonl(processed / "macocu_sl.jsonl", [
            {"text": "Hi.", "source": "macocu_sl", "domain": "web",
             "doc_id": "d-1", "uid": "macocu_sl:d-1",
             "metadata": {"url": "https://x"}},
        ])

        n = to_datatrove.convert_dataset("macocu_sl", processed, out_dir)
        assert n == 1

        with gzip.open(out_dir / "macocu_sl.jsonl.gz", "rt") as fh:
            rec = json.loads(fh.readline())
        assert rec["url"] == "https://x"
        assert "annotations" not in rec

    def test_missing_text_returns_none(self, tmp_path: Path) -> None:
        """No <key>.jsonl → convert_dataset returns None (skip)."""
        processed = tmp_path / "processed"
        processed.mkdir()
        result = to_datatrove.convert_dataset(
            "ghost", processed, tmp_path / "out"
        )
        assert result is None

    def test_skips_when_output_exists(self, tmp_path: Path) -> None:
        """Without force, an existing output is left untouched."""
        processed = tmp_path / "processed"
        out_dir = tmp_path / "out"
        _write_jsonl(processed / "kzb.jsonl", [
            {"text": "a", "source": "kzb", "domain": "sci",
             "doc_id": "1", "uid": "kzb:1"},
        ])
        out_dir.mkdir(parents=True, exist_ok=True)
        sentinel = out_dir / "kzb.jsonl.gz"
        sentinel.write_bytes(b"sentinel-content")

        result = to_datatrove.convert_dataset("kzb", processed, out_dir)

        assert result == 0
        assert sentinel.read_bytes() == b"sentinel-content"

    def test_force_overwrites_output(self, tmp_path: Path) -> None:
        """With force=True the existing output is overwritten."""
        processed = tmp_path / "processed"
        out_dir = tmp_path / "out"
        _write_jsonl(processed / "kzb.jsonl", [
            {"text": "a", "source": "kzb", "domain": "sci",
             "doc_id": "1", "uid": "kzb:1"},
        ])
        out_dir.mkdir(parents=True, exist_ok=True)
        sentinel = out_dir / "kzb.jsonl.gz"
        sentinel.write_bytes(b"sentinel-content")

        result = to_datatrove.convert_dataset(
            "kzb", processed, out_dir, force=True
        )
        assert result == 1
        with gzip.open(sentinel, "rt") as fh:
            rec = json.loads(fh.readline())
        assert rec["id"] == "kzb:1"


class TestParseArgs:
    """Tests for to_datatrove.parse_args."""

    def test_force_default_false(self) -> None:
        """--force defaults to False when omitted."""
        args = to_datatrove.parse_args(["kzb"])
        assert args.force is False

    def test_force_flag_sets_true(self) -> None:
        """Passing --force sets the flag to True."""
        args = to_datatrove.parse_args(["kzb", "--force"])
        assert args.force is True


class TestDatatroveRoundTrip:
    """Verifies that the produced JSONL is consumable by datatrove."""

    def test_jsonl_reader_round_trip(self, tmp_path: Path) -> None:
        """datatrove's JsonlReader yields Documents matching our shape."""
        datatrove = pytest.importorskip("datatrove")
        from datatrove.pipeline.readers import JsonlReader

        out_path = tmp_path / "kzb.jsonl.gz"
        with gzip.open(out_path, "wt", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "text": "Lepa beseda.",
                "id": "kzb:s1",
                "dataset": "kzb",
                "domain": "scientific",
                "doc_id": "s1",
                "annotations": {"forms": ["Lepa", "beseda", "."]},
            }, ensure_ascii=False) + "\n")

        reader = JsonlReader(str(tmp_path), glob_pattern="*.jsonl.gz")
        docs = list(reader.run())

        assert len(docs) == 1
        doc = docs[0]
        assert doc.text == "Lepa beseda."
        assert doc.id == "kzb:s1"
        assert doc.metadata["dataset"] == "kzb"
        assert doc.metadata["domain"] == "scientific"
        assert doc.metadata["annotations"]["forms"] == [
            "Lepa", "beseda", ".",
        ]
        del datatrove
