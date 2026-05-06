"""Tests for scripts/data/to_spans.py."""

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
import to_spans  # noqa: E402


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


def _record_with_spans(**overrides):
    """Build a joined record carrying token-level span annotations."""
    base = {
        "text": "John lives in Paris.",
        "source": "kzb",
        "domain": "scientific",
        "doc_id": "s1",
        "uid": "kzb:s1",
        "annotations": {
            "forms": ["John", "lives", "in", "Paris", "."],
            "spans": [
                [0, 1, "PER"],
                [3, 4, "LOC"],
            ],
        },
    }
    base.update(overrides)
    return base


class TestNormalizeSpans:
    """Tests for to_spans._normalize_spans."""

    def test_accepts_triples(self) -> None:
        """List-of-lists input is normalized to tuples."""
        out = to_spans._normalize_spans([[0, 2, "PER"], [3, 5, "LOC"]])
        assert out == [(0, 2, "PER"), (3, 5, "LOC")]

    def test_accepts_dicts(self) -> None:
        """Dict input is normalized to tuples."""
        out = to_spans._normalize_spans([
            {"start": 0, "end": 2, "label": "PER"},
        ])
        assert out == [(0, 2, "PER")]

    def test_rejects_malformed(self) -> None:
        """Unrecognized shapes raise ValueError."""
        with pytest.raises(ValueError, match="Unrecognized span shape"):
            to_spans._normalize_spans(["not a span"])


class TestToGliner:
    """Tests for to_spans.to_gliner."""

    def test_converts_to_inclusive_indices(self) -> None:
        """Token indices become GLiNER's end-inclusive convention."""
        out = to_spans.to_gliner(_record_with_spans(), 0)
        assert out is not None
        assert out["id"] == "kzb:s1"
        assert out["tokenized_text"] == [
            "John", "lives", "in", "Paris", "."
        ]
        assert out["ner"] == [[0, 0, "PER"], [3, 3, "LOC"]]

    def test_returns_none_without_spans(self) -> None:
        """Records without a spans field are skipped."""
        record = _record_with_spans()
        record["annotations"].pop("spans")
        assert to_spans.to_gliner(record, 0) is None


class TestToConll:
    """Tests for to_spans.to_conll."""

    def test_emits_iob2_tags(self) -> None:
        """Tags follow B-/I-/O conventions."""
        out = to_spans.to_conll(_record_with_spans(), 0)
        assert out is not None
        assert out["tokens"] == ["John", "lives", "in", "Paris", "."]
        assert out["ner_tags"] == ["B-PER", "O", "O", "B-LOC", "O"]

    def test_multi_token_span_gets_inside_tag(self) -> None:
        """Spans covering multiple tokens get B- followed by I- tags."""
        record = _record_with_spans()
        record["annotations"]["spans"] = [[0, 3, "ORG"]]
        out = to_spans.to_conll(record, 0)
        assert out is not None
        assert out["ner_tags"][:3] == ["B-ORG", "I-ORG", "I-ORG"]
        assert out["ner_tags"][3:] == ["O", "O"]

    def test_skips_out_of_bounds_span(self) -> None:
        """A span past the token boundary is logged and skipped."""
        record = _record_with_spans()
        record["annotations"]["spans"] = [[0, 99, "PER"]]
        out = to_spans.to_conll(record, 0)
        assert out is not None
        assert all(t == "O" for t in out["ner_tags"])


class TestToGeneric:
    """Tests for to_spans.to_generic."""

    def test_lossless_shape(self) -> None:
        """All input fields needed downstream are preserved."""
        out = to_spans.to_generic(_record_with_spans(), 0)
        assert out is not None
        assert out["id"] == "kzb:s1"
        assert out["text"] == "John lives in Paris."
        assert out["tokens"] == ["John", "lives", "in", "Paris", "."]
        assert out["spans"] == [
            {"start": 0, "end": 1, "label": "PER"},
            {"start": 3, "end": 4, "label": "LOC"},
        ]
        assert out["dataset"] == "kzb"
        assert out["domain"] == "scientific"


class TestConvertRecord:
    """Tests for to_spans.convert_record (schema dispatch)."""

    @pytest.mark.parametrize("schema", ["gliner", "conll", "generic"])
    def test_dispatches_each_schema(self, schema: str) -> None:
        """Each known schema produces a non-None result."""
        out = to_spans.convert_record(_record_with_spans(), 0, schema)
        assert out is not None

    def test_unknown_schema_raises(self) -> None:
        """Unknown schema names raise ValueError."""
        with pytest.raises(ValueError, match="Unknown schema"):
            to_spans.convert_record(_record_with_spans(), 0, "bogus")

    def test_fallback_id_when_uid_missing(self) -> None:
        """Without uid, id falls back to '<source>:idx-<index>'."""
        record = _record_with_spans()
        record.pop("uid")
        out = to_spans.convert_record(record, 7, "generic")
        assert out is not None
        assert out["id"] == "kzb:idx-00000000000007"


class TestConvertStream:
    """Tests for to_spans.convert_stream."""

    def test_counts_written_and_skipped(self) -> None:
        """Records without spans are counted as skipped, not written."""
        records = [
            _record_with_spans(),
            {"text": "no spans", "source": "kzb", "domain": "sci",
             "annotations": {"forms": ["no", "spans"]}},
        ]
        out = io.StringIO()
        written, skipped = to_spans.convert_stream(records, out, "generic")
        assert written == 1
        assert skipped == 1
        line = out.getvalue().strip()
        assert json.loads(line)["dataset"] == "kzb"


class TestConvertDataset:
    """End-to-end tests for to_spans.convert_dataset."""

    def test_writes_gzipped_output(self, tmp_path: Path) -> None:
        """convert_dataset reads text + annotations and writes gz JSONL."""
        processed = tmp_path / "processed"
        out_dir = tmp_path / "out"
        _write_jsonl(processed / "kzb.jsonl", [
            {"text": "John lives in Paris.", "source": "kzb",
             "domain": "scientific", "doc_id": "s1", "uid": "kzb:s1"},
        ])
        _write_gz_jsonl(processed / "kzb.annotations.jsonl.gz", [
            {"doc_id": "s1", "uid": "kzb:s1",
             "forms": ["John", "lives", "in", "Paris", "."],
             "spans": [[0, 1, "PER"], [3, 4, "LOC"]]},
        ])

        result = to_spans.convert_dataset(
            "kzb", processed, out_dir, "gliner"
        )
        assert result == (1, 0)

        with gzip.open(out_dir / "kzb.jsonl.gz", "rt") as fh:
            rec = json.loads(fh.readline())
        assert rec["ner"] == [[0, 0, "PER"], [3, 3, "LOC"]]

    def test_missing_input_returns_none(self, tmp_path: Path) -> None:
        """No <key>.jsonl → returns None (skip, not error)."""
        processed = tmp_path / "processed"
        processed.mkdir()
        result = to_spans.convert_dataset(
            "ghost", processed, tmp_path / "out", "generic"
        )
        assert result is None

    def test_skips_when_output_exists(self, tmp_path: Path) -> None:
        """Without force, an existing output is left untouched."""
        processed = tmp_path / "processed"
        out_dir = tmp_path / "out"
        _write_jsonl(processed / "kzb.jsonl", [
            {"text": "John lives.", "source": "kzb", "domain": "sci",
             "doc_id": "s1", "uid": "kzb:s1"},
        ])
        _write_gz_jsonl(processed / "kzb.annotations.jsonl.gz", [
            {"doc_id": "s1", "uid": "kzb:s1",
             "forms": ["John", "lives", "."],
             "spans": [[0, 1, "PER"]]},
        ])
        out_dir.mkdir(parents=True, exist_ok=True)
        sentinel = out_dir / "kzb.jsonl.gz"
        sentinel.write_bytes(b"sentinel-content")

        result = to_spans.convert_dataset(
            "kzb", processed, out_dir, "gliner"
        )

        assert result == (0, 0)
        assert sentinel.read_bytes() == b"sentinel-content"

    def test_force_overwrites_output(self, tmp_path: Path) -> None:
        """With force=True the existing output is overwritten."""
        processed = tmp_path / "processed"
        out_dir = tmp_path / "out"
        _write_jsonl(processed / "kzb.jsonl", [
            {"text": "John lives.", "source": "kzb", "domain": "sci",
             "doc_id": "s1", "uid": "kzb:s1"},
        ])
        _write_gz_jsonl(processed / "kzb.annotations.jsonl.gz", [
            {"doc_id": "s1", "uid": "kzb:s1",
             "forms": ["John", "lives", "."],
             "spans": [[0, 1, "PER"]]},
        ])
        out_dir.mkdir(parents=True, exist_ok=True)
        sentinel = out_dir / "kzb.jsonl.gz"
        sentinel.write_bytes(b"sentinel-content")

        result = to_spans.convert_dataset(
            "kzb", processed, out_dir, "gliner", force=True
        )

        assert result == (1, 0)
        with gzip.open(sentinel, "rt") as fh:
            rec = json.loads(fh.readline())
        assert rec["id"] == "kzb:s1"


class TestParseArgs:
    """Tests for to_spans.parse_args."""

    def test_schema_default_is_generic(self) -> None:
        """Default schema is generic when --schema is omitted."""
        args = to_spans.parse_args(["kzb"])
        assert args.schema == "generic"
        assert args.dataset == "kzb"
        assert args.all is False

    def test_all_flag(self) -> None:
        """--all sets the all bool and leaves dataset as None."""
        args = to_spans.parse_args(["--all", "--schema", "gliner"])
        assert args.all is True
        assert args.dataset is None
        assert args.schema == "gliner"

    def test_invalid_schema_rejected(self) -> None:
        """argparse rejects unknown schema names."""
        with pytest.raises(SystemExit):
            to_spans.parse_args(["kzb", "--schema", "bogus"])

    def test_force_default_false(self) -> None:
        """--force defaults to False when omitted."""
        args = to_spans.parse_args(["kzb"])
        assert args.force is False

    def test_force_flag_sets_true(self) -> None:
        """Passing --force sets the flag to True."""
        args = to_spans.parse_args(["kzb", "--force"])
        assert args.force is True
