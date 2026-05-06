"""Tests for scripts/data/to_sentiment.py."""

import gzip
import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2] / "scripts" / "data"),
)
import to_sentiment  # noqa: E402


def _write_sentinews_file(
    path: Path,
    rows,
    columns,
) -> None:
    """Write a tab-separated SentiNews-style file at *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(columns) + "\n")
        for row in rows:
            fh.write("\t".join(row.get(c, "") for c in columns) + "\n")


def _read_jsonl_gz(path: Path):
    """Yield JSON-decoded records from a gzipped JSONL file."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


class TestNormalizeLabel:
    """Unit tests for to_sentiment._normalize_label."""

    def test_canonical_passes_through(self):
        for label in ("negative", "neutral", "positive"):
            assert to_sentiment._normalize_label(label) == label

    def test_uppercase_normalized(self):
        assert to_sentiment._normalize_label("Positive") == "positive"

    def test_short_form_normalized(self):
        assert to_sentiment._normalize_label("neg") == "negative"
        assert to_sentiment._normalize_label("pos") == "positive"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unrecognized SA label"):
            to_sentiment._normalize_label("very-positive")


class TestLabelId:
    """Unit tests for to_sentiment._label_id."""

    def test_encoding_is_stable(self):
        assert to_sentiment._label_id("negative") == 0
        assert to_sentiment._label_id("neutral") == 1
        assert to_sentiment._label_id("positive") == 2


class TestReadSentinews:
    """Tests for the SentiNews reader."""

    def test_document_level(self, tmp_path: Path):
        raw_dir = tmp_path / "sentinews"
        _write_sentinews_file(
            raw_dir / "SentiNews_document-level.txt",
            [
                {"nid": "1", "content": "Lepa novica.", "sentiment": "positive"},
                {"nid": "2", "content": "Slaba novica.", "sentiment": "negative"},
            ],
            columns=["nid", "content", "sentiment"],
        )
        records = list(to_sentiment._read_sentinews(raw_dir))
        assert len(records) == 2
        first = records[0]
        assert first["id"] == "sentinews:1"
        assert first["text"] == "Lepa novica."
        assert first["label"] == "positive"
        assert first["label_id"] == 2
        assert first["dataset"] == "sentinews"
        assert first["task"] == "SA"
        assert first["level"] == "document"

    def test_paragraph_level_id_includes_pid(self, tmp_path: Path):
        raw_dir = tmp_path / "sentinews"
        _write_sentinews_file(
            raw_dir / "SentiNews_paragraph-level.txt",
            [
                {"nid": "1", "pid": "0", "content": "Prvi odstavek.", "sentiment": "neutral"},
            ],
            columns=["nid", "pid", "content", "sentiment"],
        )
        records = list(to_sentiment._read_sentinews(raw_dir))
        assert records[0]["id"] == "sentinews:1-p0"
        assert records[0]["level"] == "paragraph"

    def test_levels_filter(self, tmp_path: Path):
        raw_dir = tmp_path / "sentinews"
        _write_sentinews_file(
            raw_dir / "SentiNews_document-level.txt",
            [{"nid": "1", "content": "doc", "sentiment": "neutral"}],
            columns=["nid", "content", "sentiment"],
        )
        _write_sentinews_file(
            raw_dir / "SentiNews_sentence-level.txt",
            [{"nid": "1", "pid": "0", "sid": "0", "content": "sent",
              "sentiment": "neutral"}],
            columns=["nid", "pid", "sid", "content", "sentiment"],
        )
        records = list(
            to_sentiment._read_sentinews(raw_dir, levels=["document"])
        )
        assert len(records) == 1
        assert records[0]["level"] == "document"

    def test_unknown_label_skipped(self, tmp_path: Path):
        raw_dir = tmp_path / "sentinews"
        _write_sentinews_file(
            raw_dir / "SentiNews_document-level.txt",
            [
                {"nid": "1", "content": "ok", "sentiment": "neutral"},
                {"nid": "2", "content": "bad", "sentiment": "ambivalent"},
            ],
            columns=["nid", "content", "sentiment"],
        )
        records = list(to_sentiment._read_sentinews(raw_dir))
        assert len(records) == 1
        assert records[0]["id"] == "sentinews:1"

    def test_missing_files_raise(self, tmp_path: Path):
        raw_dir = tmp_path / "sentinews"
        raw_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="No SentiNews"):
            list(to_sentiment._read_sentinews(raw_dir))


class TestConvertDataset:
    """End-to-end conversion via convert_dataset."""

    def test_writes_jsonl_gz_and_label_map(self, tmp_path: Path):
        raw_dir = tmp_path / "sentinews"
        _write_sentinews_file(
            raw_dir / "SentiNews_document-level.txt",
            [
                {"nid": "1", "content": "A", "sentiment": "negative"},
                {"nid": "2", "content": "B", "sentiment": "neutral"},
                {"nid": "3", "content": "C", "sentiment": "positive"},
            ],
            columns=["nid", "content", "sentiment"],
        )
        out_dir = tmp_path / "out"
        count = to_sentiment.convert_dataset(
            "sentinews", raw_dir, out_dir,
        )
        assert count == 3

        out_path = out_dir / "sentinews.jsonl.gz"
        records = list(_read_jsonl_gz(out_path))
        assert {r["label"] for r in records} == {
            "negative", "neutral", "positive"
        }

        label_map = json.loads(
            (out_dir / "label_map.json").read_text(encoding="utf-8")
        )
        assert label_map == {"negative": 0, "neutral": 1, "positive": 2}

    def test_unknown_dataset_returns_none(self, tmp_path: Path):
        result = to_sentiment.convert_dataset(
            "nonexistent", tmp_path, tmp_path / "out",
        )
        assert result is None

    def test_existing_output_skipped_without_force(
        self, tmp_path: Path,
    ):
        raw_dir = tmp_path / "sentinews"
        _write_sentinews_file(
            raw_dir / "SentiNews_document-level.txt",
            [{"nid": "1", "content": "A", "sentiment": "negative"}],
            columns=["nid", "content", "sentiment"],
        )
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        out_path = out_dir / "sentinews.jsonl.gz"
        with gzip.open(out_path, "wt", encoding="utf-8") as fh:
            fh.write("placeholder\n")

        result = to_sentiment.convert_dataset(
            "sentinews", raw_dir, out_dir,
        )
        assert result == 0
        assert out_path.read_bytes()[:2] == b"\x1f\x8b"  # still gzip


class TestListSaDatasetsFromConfig:
    """Tests for filtering SA-tagged benchmark datasets."""

    def test_returns_only_sa_benchmarks(self, tmp_path: Path):
        config = {
            "output_dir": str(tmp_path / "raw"),
            "datasets": {
                "pretrain_ds": {
                    "enabled": True,
                    "source": "clarin",
                    "name": "Pretrain",
                    "urls": ["https://example.com/p.gz"],
                    "output_dir": "pretrain_ds",
                },
                "sentinews": {
                    "enabled": True,
                    "benchmark": True,
                    "source": "clarin",
                    "name": "SentiNews",
                    "urls": ["https://example.com/s.txt"],
                    "output_dir": "sentinews",
                    "tasks": ["SA"],
                },
                "ssj500k": {
                    "enabled": True,
                    "benchmark": True,
                    "source": "clarin",
                    "name": "ssj500k",
                    "urls": ["https://example.com/sj.zip"],
                    "output_dir": "ssj500k",
                    "tasks": ["POS", "NER"],
                },
            },
        }
        config_file = tmp_path / "download.yaml"
        config_file.write_text(yaml.dump(config))
        keys = to_sentiment.list_sa_datasets_from_config(config_file)
        assert keys == ["sentinews"]
