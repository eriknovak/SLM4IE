"""Tests for the sentiment backend (slm4ie/data/task_converters/sentiment.py)."""

import gzip
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

from slm4ie.data.task_converter import run_converter
from slm4ie.data.task_converters import sentiment


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    """Write *records* as JSONL to *path*, creating parents.

    Args:
        path: Destination path.
        records: Records to serialize.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False))
            fh.write("\n")


def _read_jsonl_gz(path: Path) -> List[Dict[str, Any]]:
    """Read a gzipped JSONL file into a list.

    Args:
        path: Gzipped JSONL path.

    Returns:
        Decoded records.
    """
    out: List[Dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _write_sentinews_tsv(path: Path, rows: List[Dict[str, str]], columns: List[str]) -> None:
    """Write a tab-separated SentiNews-style file at *path*.

    Args:
        path: Output TSV path.
        rows: Row dicts; missing columns become empty strings.
        columns: Column header order.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(columns) + "\n")
        for row in rows:
            fh.write("\t".join(row.get(c, "") for c in columns) + "\n")


def _make_extracted_layout(tmp_path: Path, dataset_key: str, records: List[Dict[str, Any]]) -> Path:
    """Build a tasks.yaml + matching ``extracted/<key>.jsonl`` source tree.

    Args:
        tmp_path: pytest tmp_path root.
        dataset_key: Source key written under ``extracted/<key>.jsonl``.
        records: Joined extraction records carrying a ``label`` / ``sentiment``.

    Returns:
        Path to the written tasks.yaml.
    """
    extracted = tmp_path / "extracted"
    raw = tmp_path / "raw"
    tasks_root = tmp_path / "tasks"
    raw.mkdir(parents=True, exist_ok=True)
    tasks_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(extracted / f"{dataset_key}.jsonl", records)

    tasks_yaml = {
        "roots": {"extracted": str(extracted), "raw": str(raw), "tasks": str(tasks_root)},
        "converters": {"sentiment": "to_sentiment"},
        "entries": {
            f"sentiment/{dataset_key}": {
                "role": "finetune_and_eval",
                "source": {"kind": "extracted", "keys": [dataset_key]},
                "splits": {
                    "train": "train.jsonl.gz",
                    "val": "val.jsonl.gz",
                    "test": "test.jsonl.gz",
                },
                "labels": ["negative", "neutral", "positive"],
                "suite": None,
                "language": "sl",
                "license": "cc-by-sa-4.0",
            },
        },
    }
    config_path = tmp_path / "tasks.yaml"
    config_path.write_text(yaml.safe_dump(tasks_yaml))
    return config_path


def _make_raw_layout(tmp_path: Path, dataset_key: str, rows: List[Dict[str, str]], columns: List[str]) -> Path:
    """Build a tasks.yaml pointing at a SentiNews-style raw TSV.

    Args:
        tmp_path: pytest tmp_path root.
        dataset_key: Subdirectory name under ``raw/<key>/``.
        rows: TSV rows.
        columns: TSV column order.

    Returns:
        Path to the written tasks.yaml.
    """
    extracted = tmp_path / "extracted"
    raw = tmp_path / "raw"
    tasks_root = tmp_path / "tasks"
    extracted.mkdir(parents=True, exist_ok=True)
    tasks_root.mkdir(parents=True, exist_ok=True)
    _write_sentinews_tsv(raw / dataset_key / "SentiNews_document-level.txt", rows, columns)

    tasks_yaml = {
        "roots": {"extracted": str(extracted), "raw": str(raw), "tasks": str(tasks_root)},
        "converters": {"sentiment": "to_sentiment"},
        "entries": {
            f"sentiment/{dataset_key}": {
                "role": "held_out",
                "source": {"kind": "raw", "keys": [dataset_key]},
                "splits": {"test": "test.jsonl.gz"},
                "labels": ["negative", "neutral", "positive"],
                "suite": None,
                "language": "sl",
                "license": "cc-by-4.0",
            },
        },
    }
    config_path = tmp_path / "tasks.yaml"
    config_path.write_text(yaml.safe_dump(tasks_yaml))
    return config_path


class TestNormalizeLabel:
    """Unit tests for `sentiment.normalize_label`."""

    def test_canonical_passes_through(self) -> None:
        """Canonical labels are returned unchanged."""
        for label in ("negative", "neutral", "positive"):
            assert sentiment.normalize_label(label, None) == label

    def test_mixed_case_normalized(self) -> None:
        """Mixed-case labels lowercase to canonical labels."""
        assert sentiment.normalize_label("Positive", None) == "positive"

    def test_short_form_normalized(self) -> None:
        """`neg` / `pos` / `neu` expand to canonical labels."""
        assert sentiment.normalize_label("neg", None) == "negative"
        assert sentiment.normalize_label("pos", None) == "positive"
        assert sentiment.normalize_label("neu", None) == "neutral"

    def test_unknown_returns_none(self) -> None:
        """Unknown labels become None instead of raising."""
        assert sentiment.normalize_label("very-positive", None) is None

    def test_allow_list_filters(self) -> None:
        """A canonical label not in the allow-list returns None."""
        assert sentiment.normalize_label("neutral", {"negative", "positive"}) is None


class TestConvertExtracted:
    """End-to-end tests for the `extracted` source path."""

    def test_writes_split_files(self, tmp_path: Path) -> None:
        """Records are bucketed across train/val/test and written gzipped."""
        records = [
            {
                "text": f"text {i}",
                "source": "sentinews",
                "doc_id": f"d{i}",
                "uid": f"sentinews:d{i}",
                "label": ["negative", "neutral", "positive"][i % 3],
            }
            for i in range(30)
        ]
        config_path = _make_extracted_layout(tmp_path, "sentinews", records)
        run_converter("to_sentiment", ["sentiment/sentinews", "--config", str(config_path), "--max-workers", "1"])

        out_dir = tmp_path / "tasks" / "sentiment" / "sentinews"
        total = 0
        sample = None
        for split in ("train", "val", "test"):
            rows = _read_jsonl_gz(out_dir / f"{split}.jsonl.gz")
            total += len(rows)
            if rows and sample is None:
                sample = rows[0]
        assert total == 30
        assert sample is not None
        assert set(sample.keys()) == {"id", "text", "label"}
        assert sample["label"] in {"negative", "neutral", "positive"}


class TestConvertRaw:
    """End-to-end tests for the `raw` single-split source path."""

    def test_raw_records_go_to_single_split(self, tmp_path: Path) -> None:
        """A single declared split collects every raw record."""
        rows = [
            {"nid": "1", "content": "Lepo.", "sentiment": "positive"},
            {"nid": "2", "content": "Slabo.", "sentiment": "negative"},
            {"nid": "3", "content": "OK.", "sentiment": "neutral"},
        ]
        config_path = _make_raw_layout(tmp_path, "twitter_sentiment_15eu", rows, ["nid", "content", "sentiment"])
        run_converter(
            "to_sentiment",
            ["sentiment/twitter_sentiment_15eu", "--config", str(config_path), "--max-workers", "1"],
        )

        out_dir = tmp_path / "tasks" / "sentiment" / "twitter_sentiment_15eu"
        assert not (out_dir / "train.jsonl.gz").exists()
        records = _read_jsonl_gz(out_dir / "test.jsonl.gz")
        assert len(records) == 3
        assert {r["label"] for r in records} == {"negative", "neutral", "positive"}
        for record in records:
            assert set(record.keys()) == {"id", "text", "label"}
