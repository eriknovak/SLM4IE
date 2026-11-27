"""Tests for the sentiment backend (slm4ie/data/tasks/converters/sentiment.py)."""

import gzip
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml

from slm4ie.data.tasks.run import convert_tasks
from slm4ie.data.tasks.converters import sentiment


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
        "converters": {"sentiment": "sentiment"},
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
        convert_tasks(config_path, entry_keys=["sentiment/sentinews"], max_workers=1)

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

    def test_parlasent_shaped_records_fill_every_split_with_lowercase_labels(self, tmp_path: Path) -> None:
        """Capitalised `metadata.label` values and positional ids convert into all three splits."""
        records = [
            {
                "text": f"Poslanec je rekel stavek {i}.",
                "source": "parlasent",
                "domain": "parliamentary",
                "doc_id": f"ParlaSent_SL:{i}",
                "metadata": {"label": ["Neutral", "Negative", "Positive"][i % 3], "split": "train"},
            }
            for i in range(60)
        ]
        config_path = _make_extracted_layout(tmp_path, "parlasent", records)
        convert_tasks(config_path, entry_keys=["sentiment/parlasent"], max_workers=1)

        out_dir = tmp_path / "tasks" / "sentiment" / "parlasent"
        splits = {split: _read_jsonl_gz(out_dir / f"{split}.jsonl.gz") for split in ("train", "val", "test")}
        assert all(splits.values())
        assert {row["label"] for rows in splits.values() for row in rows} == {"negative", "neutral", "positive"}

    def test_unmapped_labels_are_dropped_and_counted(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """A record whose label maps to no class is left out, and the run log counts it."""
        records = [
            {"text": "Dobro.", "source": "s", "doc_id": "a", "metadata": {"label": "Positive"}},
            {"text": "Mešano.", "source": "s", "doc_id": "b", "metadata": {"label": "Mixed"}},
        ]
        config_path = _make_extracted_layout(tmp_path, "s", records)
        with caplog.at_level("WARNING"):
            summary = convert_tasks(config_path, entry_keys=["sentiment/s"], max_workers=1)
        assert summary.failed == []
        assert "sentiment/s: dropped 1 record(s)" in caplog.text


class TestRejectRaw:
    """The sentiment family reads extracted sources only."""

    def test_raw_source_entry_is_reported_failed(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """An entry with `source.kind: raw` fails on its source kind, not on a missing file."""
        config_path = _make_extracted_layout(tmp_path, "tweets", [])
        config = yaml.safe_load(config_path.read_text())
        config["entries"]["sentiment/tweets"]["source"]["kind"] = "raw"
        config_path.write_text(yaml.safe_dump(config))
        tsv = tmp_path / "raw" / "tweets" / "tweets.tsv"
        tsv.parent.mkdir(parents=True)
        tsv.write_text("id\tcontent\tsentiment\n1\tLepo.\tpositive\n", encoding="utf-8")

        summary = convert_tasks(config_path, entry_keys=["sentiment/tweets"], max_workers=1)

        assert summary.failed == ["sentiment/tweets"]
        assert "sentiment/tweets has source kind 'raw'" in caplog.text
