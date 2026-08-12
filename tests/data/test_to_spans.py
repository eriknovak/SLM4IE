"""Tests for the NER backend (slm4ie/data/task_converters/spans.py)."""

import gzip
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml

from slm4ie.data.task_converter import run_converter
from slm4ie.data.task_converters import spans


def _write_jsonl(path: Path, records: List[Dict]) -> None:
    """Write *records* as JSONL to *path*, creating parents.

    Args:
        path: Destination file path.
        records: Records to serialize.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False))
            fh.write("\n")


def _write_gz_jsonl(path: Path, records: List[Dict]) -> None:
    """Write *records* as gzipped JSONL to *path*, creating parents.

    Args:
        path: Destination ``.jsonl.gz`` path.
        records: Records to serialize.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False))
            fh.write("\n")


def _read_jsonl_gz(path: Path) -> List[Dict]:
    """Read a gzipped JSONL file into a list.

    Args:
        path: Gzipped JSONL path.

    Returns:
        Decoded records.
    """
    out: List[Dict] = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _make_synthetic_layout(tmp_path: Path, dataset_key: str, records: List[Dict[str, Any]]) -> Path:
    """Build a synthetic tasks.yaml + matching extracted source tree.

    Args:
        tmp_path: pytest tmp_path fixture root.
        dataset_key: Source key written under ``extracted/<key>.jsonl``.
        records: Joined extraction records (carrying ``uid`` / ``doc_id`` /
            ``annotations.spans``).

    Returns:
        Path to the written tasks.yaml.
    """
    extracted = tmp_path / "extracted"
    raw = tmp_path / "raw"
    tasks_root = tmp_path / "tasks"
    raw.mkdir(parents=True, exist_ok=True)
    tasks_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(extracted / f"{dataset_key}.jsonl", records)

    ann_records: List[Dict[str, Any]] = []
    for record in records:
        ann = dict(record.get("annotations") or {})
        ann["uid"] = record.get("uid")
        ann["doc_id"] = record.get("doc_id")
        ann_records.append(ann)
    if any(rec.get("annotations") for rec in records):
        _write_gz_jsonl(extracted / f"{dataset_key}.annotations.jsonl.gz", ann_records)

    tasks_yaml = {
        "roots": {"extracted": str(extracted), "raw": str(raw), "tasks": str(tasks_root)},
        "converters": {"ner": "to_spans"},
        "entries": {
            f"ner/{dataset_key}": {
                "role": "finetune_and_eval",
                "source": {"kind": "extracted", "keys": [dataset_key]},
                "splits": {
                    "train": "train.jsonl.gz",
                    "val": "val.jsonl.gz",
                    "test": "test.jsonl.gz",
                },
                "labels": ["PER", "LOC", "ORG"],
                "suite": None,
                "language": "sl",
                "license": "cc-by-sa-4.0",
            },
        },
    }
    config_path = tmp_path / "tasks.yaml"
    config_path.write_text(yaml.safe_dump(tasks_yaml))
    return config_path


class TestNormalizeSpans:
    """Unit tests for `spans.normalize_spans`."""

    def test_accepts_triples(self) -> None:
        """List-of-lists input is normalized to tuples."""
        assert spans.normalize_spans([[0, 2, "PER"], [3, 5, "LOC"]]) == [(0, 2, "PER"), (3, 5, "LOC")]

    def test_accepts_dicts(self) -> None:
        """Dict input is normalized to tuples."""
        assert spans.normalize_spans([{"start": 0, "end": 2, "label": "PER"}]) == [(0, 2, "PER")]

    def test_rejects_malformed(self) -> None:
        """Unrecognized shapes raise ValueError."""
        with pytest.raises(ValueError, match="Unrecognized span shape"):
            spans.normalize_spans(["not a span"])

    def test_empty_input_returns_empty(self) -> None:
        """Empty or None input produces an empty list."""
        assert spans.normalize_spans(None) == []
        assert spans.normalize_spans([]) == []


class TestConvertEndToEnd:
    """End-to-end conversion driven through `run_converter`."""

    def test_writes_split_files(self, tmp_path: Path) -> None:
        """Every declared split file is written with NerExample schema."""
        records = [
            {
                "text": "John lives in Paris.",
                "source": "kzb",
                "domain": "sci",
                "doc_id": f"s{i}",
                "uid": f"kzb:s{i}",
                "annotations": {"forms": ["John"], "spans": [[0, 4, "PER"], [14, 19, "LOC"]]},
            }
            for i in range(20)
        ]
        config_path = _make_synthetic_layout(tmp_path, "kzb", records)
        run_converter("to_spans", ["ner/kzb", "--config", str(config_path), "--max-workers", "1"])

        out_dir = tmp_path / "tasks" / "ner" / "kzb"
        total = 0
        sample = None
        for split in ("train", "val", "test"):
            rows = _read_jsonl_gz(out_dir / f"{split}.jsonl.gz")
            total += len(rows)
            if rows and sample is None:
                sample = rows[0]
        assert total == 20
        assert sample is not None
        assert set(sample.keys()) == {"id", "text", "spans"}
        assert sample["spans"][0] == {"start": 0, "end": 4, "label": "PER"}

    def test_labels_outside_allow_list_are_dropped(self, tmp_path: Path) -> None:
        """Spans with labels outside the entry's allow-list are filtered out."""
        records = [
            {
                "text": "X Y.",
                "source": "kzb",
                "doc_id": "s1",
                "uid": "kzb:s1",
                # MISC is not in the synthetic entry's labels.
                "annotations": {"forms": ["X", "Y"], "spans": [[0, 1, "PER"], [2, 3, "MISC"]]},
            }
        ]
        config_path = _make_synthetic_layout(tmp_path, "kzb", records)
        run_converter("to_spans", ["ner/kzb", "--config", str(config_path), "--max-workers", "1"])

        out_dir = tmp_path / "tasks" / "ner" / "kzb"
        rows: List[Dict] = []
        for split in ("train", "val", "test"):
            path = out_dir / f"{split}.jsonl.gz"
            if path.exists():
                rows.extend(_read_jsonl_gz(path))
        assert len(rows) == 1
        assert rows[0]["spans"] == [{"start": 0, "end": 1, "label": "PER"}]

    def test_skips_when_outputs_exist(self, tmp_path: Path) -> None:
        """A second run without --force leaves the existing outputs untouched."""
        records = [
            {
                "text": "John.",
                "source": "kzb",
                "doc_id": "s1",
                "uid": "kzb:s1",
                "annotations": {"forms": ["John"], "spans": [[0, 4, "PER"]]},
            }
        ]
        config_path = _make_synthetic_layout(tmp_path, "kzb", records)
        argv = ["ner/kzb", "--config", str(config_path), "--max-workers", "1"]
        run_converter("to_spans", argv)
        out_dir = tmp_path / "tasks" / "ner" / "kzb"
        existing = {p: p.stat().st_mtime_ns for p in out_dir.glob("*.jsonl.gz")}
        assert existing

        run_converter("to_spans", argv)
        for path, mtime in existing.items():
            assert path.stat().st_mtime_ns == mtime
