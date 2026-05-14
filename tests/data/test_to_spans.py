"""Tests for scripts/data/to_spans.py (tasks.yaml-driven NER converter)."""

import gzip
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2] / "scripts" / "data"),
)
import to_spans  # noqa: E402

from slm4ie.data.tasks import load_tasks  # noqa: E402


def _write_jsonl(path: Path, records: List[Dict]) -> None:
    """Write *records* as JSONL to *path*.

    Args:
        path: Destination file path. Parents are created as needed.
        records: Records to serialize.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False))
            fh.write("\n")


def _write_gz_jsonl(path: Path, records: List[Dict]) -> None:
    """Write *records* as one JSON object per line, gzipped.

    Args:
        path: Destination ``.jsonl.gz`` path. Parents are created as
            needed.
        records: Records to serialize.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False))
            fh.write("\n")


def _read_jsonl_gz(path: Path) -> List[Dict]:
    """Yield JSON-decoded records from a gzipped JSONL file.

    Args:
        path: Path to a gzipped JSONL file.

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


def _make_synthetic_layout(
    tmp_path: Path,
    dataset_key: str,
    records: List[Dict[str, Any]],
) -> Path:
    """Build a synthetic tasks.yaml + matching extracted source tree.

    Args:
        tmp_path: pytest tmp_path fixture root.
        dataset_key: Source key written under ``extracted/<key>.jsonl``.
        records: Joined extraction records (already carrying ``uid`` /
            ``doc_id`` / ``annotations.spans``).

    Returns:
        Path to the written tasks.yaml.
    """
    extracted = tmp_path / "extracted"
    raw = tmp_path / "raw"
    tasks_root = tmp_path / "tasks"
    raw.mkdir(parents=True, exist_ok=True)
    tasks_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(extracted / f"{dataset_key}.jsonl", records)

    # Write the annotations sidecar derived from each record's
    # `annotations` payload (mirrors what extract.py produces).
    ann_records: List[Dict[str, Any]] = []
    for record in records:
        ann = dict(record.get("annotations") or {})
        ann["uid"] = record.get("uid")
        ann["doc_id"] = record.get("doc_id")
        ann_records.append(ann)
    if any(rec.get("annotations") for rec in records):
        _write_gz_jsonl(
            extracted / f"{dataset_key}.annotations.jsonl.gz", ann_records
        )

    tasks_yaml = {
        "roots": {
            "extracted": str(extracted),
            "raw": str(raw),
            "tasks": str(tasks_root),
        },
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
    """Unit tests for `to_spans._normalize_spans`."""

    def test_accepts_triples(self) -> None:
        """List-of-lists input is normalized to tuples."""
        out = to_spans._normalize_spans([[0, 2, "PER"], [3, 5, "LOC"]])
        assert out == [(0, 2, "PER"), (3, 5, "LOC")]

    def test_accepts_dicts(self) -> None:
        """Dict input is normalized to tuples."""
        out = to_spans._normalize_spans(
            [{"start": 0, "end": 2, "label": "PER"}]
        )
        assert out == [(0, 2, "PER")]

    def test_rejects_malformed(self) -> None:
        """Unrecognized shapes raise ValueError."""
        with pytest.raises(ValueError, match="Unrecognized span shape"):
            to_spans._normalize_spans(["not a span"])

    def test_empty_input_returns_empty(self) -> None:
        """Empty or None input produces an empty list."""
        assert to_spans._normalize_spans(None) == []
        assert to_spans._normalize_spans([]) == []


class TestRecordId:
    """Unit tests for `to_spans._record_id`."""

    def test_prefers_uid(self) -> None:
        """A non-empty `uid` wins over `doc_id`."""
        record = {"uid": "kzb:s1", "source": "kzb", "doc_id": "s1"}
        assert to_spans._record_id(record, 0) == "kzb:s1"

    def test_falls_back_to_source_doc_id(self) -> None:
        """Without `uid`, the id becomes ``<source>:<doc_id>``."""
        record = {"source": "kzb", "doc_id": "s2"}
        assert to_spans._record_id(record, 0) == "kzb:s2"

    def test_falls_back_to_index(self) -> None:
        """Without `uid` and `doc_id`, the index is used."""
        record = {"source": "kzb"}
        assert to_spans._record_id(record, 7) == "kzb:idx-00000000000007"


class TestConvertEntry:
    """End-to-end conversion tests driven by a synthetic tasks.yaml."""

    def test_writes_split_files(self, tmp_path: Path) -> None:
        """`convert_entry` writes one file per declared split."""
        records = [
            {
                "text": "John lives in Paris.",
                "source": "kzb",
                "domain": "sci",
                "doc_id": f"s{i}",
                "uid": f"kzb:s{i}",
                "annotations": {
                    "forms": ["John", "lives", "in", "Paris", "."],
                    "spans": [[0, 4, "PER"], [14, 19, "LOC"]],
                },
            }
            for i in range(20)
        ]
        config_path = _make_synthetic_layout(tmp_path, "kzb", records)
        cfg = load_tasks(config_path)
        entry = next(
            e for e in cfg.entries if f"{e.task}/{e.dataset}" == "ner/kzb"
        )

        counts = to_spans.convert_entry("ner/kzb", entry, cfg.roots)
        assert counts is not None
        assert sum(counts.values()) == 20

        out_dir = tmp_path / "tasks" / "ner" / "kzb"
        assert (out_dir / "train.jsonl.gz").exists()
        rows = _read_jsonl_gz(out_dir / "train.jsonl.gz")
        sample = rows[0]
        # Schema check: matches slm4ie.data.schema.NerExample.
        assert set(sample.keys()) == {"id", "text", "spans"}
        assert isinstance(sample["spans"], list)
        assert sample["spans"][0] == {
            "start": 0, "end": 4, "label": "PER",
        }
        assert sample["spans"][1] == {
            "start": 14, "end": 19, "label": "LOC",
        }

    def test_skips_when_outputs_exist(self, tmp_path: Path) -> None:
        """`convert_entry` returns None when every split already exists."""
        records = [
            {
                "text": "X.",
                "source": "kzb",
                "doc_id": "s1",
                "uid": "kzb:s1",
                "annotations": {
                    "forms": ["X", "."],
                    "spans": [[0, 1, "PER"]],
                },
            }
        ]
        config_path = _make_synthetic_layout(tmp_path, "kzb", records)
        cfg = load_tasks(config_path)
        entry = next(
            e for e in cfg.entries if f"{e.task}/{e.dataset}" == "ner/kzb"
        )

        out_dir = tmp_path / "tasks" / "ner" / "kzb"
        out_dir.mkdir(parents=True)
        for split_filename in entry.splits.values():
            (out_dir / split_filename).write_bytes(b"\x1f\x8b")

        result = to_spans.convert_entry("ner/kzb", entry, cfg.roots)
        assert result is None

    def test_force_overwrites_outputs(self, tmp_path: Path) -> None:
        """`force=True` re-derives outputs even when files exist."""
        records = [
            {
                "text": "John.",
                "source": "kzb",
                "doc_id": "s1",
                "uid": "kzb:s1",
                "annotations": {
                    "forms": ["John", "."],
                    "spans": [[0, 4, "PER"]],
                },
            }
        ]
        config_path = _make_synthetic_layout(tmp_path, "kzb", records)
        cfg = load_tasks(config_path)
        entry = next(
            e for e in cfg.entries if f"{e.task}/{e.dataset}" == "ner/kzb"
        )

        out_dir = tmp_path / "tasks" / "ner" / "kzb"
        out_dir.mkdir(parents=True)
        for split_filename in entry.splits.values():
            (out_dir / split_filename).write_bytes(b"placeholder")

        counts = to_spans.convert_entry(
            "ner/kzb", entry, cfg.roots, force=True
        )
        assert counts is not None
        assert sum(counts.values()) == 1

    def test_labels_outside_allow_list_are_dropped(self, tmp_path: Path) -> None:
        """Spans with labels outside the entry's allow-list are filtered."""
        records = [
            {
                "text": "X Y.",
                "source": "kzb",
                "doc_id": "s1",
                "uid": "kzb:s1",
                "annotations": {
                    "forms": ["X", "Y", "."],
                    "spans": [
                        [0, 1, "PER"],
                        # MISC is *not* in the synthetic entry's labels.
                        [2, 3, "MISC"],
                    ],
                },
            }
        ]
        config_path = _make_synthetic_layout(tmp_path, "kzb", records)
        cfg = load_tasks(config_path)
        entry = next(
            e for e in cfg.entries if f"{e.task}/{e.dataset}" == "ner/kzb"
        )

        counts = to_spans.convert_entry("ner/kzb", entry, cfg.roots)
        assert counts is not None
        # Records exist; we just need to find the one row written.
        rows: List[Dict] = []
        for split_filename in entry.splits.values():
            path = tmp_path / "tasks" / "ner" / "kzb" / split_filename
            if path.exists():
                rows.extend(_read_jsonl_gz(path))
        assert len(rows) == 1
        assert rows[0]["spans"] == [{"start": 0, "end": 1, "label": "PER"}]


class TestParseArgs:
    """Tests for `to_spans.parse_args`."""

    def test_accepts_entry_keys(self) -> None:
        """Positional entry keys are gathered into `args.entries`."""
        args = to_spans.parse_args(["ner/ssj500k", "ner/suk"])
        assert args.entries == ["ner/ssj500k", "ner/suk"]
        assert args.all is False

    def test_all_flag(self) -> None:
        """`--all` is mutually exclusive with positional entries."""
        args = to_spans.parse_args(["--all"])
        assert args.all is True
        assert args.entries == []

    def test_force_default_false(self) -> None:
        """`--force` defaults to False."""
        args = to_spans.parse_args(["ner/ssj500k"])
        assert args.force is False

    def test_force_flag_sets_true(self) -> None:
        """Passing `--force` flips the flag."""
        args = to_spans.parse_args(["ner/ssj500k", "--force"])
        assert args.force is True

    def test_bare_invocation_errors(self) -> None:
        """A bare invocation requires entries or `--all`."""
        with pytest.raises(SystemExit):
            to_spans.parse_args([])
