"""Tests for scripts/data/to_sentiment.py (tasks.yaml-driven SA converter)."""

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
import to_sentiment  # noqa: E402

from slm4ie.data.tasks import load_tasks  # noqa: E402


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    """Write *records* as JSONL to *path*.

    Args:
        path: Destination path. Parents are created if missing.
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


def _write_sentinews_tsv(
    path: Path,
    rows: List[Dict[str, str]],
    columns: List[str],
) -> None:
    """Write a tab-separated SentiNews-style file at *path*.

    Args:
        path: Output TSV path. Parents are created if missing.
        rows: Row dicts; missing columns become empty strings.
        columns: Column header order.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(columns) + "\n")
        for row in rows:
            fh.write("\t".join(row.get(c, "") for c in columns) + "\n")


def _make_extracted_layout(
    tmp_path: Path,
    dataset_key: str,
    records: List[Dict[str, Any]],
) -> Path:
    """Build a tasks.yaml + matching `extracted/<key>.jsonl` source tree.

    Args:
        tmp_path: pytest tmp_path root.
        dataset_key: Source key written under ``extracted/<key>.jsonl``.
        records: Joined extraction records, each carrying a `label` /
            `sentiment` field that the converter normalizes.

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
        "roots": {
            "extracted": str(extracted),
            "raw": str(raw),
            "tasks": str(tasks_root),
        },
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


def _make_raw_layout(
    tmp_path: Path,
    dataset_key: str,
    rows: List[Dict[str, str]],
    columns: List[str],
) -> Path:
    """Build a tasks.yaml that points at a SentiNews-style raw TSV.

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
    _write_sentinews_tsv(
        raw / dataset_key / "SentiNews_document-level.txt",
        rows,
        columns,
    )

    tasks_yaml = {
        "roots": {
            "extracted": str(extracted),
            "raw": str(raw),
            "tasks": str(tasks_root),
        },
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
    """Unit tests for `to_sentiment._normalize_label`."""

    def test_canonical_passes_through(self) -> None:
        """Canonical labels are returned unchanged."""
        for label in ("negative", "neutral", "positive"):
            assert to_sentiment._normalize_label(label, None) == label

    def test_mixed_case_normalized(self) -> None:
        """Mixed-case labels lowercase to canonical labels."""
        assert to_sentiment._normalize_label("Positive", None) == "positive"

    def test_short_form_normalized(self) -> None:
        """`neg` / `pos` / `neu` expand to canonical labels."""
        assert to_sentiment._normalize_label("neg", None) == "negative"
        assert to_sentiment._normalize_label("pos", None) == "positive"
        assert to_sentiment._normalize_label("neu", None) == "neutral"

    def test_unknown_returns_none(self) -> None:
        """Unknown labels become None instead of raising."""
        assert to_sentiment._normalize_label("very-positive", None) is None

    def test_allow_list_filters(self) -> None:
        """A canonical label not in the allow-list returns None."""
        assert (
            to_sentiment._normalize_label("neutral", {"negative", "positive"})
            is None
        )


class TestConvertEntryExtracted:
    """Tests for the `extracted` source path."""

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
        cfg = load_tasks(config_path)
        entry = next(
            e for e in cfg.entries
            if f"{e.task}/{e.dataset}" == "sentiment/sentinews"
        )

        counts = to_sentiment.convert_entry(
            "sentiment/sentinews", entry, cfg.roots
        )
        assert counts is not None
        assert sum(counts.values()) == 30

        # Every declared split file must exist.
        out_dir = tmp_path / "tasks" / "sentiment" / "sentinews"
        for split_filename in entry.splits.values():
            assert (out_dir / split_filename).exists()

        # Spot-check the schema of one row matches SentimentExample.
        records_train = _read_jsonl_gz(out_dir / "train.jsonl.gz")
        sample = records_train[0]
        assert set(sample.keys()) == {"id", "text", "label"}
        assert sample["label"] in {"negative", "neutral", "positive"}

    def test_skips_when_outputs_exist(self, tmp_path: Path) -> None:
        """Existing outputs short-circuit the converter."""
        records = [
            {
                "text": "x",
                "source": "sentinews",
                "doc_id": "d1",
                "uid": "sentinews:d1",
                "label": "neutral",
            }
        ]
        config_path = _make_extracted_layout(tmp_path, "sentinews", records)
        cfg = load_tasks(config_path)
        entry = next(
            e for e in cfg.entries
            if f"{e.task}/{e.dataset}" == "sentiment/sentinews"
        )

        out_dir = tmp_path / "tasks" / "sentiment" / "sentinews"
        out_dir.mkdir(parents=True)
        for split_filename in entry.splits.values():
            (out_dir / split_filename).write_bytes(b"\x1f\x8b")

        result = to_sentiment.convert_entry(
            "sentiment/sentinews", entry, cfg.roots
        )
        assert result is None


class TestConvertEntryRaw:
    """Tests for the `raw` source path (SentiNews-style TSV)."""

    def test_raw_records_go_to_single_split(self, tmp_path: Path) -> None:
        """When only one split is declared, every record lands there."""
        rows = [
            {"nid": "1", "content": "Lepo.", "sentiment": "positive"},
            {"nid": "2", "content": "Slabo.", "sentiment": "negative"},
            {"nid": "3", "content": "OK.", "sentiment": "neutral"},
        ]
        config_path = _make_raw_layout(
            tmp_path,
            "twitter_sentiment_15eu",
            rows,
            columns=["nid", "content", "sentiment"],
        )
        cfg = load_tasks(config_path)
        entry = next(
            e for e in cfg.entries
            if f"{e.task}/{e.dataset}" == "sentiment/twitter_sentiment_15eu"
        )

        counts = to_sentiment.convert_entry(
            "sentiment/twitter_sentiment_15eu", entry, cfg.roots
        )
        assert counts is not None
        assert counts == {"test": 3}

        out_dir = (
            tmp_path / "tasks" / "sentiment" / "twitter_sentiment_15eu"
        )
        records = _read_jsonl_gz(out_dir / "test.jsonl.gz")
        assert len(records) == 3
        assert {r["label"] for r in records} == {
            "negative", "neutral", "positive"
        }
        for record in records:
            # Schema parity with SentimentExample.
            assert set(record.keys()) == {"id", "text", "label"}


class TestParseArgs:
    """Tests for `to_sentiment.parse_args`."""

    def test_accepts_entry_keys(self) -> None:
        """Positional entries are gathered into `args.entries`."""
        args = to_sentiment.parse_args(["sentiment/sentinews"])
        assert args.entries == ["sentiment/sentinews"]
        assert args.all is False

    def test_all_flag(self) -> None:
        """`--all` parses without positional entries."""
        args = to_sentiment.parse_args(["--all"])
        assert args.all is True
        assert args.entries == []

    def test_force_flag_sets_true(self) -> None:
        """Passing `--force` flips the flag."""
        args = to_sentiment.parse_args(["--all", "--force"])
        assert args.force is True

    def test_bare_invocation_errors(self) -> None:
        """A bare invocation requires entries or `--all`."""
        with pytest.raises(SystemExit):
            to_sentiment.parse_args([])
