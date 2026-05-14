"""Tests for scripts/data/to_superglue.py (tasks.yaml-driven converter)."""

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
import to_superglue  # noqa: E402

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


def _write_tasks_yaml(
    tmp_path: Path,
    bundle_key: str,
    task: str,
    dataset: str,
    labels: List[Any],
) -> Path:
    """Build a single-entry tasks.yaml pointing at a raw SuperGLUE bundle.

    Args:
        tmp_path: pytest tmp_path root.
        bundle_key: Subdirectory name under ``raw/<bundle_key>/``
            holding the SuperGLUE distribution.
        task: Task family (e.g. ``nli``).
        dataset: Dataset name (e.g. ``cb``).
        labels: Label allow-list declared on the entry.

    Returns:
        Path to the written tasks.yaml.
    """
    extracted = tmp_path / "extracted"
    raw = tmp_path / "raw"
    tasks_root = tmp_path / "tasks"
    extracted.mkdir(parents=True, exist_ok=True)
    (raw / bundle_key).mkdir(parents=True, exist_ok=True)
    tasks_root.mkdir(parents=True, exist_ok=True)

    tasks_yaml = {
        "roots": {
            "extracted": str(extracted),
            "raw": str(raw),
            "tasks": str(tasks_root),
        },
        "converters": {
            "nli": "to_superglue",
            "qa": "to_superglue",
            "coref": "to_superglue",
            "wsd": "to_superglue",
            "commonsense": "to_superglue",
        },
        "entries": {
            f"{task}/{dataset}": {
                "role": "held_out",
                "source": {"kind": "raw", "keys": [bundle_key]},
                "splits": {
                    "train": "train.jsonl.gz",
                    "val": "val.jsonl.gz",
                    "test": "test.jsonl.gz",
                },
                "labels": labels,
                "suite": "superglue_sl",
                "language": "sl",
                "license": "cc-by-4.0",
            },
        },
    }
    config_path = tmp_path / "tasks.yaml"
    config_path.write_text(yaml.safe_dump(tasks_yaml))
    return config_path


def _make_subtask_layout(
    raw_dir: Path,
    bundle_key: str,
    subtask_dir: str,
    splits: Dict[str, List[Dict[str, Any]]],
) -> None:
    """Build a fake SuperGLUE-SL subtask tree.

    Args:
        raw_dir: Root of the raw bundle directory.
        bundle_key: Subdirectory under ``raw/`` holding the bundle.
        subtask_dir: Canonical SuperGLUE subtask name (e.g. ``CB``).
        splits: Mapping ``{split: [records]}``. Each split is written
            as ``<bundle>/SuperGLUE-HumanT/<Subtask>/<split>.jsonl``.
    """
    variant_root = raw_dir / bundle_key / "SuperGLUE-HumanT"
    for split, records in splits.items():
        _write_jsonl(
            variant_root / subtask_dir / f"{split}.jsonl",
            records,
        )


class TestVariantDiscovery:
    """Tests for the variant-root discovery helper."""

    def test_finds_humant(self, tmp_path: Path) -> None:
        """The HumanT variant root is located by directory name."""
        bundle = tmp_path / "bundle"
        _make_subtask_layout(
            bundle.parent,
            bundle.name,
            "BoolQ",
            {"train": [{"idx": 0, "label": True}]},
        )
        root = to_superglue._find_variant_root(bundle, "humant")
        assert root.name == "SuperGLUE-HumanT"

    def test_missing_raises(self, tmp_path: Path) -> None:
        """A missing variant root raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="SuperGLUE"):
            to_superglue._find_variant_root(tmp_path, "humant")


class TestConvertEntryNli:
    """End-to-end tests for the NLI (CB / RTE) task family."""

    def test_cb_records_match_nli_schema(self, tmp_path: Path) -> None:
        """Converted CB records satisfy NliExample (`id/premise/hypothesis/label`)."""
        _make_subtask_layout(
            tmp_path / "raw",
            "superglue_sl",
            "CB",
            {
                "train": [
                    {
                        "idx": 0,
                        "premise": "Misli, da je doma.",
                        "hypothesis": "Je doma.",
                        "label": "entailment",
                    },
                    {
                        "idx": 1,
                        "premise": "Pravi, da bo prišel.",
                        "hypothesis": "Ne bo prišel.",
                        "label": "contradiction",
                    },
                ],
            },
        )
        config_path = _write_tasks_yaml(
            tmp_path,
            "superglue_sl",
            "nli",
            "cb",
            ["entailment", "neutral", "contradiction"],
        )
        cfg = load_tasks(config_path)
        entry = next(
            e for e in cfg.entries if f"{e.task}/{e.dataset}" == "nli/cb"
        )

        counts = to_superglue.convert_entry(
            "nli/cb", entry, cfg.roots, variant="humant"
        )
        assert counts is not None
        assert counts["train"] == 2

        out_dir = tmp_path / "tasks" / "nli" / "cb"
        rows = _read_jsonl_gz(out_dir / "train.jsonl.gz")
        assert len(rows) == 2
        assert set(rows[0].keys()) == {"id", "premise", "hypothesis", "label"}
        assert rows[0]["label"] in {"entailment", "contradiction"}


class TestConvertEntryBoolq:
    """End-to-end test for the BoolQ converter."""

    def test_boolq_records_match_schema(self, tmp_path: Path) -> None:
        """Converted BoolQ records satisfy QaBooleanExample."""
        _make_subtask_layout(
            tmp_path / "raw",
            "superglue_sl",
            "BoolQ",
            {
                "val": [
                    {
                        "idx": 5,
                        "passage": "Janez gre v trgovino vsak teden.",
                        "question": "Ali gre Janez v trgovino?",
                        "label": True,
                    },
                ],
            },
        )
        config_path = _write_tasks_yaml(
            tmp_path,
            "superglue_sl",
            "qa",
            "boolq",
            [True, False],
        )
        cfg = load_tasks(config_path)
        entry = next(
            e for e in cfg.entries if f"{e.task}/{e.dataset}" == "qa/boolq"
        )

        counts = to_superglue.convert_entry(
            "qa/boolq", entry, cfg.roots, variant="humant"
        )
        assert counts is not None
        assert counts["val"] == 1

        out_dir = tmp_path / "tasks" / "qa" / "boolq"
        rows = _read_jsonl_gz(out_dir / "val.jsonl.gz")
        assert set(rows[0].keys()) == {"id", "passage", "question", "label"}
        assert isinstance(rows[0]["label"], bool)


class TestConvertEntryMultirc:
    """MultiRC flattens nested answers into one row per (passage, q, a)."""

    def test_multirc_flatten(self, tmp_path: Path) -> None:
        """Each answer becomes its own QaBooleanExample row."""
        _make_subtask_layout(
            tmp_path / "raw",
            "superglue_sl",
            "MultiRC",
            {
                "val": [
                    {
                        "idx": 0,
                        "passage": {
                            "text": "Janez gre v trgovino.",
                            "questions": [
                                {
                                    "idx": 0,
                                    "question": "Kam gre Janez?",
                                    "answers": [
                                        {"idx": 0, "text": "v trgovino", "label": 1},
                                        {"idx": 1, "text": "domov", "label": 0},
                                    ],
                                },
                            ],
                        },
                    },
                ],
            },
        )
        config_path = _write_tasks_yaml(
            tmp_path,
            "superglue_sl",
            "qa",
            "multirc",
            [True, False],
        )
        cfg = load_tasks(config_path)
        entry = next(
            e for e in cfg.entries if f"{e.task}/{e.dataset}" == "qa/multirc"
        )

        counts = to_superglue.convert_entry(
            "qa/multirc", entry, cfg.roots, variant="humant"
        )
        assert counts is not None
        assert counts["val"] == 2

        out_dir = tmp_path / "tasks" / "qa" / "multirc"
        rows = _read_jsonl_gz(out_dir / "val.jsonl.gz")
        assert {r["label"] for r in rows} == {True, False}
        for record in rows:
            assert set(record.keys()) == {"id", "passage", "question", "label"}
            assert record["passage"] == "Janez gre v trgovino."


class TestSkipExistingOutputs:
    """`convert_entry` short-circuits when every split file exists."""

    def test_skip_returns_none(self, tmp_path: Path) -> None:
        """Pre-existing outputs cause convert_entry to skip and return None."""
        _make_subtask_layout(
            tmp_path / "raw",
            "superglue_sl",
            "CB",
            {
                "train": [
                    {
                        "idx": 0,
                        "premise": "a",
                        "hypothesis": "b",
                        "label": "entailment",
                    }
                ],
            },
        )
        config_path = _write_tasks_yaml(
            tmp_path,
            "superglue_sl",
            "nli",
            "cb",
            ["entailment", "neutral", "contradiction"],
        )
        cfg = load_tasks(config_path)
        entry = next(
            e for e in cfg.entries if f"{e.task}/{e.dataset}" == "nli/cb"
        )

        out_dir = tmp_path / "tasks" / "nli" / "cb"
        out_dir.mkdir(parents=True)
        for split_filename in entry.splits.values():
            (out_dir / split_filename).write_bytes(b"\x1f\x8b")

        result = to_superglue.convert_entry(
            "nli/cb", entry, cfg.roots, variant="humant"
        )
        assert result is None


class TestParseArgs:
    """Tests for `to_superglue.parse_args`."""

    def test_accepts_entry_keys(self) -> None:
        """Positional entries are gathered into `args.entries`."""
        args = to_superglue.parse_args(["nli/cb", "qa/boolq"])
        assert args.entries == ["nli/cb", "qa/boolq"]
        assert args.all is False

    def test_all_flag(self) -> None:
        """`--all` parses without positional entries."""
        args = to_superglue.parse_args(["--all"])
        assert args.all is True
        assert args.entries == []

    def test_variant_default_humant(self) -> None:
        """The default variant is humant."""
        args = to_superglue.parse_args(["--all"])
        assert args.variant == "humant"

    def test_variant_googlemt(self) -> None:
        """`--variant googlemt` is accepted."""
        args = to_superglue.parse_args(["--all", "--variant", "googlemt"])
        assert args.variant == "googlemt"

    def test_invalid_variant_rejected(self) -> None:
        """Unknown variants are rejected by argparse."""
        with pytest.raises(SystemExit):
            to_superglue.parse_args(["--all", "--variant", "bogus"])

    def test_force_flag_sets_true(self) -> None:
        """Passing `--force` flips the flag."""
        args = to_superglue.parse_args(["--all", "--force"])
        assert args.force is True

    def test_bare_invocation_errors(self) -> None:
        """A bare invocation requires entries or `--all`."""
        with pytest.raises(SystemExit):
            to_superglue.parse_args([])
