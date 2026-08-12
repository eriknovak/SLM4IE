"""Tests for the SuperGLUE backend (slm4ie/data/task_converters/superglue.py)."""

import gzip
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml

from slm4ie.data.task_converter import run_converter
from slm4ie.data.task_converters import superglue


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


def _write_tasks_yaml(tmp_path: Path, bundle_key: str, task: str, dataset: str, labels: List[Any]) -> Path:
    """Build a single-entry tasks.yaml pointing at a raw SuperGLUE bundle.

    The entry declares only ``val``/``test`` splits, matching a held_out entry
    after the role gate.

    Args:
        tmp_path: pytest tmp_path root.
        bundle_key: Subdirectory name under ``raw/<bundle_key>/``.
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
        "roots": {"extracted": str(extracted), "raw": str(raw), "tasks": str(tasks_root)},
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
                "splits": {"val": "val.jsonl.gz", "test": "test.jsonl.gz"},
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
    """Build a fake SuperGLUE-SL subtask tree under the HumanT variant.

    Args:
        raw_dir: Root of the raw bundle directory.
        bundle_key: Subdirectory under ``raw/`` holding the bundle.
        subtask_dir: Canonical SuperGLUE subtask name (e.g. ``CB``).
        splits: Mapping ``{split: [records]}`` written as
            ``<bundle>/SuperGLUE-HumanT/<Subtask>/<split>.jsonl``.
    """
    variant_root = raw_dir / bundle_key / "SuperGLUE-HumanT"
    for split, records in splits.items():
        _write_jsonl(variant_root / subtask_dir / f"{split}.jsonl", records)


class TestStableId:
    """Unit tests for `superglue.stable_id`."""

    def test_int_idx_prefixed(self) -> None:
        """An integer ``idx`` is prefixed with the dataset name."""
        assert superglue.stable_id({"idx": 5}, "cb", 0) == "cb:5"

    def test_id_with_colon_kept(self) -> None:
        """An id already containing a colon is kept verbatim."""
        assert superglue.stable_id({"id": "cb:7"}, "cb", 0) == "cb:7"

    def test_dict_id_joined(self) -> None:
        """A dict id is joined with '-' under the dataset prefix."""
        assert superglue.stable_id({"id": {"p": 1, "q": 2}}, "cb", 0) == "cb:1-2"

    def test_index_fallback(self) -> None:
        """Without id/idx, the 8-digit index fallback is used."""
        assert superglue.stable_id({}, "cb", 3) == "cb:idx-00000003"


class TestVariantDiscovery:
    """Tests for the variant-root discovery helper."""

    def test_finds_humant(self, tmp_path: Path) -> None:
        """The HumanT variant root is located by directory name."""
        bundle = tmp_path / "bundle"
        _make_subtask_layout(bundle.parent, bundle.name, "BoolQ", {"val": [{"idx": 0, "label": True}]})
        root = superglue._find_variant_root(bundle, "humant")
        assert root.name == "SuperGLUE-HumanT"

    def test_missing_raises(self, tmp_path: Path) -> None:
        """A missing variant root raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="SuperGLUE"):
            superglue._find_variant_root(tmp_path, "humant")


class TestConvertNli:
    """End-to-end tests for the NLI (CB / RTE) family."""

    def test_cb_records_match_schema(self, tmp_path: Path) -> None:
        """Converted CB records satisfy NliExample and keep their source split."""
        _make_subtask_layout(
            tmp_path / "raw",
            "superglue_sl",
            "CB",
            {
                "val": [
                    {"idx": 0, "premise": "Misli, da je doma.", "hypothesis": "Je doma.", "label": "entailment"},
                    {
                        "idx": 1,
                        "premise": "Pravi, da bo prisel.",
                        "hypothesis": "Ne bo prisel.",
                        "label": "contradiction",
                    },
                ],
            },
        )
        config_path = _write_tasks_yaml(
            tmp_path, "superglue_sl", "nli", "cb", ["entailment", "neutral", "contradiction"]
        )
        run_converter("to_superglue", ["nli/cb", "--config", str(config_path), "--max-workers", "1"])

        out_dir = tmp_path / "tasks" / "nli" / "cb"
        rows = _read_jsonl_gz(out_dir / "val.jsonl.gz")
        assert len(rows) == 2
        assert set(rows[0].keys()) == {"id", "premise", "hypothesis", "label"}
        assert rows[0]["label"] in {"entailment", "contradiction"}


class TestConvertBoolq:
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
        config_path = _write_tasks_yaml(tmp_path, "superglue_sl", "qa", "boolq", [True, False])
        run_converter("to_superglue", ["qa/boolq", "--config", str(config_path), "--max-workers", "1"])

        out_dir = tmp_path / "tasks" / "qa" / "boolq"
        rows = _read_jsonl_gz(out_dir / "val.jsonl.gz")
        assert set(rows[0].keys()) == {"id", "passage", "question", "label"}
        assert isinstance(rows[0]["label"], bool)


class TestConvertMultirc:
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
        config_path = _write_tasks_yaml(tmp_path, "superglue_sl", "qa", "multirc", [True, False])
        run_converter("to_superglue", ["qa/multirc", "--config", str(config_path), "--max-workers", "1"])

        out_dir = tmp_path / "tasks" / "qa" / "multirc"
        rows = _read_jsonl_gz(out_dir / "val.jsonl.gz")
        assert {r["label"] for r in rows} == {True, False}
        for record in rows:
            assert set(record.keys()) == {"id", "passage", "question", "label"}
            assert record["passage"] == "Janez gre v trgovino."


class TestHeldOutSkipsTrainSource:
    """A held_out entry never reads its train source file."""

    def test_train_source_ignored(self, tmp_path: Path) -> None:
        """A train.jsonl present on disk is not read and no train output appears."""
        _make_subtask_layout(
            tmp_path / "raw",
            "superglue_sl",
            "CB",
            {
                "train": [{"idx": 9, "premise": "T", "hypothesis": "H", "label": "entailment"}],
                "val": [{"idx": 0, "premise": "a", "hypothesis": "b", "label": "entailment"}],
                "test": [{"idx": 1, "premise": "c", "hypothesis": "d", "label": "neutral"}],
            },
        )
        config_path = _write_tasks_yaml(
            tmp_path, "superglue_sl", "nli", "cb", ["entailment", "neutral", "contradiction"]
        )
        run_converter("to_superglue", ["nli/cb", "--config", str(config_path), "--max-workers", "1"])

        out_dir = tmp_path / "tasks" / "nli" / "cb"
        assert not (out_dir / "train.jsonl.gz").exists()
        assert len(_read_jsonl_gz(out_dir / "val.jsonl.gz")) == 1
        assert len(_read_jsonl_gz(out_dir / "test.jsonl.gz")) == 1
