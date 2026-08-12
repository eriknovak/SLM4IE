"""Tests for the shared task-converter driver (slm4ie/data/task_converter.py).

Covers the driver primitives (id synthesis, role gate, hash re-bucketing, arg
parsing) and the role-gating acceptance criterion: a `held_out` entry writes the
same records as a `finetune_and_eval` entry but with the `train` split dropped.
"""

import gzip
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml

from slm4ie.data.task_converter import (
    SplitPolicy,
    assign_hash_split,
    get_converter,
    parse_args,
    run_converter,
    synthesize_id,
    target_splits,
)
from slm4ie.data.tasks import load_tasks


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


def _write_gz_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
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


def _count_rows(path: Path) -> int:
    """Count JSONL rows in a gzipped file.

    Args:
        path: Gzipped JSONL path.

    Returns:
        Number of non-empty lines.
    """
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def _make_ner_layout(tmp_path: Path, dataset_key: str, role: str, records: List[Dict[str, Any]]) -> Path:
    """Build a synthetic tasks.yaml + extracted source tree for one NER entry.

    Args:
        tmp_path: pytest tmp_path root.
        dataset_key: Source key written under ``extracted/<key>.jsonl``.
        role: Entry role (``finetune_and_eval`` or ``held_out``).
        records: Joined extraction records carrying ``uid`` + ``annotations.spans``.

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
    _write_gz_jsonl(extracted / f"{dataset_key}.annotations.jsonl.gz", ann_records)

    tasks_yaml = {
        "roots": {"extracted": str(extracted), "raw": str(raw), "tasks": str(tasks_root)},
        "converters": {"ner": "to_spans"},
        "entries": {
            f"ner/{dataset_key}": {
                "role": role,
                "source": {"kind": "extracted", "keys": [dataset_key]},
                "splits": {
                    "train": "train.jsonl.gz",
                    "val": "val.jsonl.gz",
                    "test": "test.jsonl.gz",
                },
                "labels": ["PER", "LOC"],
                "suite": None,
                "language": "sl",
                "license": "cc-by-sa-4.0",
            },
        },
    }
    config_path = tmp_path / f"tasks-{role}.yaml"
    config_path.write_text(yaml.safe_dump(tasks_yaml))
    return config_path


def _ner_records(n: int) -> List[Dict[str, Any]]:
    """Build *n* synthetic NER records with distinct uids.

    Args:
        n: Number of records.

    Returns:
        Records carrying text, source, uid, and one PER span each.
    """
    return [
        {
            "text": f"Ime {i} zivi v Parizu.",
            "source": "kzb",
            "doc_id": f"s{i}",
            "uid": f"kzb:s{i}",
            "annotations": {"forms": ["Ime", str(i)], "spans": [[0, 3, "PER"]]},
        }
        for i in range(n)
    ]


class TestSynthesizeId:
    """Unit tests for `synthesize_id`."""

    def test_prefers_uid(self) -> None:
        """A non-empty ``uid`` wins over everything else."""
        assert synthesize_id({"uid": "kzb:s1", "source": "kzb", "doc_id": "s1"}, "kzb", 0) == "kzb:s1"

    def test_superglue_int_idx(self) -> None:
        """An ``idx`` int is prefixed with the dataset name."""
        assert synthesize_id({"idx": 5}, "cb", 0) == "cb:5"

    def test_superglue_id_with_colon_kept(self) -> None:
        """An ``id`` already containing a colon is kept verbatim."""
        assert synthesize_id({"id": "cb:train:7"}, "cb", 0) == "cb:train:7"

    def test_superglue_dict_id(self) -> None:
        """A dict ``idx`` is joined with '-' under the dataset prefix."""
        assert synthesize_id({"idx": {"a": 1, "b": 2}}, "cb", 0) == "cb:1-2"

    def test_source_doc_id(self) -> None:
        """Without uid/id/idx, ``<source>:<doc_id>`` is used."""
        assert synthesize_id({"source": "kzb", "doc_id": "s2"}, "kzb", 0) == "kzb:s2"

    def test_index_fallback(self) -> None:
        """The last resort is ``<source>:idx-<8-digit-index>``."""
        assert synthesize_id({"source": "kzb"}, "kzb", 7) == "kzb:idx-00000007"

    def test_index_fallback_uses_dataset_without_source(self) -> None:
        """The prefix defaults to the dataset when no ``source`` is present."""
        assert synthesize_id({}, "cb", 3) == "cb:idx-00000003"


class TestTargetSplits:
    """Unit tests for the role gate `target_splits`."""

    def test_finetune_keeps_all(self, tmp_path: Path) -> None:
        """A finetune_and_eval entry keeps every declared split."""
        cfg = load_tasks(_make_ner_layout(tmp_path, "kzb", "finetune_and_eval", _ner_records(3)))
        entry = cfg.entries[0]
        assert set(target_splits(entry)) == {"train", "val", "test"}

    def test_held_out_drops_train(self, tmp_path: Path) -> None:
        """A held_out entry drops the train split, keeping the rest."""
        cfg = load_tasks(_make_ner_layout(tmp_path, "kzb", "held_out", _ner_records(3)))
        entry = cfg.entries[0]
        assert "train" not in target_splits(entry)
        assert set(target_splits(entry)) == {"val", "test"}


class TestAssignHashSplit:
    """Unit tests for `assign_hash_split`."""

    def test_train_present_uses_standard_buckets(self) -> None:
        """With train present, every assignment is one of the three splits."""
        targets = ["train", "val", "test"]
        assigned = {assign_hash_split(f"k{i}", targets) for i in range(200)}
        assert assigned <= set(targets)
        assert "train" in assigned

    def test_train_absent_rebuckets_across_val_test(self) -> None:
        """With train dropped, the population spreads across val and test only."""
        targets = ["val", "test"]
        assigned = [assign_hash_split(f"k{i}", targets) for i in range(400)]
        assert set(assigned) == {"val", "test"}
        # ~50/50 split — both sides get a meaningful share, none collapses.
        val_share = assigned.count("val") / len(assigned)
        assert 0.35 < val_share < 0.65


class TestParseArgs:
    """Tests for the shared `parse_args`."""

    def test_spans_accepts_entries(self) -> None:
        """Positional entries are gathered for a HASH converter."""
        args = parse_args(get_converter("to_spans"), ["ner/ssj500k", "ner/suk"])
        assert args.entries == ["ner/ssj500k", "ner/suk"]
        assert args.all is False

    def test_all_flag(self) -> None:
        """``--all`` parses without positional entries."""
        args = parse_args(get_converter("to_sentiment"), ["--all"])
        assert args.all is True

    def test_bare_invocation_errors(self) -> None:
        """A bare invocation requires entries or ``--all``."""
        with pytest.raises(SystemExit):
            parse_args(get_converter("to_spans"), [])

    def test_superglue_adds_variant(self) -> None:
        """The SuperGLUE converter contributes ``--variant``."""
        args = parse_args(get_converter("to_superglue"), ["--all", "--variant", "googlemt"])
        assert args.variant == "googlemt"

    def test_spans_has_no_variant(self) -> None:
        """A converter without extra args rejects ``--variant``."""
        with pytest.raises(SystemExit):
            parse_args(get_converter("to_spans"), ["--all", "--variant", "humant"])


class TestRunConverterEndToEnd:
    """Drive the entry point `run_converter` end to end."""

    def test_writes_declared_splits(self, tmp_path: Path) -> None:
        """A finetune entry writes every declared split via run_converter."""
        cfg_path = _make_ner_layout(tmp_path, "kzb", "finetune_and_eval", _ner_records(30))
        run_converter("to_spans", ["ner/kzb", "--config", str(cfg_path), "--max-workers", "1"])

        out_dir = tmp_path / "tasks" / "ner" / "kzb"
        assert (out_dir / "train.jsonl.gz").exists()
        assert (out_dir / "val.jsonl.gz").exists()
        assert (out_dir / "test.jsonl.gz").exists()
        total = sum(_count_rows(out_dir / f"{s}.jsonl.gz") for s in ("train", "val", "test"))
        assert total == 30
        # Schema parity with NerExample on a sample row.
        with gzip.open(out_dir / "test.jsonl.gz", "rt", encoding="utf-8") as fh:
            row = json.loads(next(line for line in fh if line.strip()))
        assert set(row.keys()) == {"id", "text", "spans"}

    def test_unknown_entry_exits(self, tmp_path: Path) -> None:
        """An unknown entry key exits non-zero."""
        cfg_path = _make_ner_layout(tmp_path, "kzb", "finetune_and_eval", _ner_records(2))
        with pytest.raises(SystemExit):
            run_converter("to_spans", ["ner/missing", "--config", str(cfg_path), "--max-workers", "1"])


class TestRoleGatingAcceptance:
    """Acceptance: role changes which splits are written, nothing is dropped."""

    def test_held_out_drops_train_but_keeps_all_records(self, tmp_path: Path) -> None:
        """held_out writes {val,test} with no train.jsonl.gz; counts match finetune."""
        records = _ner_records(40)

        ft_cfg = _make_ner_layout(tmp_path / "ft", "kzb", "finetune_and_eval", records)
        run_converter("to_spans", ["ner/kzb", "--config", str(ft_cfg), "--max-workers", "1"])
        ft_dir = tmp_path / "ft" / "tasks" / "ner" / "kzb"
        ft_total = sum(_count_rows(ft_dir / f"{s}.jsonl.gz") for s in ("train", "val", "test"))
        assert (ft_dir / "train.jsonl.gz").exists()
        assert ft_total == 40

        ho_cfg = _make_ner_layout(tmp_path / "ho", "kzb", "held_out", records)
        run_converter("to_spans", ["ner/kzb", "--config", str(ho_cfg), "--max-workers", "1"])
        ho_dir = tmp_path / "ho" / "tasks" / "ner" / "kzb"
        assert not (ho_dir / "train.jsonl.gz").exists()
        assert (ho_dir / "val.jsonl.gz").exists()
        assert (ho_dir / "test.jsonl.gz").exists()
        ho_total = sum(_count_rows(ho_dir / f"{s}.jsonl.gz") for s in ("val", "test"))
        # Nothing dropped — just re-bucketed across the remaining eval splits.
        assert ho_total == ft_total == 40


def test_converters_declare_split_policy() -> None:
    """Each converter declares its split policy in one place (the class)."""
    assert get_converter("to_spans").split_policy is SplitPolicy.HASH
    assert get_converter("to_sentiment").split_policy is SplitPolicy.HASH
    assert get_converter("to_superglue").split_policy is SplitPolicy.SOURCE
