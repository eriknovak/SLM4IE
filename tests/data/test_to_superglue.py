"""Tests for scripts/data/to_superglue.py."""

import gzip
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2] / "scripts" / "data"),
)
import to_superglue  # noqa: E402


def _write_jsonl(path: Path, records) -> None:
    """Write *records* as JSONL to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False))
            fh.write("\n")


def _read_jsonl_gz(path: Path):
    """Yield JSON-decoded records from a gzipped JSONL file."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _make_layout(
    raw_dir: Path,
    variant_subdir: str,
    tasks: dict,
) -> Path:
    """Build a fake SuperGLUE-SL extraction tree.

    Args:
        raw_dir: Top-level raw download directory.
        variant_subdir: Variant root name (e.g. ``SuperGLUE-HumanT``).
        tasks: Mapping from task name to ``{split: [records]}``.

    Returns:
        Path: The variant root that was created.
    """
    variant_root = raw_dir / variant_subdir
    for task_name, splits in tasks.items():
        for split, records in splits.items():
            _write_jsonl(
                variant_root / task_name / f"{split}.jsonl",
                records,
            )
    return variant_root


class TestVariantDiscovery:
    """Tests for the variant-root discovery helper."""

    def test_finds_humant(self, tmp_path: Path):
        _make_layout(
            tmp_path,
            "SuperGLUE-HumanT",
            {"BoolQ": {"train": [{"idx": 0, "label": True}]}},
        )
        root = to_superglue._find_variant_root(tmp_path, "humant")
        assert root.name == "SuperGLUE-HumanT"

    def test_finds_googlemt(self, tmp_path: Path):
        _make_layout(
            tmp_path,
            "SuperGLUE-GoogleMT",
            {"BoolQ": {"train": [{"idx": 0, "label": True}]}},
        )
        root = to_superglue._find_variant_root(tmp_path, "googlemt")
        assert root.name == "SuperGLUE-GoogleMT"

    def test_falls_back_to_raw_dir_when_flat(self, tmp_path: Path):
        _write_jsonl(
            tmp_path / "BoolQ" / "train.jsonl",
            [{"idx": 0, "label": True}],
        )
        root = to_superglue._find_variant_root(tmp_path, "humant")
        assert root == tmp_path

    def test_missing_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="SuperGLUE"):
            to_superglue._find_variant_root(tmp_path, "humant")


class TestPassthrough:
    """Pass-through tasks should preserve native shape verbatim."""

    def test_cb_passthrough(self, tmp_path: Path):
        _make_layout(
            tmp_path,
            "SuperGLUE-HumanT",
            {
                "CB": {
                    "train": [
                        {
                            "idx": 0,
                            "premise": "Misli, da je doma.",
                            "hypothesis": "Je doma.",
                            "label": "entailment",
                        },
                    ],
                },
            },
        )
        out_dir = tmp_path / "out"
        written = to_superglue.convert_dataset(
            tmp_path, out_dir, variant="humant",
            tasks=["CB"], splits=["train"],
        )
        assert written == {("CB", "train"): 1}
        records = list(_read_jsonl_gz(out_dir / "CB" / "train.jsonl.gz"))
        assert records[0]["premise"] == "Misli, da je doma."
        assert records[0]["hypothesis"] == "Je doma."
        assert records[0]["label"] == "entailment"


class TestMultircFlatten:
    """MultiRC should flatten one row per answer by default."""

    def test_default_flattens(self, tmp_path: Path):
        _make_layout(
            tmp_path,
            "SuperGLUE-HumanT",
            {
                "MultiRC": {
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
            },
        )
        out_dir = tmp_path / "out"
        written = to_superglue.convert_dataset(
            tmp_path, out_dir, variant="humant",
            tasks=["MultiRC"], splits=["val"],
        )
        assert written == {("MultiRC", "val"): 2}
        records = list(_read_jsonl_gz(out_dir / "MultiRC" / "val.jsonl.gz"))
        assert {r["answer"] for r in records} == {"v trgovino", "domov"}
        assert {r["label"] for r in records} == {0, 1}
        for r in records:
            assert r["paragraph"] == "Janez gre v trgovino."
            assert r["question"] == "Kam gre Janez?"

    def test_no_flatten_passes_through(self, tmp_path: Path):
        nested = {
            "idx": 0,
            "passage": {
                "text": "X.",
                "questions": [
                    {
                        "idx": 0,
                        "question": "Q",
                        "answers": [{"idx": 0, "text": "A", "label": 1}],
                    },
                ],
            },
        }
        _make_layout(
            tmp_path,
            "SuperGLUE-HumanT",
            {"MultiRC": {"val": [nested]}},
        )
        out_dir = tmp_path / "out"
        to_superglue.convert_dataset(
            tmp_path, out_dir, variant="humant",
            tasks=["MultiRC"], splits=["val"],
            flatten_multirc=False,
        )
        records = list(_read_jsonl_gz(out_dir / "MultiRC" / "val.jsonl.gz"))
        assert records[0] == nested


class TestSplitFiltering:
    """--tasks and --splits should restrict what gets emitted."""

    def test_only_selected_splits_written(self, tmp_path: Path):
        _make_layout(
            tmp_path,
            "SuperGLUE-HumanT",
            {
                "RTE": {
                    "train": [{"idx": 0, "premise": "p", "hypothesis": "h", "label": 0}],
                    "val": [{"idx": 1, "premise": "p", "hypothesis": "h", "label": 1}],
                    "test": [{"idx": 2, "premise": "p", "hypothesis": "h"}],
                },
            },
        )
        out_dir = tmp_path / "out"
        written = to_superglue.convert_dataset(
            tmp_path, out_dir, variant="humant",
            tasks=["RTE"], splits=["val"],
        )
        assert written == {("RTE", "val"): 1}
        assert (out_dir / "RTE" / "val.jsonl.gz").exists()
        assert not (out_dir / "RTE" / "train.jsonl.gz").exists()
        assert not (out_dir / "RTE" / "test.jsonl.gz").exists()


class TestForceOverwrite:
    """Existing outputs are skipped unless --force is passed."""

    def test_skip_existing(self, tmp_path: Path):
        _make_layout(
            tmp_path,
            "SuperGLUE-HumanT",
            {"CB": {"train": [{"idx": 0, "premise": "a", "hypothesis": "b", "label": 0}]}},
        )
        out_dir = tmp_path / "out"
        out_path = out_dir / "CB" / "train.jsonl.gz"
        out_path.parent.mkdir(parents=True)
        with gzip.open(out_path, "wt", encoding="utf-8") as fh:
            fh.write("placeholder\n")

        written = to_superglue.convert_dataset(
            tmp_path, out_dir, variant="humant",
            tasks=["CB"], splits=["train"],
        )
        assert written == {("CB", "train"): 0}

    def test_force_overwrites(self, tmp_path: Path):
        _make_layout(
            tmp_path,
            "SuperGLUE-HumanT",
            {"CB": {"train": [{"idx": 0, "premise": "a", "hypothesis": "b", "label": 0}]}},
        )
        out_dir = tmp_path / "out"
        out_path = out_dir / "CB" / "train.jsonl.gz"
        out_path.parent.mkdir(parents=True)
        with gzip.open(out_path, "wt", encoding="utf-8") as fh:
            fh.write("placeholder\n")

        written = to_superglue.convert_dataset(
            tmp_path, out_dir, variant="humant",
            tasks=["CB"], splits=["train"], force=True,
        )
        assert written == {("CB", "train"): 1}
        records = list(_read_jsonl_gz(out_path))
        assert records[0]["premise"] == "a"
