"""Tests for the atomic split writer in `slm4ie.data.tasks.splits`."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from slm4ie.data.tasks.splits import all_outputs_exist, iter_jsonl, write_jsonl_splits


def _failing_examples() -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield two examples, then raise."""
    yield "train", {"id": 1}
    yield "val", {"id": 2}
    raise RuntimeError("boom")


def test_writes_final_files_and_returns_counts(tmp_path: Path) -> None:
    """A finished stream lands in the final files and returns per-split counts."""
    outputs = {"train": tmp_path / "train.jsonl.gz", "val": tmp_path / "val.jsonl.gz"}

    counts = write_jsonl_splits([("train", {"id": 1}), ("train", {"id": 2}), ("val", {"id": 3})], outputs)

    assert counts == {"train": 2, "val": 1}
    assert list(iter_jsonl(outputs["train"])) == [{"id": 1}, {"id": 2}]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["train.jsonl.gz", "val.jsonl.gz"]


def test_failure_leaves_no_new_files_and_propagates(tmp_path: Path) -> None:
    """A mid-stream failure leaves no file behind and re-raises the error."""
    outputs = {"train": tmp_path / "train.jsonl.gz", "val": tmp_path / "val.jsonl.gz"}

    with pytest.raises(RuntimeError, match="boom"):
        write_jsonl_splits(_failing_examples(), outputs)

    assert list(tmp_path.iterdir()) == []


def test_failure_keeps_earlier_final_file_byte_identical(tmp_path: Path) -> None:
    """A mid-stream failure leaves an earlier final file byte-identical."""
    train = tmp_path / "train.jsonl.gz"
    outputs = {"train": train, "val": tmp_path / "val.jsonl.gz"}
    write_jsonl_splits([("train", {"id": 9})], {"train": train})
    before = train.read_bytes()

    with pytest.raises(RuntimeError):
        write_jsonl_splits(_failing_examples(), outputs)

    assert train.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["train.jsonl.gz"]


def test_stale_partial_is_ignored_and_overwritten(tmp_path: Path) -> None:
    """A partial file left by a killed run is not an output and is overwritten."""
    outputs = {"train": tmp_path / "train.jsonl.gz"}
    (tmp_path / "train.jsonl.gz.partial").write_bytes(b"garbage")

    assert not all_outputs_exist(outputs)
    write_jsonl_splits([("train", {"id": 1})], outputs)

    assert list(iter_jsonl(outputs["train"])) == [{"id": 1}]
    assert [p.name for p in tmp_path.iterdir()] == ["train.jsonl.gz"]
