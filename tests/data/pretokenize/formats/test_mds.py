"""Tests for slm4ie/data/pretokenize/formats/mds.py (the two MDS layouts)."""

import json
from pathlib import Path

import numpy as np

from slm4ie.data.pretokenize.formats import Sample, get_format


def _write_one_part(name: str, split_dir: Path, vocab_size: int) -> None:
    """Write one sample in one part of a split and merge it.

    Args:
        name: Registry name of the MDS format.
        split_dir: The split folder.
        vocab_size: Vocabulary size passed to the writer.
    """
    output_format = get_format(name)
    writer = output_format.open_part(output_format.part_path(split_dir, "00000"), vocab_size, 1 << 20)
    writer.write(Sample((2, 40, 3), "c4:a#0"))
    writer.close()
    output_format.merge_parts(split_dir, ["00000"])


def _columns(split_dir: Path) -> dict[str, str]:
    """Return the column encodings the merged index declares.

    Args:
        split_dir: The split folder.

    Returns:
        Column name to MDS encoding.
    """
    shard = json.loads((split_dir / "index.json").read_text())["shards"][0]
    return dict(zip(shard["column_names"], shard["column_encodings"], strict=True))


def test_non_streaming_layout_stores_uint16_arrays_uncompressed(tmp_path: Path):
    """The mds layout declares input_ids as a uint16 array and writes no compression."""
    _write_one_part("mds", tmp_path, vocab_size=32_000)
    shard = json.loads((tmp_path / "index.json").read_text())["shards"][0]
    assert _columns(tmp_path)["input_ids"] == "ndarray:uint16"
    assert shard["compression"] is None


def test_non_streaming_layout_reads_as_a_numpy_array(tmp_path: Path):
    """ModernBERT's non-streaming loader gets input_ids as a numpy array, which it keeps."""
    from streaming import LocalDataset

    _write_one_part("mds", tmp_path, vocab_size=32_000)
    row = LocalDataset(local=str(tmp_path))[0]
    assert isinstance(row["input_ids"], np.ndarray) and row["input_ids"].tolist() == [2, 40, 3]


def test_streaming_layout_stores_int64_bytes_and_len(tmp_path: Path):
    """The mds-streaming layout stores input_ids as int64 bytes, as ModernBERT's streaming loader reads them."""
    from streaming import LocalDataset

    _write_one_part("mds-streaming", tmp_path, vocab_size=32_000)
    row = LocalDataset(local=str(tmp_path))[0]
    assert _columns(tmp_path)["input_ids"] == "bytes"
    assert np.frombuffer(row["input_ids"], dtype=np.int64).tolist() == [2, 40, 3] and row["len"] == 3
