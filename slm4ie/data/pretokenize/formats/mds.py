"""Mosaic streaming (MDS) layouts, in the two column shapes ModernBERT's loaders read.

ModernBERT reads MDS shards two ways. Its non-streaming loader, the one that
supports sequence packing, reads `input_ids` as a numpy array and drops every
other column (`mds`). Its streaming loader reads `input_ids` as the raw bytes of
an int64 array and the length from `len` (`mds-streaming`). Both are written
uncompressed, one MDS folder per part, and a split's parts are joined by
merging their `index.json` files in part order.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from slm4ie.data.pretokenize.formats import OutputFormat, PartWriter, Sample, StoredRow, register_format, token_dtype

# ===============================
# Writers
# ===============================


class _MDSPartWriter(PartWriter):
    """Writes one part as an MDS folder through `streaming.MDSWriter`."""

    def __init__(self, path: Path, columns: dict[str, str], encode, shard_size_bytes: int) -> None:
        """Open the MDS writer.

        Args:
            path: The part folder.
            columns: MDS column name to encoding.
            encode: Maps a `Sample` to the row written.
            shard_size_bytes: Target size of one MDS shard file.
        """
        from streaming import MDSWriter

        self._encode = encode
        self._writer = MDSWriter(out=str(path), columns=columns, compression=None, size_limit=shard_size_bytes)

    def write(self, sample: Sample) -> None:
        """Append one sample.

        Args:
            sample: The sample.
        """
        self._writer.write(self._encode(sample))

    def close(self) -> None:
        """Finish the MDS folder, writing its `index.json`."""
        self._writer.finish()


class _MDSFormat(OutputFormat):
    """Shared behaviour of the MDS layouts: one folder per part, merged indexes."""

    def part_path(self, split_dir: Path, part: str) -> Path:
        """Return the part's MDS folder.

        Args:
            split_dir: The split folder.
            part: The part name.

        Returns:
            `<split_dir>/<part>`.
        """
        return split_dir / part

    def merge_parts(self, split_dir: Path, parts: list[str]) -> None:
        """Merge the parts' `index.json` files into the split's, in part order.

        An earlier merged index is replaced, so a resumed run can merge again.

        Args:
            split_dir: The split folder.
            parts: Part names, in input-shard order.
        """
        from streaming.base.util import merge_index

        (split_dir / "index.json").unlink(missing_ok=True)
        merge_index([str(split_dir / part / "index.json") for part in parts], str(split_dir), keep_local=True)

    def _read_rows(self, split_dir: Path) -> Iterator[dict]:
        """Read every row of a merged split, in order.

        Args:
            split_dir: The split folder.

        Yields:
            The decoded rows; none for a split whose parts are all empty.
        """
        from streaming import LocalDataset

        # streaming cannot open an index without shards, so an empty split yields nothing
        if not json.loads((split_dir / "index.json").read_text(encoding="utf-8"))["shards"]:
            return
        dataset = LocalDataset(local=str(split_dir))
        for index in range(len(dataset)):
            yield dataset[index]


@register_format("mds")
class MDSFormat(_MDSFormat):
    """MDS for ModernBERT's non-streaming loader: `input_ids` as uint16 or uint32 arrays."""

    def open_part(self, path: Path, vocab_size: int, shard_size_bytes: int) -> PartWriter:
        """Open a part whose `input_ids` column is a typed numpy array.

        Args:
            path: The part folder.
            vocab_size: Size of the tokenizer's vocabulary, which sets the id type.
            shard_size_bytes: Target size of one MDS shard file.

        Returns:
            The writer.
        """
        dtype = token_dtype(vocab_size)
        columns = {"input_ids": f"ndarray:{np.dtype(dtype).name}", "len": "int32", "id": "str"}

        def encode(sample: Sample) -> dict:
            return {"input_ids": np.asarray(sample.input_ids, dtype=dtype), "len": sample.len, "id": sample.id}

        return _MDSPartWriter(path, columns, encode, shard_size_bytes)

    def iter_rows(self, split_dir: Path) -> Iterator[StoredRow]:
        """Read back every stored row of a merged split, in order.

        Args:
            split_dir: The split folder.

        Yields:
            The rows, with the stored uint16 or uint32 id arrays.
        """
        for row in self._read_rows(split_dir):
            yield StoredRow(input_ids=row["input_ids"], len=int(row["len"]), id=row["id"])


@register_format("mds-streaming")
class MDSStreamingFormat(_MDSFormat):
    """MDS for ModernBERT's streaming loader: `input_ids` as int64 bytes plus `len`."""

    def open_part(self, path: Path, vocab_size: int, shard_size_bytes: int) -> PartWriter:
        """Open a part whose `input_ids` column holds the bytes of an int64 array.

        Args:
            path: The part folder.
            vocab_size: Unused; the streaming loader always reads int64.
            shard_size_bytes: Target size of one MDS shard file.

        Returns:
            The writer.
        """
        columns = {"input_ids": "bytes", "len": "int", "id": "str"}

        def encode(sample: Sample) -> dict:
            ids = np.asarray(sample.input_ids, dtype=np.int64).tobytes()
            return {"input_ids": ids, "len": sample.len, "id": sample.id}

        return _MDSPartWriter(path, columns, encode, shard_size_bytes)

    def iter_rows(self, split_dir: Path) -> Iterator[StoredRow]:
        """Read back every stored row of a merged split, in order.

        Args:
            split_dir: The split folder.

        Yields:
            The rows, with the int64 ids decoded from their bytes.
        """
        for row in self._read_rows(split_dir):
            ids = np.frombuffer(row["input_ids"], dtype=np.int64)
            yield StoredRow(input_ids=ids, len=int(row["len"]), id=row["id"])
