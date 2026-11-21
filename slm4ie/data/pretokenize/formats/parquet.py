"""Parquet layout, which HuggingFace `datasets.load_dataset("parquet", ...)` reads.

Each part is one Parquet file named for the part, so the files of a split sort
in input-shard order and need no merge step. Columns: `input_ids` (a list of
uint16 or uint32), `len` (int32) and `id` (string).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from slm4ie.data.pretokenize.formats import OutputFormat, PartWriter, Sample, StoredRow, register_format, token_dtype

# ===============================
# Constants
# ===============================

# rows buffered before a Parquet row group is written
ROW_GROUP_ROWS = 8192
# rows read back at once; bounds the check's memory
READ_BATCH_ROWS = 8192
PART_SUFFIX = ".parquet"

# ===============================
# Writer
# ===============================


class _ParquetPartWriter(PartWriter):
    """Writes one part as a Parquet file, a row group per `ROW_GROUP_ROWS` samples."""

    def __init__(self, path: Path, vocab_size: int) -> None:
        """Open the Parquet file.

        Args:
            path: The part file.
            vocab_size: Size of the tokenizer's vocabulary, which sets the id type.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        self._pa = pa
        self._id_type = pa.from_numpy_dtype(token_dtype(vocab_size))
        self._schema = pa.schema([("input_ids", pa.list_(self._id_type)), ("len", pa.int32()), ("id", pa.string())])
        path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = pq.ParquetWriter(str(path), self._schema)
        self._rows: list[Sample] = []

    def write(self, sample: Sample) -> None:
        """Append one sample, writing a row group when the buffer is full.

        Args:
            sample: The sample.
        """
        self._rows.append(sample)
        if len(self._rows) >= ROW_GROUP_ROWS:
            self._flush()

    def close(self) -> None:
        """Write the buffered rows and close the file."""
        self._flush()
        self._writer.close()

    def _flush(self) -> None:
        """Write the buffered rows as one row group."""
        if not self._rows:
            return
        pa = self._pa
        table = pa.table(
            {
                "input_ids": pa.array([list(row.input_ids) for row in self._rows], type=pa.list_(self._id_type)),
                "len": pa.array([row.len for row in self._rows], type=pa.int32()),
                "id": pa.array([row.id for row in self._rows], type=pa.string()),
            },
            schema=self._schema,
        )
        self._writer.write_table(table)
        self._rows = []


# ===============================
# Format
# ===============================


@register_format("parquet")
class ParquetFormat(OutputFormat):
    """Parquet files for HuggingFace `datasets`, one per part."""

    def part_path(self, split_dir: Path, part: str) -> Path:
        """Return the part's Parquet file.

        Args:
            split_dir: The split folder.
            part: The part name.

        Returns:
            `<split_dir>/<part>.parquet`.
        """
        return split_dir / f"{part}{PART_SUFFIX}"

    def open_part(self, path: Path, vocab_size: int, shard_size_bytes: int) -> PartWriter:
        """Open a writer for one Parquet part.

        Args:
            path: The part file.
            vocab_size: Size of the tokenizer's vocabulary, which sets the id type.
            shard_size_bytes: Unused; a part is one file.

        Returns:
            The writer.
        """
        return _ParquetPartWriter(path, vocab_size)

    def merge_parts(self, split_dir: Path, parts: list[str]) -> None:
        """Do nothing: the part files already sort in input-shard order.

        Args:
            split_dir: The split folder.
            parts: Part names, in input-shard order.
        """

    def iter_rows(self, split_dir: Path) -> Iterator[StoredRow]:
        """Read back every stored row of a split, file by file in name order.

        The id lists are sliced out of each batch's flat values array through
        its offsets, so no token becomes a Python object.

        Args:
            split_dir: The split folder.

        Yields:
            The rows, with the stored uint16 or uint32 id arrays.
        """
        import pyarrow.parquet as pq

        for path in sorted(split_dir.glob(f"*{PART_SUFFIX}")):
            for batch in pq.ParquetFile(str(path)).iter_batches(batch_size=READ_BATCH_ROWS):
                column = batch.column("input_ids")
                values, offsets = column.values.to_numpy(), column.offsets.to_numpy()
                lengths, names = batch.column("len").to_numpy(), batch.column("id").to_pylist()
                for index, name in enumerate(names):
                    ids = values[offsets[index] : offsets[index + 1]]
                    yield StoredRow(input_ids=ids, len=int(lengths[index]), id=name)
