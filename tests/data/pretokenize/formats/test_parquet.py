"""Tests for slm4ie/data/pretokenize/formats/parquet.py (the Parquet layout)."""

from pathlib import Path

from slm4ie.data.pretokenize.formats import Sample, get_format

_SAMPLES = [Sample((2, 10, 3), "c4:a#0"), Sample((2, 11, 12, 3), "c4:a#1"), Sample((2, 13, 3), "kas:b#0")]


def _write(split_dir: Path, vocab_size: int) -> None:
    """Write the fixture samples as two parts of a split.

    Args:
        split_dir: The split folder.
        vocab_size: Vocabulary size passed to the writer.
    """
    output_format = get_format("parquet")
    for part, samples in (("00000", _SAMPLES[:2]), ("00001", _SAMPLES[2:])):
        writer = output_format.open_part(output_format.part_path(split_dir, part), vocab_size, 1 << 20)
        for sample in samples:
            writer.write(sample)
        writer.close()


def test_loads_with_huggingface_datasets_in_order(tmp_path: Path):
    """`datasets.load_dataset("parquet", ...)` over the sorted part files yields the samples in order."""
    from datasets import load_dataset

    _write(tmp_path, vocab_size=32_000)
    files = sorted(str(path) for path in tmp_path.glob("*.parquet"))
    rows = load_dataset("parquet", data_files=files, split="train", cache_dir=str(tmp_path / "cache"))
    assert [(tuple(row["input_ids"]), row["len"], row["id"]) for row in rows] == [
        (sample.input_ids, sample.len, sample.id) for sample in _SAMPLES
    ]


def test_ids_use_uint16_for_small_vocabularies(tmp_path: Path):
    """A 32,000-token vocabulary stores ids as a list of uint16."""
    import pyarrow.parquet as pq

    _write(tmp_path, vocab_size=32_000)
    schema = pq.read_schema(str(tmp_path / "00000.parquet"))
    assert str(schema.field("input_ids").type) == "list<element: uint16>"
