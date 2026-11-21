"""Tests for slm4ie/data/pretokenize/formats/__init__.py (the output-format registry)."""

from pathlib import Path

import numpy as np
import pytest

from slm4ie.data.pretokenize.formats import FORMATS, Sample, get_format, token_dtype

_PARTS = {
    "00000": [Sample((2, 10, 11, 3), "c4:a#0"), Sample((2, 12, 3), "c4:a#1")],
    "00001": [],
    "00002": [Sample((2, 70_000, 3), "kas:b#0")],
}


def _write_split(name: str, split_dir: Path, vocab_size: int) -> list[Sample]:
    """Write the fixture parts of one split in a format and merge them.

    Args:
        name: Registry name of the format.
        split_dir: The split folder.
        vocab_size: Vocabulary size passed to the writers.

    Returns:
        The samples written, in part order.
    """
    output_format = get_format(name)
    for part, samples in _PARTS.items():
        writer = output_format.open_part(output_format.part_path(split_dir, part), vocab_size, 1 << 20)
        for sample in samples:
            writer.write(sample)
        writer.close()
    output_format.merge_parts(split_dir, list(_PARTS))
    return [sample for samples in _PARTS.values() for sample in samples]


def test_builtin_formats_are_registered():
    """The mds, mds-streaming and parquet formats are registered by name."""
    assert {"mds", "mds-streaming", "parquet"} <= set(FORMATS)


def test_unknown_format_names_the_registered_ones():
    """Asking for an unknown format fails and lists the known names."""
    with pytest.raises(KeyError, match="parquet"):
        get_format("arrow")


@pytest.mark.parametrize("name", ["mds", "mds-streaming", "parquet"])
def test_split_reads_back_the_samples_in_part_order(name: str, tmp_path: Path):
    """Samples written over several parts, one empty, read back unchanged and in part order."""
    written = _write_split(name, tmp_path / "train", vocab_size=100_000)
    assert list(get_format(name).read_split(tmp_path / "train")) == written


@pytest.mark.parametrize("name", ["mds", "mds-streaming", "parquet"])
def test_merging_a_split_twice_gives_the_same_samples(name: str, tmp_path: Path):
    """A second merge of the same split, as after a resumed run, succeeds and changes nothing."""
    written = _write_split(name, tmp_path / "train", vocab_size=100_000)
    get_format(name).merge_parts(tmp_path / "train", list(_PARTS))
    assert list(get_format(name).read_split(tmp_path / "train")) == written


@pytest.mark.parametrize(("vocab_size", "dtype"), [(32_000, np.uint16), (65_536, np.uint16), (65_537, np.uint32)])
def test_token_dtype_holds_every_id(vocab_size: int, dtype: type):
    """Ids are stored as uint16 up to 65,536 tokens and as uint32 above."""
    assert token_dtype(vocab_size) is dtype


def test_sample_len_counts_its_ids():
    """A sample's length is the number of its ids."""
    assert Sample((2, 5, 6, 3), "x#0").len == 4
