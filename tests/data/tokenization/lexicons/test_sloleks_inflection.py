"""Tests for slm4ie/data/tokenization/lexicons/sloleks_inflection.py and its reader."""

from pathlib import Path

import pytest

from slm4ie.data.tokenization import lexicons
from slm4ie.data.tokenization.lexicons.sloleks_inflection import (
    find_inflection_tsv,
    iter_inflection_segmentations,
    split_form,
)

_HEADER = (
    "lexeme_id\tlexeme_lemma\torth_form_enc_id\torth_form_enc\torth_form_enc_msd\torth_form_patt_code"
    "\tlexeme_resource_sloleks_status\tending\tinitial_part\tending_part\tsplit_encoding"
)

#: Two lexemes: `mačka` (manual) with a form repeated under two tags and a
#: zero-ending form, then `pes` (automatic) with one row whose split does not
#: reproduce its form.
_ROWS = [
    "1\tmačka\t10\tmačka\tSozei\tS1\tmanual\ta\tmačk\ta\tmačk__a",
    "1\tmačka\t11\tmačkah\tSozdm\tS1\tmanual\tah\tmačk\tah\tmačk__ah",
    "1\tmačka\t11\tmačkah\tSozmm\tS1\tmanual\tah\tmačk\tah\tmačk__ah",
    "1\tmačka\t12\tmačk\tSozmr\tS1\tmanual\tØ\tmačk\t\tmačk",
    "2\tpes\t20\tpes\tSomei\tS2\tautomatic\tes\tp\tes\tp__es",
    "2\tpes\t21\tpsa\tSomer\tS2\tautomatic\ta\tps\ta\tpes__a",
]


def _write_tsv(path: Path, rows=None) -> Path:
    """Write the header and fixture rows to a TSV file.

    Args:
        path (Path): Destination `.tsv` path.
        rows (Optional[List[str]]): Data rows; the default fixture when None.

    Returns:
        Path: `path`, for chaining.
    """
    path.write_text("\n".join([_HEADER, *(rows or _ROWS)]) + "\n", encoding="utf-8")
    return path


def test_split_form_returns_stem_and_ending():
    """A split value yields the stem and the ending in order."""
    assert split_form("prostozidar__jema") == ["prostozidar", "jema"]


def test_split_form_keeps_zero_ending_whole():
    """A value without the separator is one piece."""
    assert split_form("mačk") == ["mačk"]


def test_one_record_per_lexeme_with_status(tmp_path: Path):
    """Rows group into one record per lexeme carrying its checking status."""
    records = list(iter_inflection_segmentations(_write_tsv(tmp_path / "x_inflect_morph_seg.tsv")))
    assert [(r["lemma"], r["status"]) for r in records] == [("mačka", "manual"), ("pes", "automatic")]


def test_form_repeated_under_two_tags_kept_once(tmp_path: Path):
    """A form listed once per tag becomes one form entry with both tags."""
    first = next(iter_inflection_segmentations(_write_tsv(tmp_path / "x_inflect_morph_seg.tsv")))
    forms = {f["form"]: f for f in first["forms"]}
    assert len(first["forms"]) == 3
    assert forms["mačkah"]["msds"] == ["Sozdm", "Sozmm"]


def test_row_whose_split_mismatches_form_is_skipped(tmp_path: Path):
    """A split that does not spell its form is dropped."""
    records = list(iter_inflection_segmentations(_write_tsv(tmp_path / "x_inflect_morph_seg.tsv")))
    assert [f["form"] for f in records[1]["forms"]] == ["pes"]


def test_scattered_lexeme_rows_join_one_record(tmp_path: Path):
    """Rows of one lexeme split by another lexeme's rows still form one record."""
    rows = [_ROWS[0], _ROWS[4], _ROWS[1]]
    records = list(iter_inflection_segmentations(_write_tsv(tmp_path / "x_inflect_morph_seg.tsv", rows)))
    assert [(r["lemma"], len(r["forms"])) for r in records] == [("mačka", 2), ("pes", 1)]


def test_missing_column_raises(tmp_path: Path):
    """A TSV without the split column is not taken for this release."""
    path = tmp_path / "x_inflect_morph_seg.tsv"
    path.write_text("lexeme_id\tlexeme_lemma\n1\tmačka\n", encoding="utf-8")
    with pytest.raises(ValueError, match="split_encoding"):
        list(iter_inflection_segmentations(path))


def test_reader_finds_tsv_in_subfolder_and_tags_records(tmp_path: Path):
    """The registered reader finds the TSV below the raw dir and tags each record."""
    (tmp_path / "Sloleks 3.2").mkdir()
    _write_tsv(tmp_path / "Sloleks 3.2" / "DSBS__inflect_morph_seg__2026-07.tsv")
    assert find_inflection_tsv(tmp_path) is not None
    records = list(lexicons.LEXICONS["sloleks_inflection"](tmp_path))
    assert {(r["dataset"], r["task"]) for r in records} == {("sloleks_inflection", "TOKENIZER")}
