"""Tests for slm4ie/data/tokenization/lexicons/morphscore.py and its reader."""

from pathlib import Path

import pytest

from slm4ie.data.tokenization import lexicons
from slm4ie.data.tokenization.lexicons.morphscore import iter_morphscore_items

_HEADER = (
    ",language,dataset_source,data_split,wordform,unique,lemma,stem,preceding_part,following_part,pos,"
    "morph_comp,sentence,sentence_id,word_freq,word_freq_norm"
)

#: `večera` with an ending only, then `nevečer` with a part before the stem.
_ROWS = [
    '0,slv_latn,UD_Slovenian-SSJ,train,večera,unique,večer,večer,,a,NOUN,Case=Gen,"Tistega večera.",s1,13,0.1',
    '1,slv_latn,UD_Slovenian-SSJ,dev,nevečer,repeat,večer,večer,ne,,NOUN,Case=Nom,"Nevečer.",s2,1,0.01',
]


def _write_csv(path: Path, header: str = _HEADER) -> Path:
    """Write the header and fixture rows to `path`.

    Args:
        path (Path): Destination `slovenian_data.csv` path.
        header (str): The header line.

    Returns:
        Path: `path`, for chaining.
    """
    path.write_text("\n".join([header, *_ROWS]) + "\n", encoding="utf-8")
    return path


def test_item_cut_into_nonempty_pieces(tmp_path: Path):
    """The stem and the part after it become the pieces; the empty part is dropped."""
    first = next(iter_morphscore_items(_write_csv(tmp_path / "slovenian_data.csv")))
    assert first["morphemes"] == ["večer", "a"]
    assert first["stem_index"] == 0


def test_part_before_stem_moves_stem_index(tmp_path: Path):
    """A part before the stem puts the stem second."""
    second = list(iter_morphscore_items(_write_csv(tmp_path / "slovenian_data.csv")))[1]
    assert second["morphemes"] == ["ne", "večer"]
    assert second["stem_index"] == 1
    assert second["is_unique"] is False


def test_missing_column_refused(tmp_path: Path):
    """A file without the stem column is not a MorphScore file."""
    path = _write_csv(tmp_path / "slovenian_data.csv", header=_HEADER.replace(",stem,", ",root,"))
    with pytest.raises(ValueError, match="stem"):
        list(iter_morphscore_items(path))


def test_registry_tags_dataset(tmp_path: Path):
    """The registered reader tags each record with its dataset key."""
    _write_csv(tmp_path / "slovenian_data.csv")
    records = list(lexicons.LEXICONS["morphscore_slovenian"](tmp_path))
    assert {record["dataset"] for record in records} == {"morphscore_slovenian"}
