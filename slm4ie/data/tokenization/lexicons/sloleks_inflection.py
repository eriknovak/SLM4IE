"""Parser for the Sloleks stem-ending split of every inflected form.

The CJVT release `DSBS__inflect_morph_seg__2026-07.tsv`, shared by the
Sloleks authors and cited as Sloleks 3.1 (handle 11356/2080), lists one row
per inflected form and morphosyntactic tag of the Digital Dictionary Database
of Slovene. Its `split_encoding` column cuts the form into the stem shared by
the whole paradigm and the inflectional ending, joined by two underscores
(`prostozidar__jema`); a form with a zero ending carries no separator. The
`lexeme_resource_sloleks_status` column says whether the lexeme was checked by
hand (`manual`) or generated (`automatic`).

This module reads that TSV and yields one record per lexeme. Rows repeat a
form once per tag, so each (form, split) pair is kept once with its tags
collected. Unlike `slm4ie.data.tokenization.lexicons.sloleks`, no alignment
happens here: the boundary is read straight from the split column.

Used by `slm4ie/data/tokenization/run.py` to materialize the inflectional
gold for tokenizer evaluation. Like Sloleks, it never enters the pretraining
corpus.
"""

import csv
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

#: Separator between the stem and the ending in the `split_encoding` column.
SPLIT_SEPARATOR = "__"

#: Columns the parser reads; a file missing any of them is not this release.
_REQUIRED_COLUMNS = (
    "lexeme_id",
    "lexeme_lemma",
    "orth_form_enc",
    "orth_form_enc_msd",
    "lexeme_resource_sloleks_status",
    "split_encoding",
)


def split_form(split_encoding: str) -> List[str]:
    """Split a `split_encoding` value into its stem and ending.

    Args:
        split_encoding: A form such as `mačk__ah`, or `mačk` for a zero ending.

    Returns:
        List[str]: `[stem, ending]`, or `[form]` when the ending is zero.
    """
    return [piece for piece in split_encoding.split(SPLIT_SEPARATOR) if piece]


def _add_row(lexeme: Dict[str, Any], row: Dict[str, str]) -> None:
    """Add one TSV row to its lexeme's record, keeping each (form, split) once.

    Args:
        lexeme: The lexeme's record under construction, with an `_index` map
            from (form, morphemes) to the form entry.
        row: One TSV row of that lexeme.
    """
    morphemes = split_form(row["split_encoding"])
    key = (row["orth_form_enc"], tuple(morphemes))
    entry = lexeme["_index"].get(key)
    if entry is None:
        entry = {"form": row["orth_form_enc"], "morphemes": morphemes, "msds": []}
        lexeme["_index"][key] = entry
        lexeme["forms"].append(entry)
    if row["orth_form_enc_msd"] not in entry["msds"]:
        entry["msds"].append(row["orth_form_enc_msd"])


def iter_inflection_segmentations(tsv_path: Path) -> Iterator[Dict[str, Any]]:
    """Stream per-lexeme stem-ending splits from the release TSV.

    A lexeme's rows are scattered through the file, so all lexemes are
    gathered first and emitted in first-seen order. A row whose split does not
    reproduce its form is skipped and counted in a warning.

    Args:
        tsv_path: Path to the `DSBS__inflect_morph_seg` TSV file.

    Yields:
        Dict[str, Any]: One record per lexeme with `entry_id`, `lemma`,
            `status` and `forms`, where each form carries `form`, `morphemes`
            and `msds`.

    Raises:
        ValueError: If the header lacks a column the parser reads.
    """
    skipped = 0
    lexemes: Dict[str, Dict[str, Any]] = {}
    with open(tsv_path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        missing = [column for column in _REQUIRED_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{tsv_path} lacks columns {missing}; not a Sloleks inflection split file")
        for row in reader:
            if "".join(split_form(row["split_encoding"])) != row["orth_form_enc"]:
                skipped += 1
                continue
            lexeme = lexemes.get(row["lexeme_id"])
            if lexeme is None:
                lexeme = {
                    "entry_id": row["lexeme_id"],
                    "lemma": row["lexeme_lemma"],
                    "status": row["lexeme_resource_sloleks_status"],
                    "forms": [],
                    "_index": {},
                }
                lexemes[row["lexeme_id"]] = lexeme
            _add_row(lexeme, row)
    if skipped:
        logger.warning("Skipped %d rows whose split_encoding does not reproduce the form in %s", skipped, tsv_path)
    for lexeme in lexemes.values():
        del lexeme["_index"]
        yield lexeme


def find_inflection_tsv(raw_dir: Path) -> Optional[Path]:
    """Return the inflection split TSV under a raw directory.

    Args:
        raw_dir: Directory holding the release TSV, possibly in a subfolder.

    Returns:
        Optional[Path]: The lexicographically first `*inflect_morph_seg*.tsv`,
            or None when none is found.
    """
    candidates = sorted(raw_dir.rglob("*inflect_morph_seg*.tsv"))
    return candidates[0] if candidates else None
