"""Parser for the Slovene items of MorphScore v2.

MorphScore v2 (Arnett, Hudspeth & O'Connor 2025, arXiv:2507.06378) cuts word
forms from the UD treebanks into the part before the stem, the stem and the
part after it, where the stem is the substring of the form also found in its
lemma. Its Slovene file `slovenian_data.csv` comes from UD_Slovenian-SSJ, one
row per token occurrence with the `unique` flag marking the first occurrence
of each item and `word_freq` its count in the treebank.

This module reads that CSV and yields one record per row, unfiltered; the
scoring filters of MorphScore v2 are applied where the gold is built
(`slm4ie.tokenizers.morphology.build_morphscore_lexicon`). Like Sloleks, it is
an evaluation set only and never enters the pretraining corpus.
"""

import csv
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# ===============================
# Defaults
# ===============================

#: Columns the parser reads; a file missing any of them is not this release.
_REQUIRED_COLUMNS = (
    "wordform",
    "lemma",
    "stem",
    "preceding_part",
    "following_part",
    "unique",
    "pos",
    "morph_comp",
    "data_split",
    "word_freq",
)

# ===============================
# Parser functions
# ===============================


def split_item(preceding: str, stem: str, following: str) -> list[str]:
    """Return the non-empty pieces of one item in order.

    Args:
        preceding: The part of the form before the stem, possibly empty.
        stem: The stem.
        following: The part of the form after the stem, possibly empty.

    Returns:
        The pieces that concatenate to the form.
    """
    return [piece for piece in (preceding, stem, following) if piece]


def iter_morphscore_items(csv_path: Path) -> Iterator[dict[str, Any]]:
    """Yield one record per row of a MorphScore v2 language file.

    Args:
        csv_path: Path to the language CSV (e.g. `slovenian_data.csv`).

    Yields:
        `form`, `lemma`, `stem`, `morphemes`, `stem_index`
            (the stem's position in `morphemes`), `is_unique`, `pos`, `feats`,
            `data_split`, `treebank` and `word_freq`.

    Raises:
        ValueError: If the file lacks a required column.
    """
    with open(csv_path, encoding="utf-8", newline="") as handle:
        # check the header
        reader = csv.DictReader(handle)
        missing = [column for column in _REQUIRED_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{csv_path} is missing MorphScore columns {missing}")

        # yield one record per row
        for row in reader:
            yield {
                "form": row["wordform"],
                "lemma": row["lemma"],
                "stem": row["stem"],
                "morphemes": split_item(row["preceding_part"], row["stem"], row["following_part"]),
                "stem_index": 1 if row["preceding_part"] else 0,
                "is_unique": row["unique"] == "unique",
                "pos": row["pos"],
                "feats": row["morph_comp"],
                "data_split": row["data_split"],
                "treebank": row.get("dataset_source", ""),
                "word_freq": int(row["word_freq"]),
            }


def find_morphscore_csv(raw_dir: Path) -> Path | None:
    """Return the Slovene MorphScore CSV under `raw_dir`, if present.

    Args:
        raw_dir: Directory the file was downloaded to.

    Returns:
        The first `slovenian_data.csv` found, or None.
    """
    return next(iter(sorted(Path(raw_dir).rglob("slovenian_data.csv"))), None)
