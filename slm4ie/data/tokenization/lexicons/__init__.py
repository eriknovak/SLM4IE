"""Reader backends for the tokenizer-quality datasets.

Each module turns one raw lexicon download into tagged JSONL records; the
`LEXICONS` map is the backend registry the run loop resolves dataset keys
against. A new lexicon is a new module here plus one entry in that map.
"""

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from slm4ie.data.archives import unpack_archives
from slm4ie.data.tokenization.lexicons.morphscore import find_morphscore_csv, iter_morphscore_items
from slm4ie.data.tokenization.lexicons.sloleks import iter_sloleks_dir
from slm4ie.data.tokenization.lexicons.sloleks_inflection import (
    find_inflection_tsv,
    iter_inflection_segmentations,
)
from slm4ie.data.tokenization.lexicons.sloleks_relations import (
    find_word_relations_tsv,
    iter_word_relation_segmentations,
)

# ===============================
# Defaults
# ===============================

#: Task tag identifying tokenizer-evaluation datasets.
TOKENIZER_TASK = "TOKENIZER"

# ===============================
# Lexicon readers
# ===============================


def _read_sloleks(raw_dir: Path) -> Iterator[dict[str, Any]]:
    """Yield Sloleks records, tagging them with dataset/task fields.

    Any archive sitting in `raw_dir` (e.g. the downloaded Sloleks zip) is
    unpacked in place when no XML is found yet, so the download need not be
    unzipped by hand first.

    Args:
        raw_dir: Directory holding the Sloleks XML files (or the zip to unpack).

    Yields:
        Records as produced by `iter_sloleks_dir`, extended
            with `dataset` and `task` fields.

    Raises:
        FileNotFoundError: If no Sloleks XML files are found under `raw_dir`,
            even after unpacking any archive present.
    """
    # unpack any archive when no XML is found yet
    if not any(raw_dir.rglob("*.xml")):
        unpack_archives(raw_dir)
    if not any(raw_dir.rglob("*.xml")):
        raise FileNotFoundError(
            f"No XML files found under {raw_dir}. Run `prepare_datasets.py download sloleks` first."
        )

    # tag the records with dataset and task
    for record in iter_sloleks_dir(raw_dir):
        record["dataset"] = "sloleks"
        record["task"] = TOKENIZER_TASK
        yield record


def _read_sloleks_relations(raw_dir: Path) -> Iterator[dict[str, Any]]:
    """Yield Sloleks word-relation derivational segmentations.

    Each record carries one derived lemma's morpheme decomposition (read from
    the underscore column, not heuristically aligned), tagged with the dataset
    and task fields and a `verified` flag for the manually-scored subset. Any
    archive in `raw_dir` (the downloaded zip) is unpacked in place when no TSV
    is found yet, so the download need not be unzipped by hand first.

    Args:
        raw_dir: Directory holding the word-relations TSV (or the zip to unpack).

    Yields:
        Records as produced by `iter_word_relation_segmentations`,
            extended with `dataset` and `task` fields.

    Raises:
        FileNotFoundError: If no word-relations TSV is found under `raw_dir`,
            even after unpacking any archive present.
    """
    # unpack any archive when no TSV is found yet
    tsv_path = find_word_relations_tsv(raw_dir)
    if tsv_path is None:
        unpack_archives(raw_dir)
        # unpack any archive when no TSV is found yet
    tsv_path = find_word_relations_tsv(raw_dir)
    if tsv_path is None:
        raise FileNotFoundError(
            f"No word-relations TSV found under {raw_dir}. Run `prepare_datasets.py download sloleks_relations` first."
        )

    # tag the records with dataset and task
    for record in iter_word_relation_segmentations(tsv_path):
        record["dataset"] = "sloleks_relations"
        record["task"] = TOKENIZER_TASK
        yield record


def _read_sloleks_inflection(raw_dir: Path) -> Iterator[dict[str, Any]]:
    """Yield the Sloleks stem-ending split, one record per lexeme.

    The release is a TSV the Sloleks authors share directly rather than a
    public download, so it is placed in `raw_dir` by hand; any archive there
    is unpacked in place when no TSV is found yet.

    Args:
        raw_dir: Directory holding the inflection split TSV (or its zip).

    Yields:
        Records as produced by `iter_inflection_segmentations`,
            extended with `dataset` and `task` fields.

    Raises:
        FileNotFoundError: If no inflection split TSV is found under `raw_dir`,
            even after unpacking any archive present.
    """
    # unpack any archive when no TSV is found yet
    tsv_path = find_inflection_tsv(raw_dir)
    if tsv_path is None:
        unpack_archives(raw_dir)
        # unpack any archive when no TSV is found yet
    tsv_path = find_inflection_tsv(raw_dir)
    if tsv_path is None:
        raise FileNotFoundError(
            f"No *inflect_morph_seg*.tsv found under {raw_dir}. The file is shared by the Sloleks authors; place it there."
        )

    # tag the records with dataset and task
    for record in iter_inflection_segmentations(tsv_path):
        record["dataset"] = "sloleks_inflection"
        record["task"] = TOKENIZER_TASK
        yield record


def _read_morphscore_slovenian(raw_dir: Path) -> Iterator[dict[str, Any]]:
    """Yield the Slovene MorphScore v2 items, one record per treebank row.

    Args:
        raw_dir: Directory holding the downloaded `slovenian_data.csv`.

    Yields:
        Records as produced by `iter_morphscore_items`,
            extended with `dataset` and `task` fields.

    Raises:
        FileNotFoundError: If no `slovenian_data.csv` is found under `raw_dir`.
    """
    csv_path = find_morphscore_csv(raw_dir)
    if csv_path is None:
        raise FileNotFoundError(
            f"No slovenian_data.csv found under {raw_dir}. Run `prepare_datasets.py download morphscore_slovenian` first."
        )

    # tag the records with dataset and task
    for record in iter_morphscore_items(csv_path):
        record["dataset"] = "morphscore_slovenian"
        record["task"] = TOKENIZER_TASK
        yield record


# ===============================
# Registry
# ===============================

#: Registry mapping dataset key to the callable that reads its lexicon.
LEXICONS: dict[str, Callable[[Path], Iterator[dict[str, Any]]]] = {
    "sloleks": _read_sloleks,
    "sloleks_relations": _read_sloleks_relations,
    "sloleks_inflection": _read_sloleks_inflection,
    "morphscore_slovenian": _read_morphscore_slovenian,
}
