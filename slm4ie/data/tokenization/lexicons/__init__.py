"""Reader backends for the tokenizer-quality datasets.

Each module turns one raw lexicon download into tagged JSONL records; the
`LEXICONS` map is the backend registry the run loop resolves dataset keys
against. A new lexicon is a new module here plus one entry in that map.
"""

from pathlib import Path
from typing import Any, Callable, Dict, Iterator

from slm4ie.data.archives import unpack_archives
from slm4ie.data.tokenization.lexicons.sloleks import iter_sloleks_dir
from slm4ie.data.tokenization.lexicons.sloleks_relations import (
    find_word_relations_tsv,
    iter_word_relation_segmentations,
)

#: Task tag identifying tokenizer-evaluation datasets.
TOKENIZER_TASK = "TOKENIZER"


def _read_sloleks(raw_dir: Path) -> Iterator[Dict[str, Any]]:
    """Yield Sloleks records, tagging them with dataset/task fields.

    Any archive sitting in `raw_dir` (e.g. the downloaded Sloleks zip) is
    unpacked in place when no XML is found yet, so the download need not be
    unzipped by hand first.

    Args:
        raw_dir: Directory holding the Sloleks XML files (or the zip to unpack).

    Yields:
        Dict[str, Any]: Records as produced by `iter_sloleks_dir`, extended
            with `dataset` and `task` fields.

    Raises:
        FileNotFoundError: If no Sloleks XML files are found under `raw_dir`,
            even after unpacking any archive present.
    """
    if not any(raw_dir.rglob("*.xml")):
        unpack_archives(raw_dir)
    if not any(raw_dir.rglob("*.xml")):
        raise FileNotFoundError(
            f"No XML files found under {raw_dir}. Run `prepare_datasets.py download sloleks` first."
        )
    for record in iter_sloleks_dir(raw_dir):
        record["dataset"] = "sloleks"
        record["task"] = TOKENIZER_TASK
        yield record


def _read_sloleks_relations(raw_dir: Path) -> Iterator[Dict[str, Any]]:
    """Yield Sloleks word-relation derivational segmentations.

    Each record carries one derived lemma's morpheme decomposition (read from
    the underscore column, not heuristically aligned), tagged with the dataset
    and task fields and a `verified` flag for the manually-scored subset. Any
    archive in `raw_dir` (the downloaded zip) is unpacked in place when no TSV
    is found yet, so the download need not be unzipped by hand first.

    Args:
        raw_dir: Directory holding the word-relations TSV (or the zip to unpack).

    Yields:
        Dict[str, Any]: Records as produced by `iter_word_relation_segmentations`,
            extended with `dataset` and `task` fields.

    Raises:
        FileNotFoundError: If no word-relations TSV is found under `raw_dir`,
            even after unpacking any archive present.
    """
    tsv_path = find_word_relations_tsv(raw_dir)
    if tsv_path is None:
        unpack_archives(raw_dir)
        tsv_path = find_word_relations_tsv(raw_dir)
    if tsv_path is None:
        raise FileNotFoundError(
            f"No word-relations TSV found under {raw_dir}. Run `prepare_datasets.py download sloleks_relations` first."
        )
    for record in iter_word_relation_segmentations(tsv_path):
        record["dataset"] = "sloleks_relations"
        record["task"] = TOKENIZER_TASK
        yield record


#: Registry mapping dataset key to the callable that reads its lexicon.
LEXICONS: Dict[str, Callable[[Path], Iterator[Dict[str, Any]]]] = {
    "sloleks": _read_sloleks,
    "sloleks_relations": _read_sloleks_relations,
}
