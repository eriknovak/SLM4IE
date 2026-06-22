"""Parser for the Sloleks 2.0 word-relations derivational lexicon.

The CLARIN.SI resource "List of word relations from the Sloleks 2.0 lexicon"
(handle 11356/1986, DOI 10.34894/EBQJRF, CC BY-SA 4.0) is a TSV of derivational
word pairs. Each row links a base lemma to a derived (related) lemma and gives
the derived lemma already decomposed into morphemes with underscores
(`pis_at_elj`). About 5,000 pairs additionally carry a manual linguist score
(0 inadequate, 1 acceptable, 2 adequate); the remaining rows are rule-generated
silver.

This module reads that TSV and yields per-derived-lemma morpheme segmentations,
deduplicated by lemma (a verified segmentation wins over an unverified one).
Unlike `slm4ie.data.sloleks`, no heuristic alignment happens here: the morpheme
boundaries are read straight from the underscore decomposition.

The exact column layout is taken from the resource's documentation rather than a
verbatim header, so the parser is deliberately defensive: a row counts as data
only when its decomposed column, with underscores removed, reproduces the
related-lemma column. That single check rejects a header line and any malformed
row regardless of small column-order drift, and is the contract to confirm
against the real download.

Used by `scripts/data/to_tokenization.py` to materialize a derivational
tokenizer/morphology evaluation JSONL. Like Sloleks, this resource is absent
from `configs/data/extract.yaml` and never enters the pretraining corpus.
"""

import csv
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

#: Zero-based column indices in the word-relations TSV (documented layout).
_COL_RELATED_LEMMA = 1
_COL_RELATED_DECOMPOSED = 3
_COL_RELATED_MSD = 5
_COL_RULE_ID = 9

#: Optional manual-evaluation column (present for the ~5k checked pairs). A 0
#: marks an inadequate decomposition (dropped); 1 or 2 marks a verified one.
_COL_EVAL_SCORE = 11

#: Minimum field count for a row to be considered (through the rule-id column).
_MIN_FIELDS = _COL_RULE_ID + 1


def split_morphemes(decomposed: str) -> List[str]:
    """Split an underscore-decomposed word into its morpheme pieces.

    Args:
        decomposed: A decomposed surface form such as `pis_at_elj`.

    Returns:
        List[str]: The non-empty morpheme pieces in order; concatenating them
            reproduces the surface form.
    """
    return [piece for piece in decomposed.split("_") if piece]


def _parse_eval_score(fields: List[str]) -> Optional[int]:
    """Return the manual evaluation score of a row, when present.

    Args:
        fields: The tab-split fields of one TSV row.

    Returns:
        Optional[int]: The integer score (0, 1, or 2), or None when the row
            carries no evaluation column or it does not parse as a score.
    """
    if len(fields) <= _COL_EVAL_SCORE:
        return None
    raw = fields[_COL_EVAL_SCORE].strip()
    if raw not in {"0", "1", "2"}:
        return None
    return int(raw)


def _row_to_segmentation(fields: List[str]) -> Optional[Dict[str, Any]]:
    """Convert one TSV row into a derivational segmentation record.

    A row is accepted only when its decomposed column, with underscores
    removed, equals the related-lemma column; this rejects a header line and
    malformed rows. Rows scored 0 (inadequate) by a linguist are dropped.

    Args:
        fields: The tab-split fields of one TSV row.

    Returns:
        Optional[Dict[str, Any]]: A record with `lemma`, `morphemes`, `msd`,
            `verified`, and `rule_id` keys, or None when the row is not valid
            data or was judged inadequate.
    """
    if len(fields) < _MIN_FIELDS:
        return None

    lemma = fields[_COL_RELATED_LEMMA].strip()
    decomposed = fields[_COL_RELATED_DECOMPOSED].strip()
    if not lemma or not decomposed:
        return None

    morphemes = split_morphemes(decomposed)
    if "".join(morphemes) != lemma:
        return None

    score = _parse_eval_score(fields)
    if score == 0:
        return None

    return {
        "lemma": lemma,
        "morphemes": morphemes,
        "msd": fields[_COL_RELATED_MSD].strip() or None,
        "verified": score is not None,
        "rule_id": fields[_COL_RULE_ID].strip() or None,
    }


def iter_word_relations(tsv_path: Path) -> Iterator[Dict[str, Any]]:
    """Stream raw derivational segmentations from a word-relations TSV.

    Yields one record per valid row, so a derived lemma that appears in several
    pairs is yielded several times. Use `iter_word_relation_segmentations` for
    the deduplicated view.

    Args:
        tsv_path: Path to the word-relations TSV file.

    Yields:
        Dict[str, Any]: Records as produced by `_row_to_segmentation`.
    """
    with open(tsv_path, encoding="utf-8", newline="") as handle:
        for fields in csv.reader(handle, delimiter="\t"):
            record = _row_to_segmentation(fields)
            if record is not None:
                yield record


def iter_word_relation_segmentations(tsv_path: Path) -> Iterator[Dict[str, Any]]:
    """Stream per-lemma derivational segmentations, deduplicated by lemma.

    When a derived lemma occurs in multiple pairs, a verified segmentation is
    kept over an unverified one; otherwise the first occurrence wins. Records
    are emitted in first-seen order so the output is deterministic for a given
    input file.

    Args:
        tsv_path: Path to the word-relations TSV file.

    Yields:
        Dict[str, Any]: One record per distinct derived lemma, with `lemma`,
            `morphemes`, `msd`, `verified`, and `rule_id` keys.
    """
    chosen: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for record in iter_word_relations(tsv_path):
        lemma = record["lemma"]
        existing = chosen.get(lemma)
        if existing is None:
            chosen[lemma] = record
            order.append(lemma)
        elif record["verified"] and not existing["verified"]:
            chosen[lemma] = record
    for lemma in order:
        yield chosen[lemma]


def find_word_relations_tsv(raw_dir: Path) -> Optional[Path]:
    """Return the main word-relations TSV under a raw download directory.

    Prefers a file whose name contains `word_relations` (the data file) over the
    sibling `word_relation_rules` file, and returns the lexicographically first
    match for determinism.

    Args:
        raw_dir: Directory holding the unzipped word-relations download.

    Returns:
        Optional[Path]: The data TSV path, or None when none is found.
    """
    candidates = [
        path for path in sorted(raw_dir.rglob("*.tsv")) if "word_relations" in path.name and "rules" not in path.name
    ]
    return candidates[0] if candidates else None
