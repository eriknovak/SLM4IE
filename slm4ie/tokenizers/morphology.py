"""Derive an approximate morpheme segmentation from the Sloleks lexicon.

The project has no gold morpheme segmentation, so this module reconstructs an
inflectional one from Sloleks: each inflected `form` is aligned to its `lemma`
to recover a stem plus an inflectional suffix. The alignment is greedy and
handles the common Slovene stem alternations (palatalization/sibilarization and
the fleeting "polglasnik" vowel). The result is a `MorphLexicon` that serves two
roles: a morpheme table consumed by the morphological tokenizers
(`morph_bpe`, `morph_piece`) and the silver-gold reference consumed by the morph
metrics (`morph_score`, `morph_edit_distance`, `morph_consistency_score`).

The segmentation is inflectional only — derivation, prefixation and compounding
are out of scope — and the alignment is heuristic, so treat the morph metrics as
relative comparators across tokenizers rather than absolute morphological truth.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from slm4ie.data.io_utils import open_output, open_text_stream

logger = logging.getLogger(__name__)

#: Slovene vowels used by the fleeting-vowel (polglasnik) alignment rule.
_VOWELS = set("aeiou")

#: MULTEXT-East (English) category letters for invariant word classes whose
#: forms must not be split: adposition, conjunction, particle, interjection,
#: abbreviation. Assumes the positional MULTEXT-East tagset (e.g. `Ncfsn`).
_INDECLINABLE_CATEGORIES = set("SCQIY")

#: Slovene stem-final consonant alternations (palatalization / sibilarization),
#: written as `(base, alternant)` pairs. The matcher tries both directions so a
#: lemma-side base (`k`) aligns with a form-side alternant (`č`) and vice versa.
#: Multi-character pairs are listed before single ones so they match greedily.
_STEM_ALTERNATIONS: List[Tuple[str, str]] = [
    ("sk", "šč"),
    ("st", "šč"),
    ("zg", "žj"),
    ("k", "č"),
    ("k", "c"),
    ("g", "ž"),
    ("g", "z"),
    ("h", "š"),
    ("h", "s"),
    ("c", "č"),
    ("z", "ž"),
    ("s", "š"),
    ("t", "č"),
    ("d", "j"),
    ("l", "lj"),
    ("n", "nj"),
]


@dataclass
class MorphemeSegmentation:
    """One form's morpheme split derived from its lemma.

    Attributes:
        form (str): The surface inflected form.
        morphemes (List[str]): Ordered morpheme pieces; concatenating them
            reproduces `form`.
        labels (List[str]): Per-morpheme role, one of `stem`, `suffix`, or
            `whole` (the last marks an unreliable, unsegmented fallback).
        lemma (str): The lemma the form was aligned against.
        msd (Optional[str]): The form's morphosyntactic descriptor, if known.
    """

    form: str
    morphemes: List[str]
    labels: List[str]
    lemma: str
    msd: Optional[str] = None

    @property
    def is_reliable(self) -> bool:
        """Return True when the segmentation is trustworthy for metrics.

        Returns:
            bool: False when the form could not be aligned (`whole` label).
        """
        return "whole" not in self.labels

    def boundaries(self) -> List[int]:
        """Return internal morpheme boundary offsets within `form`.

        Returns:
            List[int]: Cumulative character offsets of each internal boundary
                (excluding 0 and `len(form)`).
        """
        offsets: List[int] = []
        cursor = 0
        for morpheme in self.morphemes[:-1]:
            cursor += len(morpheme)
            offsets.append(cursor)
        return offsets


@dataclass
class MorphLexicon:
    """Form-keyed gold segmentations plus a derived morpheme inventory.

    Attributes:
        by_form (Dict[str, MorphemeSegmentation]): Reliable segmentations keyed
            by surface form; the gold reference for the morph metrics.
        inventory (Dict[str, int]): Morpheme to occurrence count across all
            reliable segmentations; the morpheme table for the morph backends.
        stems (Set[str]): Distinct stem morphemes.
        suffixes (Set[str]): Distinct suffix morphemes.
        collisions (int): Count of forms seen more than once with differing
            segmentations (the first is kept).
    """

    by_form: Dict[str, MorphemeSegmentation] = field(default_factory=dict)
    inventory: Dict[str, int] = field(default_factory=dict)
    stems: Set[str] = field(default_factory=set)
    suffixes: Set[str] = field(default_factory=set)
    collisions: int = 0

    def segment(self, word: str) -> List[str]:
        """Return the gold morphemes for `word`, or `[word]` if unknown.

        Args:
            word (str): A surface word form.

        Returns:
            List[str]: The reliable morpheme pieces, or a single-element list
                with the word itself when it is not in the lexicon.
        """
        entry = self.by_form.get(word)
        return list(entry.morphemes) if entry is not None else [word]


def _match_alternation(form: str, i: int, lemma: str, j: int) -> Optional[Tuple[int, int]]:
    """Match a stem alternation at the current alignment position.

    Args:
        form (str): Lowercased surface form.
        i (int): Current index into `form`.
        lemma (str): Lowercased lemma.
        j (int): Current index into `lemma`.

    Returns:
        Optional[Tuple[int, int]]: `(form_consumed, lemma_consumed)` when an
            alternation pair matches at the position, else None.
    """
    for base, alternant in _STEM_ALTERNATIONS:
        if form.startswith(alternant, i) and lemma.startswith(base, j):
            return len(alternant), len(base)
        if form.startswith(base, i) and lemma.startswith(alternant, j):
            return len(base), len(alternant)
    return None


def _stem_cut(form: str, lemma: str) -> int:
    """Return the form-side stem length via greedy lemma alignment.

    Walks `form` and `lemma` together, consuming matching characters and
    applying stem-alternation and fleeting-vowel rules at mismatches. The walk
    stops at the first unresolved mismatch; the form index reached is the stem
    boundary.

    Args:
        form (str): Lowercased surface form.
        lemma (str): Lowercased lemma.

    Returns:
        int: Number of leading `form` characters that belong to the stem.
    """
    i = 0
    j = 0
    while i < len(form) and j < len(lemma):
        if form[i] == lemma[j]:
            i += 1
            j += 1
            continue
        alternation = _match_alternation(form, i, lemma, j)
        if alternation is not None:
            i += alternation[0]
            j += alternation[1]
            continue
        # Fleeting vowel present in the lemma but dropped in the form.
        if lemma[j] in _VOWELS and j + 1 < len(lemma) and form[i] == lemma[j + 1]:
            j += 1
            continue
        # Epenthetic vowel present in the form but absent from the lemma.
        if form[i] in _VOWELS and i + 1 < len(form) and form[i + 1] == lemma[j]:
            i += 1
            continue
        break
    return i


def _is_indeclinable(msd: Optional[str]) -> bool:
    """Return True when the MSD marks an invariant word class.

    Args:
        msd (Optional[str]): A MULTEXT-East morphosyntactic descriptor.

    Returns:
        bool: True for adpositions, conjunctions, particles, interjections,
            and abbreviations, which must not be split.
    """
    if not msd:
        return False
    return msd[0].upper() in _INDECLINABLE_CATEGORIES


def segment_form(
    form: str,
    lemma: str,
    msd: Optional[str] = None,
    *,
    min_stem_len: int = 2,
) -> Optional[MorphemeSegmentation]:
    """Segment one inflected form against its lemma.

    Args:
        form (str): Surface inflected form.
        lemma (str): Lemma (citation form).
        msd (Optional[str]): The form's morphosyntactic descriptor.
        min_stem_len (int): Minimum stem length; shorter alignments are
            treated as unreliable and returned whole.

    Returns:
        Optional[MorphemeSegmentation]: The segmentation, or None when `form`
            is empty.
    """
    if not form:
        return None

    # Indeclinables and multi-word forms are kept whole but still reliable.
    if _is_indeclinable(msd) or " " in form:
        return MorphemeSegmentation(form, [form], ["stem"], lemma, msd)

    cut = _stem_cut(form.lower(), lemma.lower())

    if cut >= len(form):
        # The form is (a prefix of) the lemma: a single clean stem.
        return MorphemeSegmentation(form, [form], ["stem"], lemma, msd)
    if cut < min_stem_len:
        # Alignment failed (suppletion, irregular stem): untrusted fallback.
        return MorphemeSegmentation(form, [form], ["whole"], lemma, msd)

    stem = form[:cut]
    suffix = form[cut:]
    return MorphemeSegmentation(form, [stem, suffix], ["stem", "suffix"], lemma, msd)


def derive_segmentation(
    record: Dict[str, Any],
    *,
    min_stem_len: int = 2,
) -> List[MorphemeSegmentation]:
    """Segment every form in one Sloleks record.

    Args:
        record (Dict[str, Any]): A Sloleks record with `lemma` and `forms`.
        min_stem_len (int): Minimum stem length forwarded to `segment_form`.

    Returns:
        List[MorphemeSegmentation]: One segmentation per non-empty form.
    """
    lemma = record.get("lemma")
    if not lemma:
        return []
    segmentations: List[MorphemeSegmentation] = []
    for entry in record.get("forms", []):
        form = entry.get("form")
        if not form:
            continue
        segmentation = segment_form(form, lemma, entry.get("msd"), min_stem_len=min_stem_len)
        if segmentation is not None:
            segmentations.append(segmentation)
    return segmentations


def _add_to_lexicon(lexicon: MorphLexicon, segmentation: MorphemeSegmentation) -> None:
    """Insert one reliable segmentation into the lexicon, updating indices.

    Args:
        lexicon (MorphLexicon): The lexicon being built (mutated in place).
        segmentation (MorphemeSegmentation): A reliable segmentation to add.
    """
    existing = lexicon.by_form.get(segmentation.form)
    if existing is not None:
        if existing.morphemes != segmentation.morphemes:
            lexicon.collisions += 1
        return

    lexicon.by_form[segmentation.form] = segmentation
    for morpheme, label in zip(segmentation.morphemes, segmentation.labels):
        lexicon.inventory[morpheme] = lexicon.inventory.get(morpheme, 0) + 1
        if label == "suffix":
            lexicon.suffixes.add(morpheme)
        else:
            lexicon.stems.add(morpheme)


def build_morph_lexicon(sloleks_path: Path, *, min_stem_len: int = 2) -> MorphLexicon:
    """Build a morpheme lexicon from a converted Sloleks JSONL(.gz) file.

    Args:
        sloleks_path (Path): Path to `tokenization/sloleks.jsonl.gz` as
            produced by `scripts/data/to_tokenization.py`.
        min_stem_len (int): Minimum stem length for a reliable split.

    Returns:
        MorphLexicon: The form lookup plus the derived morpheme inventory.

    Raises:
        FileNotFoundError: If `sloleks_path` does not exist.
    """
    sloleks_path = Path(sloleks_path)
    if not sloleks_path.exists():
        raise FileNotFoundError(f"Sloleks lexicon not found: {sloleks_path}")

    lexicon = MorphLexicon()
    with open_text_stream(sloleks_path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for segmentation in derive_segmentation(record, min_stem_len=min_stem_len):
                if segmentation.is_reliable:
                    _add_to_lexicon(lexicon, segmentation)

    logger.info(
        "Built morpheme lexicon: %d forms, %d morphemes (%d collisions).",
        len(lexicon.by_form),
        len(lexicon.inventory),
        lexicon.collisions,
    )
    return lexicon


def save_lexicon(lexicon: MorphLexicon, path: Path) -> Path:
    """Persist a lexicon's reliable segmentations to JSONL(.gz).

    The inventory, stems, and suffixes are rebuilt from the saved forms on
    load, so only the per-form segmentations are written.

    Args:
        lexicon (MorphLexicon): The lexicon to serialize.
        path (Path): Destination `.jsonl`/`.jsonl.gz` path.

    Returns:
        Path: `path`, for convenient chaining.
    """
    path = Path(path)
    with open_output(path) as out:
        for entry in lexicon.by_form.values():
            payload = {
                "form": entry.form,
                "morphemes": entry.morphemes,
                "labels": entry.labels,
                "lemma": entry.lemma,
                "msd": entry.msd,
            }
            out.write(json.dumps(payload, ensure_ascii=False))
            out.write("\n")
    return path


def load_lexicon(path: Path) -> MorphLexicon:
    """Load a lexicon previously written by `save_lexicon`.

    Args:
        path (Path): Path to the `.jsonl`/`.jsonl.gz` lexicon file.

    Returns:
        MorphLexicon: Reconstructed lexicon with indices rebuilt.

    Raises:
        FileNotFoundError: If `path` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Lexicon file not found: {path}")

    lexicon = MorphLexicon()
    with open_text_stream(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            segmentation = MorphemeSegmentation(
                form=payload["form"],
                morphemes=payload["morphemes"],
                labels=payload["labels"],
                lemma=payload.get("lemma", ""),
                msd=payload.get("msd"),
            )
            _add_to_lexicon(lexicon, segmentation)
    return lexicon


def iter_lexicon_forms(lexicon: MorphLexicon) -> Iterator[MorphemeSegmentation]:
    """Yield the reliable segmentations stored in a lexicon.

    Args:
        lexicon (MorphLexicon): The lexicon to iterate.

    Yields:
        MorphemeSegmentation: Each stored per-form segmentation.
    """
    yield from lexicon.by_form.values()
