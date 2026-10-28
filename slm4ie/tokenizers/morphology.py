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

`build_inflection_lexicon` reads the stem-ending split the Sloleks authors
released instead, with no alignment, and `split_lexemes` holds a share of its
lexemes out of the morpheme table so the metrics score forms no table saw.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from slm4ie.tokenizers.corpus import stable_rank
from slm4ie.utils.io import open_output, open_text_stream

# ===============================
# Logging
# ===============================

logger = logging.getLogger(__name__)

# ===============================
# Defaults
# ===============================

#: Slovene vowels used by the fleeting-vowel (polglasnik) alignment rule.
_VOWELS = set("aeiou")

#: JOS (Slovene MULTEXT-East) category letters for invariant word classes whose
#: forms must not be split: preposition (D), conjunction (V), particle (L),
#: interjection (M), abbreviation (O). Sloleks carries the Slovene tags
#: (`Sozei`), where `S` is a noun.
_INDECLINABLE_CATEGORIES = set("DVLMO")

#: Slovene stem-final consonant alternations (palatalization / sibilarization),
#: written as `(base, alternant)` pairs. The matcher tries both directions so a
#: lemma-side base (`k`) aligns with a form-side alternant (`č`) and vice versa.
#: Multi-character pairs are listed before single ones so they match greedily.
_STEM_ALTERNATIONS: list[tuple[str, str]] = [
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


# ===============================
# Segmentation types
# ===============================


@dataclass
class MorphemeSegmentation:
    """One form's morpheme split derived from its lemma.

    Attributes:
        form (str): The surface inflected form.
        morphemes (list[str]): Ordered morpheme pieces; concatenating them
            reproduces `form`.
        labels (list[str]): Per-morpheme role, one of `prefix`, `stem`,
            `suffix`, `whole` (the last marks an unreliable, unsegmented
            fallback), or `morph` (a role-agnostic derivational piece).
        lemma (str): The lemma the form was aligned against.
        msd (str | None): The form's morphosyntactic descriptor, if known.
        verified (bool | None): For derivational segmentations, whether the
            split carries a manual linguist evaluation; None when not
            applicable (e.g. inflectional segmentations).
        lexeme (str | None): Id of the Sloleks lexeme the form is scored
            under, the unit a bootstrap resamples; None when not assigned.
    """

    form: str
    morphemes: list[str]
    labels: list[str]
    lemma: str
    msd: str | None = None
    verified: bool | None = None
    lexeme: str | None = None

    @property
    def is_reliable(self) -> bool:
        """Return True when the segmentation is trustworthy for metrics.

        Returns:
            False when the form could not be aligned (`whole` label).
        """
        return "whole" not in self.labels

    def boundaries(self) -> list[int]:
        """Return internal morpheme boundary offsets within `form`.

        Returns:
            Cumulative character offsets of each internal boundary
                (excluding 0 and `len(form)`).
        """
        offsets: list[int] = []
        cursor = 0

        # accumulate the morpheme lengths
        for morpheme in self.morphemes[:-1]:
            cursor += len(morpheme)
            offsets.append(cursor)
        return offsets


@dataclass
class MorphLexicon:
    """Form-keyed gold segmentations plus a derived morpheme inventory.

    Attributes:
        by_form (dict[str, MorphemeSegmentation]): Reliable segmentations keyed
            by surface form; the gold reference for the morph metrics.
        inventory (dict[str, int]): Morpheme to occurrence count across all
            reliable segmentations; the morpheme table for the morph backends.
        stems (set[str]): Distinct stem morphemes.
        suffixes (set[str]): Distinct suffix morphemes.
        collisions (int): Count of forms seen more than once with differing
            segmentations (the first is kept).
    """

    by_form: dict[str, MorphemeSegmentation] = field(default_factory=dict)
    inventory: dict[str, int] = field(default_factory=dict)
    stems: set[str] = field(default_factory=set)
    suffixes: set[str] = field(default_factory=set)
    collisions: int = 0

    def segment(self, word: str) -> list[str]:
        """Return the gold morphemes for `word`, or `[word]` if unknown.

        Args:
            word: A surface word form.

        Returns:
            The reliable morpheme pieces, or a single-element list
                with the word itself when it is not in the lexicon.
        """
        entry = self.by_form.get(word)
        return list(entry.morphemes) if entry is not None else [word]


# ===============================
# Inflectional alignment
# ===============================


def _match_alternation(form: str, i: int, lemma: str, j: int) -> tuple[int, int] | None:
    """Match a stem alternation at the current alignment position.

    Args:
        form: Lowercased surface form.
        i: Current index into `form`.
        lemma: Lowercased lemma.
        j: Current index into `lemma`.

    Returns:
        `(form_consumed, lemma_consumed)` when an
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
        form: Lowercased surface form.
        lemma: Lowercased lemma.

    Returns:
        Number of leading `form` characters that belong to the stem.
    """
    i = 0
    j = 0

    # walk form and lemma together until the first unresolved mismatch
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
        # fleeting vowel present in the lemma but dropped in the form
        if lemma[j] in _VOWELS and j + 1 < len(lemma) and form[i] == lemma[j + 1]:
            j += 1
            continue
        # epenthetic vowel present in the form but absent from the lemma
        if form[i] in _VOWELS and i + 1 < len(form) and form[i + 1] == lemma[j]:
            i += 1
            continue
        break
    return i


def _is_indeclinable(msd: str | None) -> bool:
    """Return True when the MSD marks an invariant word class.

    Args:
        msd: A MULTEXT-East morphosyntactic descriptor.

    Returns:
        True for adpositions, conjunctions, particles, interjections,
            and abbreviations, which must not be split.
    """
    if not msd:
        return False
    return msd[0].upper() in _INDECLINABLE_CATEGORIES


def segment_form(
    form: str,
    lemma: str,
    msd: str | None = None,
    *,
    min_stem_len: int = 2,
) -> MorphemeSegmentation | None:
    """Segment one inflected form against its lemma.

    Args:
        form: Surface inflected form.
        lemma: Lemma (citation form).
        msd: The form's morphosyntactic descriptor.
        min_stem_len: Minimum stem length; shorter alignments are
            treated as unreliable and returned whole.

    Returns:
        The segmentation, or None when `form`
            is empty.
    """
    if not form:
        return None

    # keep indeclinables and multi-word forms whole but reliable
    if _is_indeclinable(msd) or " " in form:
        return MorphemeSegmentation(form, [form], ["stem"], lemma, msd)

    # align the form to the lemma
    cut = _stem_cut(form.lower(), lemma.lower())

    if cut >= len(form):
        # the form is (a prefix of) the lemma: a single clean stem
        return MorphemeSegmentation(form, [form], ["stem"], lemma, msd)
    if cut < min_stem_len:
        # alignment failed (suppletion, irregular stem): untrusted fallback
        return MorphemeSegmentation(form, [form], ["whole"], lemma, msd)

    # split into stem and suffix
    stem = form[:cut]
    suffix = form[cut:]
    return MorphemeSegmentation(form, [stem, suffix], ["stem", "suffix"], lemma, msd)


def derive_segmentation(
    record: dict[str, Any],
    *,
    min_stem_len: int = 2,
) -> list[MorphemeSegmentation]:
    """Segment every form in one Sloleks record.

    Args:
        record: A Sloleks record with `lemma` and `forms`.
        min_stem_len: Minimum stem length forwarded to `segment_form`.

    Returns:
        One segmentation per non-empty form.
    """
    lemma = record.get("lemma")
    if not lemma:
        return []

    # segment each form of the record
    segmentations: list[MorphemeSegmentation] = []
    for entry in record.get("forms", []):
        form = entry.get("form")
        if not form:
            continue
        segmentation = segment_form(form, lemma, entry.get("msd"), min_stem_len=min_stem_len)
        if segmentation is not None:
            segmentations.append(segmentation)
    return segmentations


# ===============================
# Lexicon builders
# ===============================


def _add_to_lexicon(lexicon: MorphLexicon, segmentation: MorphemeSegmentation) -> None:
    """Insert one reliable segmentation into the lexicon, updating indices.

    Args:
        lexicon: The lexicon being built (mutated in place).
        segmentation: A reliable segmentation to add.
    """
    # keep the first segmentation of a form, counting conflicting repeats
    existing = lexicon.by_form.get(segmentation.form)
    if existing is not None:
        if existing.morphemes != segmentation.morphemes:
            lexicon.collisions += 1
        return

    # index the new segmentation
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
        sloleks_path: Path to `tokenization/sloleks.jsonl.gz` as
            produced by `slm4ie/data/tokenization/run.py`.
        min_stem_len: Minimum stem length for a reliable split.

    Returns:
        The form lookup plus the derived morpheme inventory.

    Raises:
        FileNotFoundError: If `sloleks_path` does not exist.
    """
    sloleks_path = Path(sloleks_path)
    if not sloleks_path.exists():
        raise FileNotFoundError(f"Sloleks lexicon not found: {sloleks_path}")

    # derive and add the reliable segmentations
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


def build_derivational_lexicon(path: Path) -> MorphLexicon:
    """Build a derivational morpheme lexicon from a word-relations JSONL.

    Reads the `sloleks_relations` JSONL produced by
    `slm4ie/data/tokenization/run.py`, where each record carries a derived
    `lemma` and its `morphemes` already split on the resource's underscore
    decomposition. Unlike `build_morph_lexicon`, no heuristic alignment runs:
    the boundaries are taken as given. The keyed form is the lemma itself, so
    this lexicon scores derivational (not inflectional) segmentation. Pieces are
    labelled `morph` because the resource gives boundaries, not per-morpheme
    roles, and the `verified` flag is carried through for later verified-only
    reporting.

    Args:
        path: Path to the `sloleks_relations` JSONL(.gz).

    Returns:
        Lemma-keyed derivational segmentations plus the derived
            morpheme inventory.

    Raises:
        FileNotFoundError: If `path` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Derivational lexicon not found: {path}")

    # take the boundaries as given
    lexicon = MorphLexicon()
    with open_text_stream(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            lemma = record.get("lemma")
            morphemes = record.get("morphemes")
            if not lemma or not morphemes:
                continue
            segmentation = MorphemeSegmentation(
                form=lemma,
                morphemes=list(morphemes),
                labels=["morph"] * len(morphemes),
                lemma=lemma,
                msd=record.get("msd"),
                verified=record.get("verified"),
            )
            _add_to_lexicon(lexicon, segmentation)

    logger.info(
        "Built derivational lexicon: %d forms, %d morphemes (%d collisions).",
        len(lexicon.by_form),
        len(lexicon.inventory),
        lexicon.collisions,
    )
    return lexicon


def build_inflection_lexicon(path: Path) -> MorphLexicon:
    """Build the inflectional gold from the Sloleks stem-ending split.

    Reads `tokenization/sloleks_inflection.jsonl.gz`, whose forms are already
    cut into the paradigm stem and the ending, so no alignment runs. A form
    that two lexemes split differently has no single gold cut and is dropped;
    it is counted in `collisions`. A form with a zero ending is kept whole. A
    form is `verified` when every lexeme carrying it was checked by hand.

    Args:
        path: Path to the converted `sloleks_inflection` JSONL(.gz).

    Returns:
        One segmentation per unambiguous form.

    Raises:
        FileNotFoundError: If `path` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Sloleks inflection split not found: {path}")

    # collect each form's splits and manual status
    splits: dict[str, set[tuple[str, ...]]] = {}
    first: dict[str, MorphemeSegmentation] = {}
    manual: dict[str, bool] = {}
    with open_text_stream(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            is_manual = record["status"] == "manual"
            for form in record["forms"]:
                surface = form["form"]
                morphemes = form["morphemes"]
                splits.setdefault(surface, set()).add(tuple(morphemes))
                manual[surface] = manual.get(surface, True) and is_manual
                if surface not in first:
                    labels = ["stem", "suffix"] if len(morphemes) == 2 else ["stem"]
                    msd = form["msds"][0] if form["msds"] else None
                    first[surface] = MorphemeSegmentation(surface, morphemes, labels, record["lemma"], msd)

    # drop forms split two ways
    lexicon = MorphLexicon()
    for surface, segmentation in first.items():
        if len(splits[surface]) > 1:
            lexicon.collisions += 1
            continue
        segmentation.verified = manual[surface]
        _add_to_lexicon(lexicon, segmentation)

    logger.info(
        "Built inflection gold: %d forms (%d manual), %d dropped as split two ways.",
        len(lexicon.by_form),
        sum(1 for s in lexicon.by_form.values() if s.verified),
        lexicon.collisions,
    )
    return lexicon


def build_morphscore_lexicon(path: Path) -> MorphLexicon:
    """Build the outside gold from the converted MorphScore v2 items.

    Applies MorphScore v2's default item filters: the first occurrence of
    each item only, the stem equal to the lemma, and no digit in the form.
    A form kept with two different cuts is dropped and counted in
    `collisions`. A form equal to its stem is kept whole.

    Args:
        path: Path to the converted `morphscore_slovenian` JSONL(.gz).

    Returns:
        One segmentation per form, with the UD part of speech
            as its `msd`.

    Raises:
        FileNotFoundError: If `path` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MorphScore items not found: {path}")

    # collect each scored form's splits
    splits: dict[str, set[tuple[str, ...]]] = {}
    first: dict[str, MorphemeSegmentation] = {}
    with open_text_stream(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if not is_morphscore_scored(item):
                continue
            form, morphemes = item["form"], item["morphemes"]
            splits.setdefault(form, set()).add(tuple(morphemes))
            if form not in first:
                stem_at = item["stem_index"]
                labels = ["prefix"] * stem_at + ["stem"] + ["suffix"] * (len(morphemes) - stem_at - 1)
                first[form] = MorphemeSegmentation(form, morphemes, labels, item["lemma"], item["pos"])

    # drop forms cut two ways
    lexicon = MorphLexicon()
    for form, segmentation in first.items():
        if len(splits[form]) > 1:
            lexicon.collisions += 1
            continue
        _add_to_lexicon(lexicon, segmentation)
    logger.info(
        "Built MorphScore gold: %d forms, %d dropped as cut two ways.", len(lexicon.by_form), lexicon.collisions
    )
    return lexicon


def is_morphscore_scored(item: dict[str, Any]) -> bool:
    """Return whether a MorphScore v2 item passes its default scoring filters.

    Args:
        item: One converted `morphscore_slovenian` record.

    Returns:
        True for a first occurrence whose stem equals its lemma and
            whose form has no digit.
    """
    return item["is_unique"] and item["stem"] == item["lemma"] and not any(ch.isdigit() for ch in item["form"])


# ===============================
# Lexeme split
# ===============================


@dataclass(frozen=True)
class LexemeSplit:
    """Sloleks lexemes split into a table side and a held-out side.

    Attributes:
        heldout_forms (frozenset[str]): Forms of held-out lexemes that no
            table-side lexeme also carries; the only forms scored.
        table_forms (frozenset[str]): Every form of a table-side lexeme.
        lexeme_of (dict[str, str]): Each held-out form to the id of the one
            held-out lexeme it is scored under; a form two held-out lexemes
            carry goes to the lexeme `stable_rank` of the pair ranks lowest.
        counts (list[dict[str, Any]]): One row per side, word class and status
            with its `lexemes` and `forms`.
    """

    heldout_forms: frozenset[str]
    table_forms: frozenset[str]
    lexeme_of: dict[str, str]
    counts: list[dict[str, Any]]


def split_lexemes(path: Path, heldout_share: float) -> LexemeSplit:
    """Hold out a share of the Sloleks lexemes, exactly within each stratum.

    Lexemes are grouped by word class (the first letter of their first form's
    tag) and checking status, ordered within each group by `stable_rank` of
    their id, and the lowest `heldout_share` of each group is held out. A whole
    paradigm sits on one side. A held-out form that a table-side lexeme also
    carries is dropped from the held-out forms. Each held-out form is assigned
    to one held-out lexeme, the unit the bootstrap resamples.

    Args:
        path: Path to the converted `sloleks_inflection` JSONL(.gz).
        heldout_share: Share of each stratum's lexemes held out.

    Returns:
        The held-out and table-side forms with per-stratum counts.

    Raises:
        FileNotFoundError: If `path` does not exist.
        ValueError: If `heldout_share` is not strictly between 0 and 1.
    """
    if not 0 < heldout_share < 1:
        raise ValueError(f"heldout_share must be in (0, 1), got {heldout_share}")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Sloleks inflection split not found: {path}")

    # group the lexemes by word class and status
    strata: dict[tuple[str, str], list[tuple[int, str, list[str]]]] = {}
    with open_text_stream(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            forms = [form["form"] for form in record["forms"]]
            first_msds = record["forms"][0]["msds"] if record["forms"] else []
            word_class = first_msds[0][:1] if first_msds else "?"
            key = (word_class, record["status"])
            entry_id = str(record["entry_id"])
            strata.setdefault(key, []).append((stable_rank(entry_id), entry_id, forms))

    # hold out the lowest-ranked share of each stratum
    sides: dict[str, dict[tuple[str, str], list[tuple[str, list[str]]]]] = {"heldout": {}, "table": {}}
    for key, lexemes in strata.items():
        lexemes.sort(key=lambda item: item[0])
        n_heldout = round(len(lexemes) * heldout_share)
        sides["heldout"][key] = [(entry_id, forms) for _rank, entry_id, forms in lexemes[:n_heldout]]
        sides["table"][key] = [(entry_id, forms) for _rank, entry_id, forms in lexemes[n_heldout:]]

    # resolve the scored forms and their lexemes
    table_forms = frozenset(form for group in sides["table"].values() for _id, forms in group for form in forms)
    heldout_all = {form for group in sides["heldout"].values() for _id, forms in group for form in forms}
    heldout_forms = frozenset(heldout_all - table_forms)
    lexeme_of = _assign_lexemes(sides["heldout"], heldout_forms)

    # count the lexemes and forms per side and stratum
    counts: list[dict[str, Any]] = []
    for side, groups in sides.items():
        for (word_class, status), group in sorted(groups.items()):
            distinct = {form for _id, forms in group for form in forms}
            kept = distinct & heldout_forms if side == "heldout" else distinct
            counts.append(
                {
                    "side": side,
                    "word_class": word_class,
                    "status": status,
                    "lexemes": len(group),
                    "forms": len(distinct),
                    "forms_kept": len(kept),
                }
            )
    logger.info(
        "Lexeme split: %d held-out forms kept, %d dropped as shared with the table side.",
        len(heldout_forms),
        len(heldout_all) - len(heldout_forms),
    )
    return LexemeSplit(heldout_forms=heldout_forms, table_forms=table_forms, lexeme_of=lexeme_of, counts=counts)


def _assign_lexemes(
    heldout: dict[tuple[str, str], list[tuple[str, list[str]]]], forms: frozenset[str]
) -> dict[str, str]:
    """Map each scored form to one held-out lexeme carrying it.

    Args:
        heldout: Held-out lexemes per stratum, as (entry id, forms) pairs.
        forms: The scored held-out forms.

    Returns:
        Form to entry id; among several carriers the one whose
            `stable_rank` of form and id is lowest.
    """
    lexeme_of: dict[str, str] = {}

    # keep the lowest-ranked carrier of each form
    for group in heldout.values():
        for entry_id, lexeme_forms in group:
            for form in lexeme_forms:
                if form not in forms:
                    continue
                current = lexeme_of.get(form)
                if current is None or stable_rank(f"{form}\t{entry_id}") < stable_rank(f"{form}\t{current}"):
                    lexeme_of[form] = entry_id
    return lexeme_of


# ===============================
# Lexicon utilities
# ===============================


def without_forms(lexicon: MorphLexicon, forms: frozenset[str]) -> MorphLexicon:
    """Return a copy of `lexicon` with the given forms left out.

    Args:
        lexicon: The lexicon to filter.
        forms: Surface forms to leave out.

    Returns:
        A new lexicon over the remaining forms, indices rebuilt.
    """
    # re-add the remaining forms to rebuild the indices
    kept = MorphLexicon()
    for form, segmentation in lexicon.by_form.items():
        if form not in forms:
            _add_to_lexicon(kept, segmentation)
    return kept


def merge_lexicons(*lexicons: MorphLexicon) -> MorphLexicon:
    """Union several lexicons into one morpheme table, preferring richer splits.

    Intended for the backend morpheme table (`morph_bpe`, `morph_piece`), which
    looks words up in `by_form`. When the same form appears in more than one
    lexicon, the segmentation with more morpheme pieces wins, so a derivational
    split of a lemma supersedes its trivial single-stem inflectional entry; ties
    keep the first-seen entry. The inventory, stems, and suffixes are rebuilt
    from the merged forms.

    Args:
        lexicons: Lexicons to union, in precedence order for
            ties (earlier wins).

    Returns:
        The merged lexicon.
    """
    # keep the richest segmentation of each form
    merged = MorphLexicon()
    for lexicon in lexicons:
        for segmentation in lexicon.by_form.values():
            existing = merged.by_form.get(segmentation.form)
            if existing is None or len(segmentation.morphemes) > len(existing.morphemes):
                merged.by_form[segmentation.form] = segmentation

    # rebuild the inventory, stems and suffixes
    for segmentation in merged.by_form.values():
        for morpheme, label in zip(segmentation.morphemes, segmentation.labels):
            merged.inventory[morpheme] = merged.inventory.get(morpheme, 0) + 1
            if label == "suffix":
                merged.suffixes.add(morpheme)
            else:
                merged.stems.add(morpheme)

    return merged


def save_lexicon(lexicon: MorphLexicon, path: Path) -> Path:
    """Persist a lexicon's reliable segmentations to JSONL(.gz).

    The inventory, stems, and suffixes are rebuilt from the saved forms on
    load, so only the per-form segmentations are written.

    Args:
        lexicon: The lexicon to serialize.
        path: Destination `.jsonl`/`.jsonl.gz` path.

    Returns:
        `path`, for convenient chaining.
    """
    # write one JSON line per segmentation
    path = Path(path)
    with open_output(path) as out:
        for entry in lexicon.by_form.values():
            payload = {
                "form": entry.form,
                "morphemes": entry.morphemes,
                "labels": entry.labels,
                "lemma": entry.lemma,
                "msd": entry.msd,
                "verified": entry.verified,
                "lexeme": entry.lexeme,
            }
            out.write(json.dumps(payload, ensure_ascii=False))
            out.write("\n")

    return path


def load_lexicon(path: Path) -> MorphLexicon:
    """Load a lexicon previously written by `save_lexicon`.

    Args:
        path: Path to the `.jsonl`/`.jsonl.gz` lexicon file.

    Returns:
        Reconstructed lexicon with indices rebuilt.

    Raises:
        FileNotFoundError: If `path` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Lexicon file not found: {path}")

    # rebuild the lexicon from the saved segmentations
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
                verified=payload.get("verified"),
                lexeme=payload.get("lexeme"),
            )
            _add_to_lexicon(lexicon, segmentation)

    return lexicon


def sample_segmentations(lexicon: MorphLexicon, n: int | None, seed: int) -> list[MorphemeSegmentation]:
    """Draw a deterministic sample of gold segmentations for the morph metrics.

    The sample is fixed by `seed` and the lexicon's (stable) form order, so every
    tokenizer scores the identical forms in the identical order. That alignment
    is what lets the per-form morph statistics be paired across tokenizers when
    bootstrapping. Forms are returned in lexicon order (the sampled indices are
    sorted), not in draw order.

    Args:
        lexicon: The gold lexicon.
        n: Target sample size; the whole lexicon is returned
            when `n` is None or not smaller than the lexicon.
        seed: Seed for the deterministic draw.

    Returns:
        The sampled segmentations in lexicon order.
    """
    forms = list(lexicon.by_form.values())
    if n is None or n >= len(forms):
        return forms

    # draw the sample, kept in lexicon order
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(forms)), n))
    return [forms[i] for i in indices]
