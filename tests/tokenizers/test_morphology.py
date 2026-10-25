"""Tests for slm4ie/tokenizers/morphology.py."""

import gzip
import json
from pathlib import Path

from slm4ie.tokenizers.corpus import stable_rank
from slm4ie.tokenizers.morphology import (
    MorphemeSegmentation,
    MorphLexicon,
    build_derivational_lexicon,
    build_inflection_lexicon,
    build_morph_lexicon,
    build_morphscore_lexicon,
    load_lexicon,
    merge_lexicons,
    save_lexicon,
    segment_form,
    split_lexemes,
    without_forms,
)

#: Three Sloleks-style entries exercising the alignment paths: a regular
#: feminine noun (clean stem+suffix), a fleeting-vowel masculine noun, and an
#: indeclinable conjunction.
_FAKE_SLOLEKS = [
    {
        "entry_id": "hisa-1",
        "lemma": "hiša",
        "lemma_msd": "Sozei",
        "forms": [
            {"form": "hiša", "msd": "Sozei"},
            {"form": "hiše", "msd": "Sozer"},
            {"form": "hiši", "msd": "Sozed"},
        ],
    },
    {
        "entry_id": "pes-1",
        "lemma": "pes",
        "lemma_msd": "Somei",
        "forms": [
            {"form": "pes", "msd": "Somei"},
            {"form": "psa", "msd": "Somer"},
        ],
    },
    {
        "entry_id": "in-1",
        "lemma": "in",
        "lemma_msd": "Vp",
        "forms": [{"form": "in", "msd": "Vp"}],
    },
]


def _write_fake_sloleks(path: Path) -> Path:
    """Write the fake Sloleks records to a gzipped JSONL file.

    Args:
        path (Path): Destination `.jsonl.gz` path.

    Returns:
        Path: `path`, for chaining.
    """
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in _FAKE_SLOLEKS:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


class TestSegmentForm:
    """Unit tests for the per-form segmentation."""

    def test_clean_stem_suffix_split(self):
        """A regular inflected form splits into stem + suffix."""
        seg = segment_form("hiše", "hiša", "Sozer")
        assert seg.morphemes == ["hiš", "e"]
        assert seg.labels == ["stem", "suffix"]
        assert seg.boundaries() == [3]
        assert seg.is_reliable

    def test_lemma_form_is_single_stem(self):
        """A form equal to its lemma is a single stem with no boundary."""
        seg = segment_form("hiša", "hiša", "Sozei")
        assert seg.morphemes == ["hiša"]
        assert seg.boundaries() == []

    def test_fleeting_vowel_alignment(self):
        """The polglasnik rule recovers the stem for pes -> psa."""
        seg = segment_form("psa", "pes", "Somer")
        assert seg.morphemes == ["ps", "a"]

    def test_consonant_alternation(self):
        """A palatalization alternation keeps the alternant in the stem."""
        seg = segment_form("junače", "junak")
        assert seg.morphemes == ["junač", "e"]

    def test_indeclinable_kept_whole(self):
        """An indeclinable MSD is kept whole and stays reliable."""
        seg = segment_form("in", "in", "Vp")
        assert seg.morphemes == ["in"]
        assert seg.is_reliable

    def test_noun_tag_is_split(self):
        """A noun carries the Slovene tag `S` and is still split, not kept whole."""
        seg = segment_form("mačkah", "mačka", "Sozdm")
        assert seg.labels == ["stem", "suffix"]

    def test_failed_alignment_flagged_unreliable(self):
        """Suppletion (no shared stem) is flagged unreliable."""
        seg = segment_form("sem", "biti", "Gp-spe-n")
        assert seg.labels == ["whole"]
        assert not seg.is_reliable

    def test_empty_form_returns_none(self):
        """An empty form yields no segmentation."""
        assert segment_form("", "hiša") is None


class TestBuildLexicon:
    """Tests for building the lexicon from Sloleks records."""

    def test_inventory_and_indices(self, tmp_path: Path):
        """Forms, inventory, stems, and suffixes are populated."""
        path = _write_fake_sloleks(tmp_path / "sloleks.jsonl.gz")
        lex = build_morph_lexicon(path)

        assert "hiše" in lex.by_form
        assert "ps" in lex.stems
        assert "a" in lex.suffixes
        # 'hiš' stem occurs for hiše and hiši -> count 2.
        assert lex.inventory["hiš"] == 2

    def test_segment_known_and_unknown(self, tmp_path: Path):
        """Segment returns gold morphemes for known forms, [word] otherwise."""
        path = _write_fake_sloleks(tmp_path / "sloleks.jsonl.gz")
        lex = build_morph_lexicon(path)
        assert lex.segment("hiše") == ["hiš", "e"]
        assert lex.segment("nepoznana") == ["nepoznana"]

    def test_save_load_round_trip(self, tmp_path: Path):
        """save_lexicon then load_lexicon reproduces forms and indices."""
        path = _write_fake_sloleks(tmp_path / "sloleks.jsonl.gz")
        lex = build_morph_lexicon(path)

        out = save_lexicon(lex, tmp_path / "lex.jsonl.gz")
        reloaded = load_lexicon(out)

        assert reloaded.by_form.keys() == lex.by_form.keys()
        assert reloaded.inventory == lex.inventory
        assert reloaded.stems == lex.stems
        assert reloaded.suffixes == lex.suffixes


class TestMorphLexicon:
    """Tests for MorphLexicon convenience behavior."""

    def test_empty_lexicon_segments_to_word(self):
        """An empty lexicon returns the word unchanged."""
        assert MorphLexicon().segment("karkoli") == ["karkoli"]


#: Derivational records as emitted by the sloleks_relations lexicon module.
_DERIV = [
    {"lemma": "pisatelj", "morphemes": ["pis", "at", "elj"], "msd": "Som", "verified": True},
    {"lemma": "hiša", "morphemes": ["hiš", "a"], "msd": "Sozei", "verified": False},
]

#: Inflectional Sloleks records: "hiša" appears in both sources.
_SLOLEKS = [
    {
        "lemma": "hiša",
        "lemma_msd": "Sozei",
        "forms": [{"form": "hiša", "msd": "Sozei"}, {"form": "hiše", "msd": "Sozer"}],
    },
]


def _write_jsonl(path: Path, records) -> Path:
    """Write records to a gzipped JSONL file.

    Args:
        path (Path): Destination `.jsonl.gz` path.
        records: Iterable of JSON-serializable mappings.

    Returns:
        Path: `path`, for chaining.
    """
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


class TestBuildDerivationalLexicon:
    """Tests for build_derivational_lexicon."""

    def test_keyed_by_lemma_with_given_boundaries(self, tmp_path: Path):
        """Each lemma maps to its underscore-given morphemes, all reliable."""
        lexicon = build_derivational_lexicon(_write_jsonl(tmp_path / "rel.jsonl.gz", _DERIV))
        assert set(lexicon.by_form) == {"pisatelj", "hiša"}
        seg = lexicon.by_form["pisatelj"]
        assert seg.morphemes == ["pis", "at", "elj"]
        assert seg.labels == ["morph", "morph", "morph"]
        assert seg.is_reliable
        assert seg.verified is True
        assert seg.boundaries() == [3, 5]

    def test_skips_records_without_lemma_or_morphemes(self, tmp_path: Path):
        """Records missing a lemma or morphemes are skipped."""
        bad = [{"lemma": "x"}, {"morphemes": ["a", "b"]}, {"lemma": "ok", "morphemes": ["o", "k"]}]
        lexicon = build_derivational_lexicon(_write_jsonl(tmp_path / "bad.jsonl.gz", bad))
        assert set(lexicon.by_form) == {"ok"}


class TestMergeLexicons:
    """Tests for the backend-table union."""

    def test_richer_split_wins_on_collision(self, tmp_path: Path):
        """A derivational split supersedes a trivial single-stem inflection."""
        deriv = build_derivational_lexicon(_write_jsonl(tmp_path / "rel.jsonl.gz", _DERIV))
        infl = build_morph_lexicon(_write_jsonl(tmp_path / "slo.jsonl.gz", _SLOLEKS))
        # Inflectional "hiša" is the lemma form -> single "stem" piece.
        assert infl.by_form["hiša"].morphemes == ["hiša"]

        merged = merge_lexicons(deriv, infl)
        assert merged.by_form["hiša"].morphemes == ["hiš", "a"]  # deriv wins (2 > 1)
        assert merged.by_form["hiše"].morphemes == ["hiš", "e"]  # infl-only form kept
        assert "pisatelj" in merged.by_form  # deriv-only lemma kept

    def test_inventory_rebuilt_from_merged_forms(self, tmp_path: Path):
        """The merged inventory counts morphemes across the kept forms."""
        deriv = build_derivational_lexicon(_write_jsonl(tmp_path / "rel.jsonl.gz", _DERIV))
        infl = build_morph_lexicon(_write_jsonl(tmp_path / "slo.jsonl.gz", _SLOLEKS))
        merged = merge_lexicons(deriv, infl)
        assert "elj" in merged.inventory
        assert merged.inventory["hiš"] >= 1


class TestVerifiedRoundTrip:
    """Tests that the verified flag survives save/load."""

    def test_verified_persisted(self, tmp_path: Path):
        """save_lexicon then load_lexicon preserves the verified flag."""
        deriv = build_derivational_lexicon(_write_jsonl(tmp_path / "rel.jsonl.gz", _DERIV))
        path = save_lexicon(deriv, tmp_path / "out.jsonl.gz")
        reloaded = load_lexicon(path)
        assert reloaded.by_form["pisatelj"].verified is True
        assert reloaded.by_form["hiša"].verified is False


#: Converted `sloleks_inflection` records: `mačka` (manual) and `mačkah`
#: shared with an automatic lexeme that splits it the same way, plus `pesa`
#: split two ways by two lexemes.
_INFLECTION = [
    {
        "entry_id": "1",
        "lemma": "mačka",
        "status": "manual",
        "forms": [
            {"form": "mačka", "morphemes": ["mačk", "a"], "msds": ["Sozei"]},
            {"form": "mačk", "morphemes": ["mačk"], "msds": ["Sozmr"]},
            {"form": "mačkah", "morphemes": ["mačk", "ah"], "msds": ["Sozdm", "Sozmm"]},
        ],
    },
    {
        "entry_id": "2",
        "lemma": "Mačka",
        "status": "automatic",
        "forms": [{"form": "mačkah", "morphemes": ["mačk", "ah"], "msds": ["Slzmm"]}],
    },
    {
        "entry_id": "3",
        "lemma": "pes",
        "status": "automatic",
        "forms": [{"form": "pesa", "morphemes": ["p", "esa"], "msds": ["Somdi"]}],
    },
    {
        "entry_id": "4",
        "lemma": "peso",
        "status": "manual",
        "forms": [{"form": "pesa", "morphemes": ["pes", "a"], "msds": ["Somer"]}],
    },
]


class TestBuildInflectionLexicon:
    """Tests for the gold read from the Sloleks stem-ending split."""

    def test_reads_stem_and_ending_without_alignment(self, tmp_path: Path):
        """A split form keeps the stem and ending given in the file."""
        lexicon = build_inflection_lexicon(_write_jsonl(tmp_path / "infl.jsonl.gz", _INFLECTION))
        assert lexicon.by_form["mačka"].morphemes == ["mačk", "a"]
        assert lexicon.by_form["mačka"].labels == ["stem", "suffix"]

    def test_zero_ending_form_kept_whole(self, tmp_path: Path):
        """A form without an ending is one stem with no internal boundary."""
        lexicon = build_inflection_lexicon(_write_jsonl(tmp_path / "infl.jsonl.gz", _INFLECTION))
        assert lexicon.by_form["mačk"].boundaries() == []

    def test_form_split_two_ways_dropped(self, tmp_path: Path):
        """A form two lexemes split differently is left out and counted."""
        lexicon = build_inflection_lexicon(_write_jsonl(tmp_path / "infl.jsonl.gz", _INFLECTION))
        assert "pesa" not in lexicon.by_form
        assert lexicon.collisions == 1

    def test_form_manual_only_when_every_lexeme_is(self, tmp_path: Path):
        """A form shared with an automatic lexeme is not verified."""
        lexicon = build_inflection_lexicon(_write_jsonl(tmp_path / "infl.jsonl.gz", _INFLECTION))
        assert lexicon.by_form["mačka"].verified is True
        assert lexicon.by_form["mačkah"].verified is False


#: Two manual nouns sharing the form `mački`, and four automatic verbs, so
#: each stratum splits evenly at a held-out share of one half.
_SPLIT_LEXEMES = [
    {
        "entry_id": "n1",
        "lemma": "mačka",
        "status": "manual",
        "forms": [
            {"form": "mačka", "morphemes": ["mačk", "a"], "msds": ["Sozei"]},
            {"form": "mački", "morphemes": ["mačk", "i"], "msds": ["Sozed"]},
        ],
    },
    {
        "entry_id": "n2",
        "lemma": "maček",
        "status": "manual",
        "forms": [
            {"form": "maček", "morphemes": ["maček"], "msds": ["Somei"]},
            {"form": "mački", "morphemes": ["mačk", "i"], "msds": ["Sommi"]},
        ],
    },
    *[
        {
            "entry_id": f"v{i}",
            "lemma": f"delati{i}",
            "status": "automatic",
            "forms": [{"form": f"dela{i}", "morphemes": [f"dela{i}"], "msds": ["Ggnste"]}],
        }
        for i in range(4)
    ],
]


class TestSplitLexemes:
    """Tests for holding Sloleks lexemes out of the morpheme table."""

    def test_each_stratum_split_exactly(self, tmp_path: Path):
        """Every word class and status holds out round(share) of its lexemes."""
        split = split_lexemes(_write_jsonl(tmp_path / "infl.jsonl.gz", _SPLIT_LEXEMES), 0.5)
        heldout = {(r["word_class"], r["status"]): r["lexemes"] for r in split.counts if r["side"] == "heldout"}
        assert heldout == {("G", "automatic"): 2, ("S", "manual"): 1}

    def test_no_heldout_form_on_table_side(self, tmp_path: Path):
        """A held-out form never appears among the table-side forms."""
        split = split_lexemes(_write_jsonl(tmp_path / "infl.jsonl.gz", _SPLIT_LEXEMES), 0.5)
        assert not split.heldout_forms & split.table_forms

    def test_form_shared_across_sides_dropped(self, tmp_path: Path):
        """A form both nouns carry cannot be held out, since one noun is on the table side."""
        split = split_lexemes(_write_jsonl(tmp_path / "infl.jsonl.gz", _SPLIT_LEXEMES), 0.5)
        assert "mački" not in split.heldout_forms
        assert "mački" in split.table_forms

    def test_split_is_reproducible(self, tmp_path: Path):
        """The same file and share give the same held-out forms."""
        path = _write_jsonl(tmp_path / "infl.jsonl.gz", _SPLIT_LEXEMES)
        assert split_lexemes(path, 0.5).heldout_forms == split_lexemes(path, 0.5).heldout_forms

    def test_without_forms_drops_only_named_forms(self, tmp_path: Path):
        """Filtering a lexicon leaves the other forms and rebuilds the inventory."""
        lexicon = build_inflection_lexicon(_write_jsonl(tmp_path / "infl.jsonl.gz", _INFLECTION))
        kept = without_forms(lexicon, frozenset({"mačka"}))
        assert "mačka" not in kept.by_form
        assert "mačkah" in kept.by_form
        assert "a" not in kept.inventory

    def test_every_heldout_form_has_one_lexeme(self, tmp_path: Path):
        """Each scored form maps to the held-out lexeme it is resampled under."""
        split = split_lexemes(_write_jsonl(tmp_path / "infl.jsonl.gz", _SPLIT_LEXEMES), 0.5)
        assert set(split.lexeme_of) == split.heldout_forms

    def test_form_of_two_heldout_lexemes_goes_to_lower_ranked(self, tmp_path: Path):
        """A form both held-out nouns carry goes to the lexeme its hash ranks lower."""
        split = split_lexemes(_write_jsonl(tmp_path / "infl.jsonl.gz", _SPLIT_LEXEMES), 0.9)
        assert split.lexeme_of["mački"] == min(("n1", "n2"), key=lambda entry: stable_rank(f"mački\t{entry}"))

    def test_lexeme_assignment_ignores_file_order(self, tmp_path: Path):
        """Reversing the lexemes in the file gives the same assignment."""
        forward = split_lexemes(_write_jsonl(tmp_path / "a.jsonl.gz", _SPLIT_LEXEMES), 0.9)
        backward = split_lexemes(_write_jsonl(tmp_path / "b.jsonl.gz", _SPLIT_LEXEMES[::-1]), 0.9)
        assert forward.lexeme_of == backward.lexeme_of

    def test_lexeme_survives_save_and_load(self, tmp_path: Path):
        """save_lexicon then load_lexicon keeps each form's lexeme id."""
        lexicon = MorphLexicon()
        lexicon.by_form["mačka"] = MorphemeSegmentation(
            "mačka", ["mačk", "a"], ["stem", "suffix"], "mačka", lexeme="n1"
        )
        reloaded = load_lexicon(save_lexicon(lexicon, tmp_path / "gold.jsonl.gz"))
        assert reloaded.by_form["mačka"].lexeme == "n1"


def _morphscore_item(form, morphemes, stem_index, *, lemma=None, unique=True):
    """Return one converted MorphScore record; the stem doubles as the lemma unless given."""
    stem = morphemes[stem_index]
    return {
        "form": form,
        "lemma": stem if lemma is None else lemma,
        "stem": stem,
        "morphemes": morphemes,
        "stem_index": stem_index,
        "is_unique": unique,
        "pos": "NOUN",
    }


#: Kept: `večera` and `nevečer` (a prefix). Dropped: a repeat, a stem that
#: differs from its lemma, and a form with a digit.
_MORPHSCORE = [
    _morphscore_item("večera", ["večer", "a"], 0),
    _morphscore_item("nevečer", ["ne", "večer"], 1),
    _morphscore_item("večera", ["večer", "a"], 0, unique=False),
    _morphscore_item("dni", ["d", "ni"], 0, lemma="dan"),
    _morphscore_item("2a", ["2", "a"], 0),
]


class TestBuildMorphscoreLexicon:
    """Tests for the outside gold built from the MorphScore v2 items."""

    def test_default_filters_keep_scored_items(self, tmp_path: Path):
        """Only first occurrences with the stem equal to the lemma and no digit remain."""
        lexicon = build_morphscore_lexicon(_write_jsonl(tmp_path / "ms.jsonl.gz", _MORPHSCORE))
        assert sorted(lexicon.by_form) == ["nevečer", "večera"]

    def test_piece_before_stem_labelled_prefix(self, tmp_path: Path):
        """A piece before the stem is a prefix and the stem keeps its place."""
        lexicon = build_morphscore_lexicon(_write_jsonl(tmp_path / "ms.jsonl.gz", _MORPHSCORE))
        assert lexicon.by_form["nevečer"].labels == ["prefix", "stem"]
