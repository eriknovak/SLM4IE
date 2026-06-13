"""Tests for slm4ie/tokenizers/morphology.py."""

import gzip
import json
from pathlib import Path

from slm4ie.tokenizers.morphology import (
    MorphLexicon,
    build_morph_lexicon,
    load_lexicon,
    save_lexicon,
    segment_form,
)

#: Three Sloleks-style entries exercising the alignment paths: a regular
#: feminine noun (clean stem+suffix), a fleeting-vowel masculine noun, and an
#: indeclinable conjunction.
_FAKE_SLOLEKS = [
    {
        "entry_id": "hisa-1",
        "lemma": "hiša",
        "lemma_msd": "Ncfsn",
        "forms": [
            {"form": "hiša", "msd": "Ncfsn"},
            {"form": "hiše", "msd": "Ncfsg"},
            {"form": "hiši", "msd": "Ncfsd"},
        ],
    },
    {
        "entry_id": "pes-1",
        "lemma": "pes",
        "lemma_msd": "Ncmsn",
        "forms": [
            {"form": "pes", "msd": "Ncmsn"},
            {"form": "psa", "msd": "Ncmsg"},
        ],
    },
    {
        "entry_id": "in-1",
        "lemma": "in",
        "lemma_msd": "Cc",
        "forms": [{"form": "in", "msd": "Cc"}],
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
        seg = segment_form("hiše", "hiša", "Ncfsg")
        assert seg.morphemes == ["hiš", "e"]
        assert seg.labels == ["stem", "suffix"]
        assert seg.boundaries() == [3]
        assert seg.is_reliable

    def test_lemma_form_is_single_stem(self):
        """A form equal to its lemma is a single stem with no boundary."""
        seg = segment_form("hiša", "hiša", "Ncfsn")
        assert seg.morphemes == ["hiša"]
        assert seg.boundaries() == []

    def test_fleeting_vowel_alignment(self):
        """The polglasnik rule recovers the stem for pes -> psa."""
        seg = segment_form("psa", "pes", "Ncmsg")
        assert seg.morphemes == ["ps", "a"]

    def test_consonant_alternation(self):
        """A palatalization alternation keeps the alternant in the stem."""
        seg = segment_form("junače", "junak")
        assert seg.morphemes == ["junač", "e"]

    def test_indeclinable_kept_whole(self):
        """An indeclinable MSD is kept whole and stays reliable."""
        seg = segment_form("in", "in", "Cc")
        assert seg.morphemes == ["in"]
        assert seg.is_reliable

    def test_failed_alignment_flagged_unreliable(self):
        """Suppletion (no shared stem) is flagged unreliable."""
        seg = segment_form("sem", "biti", "Va")
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
        """segment returns gold morphemes for known forms, [word] otherwise."""
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
