"""Tests for the derivational morpheme lexicon and the union helper."""

import gzip
import json
from pathlib import Path

from slm4ie.tokenizers.morphology import (
    build_derivational_lexicon,
    build_morph_lexicon,
    load_lexicon,
    merge_lexicons,
    save_lexicon,
)

#: Derivational records as emitted by the sloleks_relations reader.
_DERIV = [
    {"lemma": "pisatelj", "morphemes": ["pis", "at", "elj"], "msd": "Som", "verified": True},
    {"lemma": "hiša", "morphemes": ["hiš", "a"], "msd": "Sozei", "verified": False},
]

#: Inflectional Sloleks records: "hiša" appears in both sources.
_SLOLEKS = [
    {
        "lemma": "hiša",
        "lemma_msd": "Ncfsn",
        "forms": [{"form": "hiša", "msd": "Ncfsn"}, {"form": "hiše", "msd": "Ncfsg"}],
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
