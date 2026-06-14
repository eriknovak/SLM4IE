"""Tests for slm4ie/tokenizers/metrics.py with hand-checked values."""

from typing import Dict, List, Tuple

from slm4ie.tokenizers import metrics
from slm4ie.tokenizers.morphology import MorphemeSegmentation, MorphLexicon, _add_to_lexicon


class _FakeTokenizer:
    """A tokenizer whose segmentation is fixed by a lookup table."""

    name = "fake"

    def __init__(self, table: Dict[str, List[str]]):
        """Store the per-word piece table.

        Args:
            table (Dict[str, List[str]]): Word to its fixed pieces; words not
                present fall back to characters.
        """
        self._table = table

    def encode(self, text: str) -> List[str]:
        """Return the fixed pieces for `text`, else its characters."""
        return self._table.get(text, list(text))

    def encode_ids(self, text: str) -> List[int]:
        """Return positional ids for the pieces of `text`."""
        return list(range(len(self.encode(text))))

    def encode_offsets(self, text: str) -> List[Tuple[str, int, int]]:
        """Return each fixed piece with its character span in `text`."""
        spans: List[Tuple[str, int, int]] = []
        cursor = 0
        for piece in self.encode(text):
            spans.append((piece, cursor, cursor + len(piece)))
            cursor += len(piece)
        return spans

    @property
    def vocab(self) -> Dict[str, int]:
        """Return an empty vocabulary (unused by these tests)."""
        return {}


def _lexicon(entries) -> MorphLexicon:
    """Build a lexicon from `(form, morphemes)` pairs.

    Args:
        entries: Iterable of `(form, morphemes)` tuples.

    Returns:
        MorphLexicon: The populated lexicon.
    """
    lexicon = MorphLexicon()
    for form, morphemes in entries:
        labels = ["stem"] + ["suffix"] * (len(morphemes) - 1)
        _add_to_lexicon(lexicon, MorphemeSegmentation(form, list(morphemes), labels, form))
    return lexicon


class TestRenyi:
    """Tests for Renyi entropy and efficiency."""

    def test_uniform_efficiency_is_one(self):
        """A uniform distribution has Renyi efficiency 1."""
        freqs = {"a": 1, "b": 1, "c": 1, "d": 1}
        assert abs(metrics.renyi_efficiency(freqs, alpha=2.5) - 1.0) < 1e-9

    def test_single_type_efficiency_is_zero(self):
        """A single token type yields efficiency 0."""
        assert metrics.renyi_efficiency({"a": 5}) == 0.0

    def test_skewed_below_uniform(self):
        """A skewed distribution scores below 1."""
        assert metrics.renyi_efficiency({"a": 97, "b": 1, "c": 1, "d": 1}, alpha=2.5) < 1.0


class TestCompressionAndFertility:
    """Tests for the compression and fertility helpers."""

    def test_fertility(self):
        """Fertility is tokens divided by words."""
        assert metrics.fertility(10, 4) == 2.5
        assert metrics.fertility(0, 0) == 0.0

    def test_compression_stats(self):
        """Compression stats compute the expected ratios."""
        stats = metrics.compression_stats(n_tokens=4, n_chars=8, n_bytes=8)
        assert stats["ctc_total"] == 4.0
        assert stats["chars_per_token"] == 2.0
        assert stats["tokens_per_byte"] == 0.5

    def test_corpus_token_stats_single_pass(self):
        """corpus_token_stats accumulates pieces, words, chars, and bytes."""
        tok = _FakeTokenizer({"hiša": ["hiš", "a"], "pes": ["pes"]})
        stats = metrics.corpus_token_stats(tok, ["hiša", "pes"])
        assert stats["n_tokens"] == 3
        assert stats["n_words"] == 2
        assert stats["n_chars"] == len("hiša") + len("pes")


class TestMorphScore:
    """Tests for boundary-based MorphScore."""

    def test_perfect_match_f1_one(self):
        """Matching the gold boundaries exactly gives F1 = 1."""
        lex = _lexicon([("hiše", ["hiš", "e"]), ("hiši", ["hiš", "i"])])
        tok = _FakeTokenizer({"hiše": ["hiš", "e"], "hiši": ["hiš", "i"]})
        result = metrics.morph_score(tok, lex)
        assert result["f1"] == 1.0
        assert result["coverage"] == 1.0

    def test_oversplit_lowers_precision(self):
        """Splitting a monomorphemic word adds a false-positive boundary."""
        lex = _lexicon([("hiše", ["hiš", "e"]), ("hiša", ["hiša"])])
        tok = _FakeTokenizer({"hiše": ["hiš", "e"], "hiša": ["hi", "ša"]})
        result = metrics.morph_score(tok, lex)
        assert result["precision"] < 1.0


class TestMorphEditDistance:
    """Tests for the morpheme edit-distance score."""

    def test_exact_match_is_one(self):
        """Pieces equal to gold morphemes score 1."""
        lex = _lexicon([("hiše", ["hiš", "e"])])
        tok = _FakeTokenizer({"hiše": ["hiš", "e"]})
        assert metrics.morph_edit_distance_score(tok, lex) == 1.0

    def test_mismatch_below_one(self):
        """A different segmentation scores below 1."""
        lex = _lexicon([("hiše", ["hiš", "e"])])
        tok = _FakeTokenizer({"hiše": ["h", "i", "š", "e"]})
        assert metrics.morph_edit_distance_score(tok, lex) < 1.0


class TestMorphConsistency:
    """Tests for cross-form morpheme consistency."""

    def test_consistent_morpheme_scores_one(self):
        """A morpheme tokenized identically across forms scores 1."""
        lex = _lexicon([("hiše", ["hiš", "e"]), ("hiši", ["hiš", "i"])])
        tok = _FakeTokenizer({"hiše": ["hiš", "e"], "hiši": ["hiš", "i"]})
        assert metrics.morph_consistency_score(tok, lex) == 1.0

    def test_inconsistent_morpheme_scores_below_one(self):
        """A morpheme tokenized differently across forms scores below 1."""
        lex = _lexicon([("hiše", ["hiš", "e"]), ("hiši", ["hiš", "i"])])
        # 'hiš' is clean in one form but straddled (hi|ši) in the other.
        tok = _FakeTokenizer({"hiše": ["hiš", "e"], "hiši": ["hi", "ši"]})
        assert metrics.morph_consistency_score(tok, lex) < 1.0
