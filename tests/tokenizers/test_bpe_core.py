"""Tests for slm4ie/tokenizers/bpe_core.py."""

from slm4ie.tokenizers.bpe_core import encode_bpe, learn_bpe, merge_ranks


class TestLearnBpe:
    """Tests for learn_bpe merge learning."""

    def test_learns_frequent_pair(self):
        """The most frequent adjacent pair is merged first."""
        tokens, merges = learn_bpe({"ab": 10, "ac": 1}, target_size=10)
        assert ("a", "b") == merges[0]
        assert "ab" in tokens

    def test_base_characters_present(self):
        """All base characters appear in the token list."""
        tokens, _ = learn_bpe({"abc": 5}, target_size=3)
        assert set("abc").issubset(set(tokens))

    def test_no_merges_below_char_budget(self):
        """A budget at or below the character count produces no merges."""
        tokens, merges = learn_bpe({"abc": 5}, target_size=3)
        assert merges == []
        assert sorted(tokens) == ["a", "b", "c"]

    def test_does_not_cross_chunk_boundary(self):
        """Pairs spanning two separate chunks are never learned."""
        # 'ab' and 'cd' are separate chunks; the ('b','c') pair never co-occurs.
        _tokens, merges = learn_bpe({"ab": 100, "cd": 100}, target_size=50)
        assert ("b", "c") not in merges

    def test_deterministic(self):
        """Training twice yields identical merges."""
        first = learn_bpe({"banana": 3, "ananas": 2}, target_size=12)
        second = learn_bpe({"banana": 3, "ananas": 2}, target_size=12)
        assert first == second


class TestEncodeBpe:
    """Tests for encode_bpe application."""

    def test_applies_merges(self):
        """Learned merges are applied when encoding."""
        _tokens, merges = learn_bpe({"ab": 10}, target_size=10)
        ranks = merge_ranks(merges)
        assert encode_bpe("ab", ranks) == ["ab"]

    def test_empty_chunk(self):
        """Encoding an empty chunk yields no pieces."""
        assert encode_bpe("", {}) == []

    def test_unmergeable_falls_back_to_chars(self):
        """Without applicable merges, encoding returns characters."""
        assert encode_bpe("xy", {}) == ["x", "y"]
