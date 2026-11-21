"""Tests for slm4ie/data/pretokenize/chunking.py (sentence splitting and chunk packing)."""

import pytest

from slm4ie.data.pretokenize.chunking import chunk_document, split_sentences


def _flatten(chunks: list[list[int]]) -> list[int]:
    """Concatenate chunks into one id list.

    Args:
        chunks: Token id lists.

    Returns:
        The ids in order.
    """
    return [token for chunk in chunks for token in chunk]


class TestSplitSentences:
    """split_sentences cuts a document's ids where a new line starts."""

    def test_cuts_before_the_first_token_of_each_line(self):
        """Tokens of the second line form the second sentence."""
        text = "ab cd\nef"
        ids, offsets = [1, 2, 3, 4], [(0, 2), (2, 5), (5, 6), (6, 8)]
        assert split_sentences(ids, offsets, text) == [[1, 2, 3], [4]]

    def test_newline_token_stays_with_the_line_it_ends(self):
        """A token spanning the newline belongs to the line before it."""
        text = "ab\ncd"
        ids, offsets = [1, 2, 3], [(0, 2), (2, 3), (3, 5)]
        assert split_sentences(ids, offsets, text) == [[1, 2], [3]]

    def test_blank_lines_give_no_empty_sentence(self):
        """Consecutive newlines do not create empty sentences."""
        text = "ab\n\n\ncd"
        ids, offsets = [1, 2, 3], [(0, 2), (2, 5), (5, 7)]
        assert split_sentences(ids, offsets, text) == [[1, 2], [3]]

    def test_keeps_every_id_in_order(self):
        """The sentences hold every input id, in order."""
        text = "a b\nc d\ne"
        ids, offsets = [1, 2, 3, 4, 5], [(0, 1), (1, 3), (4, 5), (5, 7), (8, 9)]
        assert _flatten(split_sentences(ids, offsets, text)) == ids

    def test_mismatched_offsets_rejected(self):
        """Ids and offsets of different lengths fail."""
        with pytest.raises(ValueError, match="offsets"):
            split_sentences([1, 2], [(0, 1)], "ab")


class TestChunkDocument:
    """chunk_document packs sentences into chunks within the length limit."""

    def test_packs_whole_sentences_while_they_fit(self):
        """Sentences join one chunk until the next would overflow it."""
        assert chunk_document([[1, 2], [3], [4, 5, 6]], body_len=4) == [[1, 2, 3], [4, 5, 6]]

    def test_long_sentence_is_cut_at_token_level(self):
        """A sentence longer than the limit is cut into full pieces, its rest starting the next chunk."""
        assert chunk_document([[1], [2, 3, 4, 5, 6, 7, 8]], body_len=3) == [[1], [2, 3, 4], [5, 6, 7], [8]]

    def test_no_chunk_exceeds_the_limit(self):
        """Every chunk is at most body_len tokens long."""
        sentences = [[i] * length for i, length in enumerate([5, 1, 9, 2, 2, 7, 3])]
        assert all(len(chunk) <= 4 for chunk in chunk_document(sentences, body_len=4))

    def test_no_token_is_lost_or_reordered(self):
        """The chunks concatenate back to the sentences."""
        sentences = [[1, 2, 3], [4], [5, 6, 7, 8, 9, 10], [11, 12]]
        assert _flatten(chunk_document(sentences, body_len=4)) == list(range(1, 13))

    def test_empty_sentences_are_skipped(self):
        """Empty sentences add no empty chunk."""
        assert chunk_document([[], [1], []], body_len=2) == [[1]]

    def test_non_positive_length_rejected(self):
        """A non-positive body length fails."""
        with pytest.raises(ValueError, match="body_len"):
            chunk_document([[1]], body_len=0)
