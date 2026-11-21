"""Cut a tokenized document into training chunks at sentence boundaries.

A corpus document holds one sentence per line. The document is tokenized once,
so its newlines keep their own tokens, and `split_sentences` cuts the token ids
where a new line starts. `chunk_document` then packs whole sentences into
chunks of at most a given length, cutting a sentence only when it alone is too
long. Both are pure, so the conversion and its tests share them.
"""

from __future__ import annotations

from collections.abc import Sequence

# ===============================
# Chunking functions
# ===============================


def split_sentences(ids: Sequence[int], offsets: Sequence[tuple[int, int]], text: str) -> list[list[int]]:
    """Split a document's token ids into one list per line of its text.

    A token starts a new sentence when it begins at or after the character that
    follows a newline. A token covering the newline itself stays with the line
    it ends.

    Args:
        ids: The document's token ids, without special tokens.
        offsets: The `(start, end)` character span of each token in `text`.
        text: The document text.

    Returns:
        The token ids of each non-empty sentence, in order; together they hold
        every id of `ids`.

    Raises:
        ValueError: If `ids` and `offsets` differ in length.
    """
    if len(ids) != len(offsets):
        raise ValueError(f"{len(ids)} token ids but {len(offsets)} offsets")
    line_starts = [position + 1 for position, char in enumerate(text) if char == "\n"]

    # cut before the first token of each new line
    sentences: list[list[int]] = []
    current: list[int] = []
    next_start = 0
    for token_id, (start, _end) in zip(ids, offsets, strict=True):
        if next_start < len(line_starts) and start >= line_starts[next_start]:
            if current:
                sentences.append(current)
                current = []
            while next_start < len(line_starts) and start >= line_starts[next_start]:
                next_start += 1
        current.append(token_id)
    if current:
        sentences.append(current)
    return sentences


def chunk_document(sentences: Sequence[Sequence[int]], body_len: int) -> list[list[int]]:
    """Pack sentences into chunks of at most `body_len` tokens.

    Whole sentences are added to the current chunk while they fit; a sentence
    that does not fit starts the next chunk. A sentence longer than `body_len`
    is cut into pieces of `body_len` tokens, its last piece starting the next
    chunk. No token is dropped or reordered.

    Args:
        sentences: Token ids of each sentence, in document order.
        body_len: Longest chunk in tokens, before the template's special tokens.

    Returns:
        The chunks, in order; their concatenation equals that of `sentences`.

    Raises:
        ValueError: If `body_len` is not positive.
    """
    if body_len <= 0:
        raise ValueError(f"body_len must be positive, got {body_len}")
    chunks: list[list[int]] = []
    current: list[int] = []

    # add each sentence whole, or cut it when it alone is too long
    for sentence in sentences:
        if not sentence:
            continue
        if len(current) + len(sentence) <= body_len:
            current.extend(sentence)
            continue
        if current:
            chunks.append(current)
        current = list(sentence)
        while len(current) > body_len:
            chunks.append(current[:body_len])
            current = current[body_len:]
    if current:
        chunks.append(current)
    return chunks
