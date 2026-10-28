"""A small, dependency-free Byte-Pair Encoding core.

Shared by the from-scratch morphological backends (`morph_bpe`, `morph_piece`).
`learn_bpe` trains merges over a bag of pre-segmented chunks using incremental
pair-count updates (subword-nmt style) so a single merge touches only the chunks
that contain the merged pair, keeping large vocab sweeps tractable in pure
Python. `encode_bpe` applies a learned merge table to one chunk.

Morpheme gating lives in the callers: they decide what a "chunk" is (a whole
morpheme, a whole OOV word, ...). This module never crosses a chunk boundary,
so any boundary the caller encodes as a separate chunk is preserved.
"""

from __future__ import annotations

# ===============================
# Constants
# ===============================

#: A symbol pair considered for merging.
Pair = tuple[str, str]

# ===============================
# Pair-count helpers
# ===============================


def _remove_chunk_pairs(
    symbols: list[str],
    freq: int,
    cid: int,
    pair_freq: dict[Pair, int],
    where: dict[Pair, set],
) -> None:
    """Subtract a chunk's adjacent-pair contributions from the indices.

    Args:
        symbols: The chunk's current symbol sequence.
        freq: The chunk's corpus frequency.
        cid: The chunk's index.
        pair_freq: Pair-frequency table (mutated).
        where: Pair-to-chunk index (mutated).
    """
    for pair in zip(symbols, symbols[1:]):
        remaining = pair_freq.get(pair, 0) - freq
        if remaining <= 0:
            pair_freq.pop(pair, None)
        else:
            pair_freq[pair] = remaining
        holders = where.get(pair)
        if holders is not None:
            holders.discard(cid)
            if not holders:
                where.pop(pair, None)


def _add_chunk_pairs(
    symbols: list[str],
    freq: int,
    cid: int,
    pair_freq: dict[Pair, int],
    where: dict[Pair, set],
) -> None:
    """Add a chunk's adjacent-pair contributions to the indices.

    Args:
        symbols: The chunk's new symbol sequence.
        freq: The chunk's corpus frequency.
        cid: The chunk's index.
        pair_freq: Pair-frequency table (mutated).
        where: Pair-to-chunk index (mutated).
    """
    for pair in zip(symbols, symbols[1:]):
        pair_freq[pair] = pair_freq.get(pair, 0) + freq
        where.setdefault(pair, set()).add(cid)


def _merge_symbols(symbols: list[str], pair: Pair) -> list[str]:
    """Merge all non-overlapping occurrences of `pair` in `symbols`.

    Args:
        symbols: Symbol sequence.
        pair: Adjacent symbol pair to merge.

    Returns:
        A new sequence with the pair merged.
    """
    merged_symbol = pair[0] + pair[1]
    result: list[str] = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
            result.append(merged_symbol)
            i += 2
        else:
            result.append(symbols[i])
            i += 1
    return result


# ===============================
# Learning and encoding
# ===============================


def learn_bpe(chunk_freqs: dict[str, int], target_size: int) -> tuple[list[str], list[Pair]]:
    """Learn a BPE merge table over pre-segmented chunks.

    Each chunk is exploded into characters and merges are learned only within
    chunks, so any boundary the caller expressed by splitting into separate
    chunks is never crossed. Merge selection is deterministic: the highest
    frequency pair, ties broken by pair string.

    Args:
        chunk_freqs: Chunk string to corpus frequency.
        target_size: Desired number of tokens (initial characters plus
            learned merges). No merges are produced once this is reached.

    Returns:
        The ordered token list (sorted base characters followed by merged
        tokens in merge order, de-duplicated) and the ordered list of learned
        merges.
    """
    # explode the chunks into characters
    chunk_symbols: list[list[str]] = [list(chunk) for chunk in chunk_freqs]
    freqs: list[int] = list(chunk_freqs.values())

    # start from the base characters
    base_chars = sorted({ch for chunk in chunk_freqs for ch in chunk})
    tokens: list[str] = list(base_chars)
    seen = set(tokens)

    # count the adjacent pairs
    pair_freq: dict[Pair, int] = {}
    where: dict[Pair, set] = {}
    for cid, symbols in enumerate(chunk_symbols):
        _add_chunk_pairs(symbols, freqs[cid], cid, pair_freq, where)

    # merge the best pair until the target size is reached
    merges: list[Pair] = []
    while len(tokens) < target_size and pair_freq:
        best = max(pair_freq, key=lambda p: (pair_freq[p], p))
        merged_symbol = best[0] + best[1]
        for cid in list(where.get(best, ())):
            symbols = chunk_symbols[cid]
            freq = freqs[cid]
            _remove_chunk_pairs(symbols, freq, cid, pair_freq, where)
            new_symbols = _merge_symbols(symbols, best)
            chunk_symbols[cid] = new_symbols
            _add_chunk_pairs(new_symbols, freq, cid, pair_freq, where)
        pair_freq.pop(best, None)
        where.pop(best, None)
        merges.append(best)
        if merged_symbol not in seen:
            tokens.append(merged_symbol)
            seen.add(merged_symbol)

    return tokens, merges


def merge_ranks(merges: list[Pair]) -> dict[Pair, int]:
    """Map each merge to its priority rank (lower applies first).

    Args:
        merges: Ordered learned merges.

    Returns:
        Pair to rank.
    """
    return {pair: rank for rank, pair in enumerate(merges)}


def encode_bpe(chunk: str, ranks: dict[Pair, int]) -> list[str]:
    """Apply a learned merge table to a single chunk.

    Args:
        chunk: The text to encode (a morpheme or whole word).
        ranks: Merge ranks from `merge_ranks`.

    Returns:
        The BPE pieces of `chunk`.
    """
    if not chunk:
        return []

    # merge the lowest-rank pair until none applies
    symbols = list(chunk)
    while len(symbols) > 1:
        best_rank = None
        best_pair: Pair = ("", "")
        for pair in zip(symbols, symbols[1:]):
            rank = ranks.get(pair)
            if rank is not None and (best_rank is None or rank < best_rank):
                best_rank = rank
                best_pair = pair
        if best_rank is None:
            break
        symbols = _merge_symbols(symbols, best_pair)
    return symbols
