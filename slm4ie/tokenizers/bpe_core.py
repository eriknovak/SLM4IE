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

from typing import Dict, List, Tuple

#: A symbol pair considered for merging.
Pair = Tuple[str, str]


def _remove_chunk_pairs(
    symbols: List[str],
    freq: int,
    cid: int,
    pair_freq: Dict[Pair, int],
    where: Dict[Pair, set],
) -> None:
    """Subtract a chunk's adjacent-pair contributions from the indices.

    Args:
        symbols (List[str]): The chunk's current symbol sequence.
        freq (int): The chunk's corpus frequency.
        cid (int): The chunk's index.
        pair_freq (Dict[Pair, int]): Pair-frequency table (mutated).
        where (Dict[Pair, set]): Pair-to-chunk index (mutated).
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
    symbols: List[str],
    freq: int,
    cid: int,
    pair_freq: Dict[Pair, int],
    where: Dict[Pair, set],
) -> None:
    """Add a chunk's adjacent-pair contributions to the indices.

    Args:
        symbols (List[str]): The chunk's new symbol sequence.
        freq (int): The chunk's corpus frequency.
        cid (int): The chunk's index.
        pair_freq (Dict[Pair, int]): Pair-frequency table (mutated).
        where (Dict[Pair, set]): Pair-to-chunk index (mutated).
    """
    for pair in zip(symbols, symbols[1:]):
        pair_freq[pair] = pair_freq.get(pair, 0) + freq
        where.setdefault(pair, set()).add(cid)


def _merge_symbols(symbols: List[str], pair: Pair) -> List[str]:
    """Merge all non-overlapping occurrences of `pair` in `symbols`.

    Args:
        symbols (List[str]): Symbol sequence.
        pair (Pair): Adjacent symbol pair to merge.

    Returns:
        List[str]: A new sequence with the pair merged.
    """
    merged_symbol = pair[0] + pair[1]
    result: List[str] = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
            result.append(merged_symbol)
            i += 2
        else:
            result.append(symbols[i])
            i += 1
    return result


def learn_bpe(chunk_freqs: Dict[str, int], target_size: int) -> Tuple[List[str], List[Pair]]:
    """Learn a BPE merge table over pre-segmented chunks.

    Each chunk is exploded into characters and merges are learned only within
    chunks, so any boundary the caller expressed by splitting into separate
    chunks is never crossed. Merge selection is deterministic: the highest
    frequency pair, ties broken by pair string.

    Args:
        chunk_freqs (Dict[str, int]): Chunk string to corpus frequency.
        target_size (int): Desired number of tokens (initial characters plus
            learned merges). No merges are produced once this is reached.

    Returns:
        Tuple[List[str], List[Pair]]: The ordered token list (sorted base
            characters followed by merged tokens in merge order, de-duplicated)
            and the ordered list of learned merges.
    """
    chunk_symbols: List[List[str]] = [list(chunk) for chunk in chunk_freqs]
    freqs: List[int] = list(chunk_freqs.values())

    base_chars = sorted({ch for chunk in chunk_freqs for ch in chunk})
    tokens: List[str] = list(base_chars)
    seen = set(tokens)

    pair_freq: Dict[Pair, int] = {}
    where: Dict[Pair, set] = {}
    for cid, symbols in enumerate(chunk_symbols):
        _add_chunk_pairs(symbols, freqs[cid], cid, pair_freq, where)

    merges: List[Pair] = []
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


def merge_ranks(merges: List[Pair]) -> Dict[Pair, int]:
    """Map each merge to its priority rank (lower applies first).

    Args:
        merges (List[Pair]): Ordered learned merges.

    Returns:
        Dict[Pair, int]: Pair to rank.
    """
    return {pair: rank for rank, pair in enumerate(merges)}


def encode_bpe(chunk: str, ranks: Dict[Pair, int]) -> List[str]:
    """Apply a learned merge table to a single chunk.

    Args:
        chunk (str): The text to encode (a morpheme or whole word).
        ranks (Dict[Pair, int]): Merge ranks from `merge_ranks`.

    Returns:
        List[str]: The BPE pieces of `chunk`.
    """
    if not chunk:
        return []
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
