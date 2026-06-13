"""Tokenizer evaluation metrics for the Slovenian comparison.

Six metrics determine how compact and how morphologically appropriate a
tokenizer is for Slovenian. Corpus metrics (fertility, CTC compression, Renyi
efficiency) are computed over word tokens from the held-out evaluation sample.
Morphological metrics (MorphScore, Morph-Edit-Distance, Morph-Consistency) are
computed against the Sloleks-derived silver-gold lexicon and should be read as
relative comparators across tokenizers, not absolute morphological accuracy.

Direction of each metric (higher/lower is better) is recorded in
`METRIC_DIRECTIONS` and surfaced in the comparison report.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Counter as CounterType
from typing import Dict, Iterable, List, Optional, Tuple

from slm4ie.tokenizers.base import TokenizerSpec, clean_piece, split_words
from slm4ie.tokenizers.morphology import MorphLexicon

#: Whether a higher value is better for each reported metric.
METRIC_DIRECTIONS: Dict[str, str] = {
    "fertility": "lower",
    "tokens_per_byte": "lower",
    "chars_per_token": "higher",
    "renyi_efficiency": "higher",
    "morph_score_f1": "higher",
    "morph_edit_score": "higher",
    "morph_consistency": "higher",
}


def iter_words(text: str) -> List[str]:
    """Return the word tokens (alphabetic, no bare punctuation) of `text`.

    Args:
        text (str): Input text.

    Returns:
        List[str]: Tokens containing at least one alphabetic character.
    """
    return [token for token in split_words(text) if any(ch.isalpha() for ch in token)]


def corpus_token_stats(tokenizer: TokenizerSpec, words: Iterable[str]) -> Dict[str, object]:
    """Encode word tokens once, accumulating the stats every corpus metric needs.

    Args:
        tokenizer (TokenizerSpec): The tokenizer under evaluation.
        words (Iterable[str]): Word tokens from the evaluation sample.

    Returns:
        Dict[str, object]: Keys `freqs` (Counter of pieces), `n_tokens`,
            `n_words`, `n_chars`, and `n_bytes`.
    """
    freqs: CounterType[str] = Counter()
    n_tokens = 0
    n_words = 0
    n_chars = 0
    n_bytes = 0
    for word in words:
        pieces = tokenizer.encode(word)
        freqs.update(pieces)
        n_tokens += len(pieces)
        n_words += 1
        n_chars += len(word)
        n_bytes += len(word.encode("utf-8"))
    return {
        "freqs": freqs,
        "n_tokens": n_tokens,
        "n_words": n_words,
        "n_chars": n_chars,
        "n_bytes": n_bytes,
    }


def fertility(n_tokens: int, n_words: int) -> float:
    """Return mean subword tokens per word.

    Args:
        n_tokens (int): Total pieces emitted over the corpus.
        n_words (int): Total word tokens.

    Returns:
        float: Tokens per word; 0.0 when there are no words. Lower is better.
    """
    return n_tokens / n_words if n_words else 0.0


def compression_stats(n_tokens: int, n_chars: int, n_bytes: int) -> Dict[str, float]:
    """Return Corpus-Token-Count compression statistics.

    Args:
        n_tokens (int): Total pieces emitted over the corpus.
        n_chars (int): Total characters in the corpus words.
        n_bytes (int): Total UTF-8 bytes in the corpus words.

    Returns:
        Dict[str, float]: `ctc_total` (raw token count, lower is better),
            `tokens_per_byte` (lower is better), and `chars_per_token`
            (higher is better).
    """
    return {
        "ctc_total": float(n_tokens),
        "tokens_per_byte": n_tokens / n_bytes if n_bytes else 0.0,
        "chars_per_token": n_chars / n_tokens if n_tokens else 0.0,
    }


def _logsumexp(values: List[float]) -> float:
    """Return log(sum(exp(values))) computed stably.

    Args:
        values (List[float]): Log-domain values.

    Returns:
        float: The log-sum-exp, or -inf for an empty input.
    """
    if not values:
        return float("-inf")
    peak = max(values)
    if peak == float("-inf"):
        return float("-inf")
    return peak + math.log(sum(math.exp(v - peak) for v in values))


def renyi_entropy(freqs: Dict[str, int], alpha: float = 2.5) -> float:
    """Return the Renyi entropy of a token-frequency distribution (nats).

    Args:
        freqs (Dict[str, int]): Token to count.
        alpha (float): Renyi order; `alpha == 1` falls back to Shannon entropy.

    Returns:
        float: The Renyi entropy in nats, 0.0 for an empty distribution.
    """
    total = sum(freqs.values())
    if total <= 0:
        return 0.0
    probs = [count / total for count in freqs.values() if count > 0]
    if alpha == 1.0:
        return -sum(p * math.log(p) for p in probs)
    log_sum = _logsumexp([alpha * math.log(p) for p in probs])
    return log_sum / (1.0 - alpha)


def renyi_efficiency(freqs: Dict[str, int], alpha: float = 2.5) -> float:
    """Return the Renyi efficiency of a token-frequency distribution.

    Efficiency normalizes Renyi entropy by `log(V_used)`, the log of the number
    of distinct observed token types (Zouhar et al., 2023). It lies in `[0, 1]`;
    higher is better and a uniform distribution scores 1.

    Args:
        freqs (Dict[str, int]): Token to count.
        alpha (float): Renyi order.

    Returns:
        float: Efficiency in `[0, 1]`; 0.0 when fewer than two types occur.
    """
    vocab_used = sum(1 for count in freqs.values() if count > 0)
    if vocab_used <= 1:
        return 0.0
    efficiency = renyi_entropy(freqs, alpha) / math.log(vocab_used)
    # Clamp tiny floating-point overshoot; efficiency is defined on [0, 1].
    return min(1.0, max(0.0, efficiency))


def _piece_offsets(pieces: List[str]) -> List[int]:
    """Return cumulative character offsets at every piece boundary.

    Args:
        pieces (List[str]): Cleaned pieces of a single word.

    Returns:
        List[int]: Offsets `[0, len(p0), len(p0)+len(p1), ...]`.
    """
    offsets = [0]
    for piece in pieces:
        offsets.append(offsets[-1] + len(piece))
    return offsets


def _encoded_pieces(tokenizer: TokenizerSpec, form: str) -> Optional[List[str]]:
    """Encode `form` and return cleaned pieces, or None if they do not align.

    Args:
        tokenizer (TokenizerSpec): The tokenizer under evaluation.
        form (str): A surface word form.

    Returns:
        Optional[List[str]]: Cleaned pieces whose concatenation equals `form`,
            or None when reconstruction fails (e.g. an unknown character).
    """
    pieces = [clean_piece(p) for p in tokenizer.encode(form)]
    if "".join(pieces) != form:
        return None
    return pieces


def _reliable_forms(lexicon: MorphLexicon, max_forms: Optional[int]):
    """Yield reliable segmentations, optionally truncated for speed.

    Args:
        lexicon (MorphLexicon): The gold lexicon.
        max_forms (Optional[int]): Cap on the number of forms, or None.

    Yields:
        MorphemeSegmentation: Each (optionally truncated) gold segmentation.
    """
    for index, segmentation in enumerate(lexicon.by_form.values()):
        if max_forms is not None and index >= max_forms:
            return
        yield segmentation


def morph_score(
    tokenizer: TokenizerSpec,
    lexicon: MorphLexicon,
    *,
    max_forms: Optional[int] = None,
) -> Dict[str, float]:
    """Return micro-averaged morpheme-boundary precision, recall, and F1.

    A predicted boundary is correct when it coincides with a gold morpheme
    boundary. Counts are micro-aggregated over all reliable forms, so
    over-segmenting monomorphemic words lowers precision.

    Args:
        tokenizer (TokenizerSpec): The tokenizer under evaluation.
        lexicon (MorphLexicon): The gold lexicon.
        max_forms (Optional[int]): Cap on forms evaluated, or None for all.

    Returns:
        Dict[str, float]: `precision`, `recall`, `f1` in `[0, 1]`, plus
            `coverage` (fraction of forms whose pieces aligned).
    """
    true_positive = 0
    predicted = 0
    gold = 0
    evaluated = 0
    total = 0
    for segmentation in _reliable_forms(lexicon, max_forms):
        total += 1
        pieces = _encoded_pieces(tokenizer, segmentation.form)
        if pieces is None:
            continue
        evaluated += 1
        gold_boundaries = set(segmentation.boundaries())
        pred_boundaries = set(_piece_offsets(pieces)[1:-1])
        true_positive += len(pred_boundaries & gold_boundaries)
        predicted += len(pred_boundaries)
        gold += len(gold_boundaries)

    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "coverage": evaluated / total if total else 0.0,
    }


def _levenshtein(left: List[str], right: List[str]) -> int:
    """Return the segment-level Levenshtein distance between two sequences.

    Args:
        left (List[str]): First sequence of segments.
        right (List[str]): Second sequence of segments.

    Returns:
        int: Minimum single-segment insertions, deletions, and substitutions.
    """
    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, start=1):
        current = [i]
        for j, right_item in enumerate(right, start=1):
            cost = 0 if left_item == right_item else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def morph_edit_distance_score(
    tokenizer: TokenizerSpec,
    lexicon: MorphLexicon,
    *,
    max_forms: Optional[int] = None,
) -> float:
    """Return the mean normalized segment-edit similarity to gold morphemes.

    For each reliable form, the segment-level Levenshtein distance between the
    tokenizer's pieces and the gold morphemes is normalized by the longer
    sequence; the score is `1 - mean(distance)`.

    Args:
        tokenizer (TokenizerSpec): The tokenizer under evaluation.
        lexicon (MorphLexicon): The gold lexicon.
        max_forms (Optional[int]): Cap on forms evaluated, or None for all.

    Returns:
        float: Similarity in `[0, 1]`; higher is better.
    """
    total = 0.0
    count = 0
    for segmentation in _reliable_forms(lexicon, max_forms):
        pieces = _encoded_pieces(tokenizer, segmentation.form)
        if pieces is None:
            continue
        gold = segmentation.morphemes
        denominator = max(len(pieces), len(gold))
        if denominator == 0:
            continue
        distance = _levenshtein(pieces, gold) / denominator
        total += 1.0 - distance
        count += 1
    return total / count if count else 0.0


def _covering_pieces(
    pieces: List[str],
    offsets: List[int],
    start: int,
    end: int,
) -> Optional[Tuple[str, ...]]:
    """Return the pieces covering `[start, end)`, or None if a piece straddles.

    Args:
        pieces (List[str]): Cleaned pieces of a word.
        offsets (List[int]): Cumulative piece offsets from `_piece_offsets`.
        start (int): Morpheme start offset.
        end (int): Morpheme end offset.

    Returns:
        Optional[Tuple[str, ...]]: The contiguous pieces inside the span, or
            None when a piece crosses the span boundary.
    """
    boundary_set = set(offsets)
    if start not in boundary_set or end not in boundary_set:
        return None
    return tuple(piece for piece, lo in zip(pieces, offsets) if lo >= start and lo + len(piece) <= end)


def morph_consistency_score(
    tokenizer: TokenizerSpec,
    lexicon: MorphLexicon,
    *,
    max_forms: Optional[int] = None,
) -> float:
    """Return how consistently each morpheme is tokenized across forms.

    For every morpheme appearing in at least two reliable forms, the score is
    the share of forms in which the morpheme's character span is tokenized the
    same way (a straddling piece counts as its own, distinct outcome). The
    metric is the macro-average over qualifying morphemes.

    Args:
        tokenizer (TokenizerSpec): The tokenizer under evaluation.
        lexicon (MorphLexicon): The gold lexicon.
        max_forms (Optional[int]): Cap on forms evaluated, or None for all.

    Returns:
        float: Consistency in `[0, 1]`; higher is better.
    """
    outcomes: Dict[str, CounterType[Optional[Tuple[str, ...]]]] = {}
    for segmentation in _reliable_forms(lexicon, max_forms):
        pieces = _encoded_pieces(tokenizer, segmentation.form)
        if pieces is None:
            continue
        offsets = _piece_offsets(pieces)
        cursor = 0
        for morpheme in segmentation.morphemes:
            start, end = cursor, cursor + len(morpheme)
            cursor = end
            covering = _covering_pieces(pieces, offsets, start, end)
            outcomes.setdefault(morpheme, Counter())[covering] += 1

    scores = [
        max(counter.values()) / sum(counter.values()) for counter in outcomes.values() if sum(counter.values()) >= 2
    ]
    return sum(scores) / len(scores) if scores else 0.0
