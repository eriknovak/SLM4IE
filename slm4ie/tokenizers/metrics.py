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
import random
from collections import Counter
from typing import Counter as CounterType
from typing import Dict, Iterable, List, Optional, Set, Tuple

from slm4ie.tokenizers.base import TokenizerSpec, split_words
from slm4ie.tokenizers.morphology import MorphLexicon

#: Whether a higher value is better for each reported metric.
METRIC_DIRECTIONS: Dict[str, str] = {
    "fertility": "lower",
    "tokens_per_byte": "lower",
    "chars_per_token": "higher",
    "renyi_efficiency": "higher",
    "morph_score_f1": "higher",
    "morph_edit_distance": "lower",
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


def _boundary_positions(spans: List[Tuple[str, int, int]], form_len: int) -> List[int]:
    """Return the sorted character boundary positions implied by token spans.

    Works for any scheme: byte-level tokens that share a character span simply
    contribute the same positions. Includes the endpoints 0 and `form_len`.

    Args:
        spans (List[Tuple[str, int, int]]): `(piece, start, end)` token spans.
        form_len (int): Character length of the form.

    Returns:
        List[int]: Sorted distinct boundary offsets within `[0, form_len]`.
    """
    positions = {0, form_len}
    for _piece, start, end in spans:
        if 0 <= start <= form_len:
            positions.add(start)
        if 0 <= end <= form_len:
            positions.add(end)
    return sorted(positions)


def _token_spans(tokenizer: TokenizerSpec, form: str) -> Optional[List[Tuple[str, int, int]]]:
    """Return the token spans for `form`, or None when they do not tile it.

    Args:
        tokenizer (TokenizerSpec): The tokenizer under evaluation.
        form (str): A surface word form.

    Returns:
        Optional[List[Tuple[str, int, int]]]: `(piece, start, end)` spans whose
            union covers every character of `form`, or None otherwise.
    """
    spans = tokenizer.encode_offsets(form)
    if not spans:
        return None
    covered = set()
    for _piece, start, end in spans:
        covered.update(range(start, end))
    if covered != set(range(len(form))):
        return None
    return spans


def _segments(spans: List[Tuple[str, int, int]], form: str) -> List[str]:
    """Return the surface segments between consecutive token boundaries.

    Args:
        spans (List[Tuple[str, int, int]]): `(piece, start, end)` token spans.
        form (str): The form the spans cover.

    Returns:
        List[str]: Surface substrings delimited by the token boundaries.
    """
    positions = _boundary_positions(spans, len(form))
    return [form[positions[i] : positions[i + 1]] for i in range(len(positions) - 1)]


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
        spans = _token_spans(tokenizer, segmentation.form)
        if spans is None:
            continue
        evaluated += 1
        form_len = len(segmentation.form)
        gold_boundaries = set(segmentation.boundaries())
        pred_boundaries = {p for p in _boundary_positions(spans, form_len) if 0 < p < form_len}
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


def morph_edit_distance(
    tokenizer: TokenizerSpec,
    lexicon: MorphLexicon,
    *,
    max_forms: Optional[int] = None,
) -> float:
    """Return the mean raw segment-edit distance to gold morphemes.

    Implements the MorphBPE paper's Morphological Edit Distance (arXiv
    2502.00894): a dynamic-programming alignment between a word's tokenizer
    segments and its gold morphemes, kept in **raw** (unnormalized) form and
    averaged over forms. The paper gives no closed-form expression, so the
    standard segment-level Levenshtein distance is used. Lower is better; 0
    means the tokenizer's segments equal the gold morphemes for every form.

    Args:
        tokenizer (TokenizerSpec): The tokenizer under evaluation.
        lexicon (MorphLexicon): The gold lexicon.
        max_forms (Optional[int]): Cap on forms evaluated, or None for all.

    Returns:
        float: Mean raw segment-edit distance (>= 0); lower is better.
    """
    total = 0.0
    count = 0
    for segmentation in _reliable_forms(lexicon, max_forms):
        spans = _token_spans(tokenizer, segmentation.form)
        if spans is None:
            continue
        segments = _segments(spans, segmentation.form)
        total += _levenshtein(segments, segmentation.morphemes)
        count += 1
    return total / count if count else 0.0


#: Default cap on sampled word pairs per inverted-index group for consistency.
_DEFAULT_MAX_PAIRS_PER_GROUP = 200


def _build_index(keys_per_form: Dict[str, Set[str]]) -> Dict[str, List[str]]:
    """Invert a form-to-keys mapping into a key-to-forms index.

    Args:
        keys_per_form (Dict[str, Set[str]]): Form to its key set (tokens or
            morphemes).

    Returns:
        Dict[str, List[str]]: Key to the forms that contain it.
    """
    index: Dict[str, List[str]] = {}
    for form, keys in keys_per_form.items():
        for key in keys:
            index.setdefault(key, []).append(form)
    return index


def _grouped_pairs(
    index: Dict[str, List[str]],
    max_pairs_per_group: int,
    rng: random.Random,
) -> Set[Tuple[str, str]]:
    """Sample distinct form pairs that co-occur in each index group.

    Args:
        index (Dict[str, List[str]]): Key to forms (e.g. morpheme to forms).
        max_pairs_per_group (int): Cap on pairs drawn per group.
        rng (random.Random): Seeded RNG for reproducible sampling.

    Returns:
        Set[Tuple[str, str]]: Ordered `(form_a, form_b)` pairs.
    """
    pairs: Set[Tuple[str, str]] = set()
    for members in index.values():
        unique = sorted(set(members))
        if len(unique) < 2:
            continue
        total = len(unique) * (len(unique) - 1) // 2
        if total <= max_pairs_per_group:
            for i in range(len(unique)):
                for j in range(i + 1, len(unique)):
                    pairs.add((unique[i], unique[j]))
        else:
            # Sample distinct pairs deduped against THIS group only. Bounding the
            # loop by the group's own draws (not the cross-group accumulator)
            # guarantees termination: `total > max_pairs_per_group` here, so
            # `max_pairs_per_group` distinct pairs always exist within the group.
            # Dedup across groups still happens via the final set union.
            local: Set[Tuple[str, str]] = set()
            while len(local) < max_pairs_per_group:
                first, second = rng.sample(unique, 2)
                local.add((first, second) if first < second else (second, first))
            pairs |= local
    return pairs


def morph_consistency_score(
    tokenizer: TokenizerSpec,
    lexicon: MorphLexicon,
    *,
    max_forms: Optional[int] = None,
    max_pairs_per_group: int = _DEFAULT_MAX_PAIRS_PER_GROUP,
    seed: int = 0,
) -> float:
    """Return the morpheme/token consistency F1 (MorphBPE, arXiv 2502.00894).

    A pair of words is token-positive when their token sets intersect and
    morpheme-positive when their gold morpheme sets intersect. Over word pairs,
    precision is `P(share morpheme | share token)` and recall is
    `P(share token | share morpheme)`; the score is their F1. The paper samples
    pairs with k-means clustering and bootstrapping; this computes the same
    quantities directly from inverted indices (token to forms, morpheme to
    forms), sampling at most `max_pairs_per_group` pairs per group so it stays
    tractable and reproducible.

    Args:
        tokenizer (TokenizerSpec): The tokenizer under evaluation.
        lexicon (MorphLexicon): The gold lexicon.
        max_forms (Optional[int]): Cap on forms evaluated, or None for all.
        max_pairs_per_group (int): Cap on sampled pairs per index group.
        seed (int): Seed for reproducible pair sampling.

    Returns:
        float: Consistency F1 in `[0, 1]`; higher is better.
    """
    token_sets: Dict[str, Set[str]] = {}
    morph_sets: Dict[str, Set[str]] = {}
    for segmentation in _reliable_forms(lexicon, max_forms):
        form = segmentation.form
        token_sets[form] = set(tokenizer.encode(form))
        morph_sets[form] = set(segmentation.morphemes)

    morph_pairs = _grouped_pairs(_build_index(morph_sets), max_pairs_per_group, random.Random(seed))
    token_pairs = _grouped_pairs(_build_index(token_sets), max_pairs_per_group, random.Random(seed))
    if not morph_pairs or not token_pairs:
        return 0.0

    # Recall: of morpheme-sharing pairs, the share that also share a token.
    recall = sum(bool(token_sets[a] & token_sets[b]) for a, b in morph_pairs) / len(morph_pairs)
    # Precision: of token-sharing pairs, the share that also share a morpheme.
    precision = sum(bool(morph_sets[a] & morph_sets[b]) for a, b in token_pairs) / len(token_pairs)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
