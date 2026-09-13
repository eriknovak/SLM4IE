"""Bootstrap confidence intervals and paired significance for tokenizer metrics.

Pure, numpy-only statistical primitives used by the tokenizer analysis pipeline.
Decomposable metrics (fertility, compression ratios, MorphScore F1,
Morph-Edit-Distance) reduce to per-unit sufficient statistics that sum across
the resampling unit (documents for corpus metrics, forms for morph metrics),
so their sampling distribution can be estimated by resampling those units.

The module provides the building blocks only: drawing shared resample indices,
turning per-unit arrays into a bootstrap distribution, percentile confidence
intervals, paired bootstrap differences with a bootstrap p-value,
Holm-Bonferroni correction, and a compact-letter-display grouping. None of the
functions touch the filesystem or know anything about a specific metric; the
caller supplies the per-unit arrays and a `combine` callable that maps summed
components to the scalar metric (vectorized over the bootstrap axis).
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import numpy as np

#: Type of a metric `combine` callable: it receives one summed component per
#: per-unit array (scalars for the point estimate, 1-D arrays over the bootstrap
#: axis for a distribution) and returns the metric, broadcasting elementwise.
CombineFn = Callable[..., np.ndarray]


def make_resample_indices(n_units: int, n_resamples: int, seed: int) -> np.ndarray:
    """Draw a shared set of bootstrap resample indices.

    Every tokenizer compared on the same resampling unit (and vocab size) must
    reuse the identical index matrix so that paired differences are computed on
    the same resamples. The matrix is fully determined by `seed`.

    Args:
        n_units (int): Number of per-unit observations (documents or forms).
        n_resamples (int): Number of bootstrap resamples, B.
        seed (int): Seed for the deterministic generator.

    Returns:
        np.ndarray: An `(n_resamples, n_units)` int array of indices in
            `[0, n_units)`; empty with shape `(n_resamples, 0)` when there are
            no units.
    """
    rng = np.random.default_rng(seed)
    if n_units <= 0:
        return np.empty((n_resamples, 0), dtype=np.int32)
    return rng.integers(0, n_units, size=(n_resamples, n_units), dtype=np.int32)


def point_estimate(components: Sequence[np.ndarray], combine: CombineFn) -> float:
    """Return the metric computed on the full (unresampled) per-unit arrays.

    Args:
        components (Sequence[np.ndarray]): Per-unit sufficient-statistic arrays.
        combine (CombineFn): Maps the summed components to the scalar metric.

    Returns:
        float: The point estimate; 0.0 when the arrays are empty.
    """
    if not len(components[0]):
        return 0.0
    sums = tuple(np.asarray(c, dtype=np.float64).sum() for c in components)
    return float(combine(*sums))


def bootstrap_distribution(
    indices: np.ndarray,
    components: Sequence[np.ndarray],
    combine: CombineFn,
    *,
    chunk: int = 512,
) -> np.ndarray:
    """Return the bootstrap distribution of a decomposable metric.

    For each resample, every component array is summed over the drawn indices
    and the summed components are fed to `combine`. The bootstrap axis is
    processed in row chunks so the temporary `(chunk, n_units)` gather stays
    bounded for large corpora.

    Args:
        indices (np.ndarray): `(B, n_units)` resample indices from
            `make_resample_indices`.
        components (Sequence[np.ndarray]): Per-unit arrays of length `n_units`.
        combine (CombineFn): Maps summed components to the metric, vectorized
            over the bootstrap axis.
        chunk (int): Number of resamples summed per pass.

    Returns:
        np.ndarray: A length-`B` array of resampled metric values; all-zero when
            there are no units.
    """
    n_resamples = indices.shape[0]
    out = np.zeros(n_resamples, dtype=np.float64)
    if indices.shape[1] == 0:
        return out
    arrays = [np.asarray(c, dtype=np.float64) for c in components]
    for start in range(0, n_resamples, chunk):
        block = indices[start : start + chunk]
        sums = tuple(arr[block].sum(axis=1) for arr in arrays)
        out[start : start + block.shape[0]] = np.asarray(combine(*sums), dtype=np.float64)
    return out


def percentile_ci(distribution: np.ndarray, ci_level: float) -> Tuple[float, float]:
    """Return the percentile confidence interval of a bootstrap distribution.

    Args:
        distribution (np.ndarray): Bootstrap values.
        ci_level (float): Central mass to cover, e.g. 0.95.

    Returns:
        Tuple[float, float]: The `(low, high)` percentile bounds.
    """
    tail = (1.0 - ci_level) / 2.0 * 100.0
    low, high = np.percentile(distribution, [tail, 100.0 - tail])
    return float(low), float(high)


def bootstrap_p_value(diff: np.ndarray) -> float:
    """Return a two-sided bootstrap p-value for a paired difference.

    The p-value is `2 * min(P(d <= 0), P(d >= 0))`, clamped to `[0, 1]`. Using
    the closed inequalities makes an all-zero difference (identical tokenizers)
    return 1.0 rather than 0.0.

    Args:
        diff (np.ndarray): Bootstrap distribution of the paired difference
            `metric_A - metric_B`.

    Returns:
        float: The two-sided p-value in `[0, 1]`.
    """
    if not diff.size:
        return 1.0
    p_le = float(np.mean(diff <= 0.0))
    p_ge = float(np.mean(diff >= 0.0))
    return min(1.0, 2.0 * min(p_le, p_ge))


def holm_correction(pvalues: Sequence[float]) -> List[float]:
    """Return Holm-Bonferroni adjusted p-values.

    Sorts the raw p-values ascending, scales the k-th smallest by `m - k`, and
    enforces monotonicity so adjusted values never decrease along that order.
    Each adjusted value is clamped to `[0, 1]`.

    Args:
        pvalues (Sequence[float]): Raw p-values for a family of comparisons.

    Returns:
        List[float]: Adjusted p-values aligned with the input order.
    """
    raw = np.asarray(pvalues, dtype=np.float64)
    m = raw.size
    if m == 0:
        return []
    order = np.argsort(raw, kind="stable")
    adjusted = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * raw[idx])
        adjusted[idx] = min(running, 1.0)
    return [float(v) for v in adjusted]


def compact_letters(n_items: int, not_different: Sequence[Sequence[bool]]) -> List[str]:
    """Return compact-letter-display groups for ranked items.

    Items are assumed pre-sorted best to worst. Two items sharing a letter are
    NOT significantly different. Letters are the maximal runs of consecutive
    items that are pairwise non-significant, the conventional display for ranked
    comparisons.

    Args:
        n_items (int): Number of ranked items.
        not_different (Sequence[Sequence[bool]]): Symmetric matrix where
            `not_different[i][j]` is True when items `i` and `j` are NOT
            significantly different.

    Returns:
        List[str]: One letter group per item, in the input order; an item may
            carry several letters (e.g. `"ab"`).
    """
    if n_items <= 0:
        return []
    nd = [[bool(not_different[i][j]) for j in range(n_items)] for i in range(n_items)]

    intervals = []
    for start in range(n_items):
        end = start
        while end + 1 < n_items and all(nd[k][end + 1] for k in range(start, end + 1)):
            end += 1
        intervals.append((start, end))

    maximal = sorted(
        {
            (s, e)
            for (s, e) in intervals
            if not any(s2 <= s and e <= e2 and (s2, e2) != (s, e) for (s2, e2) in intervals)
        }
    )

    letters = [""] * n_items
    for group_idx, (start, end) in enumerate(maximal):
        letter = chr(ord("a") + group_idx)
        for i in range(start, end + 1):
            letters[i] += letter
    return letters
