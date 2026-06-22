"""Tests for slm4ie/tokenizers/stats.py on synthetic per-unit arrays."""

import numpy as np

from slm4ie.tokenizers import stats


def _mean(total: np.ndarray, count: np.ndarray) -> np.ndarray:
    """Combine summed value and count components into a mean.

    Args:
        total (np.ndarray): Summed values over the resample.
        count (np.ndarray): Summed unit counts (an all-ones component).

    Returns:
        np.ndarray: The mean, elementwise over the bootstrap axis.
    """
    return total / count


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Combine two summed components into their ratio.

    Args:
        numerator (np.ndarray): Summed numerator.
        denominator (np.ndarray): Summed denominator.

    Returns:
        np.ndarray: The ratio, elementwise over the bootstrap axis.
    """
    return numerator / denominator


class TestBootstrapCi:
    """Tests for the bootstrap confidence interval primitives."""

    def test_ci_covers_known_mean(self):
        """The 95% CI brackets the true mean of a synthetic sample."""
        rng = np.random.default_rng(0)
        values = rng.normal(loc=5.0, scale=2.0, size=2000)
        ones = np.ones_like(values)
        indices = stats.make_resample_indices(values.size, 2000, seed=7)
        dist = stats.bootstrap_distribution(indices, (values, ones), _mean)
        low, high = stats.percentile_ci(dist, 0.95)

        point = stats.point_estimate((values, ones), _mean)
        assert low < 5.0 < high
        assert low < point < high
        assert abs(point - values.mean()) < 1e-9

    def test_point_estimate_matches_ratio(self):
        """The point estimate of a ratio equals sum/sum of the components."""
        num = np.array([2.0, 4.0, 6.0])
        den = np.array([1.0, 1.0, 2.0])
        assert stats.point_estimate((num, den), _ratio) == 12.0 / 4.0

    def test_empty_units_are_safe(self):
        """Empty per-unit arrays yield a zero point and zero distribution."""
        empty = np.array([], dtype=np.float64)
        indices = stats.make_resample_indices(0, 100, seed=1)
        dist = stats.bootstrap_distribution(indices, (empty, empty), _ratio)
        assert dist.shape == (100,)
        assert np.all(dist == 0.0)
        assert stats.point_estimate((empty, empty), _ratio) == 0.0


class TestPairedDifference:
    """Tests for paired bootstrap differences and the bootstrap p-value."""

    def test_detects_injected_difference(self):
        """A shifted sample yields a difference CI that excludes 0."""
        rng = np.random.default_rng(1)
        base = rng.normal(0.0, 1.0, size=1500)
        shifted = base + 0.8
        ones = np.ones_like(base)
        indices = stats.make_resample_indices(base.size, 2000, seed=3)
        dist_a = stats.bootstrap_distribution(indices, (shifted, ones), _mean)
        dist_b = stats.bootstrap_distribution(indices, (base, ones), _mean)
        diff = dist_a - dist_b

        low, high = stats.percentile_ci(diff, 0.95)
        assert low > 0.0
        assert abs(float(np.median(diff)) - 0.8) < 0.1
        assert stats.bootstrap_p_value(diff) < 0.05

    def test_identical_samples_not_significant(self):
        """Identical inputs give a difference CI spanning 0 and p == 1."""
        values = np.linspace(1.0, 10.0, 400)
        ones = np.ones_like(values)
        indices = stats.make_resample_indices(values.size, 1000, seed=5)
        dist = stats.bootstrap_distribution(indices, (values, ones), _mean)
        diff = dist - dist  # identical tokenizer paired with itself
        low, high = stats.percentile_ci(diff, 0.95)
        assert low == 0.0 and high == 0.0
        assert stats.bootstrap_p_value(diff) == 1.0

    def test_p_value_bounds(self):
        """The bootstrap p-value stays within [0, 1]."""
        rng = np.random.default_rng(2)
        diff = rng.normal(0.3, 1.0, size=500)
        p = stats.bootstrap_p_value(diff)
        assert 0.0 <= p <= 1.0


class TestHolm:
    """Tests for the Holm-Bonferroni correction."""

    def test_monotone_and_at_least_raw(self):
        """Adjusted p-values are non-decreasing in rank and >= raw."""
        raw = [0.001, 0.009, 0.02, 0.04, 0.5]
        adjusted = stats.holm_correction(raw)
        assert all(a >= r - 1e-12 for a, r in zip(adjusted, raw))
        ordered = [adjusted[i] for i in np.argsort(raw)]
        assert all(ordered[i] <= ordered[i + 1] + 1e-12 for i in range(len(ordered) - 1))

    def test_clamped_to_one(self):
        """Large scaled values are clamped to 1.0."""
        assert stats.holm_correction([0.5, 0.6, 0.7])[-1] <= 1.0

    def test_empty(self):
        """An empty family returns an empty list."""
        assert stats.holm_correction([]) == []


class TestCompactLetters:
    """Tests for the compact-letter-display grouping."""

    def test_overlapping_middle_group(self):
        """Items 0-1 and 1-2 non-significant but 0-2 significant overlap on 1."""
        nd = [
            [True, True, False],
            [True, True, True],
            [False, True, True],
        ]
        letters = stats.compact_letters(3, nd)
        assert letters == ["a", "ab", "b"]

    def test_all_different_gets_unique_letters(self):
        """When every pair differs, each item gets its own letter."""
        nd = [[i == j for j in range(3)] for i in range(3)]
        letters = stats.compact_letters(3, nd)
        assert len(set(letters)) == 3

    def test_all_same_share_one_letter(self):
        """When nothing differs, all items share a single letter."""
        nd = [[True] * 3 for _ in range(3)]
        assert stats.compact_letters(3, nd) == ["a", "a", "a"]
