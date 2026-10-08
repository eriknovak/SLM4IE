"""Tests for `slm4ie.utils.stats`."""

import pytest

from slm4ie.utils.stats import wilson


def test_empty_cell_has_a_zero_interval() -> None:
    """No items gives (0, 0) rather than a division error."""
    assert wilson(0, 0) == (0.0, 0.0)


def test_interval_brackets_the_share() -> None:
    """The interval contains the observed share and matches the textbook value."""
    low, high = wilson(30, 40)
    assert low < 0.75 < high
    assert low == pytest.approx(0.5981, abs=1e-4)
    assert high == pytest.approx(0.8581, abs=1e-4)


def test_interval_stays_inside_the_unit_range() -> None:
    """All-or-nothing shares keep their bounds inside [0, 1]."""
    assert wilson(0, 5)[0] == 0.0
    assert wilson(5, 5)[1] == 1.0
