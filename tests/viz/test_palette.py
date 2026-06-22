"""Tests for the SLM4IE color system."""

from slm4ie.viz import palette


def test_categorical_colors_are_deterministic_and_complete():
    """Every key gets a color, and the mapping is stable across calls."""
    keys = ["bpe", "charbpe", "wordpiece", "unigram", "morphbpe", "morphpiece"]
    first = palette.categorical_colors(keys)
    second = palette.categorical_colors(keys)
    assert first == second
    assert set(first) == set(keys)
    assert all(color in palette.CATEGORICAL for color in first.values())


def test_categorical_colors_pin_by_position():
    """The same position yields the same color regardless of the other keys."""
    mapping = palette.categorical_colors(["a", "b", "c"])
    assert mapping["a"] == palette.CATEGORICAL[0]
    assert mapping["b"] == palette.CATEGORICAL[1]
    assert mapping["c"] == palette.CATEGORICAL[2]


def test_categorical_colors_cycle_past_the_palette():
    """More keys than colors wrap around without dropping any key."""
    keys = [str(i) for i in range(len(palette.CATEGORICAL) + 2)]
    mapping = palette.categorical_colors(keys)
    assert len(mapping) == len(keys)
    assert mapping[keys[len(palette.CATEGORICAL)]] == palette.CATEGORICAL[0]


def test_sequential_runs_from_cream_to_navy():
    """The sequential scale spans the full ramp endpoints over [0, 1]."""
    assert palette.SEQUENTIAL[0] == [0.0, palette.RAMP[0]]
    assert palette.SEQUENTIAL[-1] == [1.0, palette.RAMP[-1]]


def test_diverging_is_blue_to_amber_through_neutral():
    """The diverging scale has good at the top, bad at the bottom, neutral mid."""
    assert palette.DIVERGING[0][0] == 0.0
    assert palette.DIVERGING[1][0] == 0.5
    assert palette.DIVERGING[-1][0] == 1.0
    assert palette.DIVERGING[-1][1] == palette.RAMP[-1]  # navy = good
    assert palette.DIVERGING[0][1] != palette.DIVERGING[-1][1]
