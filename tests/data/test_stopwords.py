"""Tests for slm4ie.data.stopwords loader and parser."""

from pathlib import Path

import pytest

from slm4ie.data.stopwords import _parse_stopwords, load_stopwords


def test_load_sl_returns_set_and_bytes() -> None:
    """Loading the bundled `sl` list returns a populated set and raw bytes."""
    tokens, raw = load_stopwords("sl")

    assert isinstance(tokens, set)
    assert len(tokens) > 0
    assert all(isinstance(t, str) for t in tokens)
    assert all(t == t.lower() for t in tokens)

    assert isinstance(raw, bytes)
    assert len(raw) > 0

    on_disk = (Path(__file__).resolve().parents[2] / "slm4ie" / "data" / "stopwords" / "sl.txt").read_bytes()
    assert raw == on_disk


def test_unknown_code_raises_with_available() -> None:
    """Unknown codes raise ValueError listing available codes."""
    with pytest.raises(ValueError, match="'sl'"):
        load_stopwords("xx")


def test_comments_and_blanks_skipped() -> None:
    """Blank lines and `#` comment lines are stripped during parsing."""
    raw = b"# header comment\n\nin\nali\n   \n# another\npa\n"

    tokens = _parse_stopwords(raw)

    assert tokens == {"in", "ali", "pa"}


def test_tokens_lowercased() -> None:
    """Mixed-case tokens are normalized to lowercase."""
    raw = b"MixedCase\nUPPER\nlower\n"

    tokens = _parse_stopwords(raw)

    assert tokens == {"mixedcase", "upper", "lower"}
