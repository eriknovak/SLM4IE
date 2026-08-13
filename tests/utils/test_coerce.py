"""Tests for the shared value-coercion helpers (slm4ie/utils/coerce.py)."""

import pytest

from slm4ie.utils.coerce import coerce_bool


class TestCoerceBool:
    """Unit tests for `coerce_bool`."""

    @pytest.mark.parametrize("value", [True, 1, "1", "true", "TRUE", " yes ", "y", "t"])
    def test_truthy(self, value: object) -> None:
        """Recognized truthy spellings coerce to True."""
        assert coerce_bool(value) is True

    @pytest.mark.parametrize("value", [False, 0, "0", "false", "FALSE", " no ", "n", "f"])
    def test_falsy(self, value: object) -> None:
        """Recognized falsy spellings coerce to False."""
        assert coerce_bool(value) is False

    @pytest.mark.parametrize("value", [None, "", "maybe", "2", [], {}])
    def test_unrecognized_returns_none(self, value: object) -> None:
        """Unrecognized values return None rather than a coerced bool."""
        assert coerce_bool(value) is None
