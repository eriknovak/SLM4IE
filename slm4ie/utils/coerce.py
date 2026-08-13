"""Value-coercion helpers shared across data scripts."""

from typing import Any, Optional

#: Truthy string spellings recognized by `coerce_bool`.
_TRUE_TOKENS = frozenset({"true", "1", "yes", "y", "t"})

#: Falsy string spellings recognized by `coerce_bool`.
_FALSE_TOKENS = frozenset({"false", "0", "no", "n", "f"})


def coerce_bool(value: Any) -> Optional[bool]:
    """Coerce common boolean spellings to `bool`.

    Accepts native booleans, numeric 0/1 (as `int` or `float`), and the string
    spellings in `_TRUE_TOKENS` / `_FALSE_TOKENS` (case-insensitive, trimmed).

    Args:
        value: A value pulled from a record.

    Returns:
        `True` or `False` for recognized inputs, None otherwise.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE_TOKENS:
            return True
        if lowered in _FALSE_TOKENS:
            return False
    return None
