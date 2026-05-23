"""Bundled stopword lists keyed by language code.

Each ``<code>.txt`` file in this folder is a UTF-8 plain-text list of
stopwords (one token per line; blank lines and ``#``-prefixed lines are
ignored). Tokens are lowercased on load. New languages are added by
dropping a new ``<code>.txt`` file next to this module — no registry
update is needed.

The loader returns the original file bytes alongside the parsed set so
that callers can fold the file contents into downstream sentinel
hashes without having to re-read the file.
"""

from pathlib import Path
from typing import Set, Tuple


def _parse_stopwords(raw: bytes) -> Set[str]:
    """Parse a stopword-file payload into a lowercased token set.

    Blank lines and lines whose first non-whitespace character is ``#``
    are skipped. Remaining lines are stripped and lowercased before
    being added to the result set.

    Args:
        raw: UTF-8 encoded contents of a stopword file.

    Returns:
        Set of lowercased stopword tokens.
    """
    out: Set[str] = set()
    for line in raw.decode("utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.add(line.lower())
    return out


def load_stopwords(code: str) -> Tuple[Set[str], bytes]:
    """Load the bundled stopword list for a language code.

    Resolves ``<code>.txt`` next to this module, reads its bytes, and
    parses them into a lowercased token set. Both the set and the raw
    file bytes are returned; the bytes are intended for stable sentinel
    hashing by callers that depend on stopword contents.

    Args:
        code: Language code identifying the bundled list (e.g. ``"sl"``).
            Matched against the file stem of ``<code>.txt`` in this
            package directory.

    Returns:
        Tuple ``(stopword set, raw file bytes)``. The bytes are the
        original on-disk contents, not a re-serialization.

    Raises:
        ValueError: If ``<code>.txt`` is not present. The message lists
            available codes discovered via ``glob('*.txt')``.
    """
    folder = Path(__file__).parent
    path = folder / f"{code}.txt"
    if not path.exists():
        available = sorted(p.stem for p in folder.glob("*.txt"))
        raise ValueError(
            f"unknown stopwords language {code!r}; available: {available}"
        )
    raw = path.read_bytes()
    return _parse_stopwords(raw), raw
