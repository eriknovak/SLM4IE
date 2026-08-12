"""Shared record→Document assembly seam for the extractors.

Four extractors (json, jsonl, coleslaw, huggingface) turn a raw record
into a `Document`. Two steps of that assembly are format-independent and
were previously copied — and had diverged — across all four: probing an
identifier from candidate fields, and projecting the remaining fields into
`Document.metadata`. This module owns both so the policy lives in one place.

The doc_id policy is: walk an ordered list of candidate keys and return the
first present, non-empty value, `str()`-coerced. The metadata policy is:
keep either a whitelist of present fields or every field not in an exclude
set, always dropping `None` values and preserving record iteration order.

Text selection, annotation parsing, and any positional id fallback remain
format-specific and stay in each extractor; only these two shared steps live
here.
"""

from typing import AbstractSet, Any, Callable, Dict, Iterable, Optional

__all__ = ["probe_doc_id", "project_metadata"]


def probe_doc_id(record: Dict[str, Any], keys: Iterable[str]) -> Optional[str]:
    """Return a record's document id from an ordered list of candidate keys.

    Walks `keys` in order and returns the first value that is present and
    non-empty, coerced to `str`. A key whose value is `None` or whose
    `str()` form is empty is skipped.

    Args:
        record (Dict[str, Any]): The raw source record.
        keys (Iterable[str]): Candidate id fields, in priority order.

    Returns:
        Optional[str]: The first present, non-empty candidate value as a
            string, or None when no candidate key yields one.
    """
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        text = str(value)
        if text:
            return text
    return None


def project_metadata(
    record: Dict[str, Any],
    *,
    exclude: AbstractSet[str] = frozenset(),
    whitelist: Optional[Iterable[str]] = None,
    value_transform: Optional[Callable[[Any], Any]] = None,
) -> Dict[str, Any]:
    """Project a record's fields into a `Document.metadata` dict.

    When `whitelist` is given, exactly those listed keys that are present on
    the record are kept, in whitelist order. When `whitelist` is None, every
    key not in `exclude` is kept, in record iteration order. In both modes a
    field whose value is `None` is dropped, and `value_transform` (default
    identity) is applied to each kept value after the None check — so the
    transform is never called on `None`.

    Args:
        record (Dict[str, Any]): The raw source record.
        exclude (AbstractSet[str]): Keys to omit; used only when `whitelist`
            is None. Defaults to the empty set.
        whitelist (Optional[Iterable[str]]): Explicit keys to keep, in order.
            When None, the exclude branch runs instead.
        value_transform (Optional[Callable[[Any], Any]]): Applied to each
            kept value. When None, values pass through unchanged.

    Returns:
        Dict[str, Any]: The projected metadata, order-preserving.
    """
    transform = value_transform if value_transform is not None else (lambda v: v)
    metadata: Dict[str, Any] = {}

    if whitelist is not None:
        for key in whitelist:
            if key not in record:
                continue
            value = record[key]
            if value is None:
                continue
            metadata[key] = transform(value)
        return metadata

    for key, value in record.items():
        if key in exclude:
            continue
        if value is None:
            continue
        metadata[key] = transform(value)
    return metadata
