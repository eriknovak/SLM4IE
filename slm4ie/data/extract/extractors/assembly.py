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

The id contract every extractor honours (see `CONTEXT.md`, "Document id"):
`doc_id` is unique within its dataset and stable across re-extraction. It is
the source's native id when that is unique within the dataset, else the
positional id `<unit>:<ordinal>` built by `positional_doc_id`, where the unit
is the raw file's path relative to the dataset dir (`relative_unit`) and the
ordinal the document's 0-based position in that file. The native id, when the
source has one, travels beside it as `Document.native_id`. Extraction
(`slm4ie.data.extract`) asserts the uniqueness after writing.

Text selection and annotation parsing remain format-specific and stay in each
extractor.
"""

from pathlib import Path
from typing import AbstractSet, Any, Callable, Dict, Iterable, Optional

__all__ = ["positional_doc_id", "probe_doc_id", "project_metadata", "relative_unit"]

#: Zero-padding of a positional ordinal. Wide enough for the largest raw
#: file in any shared source; ids stay unique past it, only unaligned.
ORDINAL_WIDTH = 8


def relative_unit(filepath: Path, input_dir: Path) -> str:
    """Return the unit name of a raw file: its path under the dataset dir.

    Args:
        filepath (Path): Input file being parsed.
        input_dir (Path): Dataset root the path is expressed against.

    Returns:
        str: `filepath` relative to `input_dir`, suffix stripped and
            separators normalized to `/`. Falls back to the bare
            filename stem when `filepath` lies outside `input_dir`.
    """
    try:
        rel = filepath.relative_to(input_dir)
    except ValueError:
        rel = Path(filepath.name)
    return rel.with_suffix("").as_posix()


def positional_doc_id(unit: str, ordinal: int, width: int = ORDINAL_WIDTH) -> str:
    """Build the positional document id `<unit>:<ordinal>`.

    Used when a source has no id that is unique within the dataset. The
    ordinal counts every record of the unit, including skipped ones, so
    an id stays pinned to its raw position when filters change.

    Args:
        unit (str): The raw file's unit name, from `relative_unit`.
        ordinal (int): 0-based position of the record within the unit.
        width (int): Zero-padding of the ordinal.

    Returns:
        str: The positional id, e.g. `UradniList/ul-uredbeni:00001234`.
    """
    return f"{unit}:{ordinal:0{width}d}"


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
