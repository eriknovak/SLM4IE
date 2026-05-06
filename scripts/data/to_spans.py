"""Convert SLM4IE processed JSONL into span-level IE training files.

Reads <key>.jsonl and the optional <key>.annotations.jsonl.gz sidecar
from the extraction output directory and emits a per-dataset JSONL
whose shape matches the requested span IE schema.

The converter expects each record's annotations payload to carry a
spans field that is a list of either [start, end, label] triples or
{start, end, label} dicts, with token-level indices that are
end-exclusive (Python slice convention). Records without a spans
field are skipped with a warning -- they typically come from corpora
that have UD-only annotations and no entity layer.

Three output schemas are supported. The gliner schema produces
GLiNER-style training examples and converts token indices to GLiNER's
end-inclusive convention; output keys are id, tokenized_text, and ner
(a list of [start, end_inclusive, label] triples). The conll schema
produces CoNLL-style IOB2 token tags; output keys are id, tokens, and
ner_tags. The generic schema is a lossless dump preserving text,
tokens, spans, and provenance; output keys are id, text, tokens,
spans (as {start, end, label} dicts), dataset, and domain.

Examples:
    Convert a single dataset to GLiNER format:

        uv run python scripts/data/to_spans.py kzb --schema gliner

    Convert every dataset declared in extract.yaml to the generic
    schema:

        uv run python scripts/data/to_spans.py --all --schema generic
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import IO, Any, Callable, Dict, Iterable, List, Optional, Tuple

import yaml
from tqdm import tqdm

from slm4ie.data.io_utils import (
    find_dataset_files as _find_dataset_files,
    find_project_root as _find_project_root,
    iter_joined_records as _iter_joined_records,
    open_output as _open_output,
    resolve_processed_dir as _resolve_processed_dir,
)

logger = logging.getLogger(__name__)

SCHEMAS: Tuple[str, ...] = ("gliner", "conll", "generic")


def _normalize_spans(
    raw_spans: Iterable[Any],
) -> List[Tuple[int, int, str]]:
    """Normalize spans from list-of-lists or list-of-dicts to tuples.

    Args:
        raw_spans (Iterable[Any]): Spans as either ``[s, e, label]``
            triples or ``{"start": s, "end": e, "label": label}``
            dicts.

    Returns:
        List[Tuple[int, int, str]]: Normalized ``(start, end, label)``
            tuples with end-exclusive token indices.

    Raises:
        ValueError: If a span entry is malformed.
    """
    normalized: List[Tuple[int, int, str]] = []
    for span in raw_spans:
        if isinstance(span, dict):
            normalized.append((
                int(span["start"]),
                int(span["end"]),
                str(span["label"]),
            ))
        elif isinstance(span, (list, tuple)) and len(span) == 3:
            start, end, label = span
            normalized.append((int(start), int(end), str(label)))
        else:
            raise ValueError(f"Unrecognized span shape: {span!r}")
    return normalized


def _record_id(record: Dict[str, Any], index: int) -> str:
    """Returns a stable id for *record* with a deterministic fallback.

    Args:
        record (Dict[str, Any]): Joined record.
        index (int): Zero-based record position used to synthesize a
            fallback id when ``uid`` is missing.

    Returns:
        str: ``uid`` from the record, or
            ``"<source>:idx-<14-digit-index>"`` when absent.
    """
    source = record.get("source", "unknown")
    return record.get("uid") or f"{source}:idx-{index:014d}"


def _extract_spans_or_none(
    record: Dict[str, Any],
) -> Optional[List[Tuple[int, int, str]]]:
    """Returns normalized spans from a record, or None if absent.

    Args:
        record (Dict[str, Any]): Joined record.

    Returns:
        Optional[List[Tuple[int, int, str]]]: Normalized spans, or
            None when the record carries no ``spans`` field.
    """
    annotations = record.get("annotations") or {}
    raw = annotations.get("spans")
    if raw is None:
        return None
    return _normalize_spans(raw)


def _tokens_for(record: Dict[str, Any]) -> List[str]:
    """Returns the token list for *record* (the ``forms`` array).

    Args:
        record (Dict[str, Any]): Joined record.

    Returns:
        List[str]: Token surface forms.

    Raises:
        ValueError: If the record has spans but no ``forms`` array.
    """
    annotations = record.get("annotations") or {}
    forms = annotations.get("forms")
    if forms is None:
        raise ValueError(
            "Record has spans but no token forms; cannot emit "
            "token-level output."
        )
    return list(forms)


def to_gliner(
    record: Dict[str, Any],
    index: int,
) -> Optional[Dict[str, Any]]:
    """Converts a record to a GLiNER training example.

    Args:
        record (Dict[str, Any]): Joined record (text + annotations).
        index (int): Zero-based record position.

    Returns:
        Optional[Dict[str, Any]]: Dict with ``id``, ``tokenized_text``,
            ``ner`` (with end-inclusive indices), or None when the
            record has no spans.
    """
    spans = _extract_spans_or_none(record)
    if spans is None:
        return None
    tokens = _tokens_for(record)
    ner = [[s, e - 1, label] for s, e, label in spans]
    return {
        "id": _record_id(record, index),
        "tokenized_text": tokens,
        "ner": ner,
    }


def to_conll(
    record: Dict[str, Any],
    index: int,
) -> Optional[Dict[str, Any]]:
    """Converts a record to a CoNLL-style IOB2-tagged example.

    Args:
        record (Dict[str, Any]): Joined record (text + annotations).
        index (int): Zero-based record position.

    Returns:
        Optional[Dict[str, Any]]: Dict with ``id``, ``tokens``, and
            ``ner_tags`` (IOB2-tagged), or None when the record has
            no spans.
    """
    spans = _extract_spans_or_none(record)
    if spans is None:
        return None
    tokens = _tokens_for(record)
    tags = ["O"] * len(tokens)
    for start, end, label in sorted(spans, key=lambda s: (s[0], -s[1])):
        if not (0 <= start < end <= len(tokens)):
            logger.warning(
                "Span (%d, %d, %r) out of bounds for %d tokens; skipping.",
                start, end, label, len(tokens),
            )
            continue
        tags[start] = f"B-{label}"
        for i in range(start + 1, end):
            tags[i] = f"I-{label}"
    return {
        "id": _record_id(record, index),
        "tokens": tokens,
        "ner_tags": tags,
    }


def to_generic(
    record: Dict[str, Any],
    index: int,
) -> Optional[Dict[str, Any]]:
    """Converts a record to the lossless generic span shape.

    Args:
        record (Dict[str, Any]): Joined record (text + annotations).
        index (int): Zero-based record position.

    Returns:
        Optional[Dict[str, Any]]: Dict with ``id``, ``text``,
            ``tokens``, ``spans`` (as dicts), ``dataset``, ``domain``,
            or None when the record has no spans.
    """
    spans = _extract_spans_or_none(record)
    if spans is None:
        return None
    tokens = _tokens_for(record)
    return {
        "id": _record_id(record, index),
        "text": record["text"],
        "tokens": tokens,
        "spans": [
            {"start": s, "end": e, "label": label}
            for s, e, label in spans
        ],
        "dataset": record.get("source", "unknown"),
        "domain": record.get("domain", "unknown"),
    }


_CONVERTERS: Dict[
    str, Callable[[Dict[str, Any], int], Optional[Dict[str, Any]]]
] = {
    "gliner": to_gliner,
    "conll": to_conll,
    "generic": to_generic,
}


def convert_record(
    record: Dict[str, Any],
    index: int,
    schema: str,
) -> Optional[Dict[str, Any]]:
    """Dispatches to the appropriate per-schema converter.

    Args:
        record (Dict[str, Any]): Joined record (text + annotations).
        index (int): Zero-based record position.
        schema (str): One of ``gliner``, ``conll``, ``generic``.

    Returns:
        Optional[Dict[str, Any]]: Converted record, or None when the
            input lacks span annotations and should be skipped.

    Raises:
        ValueError: If *schema* is not a known schema name.
    """
    try:
        converter = _CONVERTERS[schema]
    except KeyError as exc:
        raise ValueError(
            f"Unknown schema {schema!r}. Choose from {SCHEMAS}."
        ) from exc
    return converter(record, index)


def convert_stream(
    records: Iterable[Dict[str, Any]],
    out_stream: IO[str],
    schema: str,
) -> Tuple[int, int]:
    """Converts each input record and writes it as a JSONL line.

    Args:
        records (Iterable[Dict[str, Any]]): Iterable of joined records.
        out_stream (IO[str]): Writable text stream for converted JSONL.
        schema (str): One of ``gliner``, ``conll``, ``generic``.

    Returns:
        Tuple[int, int]: ``(written, skipped)`` counts.
    """
    written = 0
    skipped = 0
    for index, record in enumerate(records):
        converted = convert_record(record, index, schema)
        if converted is None:
            skipped += 1
            continue
        out_stream.write(json.dumps(converted, ensure_ascii=False))
        out_stream.write("\n")
        written += 1
    return written, skipped


def list_datasets_from_config(config_path: Path) -> List[str]:
    """Returns the dataset keys declared in ``extract.yaml``.

    Args:
        config_path (Path): Path to the extraction YAML config.

    Returns:
        List[str]: Dataset keys in declaration order.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open() as fh:
        cfg = yaml.safe_load(fh) or {}
    return list((cfg.get("datasets") or {}).keys())


def convert_dataset(
    key: str,
    processed_dir: Path,
    output_dir: Path,
    schema: str,
    force: bool = False,
) -> Optional[Tuple[int, int]]:
    """Converts a single dataset, writing ``<output_dir>/<key>.jsonl.gz``.

    Args:
        key (str): Dataset key.
        processed_dir (Path): Directory containing processed input
            files.
        output_dir (Path): Directory to write span-shaped output into.
            Created if it does not exist.
        schema (str): One of ``gliner``, ``conll``, ``generic``.
        force (bool): When True, overwrite an existing output file.
            Defaults to False (skip and return ``(0, 0)``).

    Returns:
        Optional[Tuple[int, int]]: ``(written, skipped)`` counts, or
            None when no input file exists for *key*. Returns
            ``(0, 0)`` when the output already exists and *force* is
            False.
    """
    pair = _find_dataset_files(processed_dir, key)
    if pair is None:
        logger.warning(
            "No processed input found for dataset %r in %s",
            key, processed_dir,
        )
        return None
    text_path, ann_path = pair

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{key}.jsonl.gz"

    if out_path.exists() and not force:
        logger.info(
            "Skipping %r, output already exists: %s "
            "(use --force to overwrite)",
            key, out_path,
        )
        return (0, 0)

    logger.info(
        "Converting %s%s → %s [schema=%s]",
        text_path,
        f" + {ann_path}" if ann_path else "",
        out_path,
        schema,
    )
    records = _iter_joined_records(text_path, ann_path)
    progress = tqdm(records, desc=key, unit="doc")
    with _open_output(out_path) as out_stream:
        try:
            written, skipped = convert_stream(progress, out_stream, schema)
        finally:
            progress.close()
    logger.info(
        "Wrote %d records (skipped %d without spans) to %s",
        written, skipped, out_path,
    )
    return written, skipped


def parse_args(argv=None) -> argparse.Namespace:
    """Parses command-line arguments.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Convert SLM4IE processed JSONL into span IE training "
            "files (gliner / conll / generic)."
        )
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "dataset",
        nargs="?",
        help="Dataset key (e.g. 'kzb'). Mutually exclusive with --all.",
    )
    target.add_argument(
        "--all",
        action="store_true",
        help="Convert every dataset declared in extract.yaml.",
    )
    parser.add_argument(
        "--schema",
        choices=SCHEMAS,
        default="generic",
        help="Output schema. Defaults to 'generic'.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing the processed <key>.jsonl files. "
            "Defaults to output_dir from configs/data/extract.yaml."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory to write <key>.jsonl.gz into. Defaults to "
            "<processed-dir>/spans/<schema>."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Path to extract.yaml (default: configs/data/extract.yaml)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing <key>.jsonl.gz outputs.",
    )
    return parser.parse_args(argv)


def main():
    """Runs the conversion from CLI arguments."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args = parse_args()
    project_root = _find_project_root()

    config_path = (
        args.config
        if args.config
        else project_root / "configs" / "data" / "extract.yaml"
    )
    processed_dir = _resolve_processed_dir(config_path, args.processed_dir)
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else processed_dir / "spans" / args.schema
    )

    if args.all:
        keys = list_datasets_from_config(config_path)
    else:
        keys = [args.dataset]

    total_written = 0
    total_skipped = 0
    missing: List[str] = []
    for key in keys:
        result = convert_dataset(
            key, processed_dir, output_dir, args.schema, force=args.force
        )
        if result is None:
            missing.append(key)
        else:
            written, skipped = result
            total_written += written
            total_skipped += skipped

    logger.info(
        "Done. Converted %d dataset(s), %d records written, "
        "%d skipped (no spans). Missing inputs: %s",
        len(keys) - len(missing), total_written, total_skipped,
        missing or "none",
    )
    if not args.all and missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
