"""Convert raw SA benchmark downloads into evaluation-ready JSONL.

Reads sentiment-analysis benchmark datasets directly from their raw
download directories (i.e. it bypasses the unified ``<key>.jsonl``
extraction layer used by to_datatrove.py and to_spans.py) and emits
per-dataset JSONL files whose shape is suitable for downstream
classification training and SloBENCH-style evaluation.

Each output line has this shape:

    {
        "id":       "sentinews:doc-00042",
        "text":     "<text content>",
        "label":    "negative",
        "label_id": 0,
        "dataset":  "sentinews",
        "task":     "SA",
        "level":    "document",
        "metadata": {...}
    }

A sibling ``label_map.json`` is written next to the output file so the
``label_id`` integer encoding is traceable.

Examples:
    Convert the SentiNews document-level split:

        uv run python scripts/data/to_sentiment.py sentinews

    Convert every SA benchmark dataset declared in download.yaml:

        uv run python scripts/data/to_sentiment.py --all
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import (
    IO,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
)

from tqdm import tqdm

from slm4ie.data.download import DatasetConfig, load_config
from slm4ie.data.io_utils import find_project_root, open_output

logger = logging.getLogger(__name__)

#: Canonical 3-class label vocabulary for sentiment analysis. Order
#: defines the integer encoding used in the ``label_id`` field.
CANONICAL_LABELS: Tuple[str, ...] = ("negative", "neutral", "positive")

#: Map common SentiNews label spellings to canonical labels.
_LABEL_NORMALIZATION: Dict[str, str] = {
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
    "neg": "negative",
    "pos": "positive",
    "neu": "neutral",
}

#: Levels supported by SentiNews-style corpora.
SENTINEWS_LEVELS: Tuple[str, ...] = ("document", "paragraph", "sentence")


def _normalize_label(raw_label: str) -> str:
    """Map a raw label string to one of CANONICAL_LABELS.

    Args:
        raw_label (str): Label as it appears in the source file.

    Returns:
        str: Canonical lowercase label.

    Raises:
        ValueError: If the label is not recognized.
    """
    cleaned = raw_label.strip().lower()
    if cleaned not in _LABEL_NORMALIZATION:
        raise ValueError(f"Unrecognized SA label: {raw_label!r}")
    return _LABEL_NORMALIZATION[cleaned]


def _label_id(label: str) -> int:
    """Return the integer encoding for *label*.

    Args:
        label (str): Canonical label.

    Returns:
        int: Index into CANONICAL_LABELS.
    """
    return CANONICAL_LABELS.index(label)


def _sentinews_files(raw_dir: Path) -> Dict[str, Path]:
    """Discover SentiNews files keyed by annotation level.

    Args:
        raw_dir (Path): Directory containing the SentiNews downloads.

    Returns:
        Dict[str, Path]: Mapping from level name (one of
            SENTINEWS_LEVELS) to the matching file path. Only levels
            actually present on disk are included.
    """
    available: Dict[str, Path] = {}
    for level in SENTINEWS_LEVELS:
        for candidate in raw_dir.glob(f"SentiNews_{level}-level.*"):
            if candidate.is_file():
                available[level] = candidate
                break
    return available


def _open_tsv(path: Path) -> IO[str]:
    """Open a SentiNews tab-separated text file.

    Args:
        path (Path): Path to the SentiNews file.

    Returns:
        IO[str]: A readable text stream. Caller is responsible for
            closing it.
    """
    return path.open(encoding="utf-8", newline="")


def _read_sentinews_level(
    path: Path,
    level: str,
) -> Iterator[Dict[str, Any]]:
    """Yield normalized records from a single SentiNews file.

    Args:
        path (Path): Path to one of the SentiNews_*-level.txt files.
        level (str): Annotation level (document / paragraph / sentence).

    Yields:
        Dict[str, Any]: Normalized record with id, text, label,
            label_id, dataset, task, level, and metadata fields.
    """
    with _open_tsv(path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        text_keys = ("content", "text", "sentence", "paragraph")
        for index, row in enumerate(reader):
            text = next(
                (row[k] for k in text_keys if k in row and row[k]),
                None,
            )
            raw_label = row.get("sentiment") or row.get("label")
            if text is None or raw_label is None:
                logger.warning(
                    "Skipping malformed %s row %d in %s",
                    level, index, path.name,
                )
                continue

            try:
                label = _normalize_label(raw_label)
            except ValueError as exc:
                logger.warning(
                    "Skipping row %d (%s): %s",
                    index, path.name, exc,
                )
                continue

            nid = row.get("nid") or row.get("doc_id") or f"idx-{index:08d}"
            id_parts = [str(nid)]
            if level in ("paragraph", "sentence"):
                pid = row.get("pid")
                if pid is not None:
                    id_parts.append(f"p{pid}")
            if level == "sentence":
                sid = row.get("sid")
                if sid is not None:
                    id_parts.append(f"s{sid}")

            metadata = {
                k: v
                for k, v in row.items()
                if k
                not in {
                    "content",
                    "text",
                    "sentence",
                    "paragraph",
                    "sentiment",
                    "label",
                }
                and v not in (None, "")
            }

            yield {
                "id": f"sentinews:{'-'.join(id_parts)}",
                "text": text,
                "label": label,
                "label_id": _label_id(label),
                "dataset": "sentinews",
                "task": "SA",
                "level": level,
                "metadata": metadata,
            }


def _read_sentinews(
    raw_dir: Path,
    levels: Optional[List[str]] = None,
) -> Iterator[Dict[str, Any]]:
    """Yield normalized records from a SentiNews download.

    Args:
        raw_dir (Path): Directory containing the SentiNews files.
        levels (Optional[List[str]]): Annotation levels to emit. When
            None, emits whichever levels are present in *raw_dir*.

    Yields:
        Dict[str, Any]: Normalized records (one per item per level).

    Raises:
        FileNotFoundError: If no SentiNews_*-level.txt files are found.
    """
    available = _sentinews_files(raw_dir)
    if not available:
        raise FileNotFoundError(
            f"No SentiNews_*-level.txt files found in {raw_dir}. "
            "Run scripts/data/download.py --datasets sentinews first."
        )

    selected_levels = levels or list(available.keys())
    for level in selected_levels:
        if level not in available:
            logger.warning(
                "SentiNews level %r not found in %s; skipping.",
                level, raw_dir,
            )
            continue
        yield from _read_sentinews_level(available[level], level)


#: Registry mapping dataset key to a reader callable.
_READERS: Dict[
    str,
    Callable[..., Iterator[Dict[str, Any]]],
] = {
    "sentinews": _read_sentinews,
}


def list_sa_datasets_from_config(
    config_path: Path,
) -> List[str]:
    """Return SA-tagged benchmark dataset keys from a download config.

    Args:
        config_path (Path): Path to the download YAML config.

    Returns:
        List[str]: Dataset keys whose ``benchmark`` field is true and
            whose ``tasks`` list contains ``"SA"``.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    _, datasets = load_config(config_path)
    return [
        key
        for key, cfg in datasets.items()
        if cfg.benchmark and "SA" in cfg.tasks
    ]


def write_records(
    records: Iterable[Dict[str, Any]],
    out_stream: IO[str],
) -> int:
    """Write *records* as JSONL lines and return the count.

    Args:
        records (Iterable[Dict[str, Any]]): Records to serialize.
        out_stream (IO[str]): Writable text stream.

    Returns:
        int: Number of records written.
    """
    count = 0
    for record in records:
        out_stream.write(json.dumps(record, ensure_ascii=False))
        out_stream.write("\n")
        count += 1
    return count


def write_label_map(output_dir: Path) -> Path:
    """Persist the label_id encoding next to the output file.

    Args:
        output_dir (Path): Directory the JSONL output is being written
            to. The label map lands at ``<output_dir>/label_map.json``.

    Returns:
        Path: Path to the written label map.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "label_map.json"
    payload = {label: idx for idx, label in enumerate(CANONICAL_LABELS)}
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def convert_dataset(
    key: str,
    raw_dir: Path,
    output_dir: Path,
    levels: Optional[List[str]] = None,
    force: bool = False,
) -> Optional[int]:
    """Convert one SA dataset to ``<output_dir>/<key>.jsonl.gz``.

    Args:
        key (str): Dataset key (must be registered in _READERS).
        raw_dir (Path): Directory holding the raw download for *key*.
        output_dir (Path): Directory to write the JSONL output into.
            Created if missing.
        levels (Optional[List[str]]): Annotation levels to emit (only
            relevant for multi-level corpora like SentiNews).
        force (bool): When True, overwrite an existing output file.

    Returns:
        Optional[int]: Number of records written, or None when no
            reader is registered for *key*. Returns 0 when the output
            already exists and *force* is False.
    """
    reader = _READERS.get(key)
    if reader is None:
        logger.warning(
            "No SA reader registered for dataset %r; skipping.", key,
        )
        return None

    if not raw_dir.exists():
        logger.warning(
            "Raw dir for %r does not exist: %s", key, raw_dir,
        )
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{key}.jsonl.gz"

    if out_path.exists() and not force:
        logger.info(
            "Skipping %r, output already exists: %s "
            "(use --force to overwrite)",
            key, out_path,
        )
        return 0

    logger.info("Converting %s → %s", raw_dir, out_path)
    records = reader(raw_dir, levels=levels) if key == "sentinews" else reader(raw_dir)
    progress = tqdm(records, desc=key, unit="rec")
    with open_output(out_path) as out_stream:
        try:
            count = write_records(progress, out_stream)
        finally:
            progress.close()

    write_label_map(output_dir)
    logger.info("Wrote %d records to %s", count, out_path)
    return count


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Convert SA benchmark downloads into evaluation-ready "
            "<output_dir>/<key>.jsonl.gz files."
        )
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "dataset",
        nargs="?",
        help=(
            "Dataset key (e.g. 'sentinews'). Mutually exclusive "
            "with --all."
        ),
    )
    target.add_argument(
        "--all",
        action="store_true",
        help=(
            "Convert every SA-tagged benchmark dataset declared in "
            "the download config."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Path to download.yaml (default: "
            "configs/data/download.yaml)."
        ),
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing the raw downloads. Defaults to the "
            "output_dir from the download config."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory to write <key>.jsonl.gz into. Defaults to "
            "<raw-dir>/eval/sentiment."
        ),
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        choices=SENTINEWS_LEVELS,
        default=None,
        help=(
            "SentiNews annotation levels to emit (default: all "
            "levels present in the raw download)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing <key>.jsonl.gz outputs.",
    )
    return parser.parse_args(argv)


def _resolve_raw_dir(
    config_path: Path,
    override: Optional[Path],
) -> Path:
    """Return the base raw-downloads directory.

    Args:
        config_path (Path): Path to download.yaml.
        override (Optional[Path]): Explicit ``--raw-dir`` value.

    Returns:
        Path: Directory containing per-dataset subdirectories.

    Raises:
        FileNotFoundError: If *override* is None and *config_path*
            does not exist.
    """
    if override is not None:
        return override
    base, _ = load_config(config_path)
    return Path(base)


def _resolve_dataset_dir(
    base_raw_dir: Path,
    config_path: Path,
    key: str,
) -> Path:
    """Return the raw-input directory for a specific dataset.

    Args:
        base_raw_dir (Path): Base raw downloads directory.
        config_path (Path): Path to download.yaml.
        key (str): Dataset key.

    Returns:
        Path: Per-dataset raw directory (``<base>/<output_dir>``).
    """
    _, datasets = load_config(config_path)
    cfg: Optional[DatasetConfig] = datasets.get(key)
    subdir = cfg.output_dir if cfg and cfg.output_dir else key
    return base_raw_dir / subdir


def main():
    """Run the SA conversion from CLI arguments."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args = parse_args()
    project_root = find_project_root()
    config_path = (
        args.config
        if args.config
        else project_root / "configs" / "data" / "download.yaml"
    )

    base_raw_dir = _resolve_raw_dir(config_path, args.raw_dir)
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else base_raw_dir / "eval" / "sentiment"
    )

    if args.all:
        keys = list_sa_datasets_from_config(config_path)
    else:
        keys = [args.dataset]

    total = 0
    skipped: List[str] = []
    for key in keys:
        dataset_raw_dir = _resolve_dataset_dir(base_raw_dir, config_path, key)
        result = convert_dataset(
            key,
            dataset_raw_dir,
            output_dir,
            levels=args.levels,
            force=args.force,
        )
        if result is None:
            skipped.append(key)
        else:
            total += result

    logger.info(
        "Done. Converted %d dataset(s), %d records total. Skipped: %s",
        len(keys) - len(skipped), total, skipped or "none",
    )
    if not args.all and skipped:
        sys.exit(1)


if __name__ == "__main__":
    main()
