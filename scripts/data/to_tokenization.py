"""Convert raw tokenizer-evaluation downloads into JSONL records.

This script materializes datasets tagged for tokenizer / morphology
evaluation. It bypasses the extract -> datatrove -> curate pipeline used
by pretraining corpora and reads the raw download directly, mirroring
the structure of `scripts/data/to_sentiment.py` and
`scripts/data/to_superglue.py`.

Configuration is read from `configs/data/tokenization.yaml`, which
declares `input_dir`, `output_dir`, and the dataset keys to convert.
Each entry in `datasets` must:

  1. be declared in `configs/data/download.yaml` so the raw subdirectory
     can be resolved via its `output_dir` field, and
  2. have a reader registered in `_READERS` below.

Sloleks 3.1 is the seed dataset; new tokenizer-eval lexicons can be
added by registering a reader and listing the key in
`configs/data/tokenization.yaml`.

Each output line has this shape:

    {
        "entry_id": "...",
        "lemma":    "hiša",
        "lemma_msd": "Ncfsn",
        "forms": [
            {"form": "hiša", "msd": "Ncfsn"},
            {"form": "hiše", "msd": "Ncfsg"}
        ],
        "dataset": "sloleks",
        "task":    "TOKENIZER"
    }

Examples:
    Convert Sloleks 3.1:

        uv run python scripts/data/to_tokenization.py sloleks

    Convert every dataset declared in tokenization.yaml:

        uv run python scripts/data/to_tokenization.py --all
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, IO, Iterable, Iterator, List, Optional

import yaml
from tqdm import tqdm

from slm4ie.data.catalog import DatasetConfig, load_config
from slm4ie.data.io_utils import find_project_root, open_output
from slm4ie.data.parallel import (
    configure_script_logging,
    cpu_default,
    resolve_workers,
    run_parallel,
    workers_quiet,
)
from slm4ie.data.sloleks import iter_sloleks_dir
from slm4ie.data.sloleks_relations import (
    find_word_relations_tsv,
    iter_word_relation_segmentations,
)

logger = logging.getLogger(__name__)

#: Task tag identifying tokenizer-evaluation datasets.
TOKENIZER_TASK = "TOKENIZER"

#: Default location of the tokenization config relative to the project root.
DEFAULT_CONFIG_RELPATH = Path("configs") / "data" / "tokenization.yaml"

#: Default location of the download config relative to the project root.
DEFAULT_DOWNLOAD_CONFIG_RELPATH = Path("configs") / "data" / "download.yaml"


def _read_sloleks(raw_dir: Path) -> Iterator[Dict[str, Any]]:
    """Yield Sloleks records, tagging them with dataset/task fields.

    Args:
        raw_dir: Directory holding the unzipped Sloleks XML files.

    Yields:
        Dict[str, Any]: Records as produced by `iter_sloleks_dir`,
            extended with `dataset` and `task` fields.

    Raises:
        FileNotFoundError: If no Sloleks XML files are found under
            `raw_dir`.
    """
    if not any(raw_dir.rglob("*.xml")):
        raise FileNotFoundError(
            f"No XML files found under {raw_dir}. "
            "Run scripts/data/download.py --datasets sloleks first "
            "and unzip Sloleks.3.1.zip."
        )
    for record in iter_sloleks_dir(raw_dir):
        record["dataset"] = "sloleks"
        record["task"] = TOKENIZER_TASK
        yield record


def _read_sloleks_relations(raw_dir: Path) -> Iterator[Dict[str, Any]]:
    """Yield Sloleks word-relation derivational segmentations.

    Each record carries one derived lemma's morpheme decomposition (read from
    the underscore column, not heuristically aligned), tagged with the dataset
    and task fields and a `verified` flag for the manually-scored subset.

    Args:
        raw_dir: Directory holding the unzipped word-relations download.

    Yields:
        Dict[str, Any]: Records as produced by
            `iter_word_relation_segmentations`, extended with `dataset` and
            `task` fields.

    Raises:
        FileNotFoundError: If no word-relations TSV is found under `raw_dir`.
    """
    tsv_path = find_word_relations_tsv(raw_dir)
    if tsv_path is None:
        raise FileNotFoundError(
            f"No word-relations TSV found under {raw_dir}. "
            "Run scripts/data/download.py --datasets sloleks_relations first "
            "and unzip the archive."
        )
    for record in iter_word_relation_segmentations(tsv_path):
        record["dataset"] = "sloleks_relations"
        record["task"] = TOKENIZER_TASK
        yield record


#: Registry mapping dataset key to a reader callable.
_READERS: Dict[
    str,
    Callable[[Path], Iterator[Dict[str, Any]]],
] = {
    "sloleks": _read_sloleks,
    "sloleks_relations": _read_sloleks_relations,
}


def load_tokenization_config(config_path: Path) -> Dict[str, Any]:
    """Load and validate the tokenization YAML config.

    Args:
        config_path: Path to `configs/data/tokenization.yaml`.

    Returns:
        Dict[str, Any]: Parsed config with `input_dir`, `output_dir`,
            and `datasets` (list of dataset keys).

    Raises:
        FileNotFoundError: If `config_path` does not exist.
        ValueError: If required fields are missing or malformed.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Tokenization config not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    input_dir = raw.get("input_dir")
    output_dir = raw.get("output_dir")
    datasets = raw.get("datasets") or []

    missing: List[str] = []
    if not input_dir:
        missing.append("input_dir")
    if not output_dir:
        missing.append("output_dir")
    if missing:
        raise ValueError(f"Tokenization config {config_path} is missing required fields: {', '.join(missing)}")
    if not isinstance(input_dir, str) or not isinstance(output_dir, str):
        raise ValueError(f"Tokenization config {config_path}: `input_dir` and `output_dir` must be strings.")
    if not isinstance(datasets, list) or not all(isinstance(k, str) for k in datasets):
        raise ValueError(f"Tokenization config {config_path}: `datasets` must be a list of strings.")

    return {
        "input_dir": Path(input_dir),
        "output_dir": Path(output_dir),
        "datasets": list(datasets),
    }


def write_records(
    records: Iterable[Dict[str, Any]],
    out_stream: IO[str],
) -> int:
    """Write `records` as JSONL lines and return the count.

    Args:
        records: Records to serialize.
        out_stream: Writable text stream.

    Returns:
        int: Number of records written.
    """
    count = 0
    for record in records:
        out_stream.write(json.dumps(record, ensure_ascii=False))
        out_stream.write("\n")
        count += 1
    return count


def convert_dataset(
    key: str,
    raw_dir: Path,
    output_dir: Path,
    force: bool = False,
) -> Optional[int]:
    """Convert one tokenizer-eval dataset to `<output_dir>/<key>.jsonl.gz`.

    Args:
        key: Dataset key (must be registered in `_READERS`).
        raw_dir: Directory holding the raw download for `key`.
        output_dir: Directory to write the JSONL output into. Created
            if missing.
        force: When True, overwrite an existing output file.

    Returns:
        Optional[int]: Number of records written, or None when no
            reader is registered for `key` or the raw directory is
            absent. Returns 0 when the output already exists and
            `force` is False.

    Raises:
        ValueError: If the conversion ran but produced zero records,
            which signals a reader/format mismatch rather than success.
    """
    reader = _READERS.get(key)
    if reader is None:
        logger.warning(
            "No tokenizer-eval reader registered for dataset %r; skipping.",
            key,
        )
        return None

    if not raw_dir.exists():
        logger.warning("Raw dir for %r does not exist: %s", key, raw_dir)
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{key}.jsonl.gz"

    if out_path.exists() and not force:
        logger.info(
            "Skipping %r, output already exists: %s (use --force to overwrite)",
            key,
            out_path,
        )
        return 0

    logger.info("Converting %s -> %s", raw_dir, out_path)
    progress = tqdm(reader(raw_dir), desc=key, unit="entry", disable=workers_quiet())
    with open_output(out_path) as out_stream:
        try:
            count = write_records(progress, out_stream)
        finally:
            progress.close()

    if count == 0:
        # An empty output almost always means a reader/format mismatch;
        # remove the bogus file and fail loudly instead of exiting 0.
        out_path.unlink(missing_ok=True)
        raise ValueError(
            f"Conversion of {key!r} produced 0 records from {raw_dir}; "
            "likely a reader/format mismatch. No output written."
        )

    logger.info("Wrote %d records to %s", count, out_path)
    return count


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to `sys.argv`).

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Convert tokenizer-evaluation benchmark downloads (e.g. Sloleks 3.1) "
            "into <output_dir>/<key>.jsonl.gz, driven by configs/data/tokenization.yaml."
        )
    )
    parser.add_argument(
        "dataset_keys",
        nargs="*",
        help=(
            "Dataset keys to convert (subset of the `datasets` list in "
            "tokenization.yaml). Mutually exclusive with --all."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Convert every dataset declared in tokenization.yaml.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to tokenization.yaml (default: configs/data/tokenization.yaml).",
    )
    parser.add_argument(
        "--download-config",
        type=Path,
        default=None,
        help=(
            "Path to download.yaml, used to resolve per-dataset raw "
            "subdirectories (default: configs/data/download.yaml)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing <key>.jsonl.gz outputs.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=0,
        help=(
            "Process datasets in parallel. 0=auto (cpu_count // 2), "
            "1=serial, N=N workers. Capped at the number of datasets."
        ),
    )
    return parser.parse_args(argv)


def _resolve_dataset_subdir(
    download_config_path: Path,
    key: str,
) -> str:
    """Return the per-dataset raw subdirectory name from `download.yaml`.

    Args:
        download_config_path: Path to download.yaml.
        key: Dataset key.

    Returns:
        str: The `output_dir` value declared for `key` in download.yaml,
            or `key` itself if no per-dataset override exists.
    """
    _, datasets = load_config(download_config_path)
    cfg: Optional[DatasetConfig] = datasets.get(key)
    return cfg.output_dir if cfg and cfg.output_dir else key


def main() -> None:
    """Run the tokenizer-eval conversion from CLI arguments."""
    args = parse_args()
    project_root = find_project_root()

    config_path = args.config if args.config else project_root / DEFAULT_CONFIG_RELPATH
    download_config_path = (
        args.download_config if args.download_config else project_root / DEFAULT_DOWNLOAD_CONFIG_RELPATH
    )

    config = load_tokenization_config(config_path)
    declared_keys: List[str] = config["datasets"]
    input_dir: Path = config["input_dir"]
    output_dir: Path = config["output_dir"]

    if args.all and args.dataset_keys:
        logger.error("Pass either positional dataset keys or --all, not both.")
        sys.exit(2)

    if args.all:
        keys: List[str] = list(declared_keys)
        if not keys:
            logger.warning(
                "No datasets declared in %s; nothing to do.",
                config_path,
            )
            return
    elif args.dataset_keys:
        keys = list(args.dataset_keys)
        unknown = [k for k in keys if k not in declared_keys]
        if unknown:
            logger.warning(
                "Dataset key(s) %s not listed in %s; proceeding anyway.",
                unknown,
                config_path,
            )
    else:
        logger.error(
            "Specify at least one dataset key (positional) or pass --all.",
        )
        sys.exit(2)

    # Resolve per-dataset raw dirs in the parent (avoids re-parsing
    # download.yaml inside every worker).
    dataset_dirs: Dict[str, Path] = {
        key: input_dir / _resolve_dataset_subdir(download_config_path, key) for key in keys
    }

    workers = resolve_workers(args.max_workers, len(keys), cpu_default(len(keys)))
    configure_script_logging(parallel=workers > 1)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = project_root / "logs" / Path(__file__).stem / stamp

    def kwargs_for(key: str) -> Dict[str, Any]:
        """Return per-worker kwargs for `convert_dataset`.

        Args:
            key: Dataset key being processed.

        Returns:
            Dict[str, Any]: Keyword arguments for `convert_dataset`.
        """
        return {
            "raw_dir": dataset_dirs[key],
            "output_dir": output_dir,
            "force": args.force,
        }

    results, failures = run_parallel(
        convert_dataset,
        keys,
        max_workers=workers,
        desc="tokenization",
        pool="process",
        kwargs_for=kwargs_for,
        log_dir=log_dir,
    )

    skipped: List[str] = [k for k, v in results.items() if v is None]
    total = sum(v for v in results.values() if v is not None)

    logger.info(
        "Done. Converted %d dataset(s), %d records total. Skipped: %s. Failed: %s",
        len(results) - len(skipped),
        total,
        skipped or "none",
        [k for k, _ in failures] or "none",
    )
    if failures:
        sys.exit(2)
    if not args.all and skipped:
        sys.exit(1)


if __name__ == "__main__":
    main()
