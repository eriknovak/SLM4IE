"""Convert raw tokenizer-evaluation downloads into JSONL records.

This script materializes datasets tagged for tokenizer / morphology
evaluation. It bypasses the extract → datatrove → curate pipeline used
by pretraining corpora and reads the raw download directly, mirroring
the structure of `scripts/data/to_sentiment.py` and
`scripts/data/to_superglue.py`.

Datasets are selected by filtering `configs/data/download.yaml` for
entries with `benchmark: true` and `"TOKENIZER" in tasks`. Sloleks 3.1
is the seed dataset; new tokenizer-eval lexicons can be added simply
by tagging them with `TOKENIZER` and registering a reader below.

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

        uv run python scripts/data/to_tokenizer_eval.py sloleks

    Convert every TOKENIZER-tagged benchmark in the config:

        uv run python scripts/data/to_tokenizer_eval.py --all
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, IO, Iterable, Iterator, List, Optional

from tqdm import tqdm

from slm4ie.data.download import DatasetConfig, load_config
from slm4ie.data.io_utils import find_project_root, open_output
from slm4ie.data.parallel import (
    configure_script_logging,
    cpu_default,
    resolve_workers,
    run_parallel,
    workers_quiet,
)
from slm4ie.data.sloleks import iter_sloleks_dir

logger = logging.getLogger(__name__)

#: Task tag identifying tokenizer-evaluation datasets.
TOKENIZER_TASK = "TOKENIZER"


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


#: Registry mapping dataset key to a reader callable.
_READERS: Dict[
    str,
    Callable[[Path], Iterator[Dict[str, Any]]],
] = {
    "sloleks": _read_sloleks,
}


def list_tokenizer_datasets_from_config(config_path: Path) -> List[str]:
    """Return TOKENIZER-tagged benchmark dataset keys from a config.

    Args:
        config_path: Path to the download YAML config.

    Returns:
        List[str]: Dataset keys whose `benchmark` field is true and
            whose `tasks` list contains `"TOKENIZER"`.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    _, datasets = load_config(config_path)
    return [
        key
        for key, cfg in datasets.items()
        if cfg.benchmark and TOKENIZER_TASK in cfg.tasks
    ]


def write_records(
    records: Iterable[Dict[str, Any]],
    out_stream: IO[str],
) -> int:
    """Write *records* as JSONL lines and return the count.

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
        key: Dataset key (must be registered in _READERS).
        raw_dir: Directory holding the raw download for *key*.
        output_dir: Directory to write the JSONL output into. Created
            if missing.
        force: When True, overwrite an existing output file.

    Returns:
        Optional[int]: Number of records written, or None when no
            reader is registered for *key* or the raw directory is
            absent. Returns 0 when the output already exists and
            *force* is False.
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
            "Skipping %r, output already exists: %s "
            "(use --force to overwrite)",
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
            "Convert tokenizer-evaluation benchmark downloads (e.g. "
            "Sloleks 3.1) into <output_dir>/<key>.jsonl.gz."
        )
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "dataset",
        nargs="?",
        help="Dataset key (e.g. 'sloleks'). Mutually exclusive with --all.",
    )
    target.add_argument(
        "--all",
        action="store_true",
        help=(
            "Convert every TOKENIZER-tagged benchmark dataset declared "
            "in the download config."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to download.yaml (default: configs/data/download.yaml).",
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
            "/vault/data/SLM4IE/benchmarks/tokenizer/<key>."
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


def _resolve_raw_dir(
    config_path: Path,
    override: Optional[Path],
) -> Path:
    """Return the base raw-downloads directory.

    Args:
        config_path: Path to download.yaml.
        override: Explicit `--raw-dir` value.

    Returns:
        Path: Directory containing per-dataset subdirectories.
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
        base_raw_dir: Base raw downloads directory.
        config_path: Path to download.yaml.
        key: Dataset key.

    Returns:
        Path: Per-dataset raw directory (`<base>/<output_dir>`).
    """
    _, datasets = load_config(config_path)
    cfg: Optional[DatasetConfig] = datasets.get(key)
    subdir = cfg.output_dir if cfg and cfg.output_dir else key
    return base_raw_dir / subdir


def _default_output_dir(key: str) -> Path:
    """Return the default output directory for *key*.

    Args:
        key: Dataset key.

    Returns:
        Path: `/vault/data/SLM4IE/benchmarks/tokenizer/<key>`.
    """
    return Path("/vault/data/SLM4IE/benchmarks/tokenizer") / key


def main() -> None:
    """Run the tokenizer-eval conversion from CLI arguments."""
    args = parse_args()
    project_root = find_project_root()
    config_path = (
        args.config
        if args.config
        else project_root / "configs" / "data" / "download.yaml"
    )

    base_raw_dir = _resolve_raw_dir(config_path, args.raw_dir)

    if args.all:
        keys = list_tokenizer_datasets_from_config(config_path)
        if not keys:
            logger.warning(
                "No datasets in %s have benchmark: true and tasks: [%s].",
                config_path,
                TOKENIZER_TASK,
            )
            return
    else:
        keys = [args.dataset]

    # Resolve per-dataset paths in the parent (avoids re-parsing
    # download.yaml inside every worker).
    dataset_dirs: Dict[str, Path] = {
        key: _resolve_dataset_dir(base_raw_dir, config_path, key)
        for key in keys
    }
    output_dirs: Dict[str, Path] = {
        key: (
            args.output_dir
            if args.output_dir is not None
            else _default_output_dir(key)
        )
        for key in keys
    }

    workers = resolve_workers(args.max_workers, len(keys), cpu_default(len(keys)))
    configure_script_logging(parallel=workers > 1)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = project_root / "logs" / Path(__file__).stem / stamp

    def kwargs_for(key: str) -> Dict[str, Any]:
        return {
            "raw_dir": dataset_dirs[key],
            "output_dir": output_dirs[key],
            "force": args.force,
        }

    results, failures = run_parallel(
        convert_dataset,
        keys,
        max_workers=workers,
        desc="tokenizer-eval",
        pool="process",
        kwargs_for=kwargs_for,
        log_dir=log_dir,
    )

    skipped: List[str] = [k for k, v in results.items() if v is None]
    total = sum(v for v in results.values() if v is not None)

    logger.info(
        "Done. Converted %d dataset(s), %d records total. "
        "Skipped: %s. Failed: %s",
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
