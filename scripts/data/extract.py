"""Extract and convert raw datasets to unified JSONL format."""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from slm4ie.data.parallel import configure_script_logging
from slm4ie.data.processing import extract_datasets


def _find_project_root() -> Path:
    """Find the project root by locating pyproject.toml.

    Returns:
        Path: The project root directory.

    Raises:
        FileNotFoundError: If pyproject.toml cannot be found.
    """
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Could not find project root (no pyproject.toml)")


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to sys.argv).

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=("Extract raw datasets to unified JSONL format.")
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=("Dataset keys to extract (default: all configured)."),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=("Path to extraction YAML config (default: configs/data/extract.yaml)."),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=("Re-extract datasets even if their output already exists."),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=0,
        help=(
            "Extract datasets in parallel. 0=auto (cpu_count // 2), "
            "1=serial, N=N workers. Capped at the number of datasets."
        ),
    )
    return parser.parse_args(argv)


def main():
    """Run dataset extraction pipeline."""
    args = parse_args()
    project_root = _find_project_root()

    config_path = (
        Path(args.config)
        if args.config
        else project_root / "configs" / "data" / "extract.yaml"
    )

    configure_script_logging(parallel=args.max_workers > 1)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = project_root / "logs" / Path(__file__).stem / stamp

    try:
        extract_datasets(
            config_path=config_path,
            dataset_keys=args.datasets,
            force=args.force,
            max_workers=args.max_workers,
            log_dir=log_dir,
        )
    except ValueError as e:
        logging.getLogger(__name__).error(str(e))
        sys.exit(1)
    except Exception as e:
        logging.getLogger(__name__).error("Extraction failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
