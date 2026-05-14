"""Download datasets from HuggingFace Hub, Clarin.si, and other sources."""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from slm4ie.data.download import download_datasets
from slm4ie.data.io_utils import find_project_root
from slm4ie.data.parallel import configure_script_logging


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Selection is explicit: either pass one or more positional dataset
    keys, or pass `--all`. Exactly one of the two must be provided.

    Args:
        argv: Argument list (defaults to `sys.argv`).

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Download datasets for SLM4IE project."
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "datasets",
        nargs="*",
        default=[],
        help="Dataset keys to download. Use --all for the full catalog.",
    )
    target.add_argument(
        "--all",
        action="store_true",
        help="Download every dataset declared in the config.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to YAML config file. Overrides --config-name "
            "if both are set."
        ),
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="download",
        help=(
            "Name of the config in configs/data/ to use, "
            "without extension (default: 'download'). "
            "Use 'benchmarks' for evaluation datasets."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if output exists.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override base output directory.",
    )
    benchmark_group = parser.add_mutually_exclusive_group()
    benchmark_group.add_argument(
        "--only-benchmarks",
        action="store_true",
        help=(
            "Restrict default selection to datasets marked "
            "`benchmark: true` in the config."
        ),
    )
    benchmark_group.add_argument(
        "--exclude-benchmarks",
        action="store_true",
        help=(
            "Drop benchmark datasets from the default selection "
            "(pretraining-only)."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=0,
        help=(
            "Download datasets in parallel using a thread pool. "
            "0=auto (min(4, n_datasets), polite to remote servers), "
            "1=serial, N=N threads. Capped at the number of datasets."
        ),
    )
    args = parser.parse_args(argv)
    # argparse's `required=True` on a mutex group accepts a positional
    # with `nargs="*"`, but treats an empty positional as "provided" —
    # so bare invocation can slip through. Validate the xor by hand.
    if args.all and args.datasets:
        parser.error("argument --all: not allowed with positional datasets")
    if not args.all and not args.datasets:
        parser.error("one of the arguments datasets --all is required")
    return args


def main():
    """Run dataset download pipeline."""
    args = parse_args()
    project_root = find_project_root()

    config_path = (
        Path(args.config)
        if args.config
        else project_root
        / "configs"
        / "data"
        / f"{args.config_name}.yaml"
    )

    configure_script_logging(
        parallel=args.max_workers > 1,
        console_level=logging.WARNING,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = project_root / "logs" / Path(__file__).stem / stamp
    print(f"Logs: {log_dir}", file=sys.stderr)

    try:
        download_datasets(
            config_path=config_path,
            dataset_keys=args.datasets if not args.all else None,
            force=args.force,
            output_dir_override=args.output_dir,
            only_benchmarks=args.only_benchmarks,
            exclude_benchmarks=args.exclude_benchmarks,
            max_workers=args.max_workers,
            log_dir=log_dir,
        )
    except ValueError as e:
        logging.getLogger(__name__).error(str(e))
        sys.exit(1)
    except Exception as e:
        logging.getLogger(__name__).error(
            "Download failed: %s", e
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
