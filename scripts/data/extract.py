"""Extract and convert raw datasets to unified JSONL format."""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from slm4ie.data.io_utils import find_project_root
from slm4ie.data.parallel import configure_script_logging
from slm4ie.data.processing import extract_datasets


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Selection is explicit: either pass one or more positional dataset
    keys, or pass `--all`. Exactly one of the two must be provided.

    Args:
        argv: Optional argument list (defaults to `sys.argv`).

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Extract raw datasets to unified JSONL format.",
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "datasets",
        nargs="*",
        default=[],
        help="Dataset keys to extract (e.g. 'kzb gigafida').",
    )
    target.add_argument(
        "--all",
        action="store_true",
        help="Extract every dataset declared in the config.",
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
        default="extract",
        help=(
            "Name of the config in configs/data/ to use, "
            "without extension (default: 'extract')."
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Override the configured input directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override the configured output directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract datasets even if their output already exists.",
    )
    parser.add_argument(
        "--mlflow",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable/disable MLflow extraction-build tracking, overriding "
            "the config's mlflow.enabled. Default: defer to config."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=0,
        help=(
            "Shard workers used WITHIN each dataset (datasets run "
            "sequentially). 0=auto (all cores), 1=serial single pass, "
            "N=N worker processes. Only file-based extractors with "
            "enough input files are sharded."
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


def main() -> None:
    """Run dataset extraction pipeline."""
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
        extract_datasets(
            config_path=config_path,
            dataset_keys=args.datasets if not args.all else None,
            force=args.force,
            max_workers=args.max_workers,
            log_dir=log_dir,
            input_dir_override=args.input_dir,
            output_dir_override=args.output_dir,
            mlflow_enabled=args.mlflow,
        )
    except ValueError as e:
        logging.getLogger(__name__).error(str(e))
        sys.exit(1)
    except Exception as e:
        logging.getLogger(__name__).error("Extraction failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
