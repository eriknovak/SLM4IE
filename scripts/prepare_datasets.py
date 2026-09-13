"""Prepare the shared dataset tiers: download, extract, and convert.

One CLI over the shared data machinery in `slm4ie.data`, with one subcommand per
step of the pipeline. Each subcommand selects work the same way: positional keys,
or `--all` for everything the config declares.

    download      raw/<key>/...                     from configs/data/download.yaml
    extract       extracted/<key>.jsonl             from configs/data/extract.yaml
    tasks         tasks/<task>/<dataset>/<split>    from configs/data/tasks.yaml
    tokenization  tokenization/<dataset>.jsonl.gz   from configs/data/tokenization.yaml

The pretraining corpus is built by `scripts/curate_pretraining_corpus.py`: it
runs on an experiment-owned config and has its own stage machinery.

Examples:
    Fetch and normalize the whole catalog:

        uv run python scripts/prepare_datasets.py download --all
        uv run python scripts/prepare_datasets.py extract --all

    Convert task datasets (every family, or one entry):

        uv run python scripts/prepare_datasets.py tasks --all
        uv run python scripts/prepare_datasets.py tasks ner/ssj500k
        uv run python scripts/prepare_datasets.py tasks nli/cb --variant googlemt

    Convert the tokenizer-evaluation lexicons:

        uv run python scripts/prepare_datasets.py tokenization --all
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from slm4ie.data.download import download_datasets
from slm4ie.data.extract import extract_datasets
from slm4ie.data.parallel import configure_script_logging
from slm4ie.data.tasks.driver import convert_tasks
from slm4ie.data.tasks.converters.superglue import DEFAULT_VARIANT, VARIANT_DIRS
from slm4ie.data.tokenization import convert_tokenization_datasets
from slm4ie.utils.cli import (
    add_selection,
    add_workers,
    resolve_config,
    stamped_log_dir,
    validate_selection,
)

logger = logging.getLogger(__name__)


def _add_download_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the `download` subcommand.

    Args:
        subparsers: The subparser registry to extend.
    """
    parser = subparsers.add_parser("download", help="Download raw datasets from their sources.")
    add_selection(
        parser,
        "datasets",
        "Dataset keys to download. Use --all for the full catalog.",
        "Download every dataset declared in the config.",
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to the download config YAML.")
    parser.add_argument(
        "--config-name",
        default="download",
        help="Config in configs/data/ to use, without extension (default: 'download').",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if output exists.")
    parser.add_argument("--output-dir", default=None, help="Override base output directory.")
    role = parser.add_mutually_exclusive_group()
    role.add_argument(
        "--only-benchmarks",
        action="store_true",
        help="Restrict selection to non-pretraining datasets (`role: benchmark` or `role: lexicon`).",
    )
    role.add_argument(
        "--exclude-benchmarks",
        action="store_true",
        help="Keep only pretraining datasets (`role: pretrain`) in the selection.",
    )
    add_workers(
        parser,
        "Datasets downloaded in parallel. 0=auto (min(4, n_datasets), polite to remote servers), 1=serial, N=N threads.",
    )


def _add_extract_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the `extract` subcommand.

    Args:
        subparsers: The subparser registry to extend.
    """
    parser = subparsers.add_parser("extract", help="Normalize raw downloads into unified JSONL.")
    add_selection(
        parser,
        "datasets",
        "Dataset keys to extract (e.g. 'kzb gigafida').",
        "Extract every dataset declared in the config.",
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to the extract config YAML.")
    parser.add_argument(
        "--config-name",
        default="extract",
        help="Config in configs/data/ to use, without extension (default: 'extract').",
    )
    parser.add_argument("--input-dir", default=None, help="Override the configured input directory.")
    parser.add_argument("--output-dir", default=None, help="Override the configured output directory.")
    parser.add_argument("--force", action="store_true", help="Re-extract even if the output already exists.")
    parser.add_argument(
        "--mlflow",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable MLflow extraction tracking, overriding the config. Default: defer to config.",
    )
    add_workers(
        parser,
        (
            "Shard workers used WITHIN each dataset (datasets run sequentially). 0=auto (all cores), "
            "1=serial single pass, N=N worker processes. Only file-based extractors with enough input "
            "files are sharded."
        ),
    )


def _add_tasks_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the `tasks` subcommand.

    Args:
        subparsers: The subparser registry to extend.
    """
    parser = subparsers.add_parser("tasks", help="Convert task datasets into per-split JSONL.")
    add_selection(
        parser,
        "entries",
        "Entry keys to process, e.g. 'ner/ssj500k'.",
        "Process every entry declared in tasks.yaml.",
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to tasks.yaml.")
    parser.add_argument("--force", action="store_true", help="Re-derive outputs even when every split exists.")
    parser.add_argument(
        "--mlflow",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable MLflow per-dataset tracking, overriding tasks.yaml. Default: defer to config.",
    )
    parser.add_argument(
        "--variant",
        choices=tuple(VARIANT_DIRS.keys()),
        default=DEFAULT_VARIANT,
        help=f"SuperGLUE-SL variant to read; ignored by other families (default: {DEFAULT_VARIANT}).",
    )
    add_workers(parser, "Entries processed in parallel. 0=auto (cpu_count // 2), 1=serial, N=N workers.")


def _add_tokenization_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the `tokenization` subcommand.

    Args:
        subparsers: The subparser registry to extend.
    """
    parser = subparsers.add_parser("tokenization", help="Convert tokenizer-evaluation lexicons into JSONL.")
    add_selection(
        parser,
        "datasets",
        "Dataset keys to convert (subset of tokenization.yaml's `datasets`).",
        "Convert every dataset declared in tokenization.yaml.",
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to tokenization.yaml.")
    parser.add_argument(
        "--download-config",
        type=Path,
        default=None,
        help="Path to download.yaml, used to resolve per-dataset raw subdirectories.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing <key>.jsonl.gz outputs.")
    add_workers(parser, "Datasets processed in parallel. 0=auto (cpu_count // 2), 1=serial, N=N workers.")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to `sys.argv`).

    Returns:
        Parsed namespace, including the selected subcommand in `command`.

    Raises:
        SystemExit: If the selection is missing or invalid.
    """
    parser = argparse.ArgumentParser(
        description="Prepare the shared dataset tiers: download, extract, and convert.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_download_parser(subparsers)
    _add_extract_parser(subparsers)
    _add_tasks_parser(subparsers)
    _add_tokenization_parser(subparsers)

    args = parser.parse_args(argv)
    dest = "entries" if args.command == "tasks" else "datasets"
    validate_selection(parser, args, dest)
    return args


def _run_download(args: argparse.Namespace) -> int:
    """Download the selected datasets.

    Args:
        args: Parsed arguments for the `download` subcommand.

    Returns:
        Process exit code.
    """
    config_path = resolve_config(args.config, "configs", "data", f"{args.config_name}.yaml")
    configure_script_logging(parallel=args.max_workers > 1, console_level=logging.WARNING)
    log_dir = stamped_log_dir("download")
    print(f"Logs: {log_dir}", file=sys.stderr)

    try:
        download_datasets(
            config_path=config_path,
            dataset_keys=None if args.all else args.datasets,
            force=args.force,
            output_dir_override=args.output_dir,
            only_benchmarks=args.only_benchmarks,
            exclude_benchmarks=args.exclude_benchmarks,
            max_workers=args.max_workers,
            log_dir=log_dir,
        )
    except Exception as exc:
        logger.error("Download failed: %s", exc)
        return 1
    return 0


def _run_extract(args: argparse.Namespace) -> int:
    """Extract the selected datasets to unified JSONL.

    Args:
        args: Parsed arguments for the `extract` subcommand.

    Returns:
        Process exit code.
    """
    config_path = resolve_config(args.config, "configs", "data", f"{args.config_name}.yaml")
    configure_script_logging(parallel=args.max_workers > 1, console_level=logging.WARNING)
    log_dir = stamped_log_dir("extract")
    print(f"Logs: {log_dir}", file=sys.stderr)

    try:
        extract_datasets(
            config_path=config_path,
            dataset_keys=None if args.all else args.datasets,
            force=args.force,
            max_workers=args.max_workers,
            log_dir=log_dir,
            input_dir_override=args.input_dir,
            output_dir_override=args.output_dir,
            mlflow_enabled=args.mlflow,
        )
    except Exception as exc:
        logger.error("Extraction failed: %s", exc)
        return 1
    return 0


def _run_tasks(args: argparse.Namespace) -> int:
    """Convert the selected task-registry entries.

    Args:
        args: Parsed arguments for the `tasks` subcommand.

    Returns:
        Process exit code.
    """
    config_path = resolve_config(args.config, "configs", "data", "tasks.yaml")
    try:
        summary = convert_tasks(
            config_path=config_path,
            entry_keys=None if args.all else args.entries,
            force=args.force,
            mlflow_enabled=args.mlflow,
            max_workers=args.max_workers,
            options={"variant": args.variant},
        )
    except KeyError as exc:
        # KeyError's str() wraps the message in quotes; print the message itself.
        logger.error("%s", exc.args[0] if exc.args else exc)
        return 1
    return 2 if summary.failed else 0


def _run_tokenization(args: argparse.Namespace) -> int:
    """Convert the selected tokenizer-evaluation datasets.

    Args:
        args: Parsed arguments for the `tokenization` subcommand.

    Returns:
        Process exit code.
    """
    summary = convert_tokenization_datasets(
        config_path=resolve_config(args.config, "configs", "data", "tokenization.yaml"),
        download_config_path=resolve_config(args.download_config, "configs", "data", "download.yaml"),
        dataset_keys=None if args.all else args.datasets,
        force=args.force,
        max_workers=args.max_workers,
    )
    if summary.failed:
        return 2
    # An explicitly named dataset that produced nothing is a failed request,
    # unlike a catalog-wide run where a missing download is expected.
    if not args.all and summary.skipped:
        return 1
    return 0


def main() -> None:
    """Dispatch the selected subcommand."""
    args = parse_args()
    handlers = {
        "download": _run_download,
        "extract": _run_extract,
        "tasks": _run_tasks,
        "tokenization": _run_tokenization,
    }
    sys.exit(handlers[args.command](args))


if __name__ == "__main__":
    main()
