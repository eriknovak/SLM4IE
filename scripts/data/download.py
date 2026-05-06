"""Download datasets from HuggingFace Hub, Clarin.si, and other sources."""

import argparse
import logging
import sys
from pathlib import Path

from slm4ie.data.download import download_datasets


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
    raise FileNotFoundError(
        "Could not find project root (no pyproject.toml)"
    )


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to sys.argv).

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Download datasets for SLM4IE project."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset keys to download (default: all enabled).",
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
    return parser.parse_args(argv)


def main():
    """Run dataset download pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s %(levelname)s %(name)s: %(message)s"
        ),
    )

    args = parse_args()
    project_root = _find_project_root()

    config_path = (
        Path(args.config)
        if args.config
        else project_root
        / "configs"
        / "data"
        / f"{args.config_name}.yaml"
    )

    try:
        download_datasets(
            config_path=config_path,
            dataset_keys=args.datasets,
            force=args.force,
            output_dir_override=args.output_dir,
            only_benchmarks=args.only_benchmarks,
            exclude_benchmarks=args.exclude_benchmarks,
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
