"""Tokenize the curated pretraining corpus into pretokenized outputs for other projects.

One CLI over `slm4ie.data.pretokenize`, with two subcommands. Both select
entries of the registry (`configs/data/pretokenize.yaml`) the same way:
positional entry keys, or `--all`.

    run    tokenize each entry's corpus stage with its exported tokenizer and write
           pretokenized/<corpus>/<tokenizer>-<length>/<format>/{train,validation}/
           plus manifest.json; finished shards are reused after an interruption
    check  read a finished output back and verify it against its manifest

The output formats are pluggable (`mds` for ModernBERT's non-streaming loader,
`mds-streaming` for its streaming loader, `parquet` for HuggingFace `datasets`),
chosen per entry. The tokenizer is an export of `sweep_tokenizers.py export`.

Examples:
    Convert the ModernSloBERT entry, then check it:

        uv run --group pretokenize python scripts/pretokenize_corpus.py run sentence-dedup/bpe-32000-modernbert-1024
        uv run --group pretokenize python scripts/pretokenize_corpus.py check sentence-dedup/bpe-32000-modernbert-1024

    Convert every entry from scratch, 16 shards at a time:

        uv run --group pretokenize python scripts/pretokenize_corpus.py run --all --force --max-workers 16
"""

import argparse
import logging
import sys
from pathlib import Path

from slm4ie.data.pretokenize import PretokenizeConfig, check_pretokenized, load_pretokenize_config, run_pretokenize
from slm4ie.utils.cli import add_keys, add_workers, resolve_config, validate_keys
from slm4ie.utils.parallel import configure_script_logging

# ===============================
# Logging
# ===============================

logger = logging.getLogger(__name__)

# ===============================
# Defaults
# ===============================

ERROR_EXIT_CODE = 1
FAILED_EXIT_CODE = 2

# ===============================
# Helper functions
# ===============================


def _add_common_arguments(parser: argparse.ArgumentParser, verb: str) -> None:
    """Add the entry selection and the config path to a subcommand.

    Args:
        parser: The subcommand parser.
        verb: What the subcommand does to an entry, for the help text.
    """
    add_keys(parser, "entries", f"Entry keys to {verb}.", f"{verb.capitalize()} every entry of the registry.")
    parser.add_argument("--config", type=Path, default=None, help="Path to pretokenize.yaml.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to `sys.argv`).

    Returns:
        Parsed namespace, including the selected subcommand in `command`.

    Raises:
        SystemExit: If no subcommand or no entry selection is given.
    """
    parser = argparse.ArgumentParser(description="Tokenize the pretraining corpus into pretokenized outputs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Convert the selected entries.")
    _add_common_arguments(run_parser, "convert")
    run_parser.add_argument("--force", action="store_true", help="Convert from scratch, replacing existing output.")
    add_workers(
        run_parser, "Input shards converted in parallel. 0=auto (by cores, items and memory), 1=serial, N=N workers."
    )

    check_parser = subparsers.add_parser("check", help="Verify finished outputs against their manifest.")
    _add_common_arguments(check_parser, "check")

    args = parser.parse_args(argv)
    validate_keys(parser, args, "entries")
    return args


def _selected(args: argparse.Namespace) -> tuple[PretokenizeConfig, list[str]]:
    """Load the registry and resolve the selected entry keys.

    Args:
        args: Parsed arguments.

    Returns:
        The loaded registry and the selected keys.

    Raises:
        KeyError: If a selected key is not in the registry, listing the declared ones.
    """
    config = load_pretokenize_config(resolve_config(args.config, "configs", "data", "pretokenize.yaml"))
    keys = sorted(config.entries) if args.all else args.entries
    unknown = sorted(set(keys) - set(config.entries))
    if unknown:
        raise KeyError(f"Unknown pretokenize entries {unknown}; declared: {sorted(config.entries)}")
    return config, keys


def _run(args: argparse.Namespace) -> int:
    """Convert the selected entries.

    Args:
        args: Parsed arguments for the `run` subcommand.

    Returns:
        Process exit code.
    """
    configure_script_logging(parallel=args.max_workers != 1, console_level=logging.INFO)
    try:
        config, keys = _selected(args)
        summary = run_pretokenize(config, keys, force=args.force, max_workers=args.max_workers)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        logger.error("%s", exc.args[0] if exc.args else exc)
        return ERROR_EXIT_CODE
    for key, parts in summary.failed.items():
        logger.error("%s: %d input shards failed: %s", key, len(parts), ", ".join(parts))
    return FAILED_EXIT_CODE if summary.failed else 0


def _check(args: argparse.Namespace) -> int:
    """Check the selected entries' finished outputs.

    Args:
        args: Parsed arguments for the `check` subcommand.

    Returns:
        Process exit code.
    """
    configure_script_logging(parallel=False, console_level=logging.INFO)
    try:
        config, keys = _selected(args)
    except KeyError as exc:
        logger.error("%s", exc.args[0])
        return ERROR_EXIT_CODE
    failed = False
    for key in keys:
        report = check_pretokenized(config.entries[key].output_dir)
        for warning in report.warnings:
            logger.warning("%s: %s", key, warning)
        for problem in report.problems:
            logger.error("%s: %s", key, problem)
        logger.info("%s: %s", key, "ok" if report.ok else f"{len(report.problems)} problems")
        failed = failed or not report.ok
    return FAILED_EXIT_CODE if failed else 0


# ===============================
# Main function
# ===============================


def main() -> None:
    """Dispatch the selected subcommand."""
    args = parse_args()
    handlers = {"run": _run, "check": _check}
    sys.exit(handlers[args.command](args))


if __name__ == "__main__":
    main()
