"""Argument-parsing helpers shared by the scripts in `scripts/`.

Every CLI in this repo selects work the same way (positional keys or `--all`),
resolves a config path against the project root, and writes per-key logs under
`logs/<name>/<timestamp>/`. These helpers own that shape so each script only
declares what is genuinely its own.

Helpers only: nothing here parses `sys.argv`, exits, or defines a `main`, so
`slm4ie` stays an importable library.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from slm4ie.data.io_utils import find_project_root


def add_selection(
    parser: argparse.ArgumentParser,
    dest: str,
    item_help: str,
    all_help: str,
) -> None:
    """Add the positional-keys / `--all` selection pair to `parser`.

    Args:
        parser: Parser to extend.
        dest: Name of the positional argument (e.g. `datasets`, `entries`).
        item_help: Help text for the positional argument.
        all_help: Help text for `--all`.
    """
    target = parser.add_mutually_exclusive_group()
    target.add_argument(dest, nargs="*", default=[], help=item_help)
    target.add_argument("--all", action="store_true", help=all_help)


def validate_selection(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    dest: str,
) -> None:
    """Fail unless exactly one of the positional keys or `--all` was given.

    argparse accepts a `nargs="*"` positional inside a required mutually
    exclusive group even when empty, so the exclusive-or is checked by hand.

    Args:
        parser: Parser used to report the error.
        args: Parsed arguments to validate.
        dest: Name of the positional argument added by `add_selection`.

    Raises:
        SystemExit: If both or neither selection form was provided.
    """
    selected = getattr(args, dest)
    if args.all and selected:
        parser.error(f"argument --all: not allowed with positional {dest}")
    if not args.all and not selected:
        parser.error(f"one of the arguments {dest} --all is required")


def add_workers(parser: argparse.ArgumentParser, help_text: str) -> None:
    """Add the `--max-workers` flag to `parser`.

    Args:
        parser: Parser to extend.
        help_text: Help text describing what the workers do for this command.
    """
    parser.add_argument("--max-workers", type=int, default=0, help=help_text)


def resolve_config(explicit: Optional[Path], *relative_parts: str) -> Path:
    """Return `explicit` when given, else the project-relative default path.

    Args:
        explicit: Path passed on the command line, or None.
        *relative_parts: Path segments of the default, relative to the project
            root (e.g. `"configs", "data", "download.yaml"`).

    Returns:
        The resolved config path.
    """
    if explicit is not None:
        return Path(explicit)
    return find_project_root().joinpath(*relative_parts)


def stamped_log_dir(name: str) -> Path:
    """Return a fresh per-run log directory under the project root.

    Args:
        name: Command name used as the log subdirectory (e.g. `download`).

    Returns:
        Path to `logs/<name>/<UTC timestamp>/`. The directory is not created;
        `run_parallel` creates it when it writes the first per-key log.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return find_project_root() / "logs" / name / stamp
