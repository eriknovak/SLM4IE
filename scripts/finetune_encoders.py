r"""Fine-tune pretrained encoders on task registry entries and score them on test.

A thin CLI over `slm4ie.finetune`. The `run` subcommand loads the shared
fine-tuning config, merges an experiment's dial over it, and runs every
combination of encoder, entry, learning rate and seed. Each run trains the
encoder with its family's task head on the entry's train split, keeps the best
epoch on val, scores test and, with `--experiment`, logs one MLflow run. The
`search` subcommand runs the config's hyperparameter search instead: one Optuna
study per encoder and entry, picked on val, whose best setting is then
fine-tuned once per seed and scored on test.

    run      one MLflow run per (encoder, entry, learning rate, seed), no files
    search   one Optuna study per (encoder, entry) in --storage; one MLflow
             parent run per study, a child per trial and per final seed

Parameters of `run` (`search` takes the same):
    entries          Task registry keys such as `pos/ssj500k`; or `--all` for
                     every entry that writes train, val and test splits. `--all`
                     skips, with a warning, an entry whose split files are
                     missing on disk; a named entry with missing files fails.
    --encoder        HuggingFace name or local path, repeatable; defaults to the
                     dial's `encoders`.
    --config         The shared config (default: configs/evaluation/finetune.yaml).
    --dial           An experiment's dial, deep-merged over the shared config.
    --dry-run        Print the planned runs and their count; train nothing.
    --experiment     MLflow experiment name (default: no tracking).
    --tag            MLflow tag KEY=VALUE, repeatable; tracking needs `tier`.
    --device         Torch device, such as cuda:0 (default: first GPU, else CPU).

Parameters of `search` only:
    --storage        Optuna storage URL with an absolute path, such as
                     sqlite:////abs/path/optuna.db, so a stopped search resumes
                     (default: $OPTUNA_STORAGE, which tracking-env.sh --study sets).

Examples:
    List the runs of two encoders on the POS entries without training:

        uv run --group evaluation python scripts/finetune_encoders.py run \
            pos/ssj500k pos/suk --encoder EMBEDDIA/sloberta --encoder xlm-roberta-base --dry-run

    Fine-tune one encoder on every trainable entry and log each run:

        uv run --group evaluation python scripts/finetune_encoders.py run --all \
            --encoder EMBEDDIA/sloberta --experiment slm4ie/encoder-baselines --tag tier=scratch

    Search the settings of one encoder on one entry with the dial's search block:

        uv run --group evaluation python scripts/finetune_encoders.py search pos/ssj500k \
            --encoder EMBEDDIA/sloberta --dial experiments/programmes/<programme>/<slug>/configs/search-lr.yaml \
            --storage sqlite:////abs/path/optuna.db --experiment slm4ie/<slug> --tag tier=scratch
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from slm4ie.data.tasks.config import TaskEntry, TasksRoots, load_tasks_config
from slm4ie.finetune.config import FinetuneConfig, load_finetune_config, planned_runs
from slm4ie.finetune.run import missing_splits, unwritten_splits
from slm4ie.finetune.search import search_finetune, study_name
from slm4ie.finetune.tracking import run_name, track_finetune
from slm4ie.utils.cli import add_keys, add_tags, parse_tags, resolve_config, validate_keys
from slm4ie.utils.parallel import configure_script_logging

# ===============================
# Logging
# ===============================

logger = logging.getLogger(__name__)

# ===============================
# Defaults
# ===============================

ERROR_EXIT_CODE = 1
DEFAULT_CONFIG = ("configs", "evaluation", "finetune.yaml")
STORAGE_ENV = "OPTUNA_STORAGE"

# ===============================
# Helper functions
# ===============================


def _select_entries(entries: list[TaskEntry], roots: TasksRoots, keys: list[str], select_all: bool) -> list[TaskEntry]:
    """Return the named entries, or with `select_all` every entry that can be fine-tuned.

    With `select_all`, an entry whose role keeps no train split is left out, and
    an entry whose train, val or test file is missing on disk is skipped with a
    warning that names it.

    Raises:
        ValueError: If a key names no entry, or a named entry writes no train,
            val or test split, or one of those files is missing on disk.
    """
    trainable = [e for e in entries if not unwritten_splits(e)]
    if select_all:
        missing = {e.key: missing_splits(e, roots) for e in trainable}
        skipped = sorted(key for key, paths in missing.items() if paths)
        if skipped:
            logger.warning("skipped=%d entries=%s reason=split files missing on disk", len(skipped), skipped)
        return [e for e in trainable if not missing[e.key]]

    # resolve the keys in the order given, refusing any that cannot be trained
    by_key = {e.key: e for e in entries}
    unknown = [key for key in keys if key not in by_key]
    if unknown:
        raise ValueError(f"unknown task registry entries {unknown}; known: {sorted(by_key)}")
    untrainable = [key for key in keys if by_key[key] not in trainable]
    if untrainable:
        raise ValueError(f"entries {untrainable} write no train, val and test splits; they cannot be fine-tuned")
    for key in keys:
        if missing := missing_splits(by_key[key], roots):
            raise ValueError(f"entry '{key}' has split files missing on disk: {[str(path) for path in missing]}")
    return [by_key[key] for key in keys]


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the selection, config, tracking and device arguments `run` and `search` share."""
    add_keys(
        parser,
        "entries",
        "Task registry entries, such as pos/ssj500k.",
        "Fine-tune on every entry whose train, val and test splits are on disk.",
    )
    parser.add_argument(
        "--encoder",
        action="append",
        default=[],
        help="HuggingFace name or local path, repeatable (default: the dial's encoders)",
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="Shared config (default: configs/evaluation/finetune.yaml)"
    )
    parser.add_argument("--dial", type=Path, default=None, help="Experiment dial merged over the shared config")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned runs and their count; train nothing")
    parser.add_argument("--experiment", default=None, help="MLflow experiment name (default: no tracking)")
    add_tags(parser)
    parser.add_argument("--device", default=None, help="Torch device, such as cuda:0 (default: first GPU, else CPU)")


def _add_run_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the `run` subcommand.

    Args:
        subparsers: The subparser registry to extend.
    """
    parser = subparsers.add_parser("run", help="Fine-tune every encoder, entry, learning rate and seed.")
    _add_common_arguments(parser)


def _add_search_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the `search` subcommand.

    Args:
        subparsers: The subparser registry to extend.
    """
    parser = subparsers.add_parser(
        "search", help="Search each encoder's settings on each entry, then score the best on test per seed."
    )
    _add_common_arguments(parser)
    parser.add_argument(
        "--storage", default=None, help=f"Optuna storage URL with an absolute path (default: ${STORAGE_ENV})"
    )


def _load(args: argparse.Namespace) -> tuple[FinetuneConfig, TasksRoots, list[TaskEntry], list[str], dict[str, str]]:
    """Load the config, the dial and the task registry, and resolve the entries, encoders and tags.

    Raises:
        FileNotFoundError: If a config file is missing.
        KeyError: If the task registry lacks a key it needs.
        ValueError: If the config is malformed, an entry cannot be fine-tuned,
            a tag is malformed, or no encoder is named.
    """
    config = load_finetune_config(resolve_config(args.config, *DEFAULT_CONFIG), args.dial)
    tasks = load_tasks_config(config.tasks_config)
    entries = _select_entries(tasks.entries, tasks.roots, args.entries, args.all)
    tags = parse_tags(args.tag)
    encoders = args.encoder or list(config.encoders)
    if not encoders:
        raise ValueError("no encoder; pass --encoder or name encoders in the dial")
    return config, tasks.roots, entries, encoders, tags


def _tracking_ready(experiment: str | None) -> bool:
    """Tell whether tracking can start; without a tracking server MLflow would write a store inside the tree."""
    if experiment is not None and not os.environ.get("MLFLOW_TRACKING_URI"):
        logger.error("--experiment %s needs MLFLOW_TRACKING_URI: launch through tracking-env.sh", experiment)
        return False
    return True


def _run(args: argparse.Namespace) -> int:
    """Plan the grid of runs, then print it or fine-tune each run in turn.

    Args:
        args: Parsed arguments for the `run` subcommand.

    Returns:
        Process exit code.
    """
    configure_script_logging(parallel=False)

    # plan the runs from the config, the dial and the selection
    try:
        config, roots, entries, encoders, tags = _load(args)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        logger.error("Fine-tuning failed: %s", exc)
        return ERROR_EXIT_CODE
    runs = planned_runs(config, encoders, entries)

    # print the plan and stop on a dry run
    if args.dry_run:
        for planned in runs:
            print(run_name(planned))
        print(f"{len(runs)} planned runs")
        return 0

    if not _tracking_ready(args.experiment):
        return ERROR_EXIT_CODE

    # fine-tune each run, stopping at the first failure
    for index, planned in enumerate(runs, start=1):
        logger.info("run %d/%d: %s", index, len(runs), run_name(planned))
        try:
            track_finetune(planned, roots, args.experiment, tags, device=args.device)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            logger.exception("Fine-tuning failed at %s: %s", run_name(planned), exc)
            return ERROR_EXIT_CODE
    return 0


def _search(args: argparse.Namespace) -> int:
    """Plan one study per encoder and entry, then print the plan or run each study in turn.

    Args:
        args: Parsed arguments for the `search` subcommand.

    Returns:
        Process exit code.
    """
    configure_script_logging(parallel=False)

    # plan the studies from the config's search block and the selection
    try:
        config, roots, entries, encoders, tags = _load(args)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        logger.error("Search failed: %s", exc)
        return ERROR_EXIT_CODE
    search = config.search
    if search is None:
        logger.error("Search failed: the config and dial have no 'search' block")
        return ERROR_EXIT_CODE
    studies = [(encoder, entry) for encoder in encoders for entry in entries]

    # print the plan and stop on a dry run
    if args.dry_run:
        finals = 0
        for encoder, entry in studies:
            seeds = len(config.settings_for(entry.task).seeds)
            finals += seeds
            name = study_name(encoder, entry, config.settings_for(entry.task), search)
            print(f"{name}: {search.trials} trials, {seeds} final seeds")
        print(f"{len(studies)} studies, {len(studies) * search.trials} trials, {finals} final runs")
        return 0

    # a stored study resumes; one in memory would be lost with the process
    storage = args.storage or os.environ.get(STORAGE_ENV)
    if not storage:
        logger.error("Search failed: pass --storage or set %s to an absolute sqlite URL", STORAGE_ENV)
        return ERROR_EXIT_CODE
    if not _tracking_ready(args.experiment):
        return ERROR_EXIT_CODE

    # run each study, stopping at the first failure
    for index, (encoder, entry) in enumerate(studies, start=1):
        name = study_name(encoder, entry, config.settings_for(entry.task), search)
        logger.info("study %d/%d: %s", index, len(studies), name)
        try:
            search_finetune(
                encoder,
                entry,
                config.settings_for(entry.task),
                search,
                roots,
                storage=storage,
                experiment=args.experiment,
                tags=tags,
                device=args.device,
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            logger.exception("Search failed at %s: %s", name, exc)
            return ERROR_EXIT_CODE
    return 0


# ===============================
# Main function
# ===============================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to `sys.argv`).

    Returns:
        Parsed namespace, including the selected subcommand in `command`.

    Raises:
        SystemExit: If the arguments are invalid, such as both or neither of
            entries and `--all`.
    """
    parser = argparse.ArgumentParser(description="Fine-tune pretrained encoders on task registry entries.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_run_parser(subparsers)
    _add_search_parser(subparsers)
    args = parser.parse_args(argv)
    validate_keys(subparsers.choices[args.command], args, "entries")
    return args


def main(argv: list[str] | None = None) -> int:
    """Dispatch the selected subcommand.

    Args:
        argv: Optional argument list (defaults to `sys.argv`).

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    return _search(args) if args.command == "search" else _run(args)


if __name__ == "__main__":
    sys.exit(main())
