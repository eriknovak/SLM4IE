"""Run the tokenizer comparison sweep: sample, train, evaluate, export.

One CLI over `slm4ie.tokenizers`, with one subcommand per step of the sweep.
Every subcommand reads the same sweep config (`--config`) and selects runs the
same way: `--all`, or `--tokenizer` optionally narrowed by `--vocab-size`.

    sample    materialize the shared training sample + morpheme lexicon, and with a
              corpus split the split itself and the evaluation sample
    train     train the selected tokenizers, logging runs to MLflow
    evaluate  score trained artifacts and write the comparison report
    export    write HuggingFace tokenizer files into each artifact directory

`sample` is optional: `train` prepares the same inputs when they are missing.
Running it up front decouples the expensive sampling from training, so repeated
or parallel-capped sweeps reuse one identical persistent sample.

Examples:
    Materialize the shared sample (no-op if it already exists):

        uv run python scripts/sweep_tokenizers.py sample --config $SWEEP

    Train the whole sweep, or one run:

        uv run python scripts/sweep_tokenizers.py train --config $SWEEP --all
        uv run python scripts/sweep_tokenizers.py train --config $SWEEP --tokenizer bpe --vocab-size 16000

    Evaluate and export:

        uv run python scripts/sweep_tokenizers.py evaluate --config $SWEEP --all
        uv run python scripts/sweep_tokenizers.py export --config $SWEEP --all
"""

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

from slm4ie.utils.parallel import configure_script_logging
from slm4ie.tokenizers.evaluate import evaluate_sweep
from slm4ie.tokenizers.config import SweepConfig, load_sweep_config
from slm4ie.tokenizers.hf_export import export_sweep_runs
from slm4ie.tokenizers.corpus import check_sample_side, materialize_sample
from slm4ie.tokenizers.train import corpus_split_for, prepare_inputs, resolve_sweep_runs, train_sweep
from slm4ie.utils.cli import add_workers
from slm4ie.utils.mlflow import DEFAULT_TRACKING_URI, resolve_tracking_uri

# ===============================
# Logging
# ===============================

logger = logging.getLogger(__name__)

# ===============================
# Defaults
# ===============================

BYTES_PER_MIB = 1024 * 1024
FAILED_EXIT_CODE = 2
NO_ARTIFACTS_EXIT_CODE = 1

# ===============================
# Helper functions
# ===============================


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    """Add the required sweep-config argument.

    Args:
        parser: Subcommand parser to extend.
    """
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the sweep config, e.g. configs/tokenizers/sweep.yaml.",
    )


def _add_run_selection(parser: argparse.ArgumentParser, verb: str) -> None:
    """Add the `--all` / `--tokenizer` / `--vocab-size` run selection.

    Args:
        parser: Subcommand parser to extend.
        verb: What the subcommand does, used in the help text (e.g. `train`).
    """
    parser.add_argument(
        "--tokenizer",
        default=None,
        help=f"{verb.capitalize()} this one tokenizer (all its vocab sizes unless --vocab-size narrows it).",
    )
    parser.add_argument("--vocab-size", type=int, default=None, help="Narrow --tokenizer to this single vocab size.")
    parser.add_argument("--all", action="store_true", help=f"{verb.capitalize()} every run in the sweep.")


def _add_tier_argument(parser: argparse.ArgumentParser) -> None:
    """Add `--tier`, the labflow run tier tagged on every MLflow run.

    Args:
        parser: Subcommand parser to extend.
    """
    parser.add_argument(
        "--tier",
        choices=("scratch", "tracked"),
        default="scratch",
        help="labflow run tier tagged on every MLflow run: scratch for the inner loop, tracked for cited runs.",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to `sys.argv`).

    Returns:
        Parsed namespace, including the selected subcommand in `command`.

    Raises:
        SystemExit: If no subcommand is given.
    """
    parser = argparse.ArgumentParser(description="Run the tokenizer comparison sweep declared in the sweep config.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample_parser = subparsers.add_parser("sample", help="Materialize the shared training sample and lexicon.")
    _add_config_argument(sample_parser)
    sample_parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the sample and lexicon even when they already exist.",
    )
    add_workers(sample_parser, "Sources ranked at once when the corpus split is fitted. 0=1.")

    train_parser = subparsers.add_parser("train", help="Train the selected tokenizers.")
    _add_config_argument(train_parser)
    _add_run_selection(train_parser, "train")
    _add_tier_argument(train_parser)
    train_parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain existing artifacts; the sample and table are kept (redraw with `sample --force`).",
    )
    add_workers(train_parser, "Parallel runs. 0=auto (cpu_count // 2), 1=serial, N=N workers.")

    evaluate_parser = subparsers.add_parser("evaluate", help="Score trained tokenizers and write the report.")
    _add_config_argument(evaluate_parser)
    _add_run_selection(evaluate_parser, "evaluate")
    _add_tier_argument(evaluate_parser)
    evaluate_parser.add_argument(
        "--morph-form-sample",
        type=int,
        default=None,
        help="Override the morph form-sample size (config stats.morph_form_sample). Useful for quick validation.",
    )
    evaluate_parser.add_argument(
        "--n-resamples",
        type=int,
        default=None,
        help="Override the bootstrap resample count B (config stats.n_resamples). Useful for quick validation.",
    )
    evaluate_parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the gold lexicons; the evaluation sample is kept (redraw with `sample --force`).",
    )
    add_workers(evaluate_parser, "Parallel evaluations. 0=auto, 1=serial, N=N workers.")

    export_parser = subparsers.add_parser("export", help="Export trained tokenizers as HuggingFace directories.")
    _add_config_argument(export_parser)
    _add_run_selection(export_parser, "export")

    return parser.parse_args(argv)


def _select_runs(cfg: SweepConfig, args: argparse.Namespace) -> list[str]:
    """Resolve the run keys named by the CLI selection.

    Args:
        cfg: The resolved sweep configuration.
        args: Parsed arguments carrying the selection flags.

    Returns:
        The selected run keys.

    Raises:
        SystemExit: If the selection is invalid.
    """
    try:
        return resolve_sweep_runs(cfg, all_runs=args.all, tokenizer=args.tokenizer, vocab_size=args.vocab_size)
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(FAILED_EXIT_CODE)


def _trained_runs(cfg: SweepConfig, args: argparse.Namespace) -> list[str]:
    """Resolve the selection down to runs that have a trained artifact.

    Args:
        cfg: The resolved sweep configuration.
        args: Parsed arguments carrying the selection flags.

    Returns:
        The selected run keys whose artifact directory holds a `metadata.json`.

    Raises:
        SystemExit: If the selection is invalid or nothing is trained yet.
    """
    keys = [k for k in _select_runs(cfg, args) if (cfg.output_root / k / "metadata.json").exists()]
    if not keys:
        logger.error("No trained artifacts found under %s. Run `sweep_tokenizers.py train` first.", cfg.output_root)
        sys.exit(NO_ARTIFACTS_EXIT_CODE)
    return keys


def _describe(path: Path) -> str:
    """Return a human-readable `path (size)` string for logging.

    Args:
        path: An existing file path.

    Returns:
        The path followed by its size in MiB, or just the path when its size
        cannot be read.
    """
    try:
        mib = path.stat().st_size / BYTES_PER_MIB
    except OSError:
        return str(path)
    return f"{path} ({mib:.1f} MiB)"


def _run_sample(cfg: SweepConfig, args: argparse.Namespace) -> int:
    """Materialize the persistent training sample and morpheme lexicon.

    Args:
        cfg: The resolved sweep configuration.
        args: Parsed arguments for the `sample` subcommand.

    Returns:
        Process exit code.
    """
    configure_script_logging(parallel=False, console_level=logging.INFO)
    cfg.output_root.mkdir(parents=True, exist_ok=True)

    # prepare the training sample and lexicon
    max_workers = max(1, args.max_workers)
    sample_path, lexicon_path = prepare_inputs(cfg, force=args.force, max_workers=max_workers)

    # materialize the evaluation sample
    split = corpus_split_for(cfg, max_workers)
    if split is not None and (args.force or not cfg.eval_sample_path.exists()):
        materialize_sample(cfg.corpus_root, cfg.eval_budget, cfg.eval_sample_path, split, "eval")
        logger.info("Evaluation sample ready: %s", _describe(cfg.eval_sample_path))
    elif split is not None:
        check_sample_side(cfg.eval_sample_path, "eval")

    # report what is ready
    logger.info("Training sample ready: %s", _describe(sample_path))
    if lexicon_path is not None:
        logger.info("Morpheme lexicon ready: %s", _describe(lexicon_path))
    else:
        logger.info("No morphological backend in the sweep; lexicon not built.")

    # check the sample exists
    if not sample_path.exists():
        logger.error("Sample was not materialized: %s", sample_path)
        return 1
    return 0


def _run_train(cfg: SweepConfig, args: argparse.Namespace) -> int:
    """Train the selected runs.

    Args:
        cfg: The resolved sweep configuration.
        args: Parsed arguments for the `train` subcommand.

    Returns:
        Process exit code.
    """
    keys = _select_runs(cfg, args)
    if not keys:
        logger.warning("No runs selected; nothing to do.")
        return 0
    summary = train_sweep(cfg, keys, force=args.force, max_workers=args.max_workers)
    return FAILED_EXIT_CODE if summary.failed else 0


def _run_evaluate(cfg: SweepConfig, args: argparse.Namespace) -> int:
    """Evaluate the selected trained runs and write the report.

    Args:
        cfg: The resolved sweep configuration.
        args: Parsed arguments for the `evaluate` subcommand.

    Returns:
        Process exit code.
    """
    summary = evaluate_sweep(
        cfg,
        _trained_runs(cfg, args),
        force=args.force,
        morph_form_sample=args.morph_form_sample,
        n_resamples=args.n_resamples,
        max_workers=args.max_workers,
    )
    return FAILED_EXIT_CODE if summary.failed else 0


def _run_export(cfg: SweepConfig, args: argparse.Namespace) -> int:
    """Export the selected trained runs as HuggingFace tokenizer directories.

    Args:
        cfg: The resolved sweep configuration.
        args: Parsed arguments for the `export` subcommand.

    Returns:
        Process exit code.
    """
    configure_script_logging(parallel=False, console_level=logging.INFO)
    summary = export_sweep_runs(cfg.output_root, _trained_runs(cfg, args))
    return FAILED_EXIT_CODE if summary.failed else 0


# ===============================
# Main function
# ===============================


def main() -> None:
    """Dispatch the selected subcommand."""
    args = parse_args()

    # load the config and tag the run tier
    cfg = load_sweep_config(args.config)
    if getattr(args, "tier", None):
        cfg = dataclasses.replace(cfg, mlflow_tags={**cfg.mlflow_tags, "tier": args.tier})
    if getattr(args, "tier", None) == "tracked" and cfg.mlflow_enabled:
        if resolve_tracking_uri(cfg.mlflow_tracking_uri) == DEFAULT_TRACKING_URI:
            sys.exit(
                f"--tier tracked needs MLFLOW_TRACKING_URI set; refusing the local {DEFAULT_TRACKING_URI} fallback"
            )

    # dispatch the subcommand
    handlers = {
        "sample": _run_sample,
        "train": _run_train,
        "evaluate": _run_evaluate,
        "export": _run_export,
    }
    sys.exit(handlers[args.command](cfg, args))


if __name__ == "__main__":
    main()
