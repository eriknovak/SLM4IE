r"""Evaluate chat LLMs zero-shot on the NER task registry entries.

A thin CLI over `slm4ie.evaluation.run` and `slm4ie.evaluation.memory`. The `run`
subcommand prompts one model once per sentence window of every document of one
split, appends each answer to an answers file, scores the answers and logs one
MLflow run. The `memory` subcommand starts each named local model on the
vllm-service and logs its weight memory and KV-cache memory per token, read from
the container's startup log.

    run      <answers-dir>/<backend>-<model>-<task>-<dataset>-<split>-<examples>-<prompt hash>[-<options hash>].jsonl
    memory   one MLflow run per model, no files

Examples:
    Score Gemma 4 12B from the vllm-service on the SUK validation split:

        uv run --group evaluation python scripts/evaluate_llms.py run \
            --backend vllm --model-key gemma4-12b --entry ner/suk --split val \
            --answers-dir data/experiments/validation/ie-benchmark-slovenian/interim/scratch \
            --experiment slm4ie/validation/ie-benchmark-slovenian --tag tier=scratch

    Read two local models' GPU memory from the vllm-service:

        uv run --group evaluation python scripts/evaluate_llms.py memory \
            --model-key gams-9b --model-key gemma4-12b \
            --experiment slm4ie/validation/ie-benchmark-slovenian --tag tier=scratch

    Continue a Claude run stopped by the subscription's rate limit:

        uv run --group evaluation python scripts/evaluate_llms.py run \
            --backend claude --model-key claude-opus-5-5 --entry ner/slavicner --split test \
            --answers-dir data/experiments/validation/ie-benchmark-slovenian/interim/answers --resume
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from slm4ie.evaluation.backends import available_backends
from slm4ie.evaluation.memory import measure_vllm_memory
from slm4ie.evaluation.prompts import DEFAULT_EXAMPLES
from slm4ie.evaluation.run import BenchmarkRun, load_model_spec, run_benchmark
from slm4ie.utils.cli import resolve_config
from slm4ie.utils.parallel import configure_script_logging

# ===============================
# Logging
# ===============================

logger = logging.getLogger(__name__)

# ===============================
# Defaults
# ===============================

ERROR_EXIT_CODE = 1
EXAMPLE_VARIANTS = ("unseen", "common")
DEFAULT_MODELS_CONFIG = ("experiments", "validation", "ie-benchmark-slovenian", "configs", "benchmark-models.yaml")

# ===============================
# Helper functions
# ===============================


def _parse_pairs(pairs: list[str], flag: str) -> dict[str, Any]:
    """Turn repeated `KEY=VALUE` arguments into a dict, each value read as YAML (`8bit`, `false`, `0`)."""
    parsed = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise SystemExit(f"{flag} takes KEY=VALUE, got {pair!r}")
        parsed[key] = yaml.safe_load(value)
    return parsed


def _tracking_is_unset(experiment: str | None) -> bool:
    """Report and return whether `--experiment` was given without `MLFLOW_TRACKING_URI`.

    Without it, MLflow would write a local store inside the working tree, which a
    tracked run from a snapshot loses.
    """
    if experiment is not None and not os.environ.get("MLFLOW_TRACKING_URI"):
        logger.error("--experiment %s needs MLFLOW_TRACKING_URI: launch through tracking-env.sh", experiment)
        return True
    return False


def _add_run_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the `run` subcommand.

    Args:
        subparsers: The subparser registry to extend.
    """
    parser = subparsers.add_parser("run", help="Answer, score and log one model on one split.")
    parser.add_argument("--backend", required=True, choices=available_backends(), help="Chat backend")
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--model", help="Model in the backend's naming (served name, HF id, alias)")
    model.add_argument("--model-key", help="Model key in --models-config; its backend block sets model and options")
    parser.add_argument(
        "--models-config",
        type=Path,
        default=None,
        help="Model list --model-key reads (default: the ie-benchmark-slovenian benchmark-models.yaml)",
    )
    parser.add_argument("--entry", required=True, help="Task registry entry, such as ner/suk")
    parser.add_argument("--split", required=True, help="Split name, such as val or test")
    parser.add_argument("--answers-dir", type=Path, required=True, help="Directory the answers file is written to")
    parser.add_argument(
        "--examples",
        choices=EXAMPLE_VARIANTS,
        default=DEFAULT_EXAMPLES,
        help=f"Label example variant (default: {DEFAULT_EXAMPLES})",
    )
    parser.add_argument(
        "--option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Backend setting, repeatable, such as quantization=8bit or enable_thinking=false",
    )
    parser.add_argument(
        "--window-budget",
        type=int,
        default=None,
        help="Stop after the first document that brings the window count to this (default: every window)",
    )
    restart = parser.add_mutually_exclusive_group()
    restart.add_argument("--resume", action="store_true", help="Continue an existing answers file")
    restart.add_argument("--force", action="store_true", help="Discard an existing answers file and start over")
    parser.add_argument("--experiment", default=None, help="MLflow experiment name (default: no tracking)")
    parser.add_argument(
        "--tag", action="append", default=[], metavar="KEY=VALUE", help="MLflow tag, repeatable, such as tier=scratch"
    )
    parser.add_argument(
        "--tasks-config", type=Path, default=None, help="Task registry (default: configs/data/tasks.yaml)"
    )
    parser.add_argument(
        "--labels-config",
        type=Path,
        default=None,
        help="Label descriptions (default: configs/evaluation/ner-labels.yaml)",
    )


def _run(args: argparse.Namespace) -> int:
    """Run one model on one split.

    Args:
        args: Parsed arguments for the `run` subcommand.

    Returns:
        Process exit code.
    """
    configure_script_logging(parallel=False)
    if _tracking_is_unset(args.experiment):
        return ERROR_EXIT_CODE

    # take the model and its options from the model list when a key names it
    model, options = args.model, {}
    if args.model_key:
        models_config = resolve_config(args.models_config, *DEFAULT_MODELS_CONFIG)
        try:
            model, options = load_model_spec(models_config, args.model_key, args.backend)
        except KeyError as exc:
            logger.error("Evaluation failed: %s", exc)
            return ERROR_EXIT_CODE
    options.update(_parse_pairs(args.option, "--option"))
    tags = {key: str(value) for key, value in _parse_pairs(args.tag, "--tag").items()}
    if args.model_key:
        tags["model_key"] = args.model_key

    run = BenchmarkRun(
        backend=args.backend,
        model=model,
        entry=args.entry,
        split=args.split,
        tasks_config=resolve_config(args.tasks_config, "configs", "data", "tasks.yaml"),
        labels_config=resolve_config(args.labels_config, "configs", "evaluation", "ner-labels.yaml"),
        answers_dir=args.answers_dir,
        examples=args.examples,
        backend_options=options,
        window_budget=args.window_budget,
        resume=args.resume,
        force=args.force,
        experiment=args.experiment,
        tags=tags,
    )
    try:
        run_benchmark(run)
    except (KeyError, FileExistsError) as exc:
        logger.error("Evaluation failed: %s", exc)
        return ERROR_EXIT_CODE
    return 0


def _add_memory_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the `memory` subcommand.

    Args:
        subparsers: The subparser registry to extend.
    """
    parser = subparsers.add_parser("memory", help="Read local models' GPU memory from the vllm-service.")
    parser.add_argument(
        "--model-key", action="append", required=True, help="Model key in --models-config with a vllm block, repeatable"
    )
    parser.add_argument(
        "--models-config",
        type=Path,
        default=None,
        help="Model list --model-key reads (default: the ie-benchmark-slovenian benchmark-models.yaml)",
    )
    parser.add_argument("--experiment", default=None, help="MLflow experiment name (default: no tracking)")
    parser.add_argument(
        "--tag", action="append", default=[], metavar="KEY=VALUE", help="MLflow tag, repeatable, such as tier=scratch"
    )


def _memory(args: argparse.Namespace) -> int:
    """Read each named model's GPU memory, one model at a time.

    Args:
        args: Parsed arguments for the `memory` subcommand.

    Returns:
        Process exit code.
    """
    configure_script_logging(parallel=False)
    if _tracking_is_unset(args.experiment):
        return ERROR_EXIT_CODE
    models_config = resolve_config(args.models_config, *DEFAULT_MODELS_CONFIG)
    tags = {key: str(value) for key, value in _parse_pairs(args.tag, "--tag").items()}
    for model_key in args.model_key:
        try:
            measure_vllm_memory(model_key, models_config, args.experiment, tags)
        except (KeyError, ValueError, subprocess.CalledProcessError) as exc:
            logger.error("Memory reading failed for %s: %s", model_key, exc)
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
    """
    parser = argparse.ArgumentParser(description="Evaluate chat LLMs zero-shot on the NER task registry entries.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_run_parser(subparsers)
    _add_memory_parser(subparsers)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Dispatch the selected subcommand.

    Args:
        argv: Optional argument list (defaults to `sys.argv`).

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    if args.command == "memory":
        return _memory(args)
    return _run(args)


if __name__ == "__main__":
    sys.exit(main())
