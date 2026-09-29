"""CLI over the language-model document judge.

Parses arguments and dispatches into `slm4ie.data.judge`, which owns the
batching, the subprocess calls, the schema check and the resume behaviour.

Two tasks share the machinery. `--task rubric` labels a sample of documents
against a rubric prompt, one JSON verdict per line. `--task pairwise` draws a
kept and a dropped document from the same source and stage and asks which is
the better text, each pair asked twice with the two swapped; the drawing is
written beside the verdicts and joined to them at analysis time.

Verdicts already in the output file are never re-requested, so an interrupted
run continues where it stopped, and `--max-batches` bounds what a single
invocation costs.

Examples:
    export EXP=data/experiments/data/curation-quality-slovenian

    # Judge the whole sample through the Batch API (half price).
    uv run python scripts/judge_documents.py \
        --source $EXP/interim/sample.jsonl \
        --out $EXP/interim/verdicts.jsonl \
        --rubric experiments/data/curation-quality-slovenian/configs/judge-rubric.md \
        --model claude-haiku-4-5

    # Try two requests through the local CLI, no API key needed.
    uv run python scripts/judge_documents.py \
        --source $EXP/interim/sample.jsonl \
        --out $EXP/interim/verdicts.jsonl \
        --rubric experiments/data/curation-quality-slovenian/configs/judge-rubric.md \
        --backend cli --max-batches 2

    # Compare what each content filter kept against what it dropped.
    uv run python scripts/judge_documents.py \
        --source $EXP/interim/sample.jsonl \
        --out $EXP/interim/pairwise-sonnet.jsonl \
        --rubric experiments/data/curation-quality-slovenian/configs/judge-pairwise.md \
        --task pairwise --backend cli
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from slm4ie.data.judge import BACKENDS, PAIRWISE_STAGES, TASKS, judge_documents, judge_pairs


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to `sys.argv`).

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Label sampled documents with a language-model judge. Resumable: verdicts already "
            "written are skipped, so rerunning finishes what an earlier run left."
        )
    )
    parser.add_argument("--source", type=Path, required=True, help="JSONL file of sampled documents.")
    parser.add_argument("--out", type=Path, required=True, help="JSONL file of verdicts; appended to.")
    parser.add_argument("--rubric", type=Path, required=True, help="Prompt file holding the rubric.")
    parser.add_argument(
        "--task",
        choices=sorted(TASKS),
        default="rubric",
        help="rubric: label each document. pairwise: compare a kept document against a dropped one.",
    )
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default="api",
        help="api: the Batch API, half price, answers within hours. cli: one subprocess per group.",
    )
    parser.add_argument("--model", default="claude-sonnet-5", help="Model the judge should use.")
    parser.add_argument("--batch-size", type=int, default=None, help="Units per request; the task decides by default.")
    parser.add_argument("--concurrency", type=int, default=4, help="Calls in flight at once (cli).")
    parser.add_argument("--max-batches", type=int, default=None, help="Stop after this many calls.")
    parser.add_argument("--max-chars", type=int, default=2000, help="Characters of each document sent.")
    parser.add_argument("--command", default="claude", help="The judge executable.")
    parser.add_argument("--timeout", type=int, default=600, help="Seconds allowed per call (cli).")
    parser.add_argument("--poll-seconds", type=int, default=60, help="Seconds between batch checks (api).")
    parser.add_argument("--pairs", type=Path, default=None, help="Where the drawn comparisons are kept (pairwise).")
    parser.add_argument("--pairs-per-cell", type=int, default=20, help="Comparisons per source and stage (pairwise).")
    parser.add_argument("--stages", nargs="+", default=list(PAIRWISE_STAGES), help="Stages to compare (pairwise).")
    parser.add_argument("--seed", type=int, default=20260916, help="Seed fixing the drawing (pairwise).")
    return parser.parse_args(argv)


def main() -> None:
    """Run the judge over the sampled documents."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    shared = {
        "source": args.source,
        "destination": args.out,
        "rubric_path": args.rubric,
        "backend": args.backend,
        "model": args.model,
        "batch_size": args.batch_size,
        "concurrency": args.concurrency,
        "max_batches": args.max_batches,
        "max_chars": args.max_chars,
        "command": args.command,
        "timeout": args.timeout,
        "poll_seconds": args.poll_seconds,
    }
    if args.task == "pairwise":
        judge_pairs(
            pairs_path=args.pairs, stages=args.stages, pairs_per_cell=args.pairs_per_cell, seed=args.seed, **shared
        )
    else:
        judge_documents(**shared)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
