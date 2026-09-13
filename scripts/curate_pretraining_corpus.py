"""CLI over the pretraining-corpus curation pipeline.

Parses arguments and dispatches into `slm4ie.data.curate.runner`, which owns the
eight stages, their sentinels and the invalidation cascade.

Three subcommands:

* `run` builds the corpus. With positional keys (e.g. `kzb solar`) it runs the
  four scoped stages (convert, language, quality, repetition) for the named
  datasets only; corpus stages never run in that mode. With `--all` it runs
  every stage for every dataset in `extract.yaml`, skipping sentinels that are
  already current. The corpus stages (exact_dedup, sentence_dedup, statistics)
  require `--all` -- `--stage exact_dedup` without it is an error.
* `recount` backfills per-source record counts onto existing scoped sentinels
  and rewrites no data.
* `diagnose` samples the finished corpus and reports where foreign-language
  text survives the language stage. Read-only: it writes nothing.

`--config` is required: an experiment may curate its own corpus variant, so the
shared registry is never assumed.

Examples:
    export CURATION=configs/data/curate.yaml

    # Scoped stages for two datasets; corpus stages run later.
    uv run python scripts/curate_pretraining_corpus.py run --config $CURATION kzb solar

    # Run (or resume) the full pipeline over every dataset.
    uv run python scripts/curate_pretraining_corpus.py run --config $CURATION --all

    # Re-run quality and every downstream stage.
    uv run python scripts/curate_pretraining_corpus.py run --config $CURATION --all \
        --force --stage quality

    # cpu_count // 2 workers (default is serial: 1 worker).
    uv run python scripts/curate_pretraining_corpus.py run --config $CURATION --all \
        --max-workers 0

    # Backfill per-source counts onto existing sentinels; reprocesses no data.
    uv run python scripts/curate_pretraining_corpus.py recount --config $CURATION

    # Report foreign-language leakage in the finished corpus.
    uv run python scripts/curate_pretraining_corpus.py diagnose --config $CURATION \
        --candidates sl,en,de,hr,sr,it,fr
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from slm4ie.data.curate import ALL_STAGE_NAMES
from slm4ie.data.curate.diagnose import diagnose_language_leakage
from slm4ie.data.curate.runner import curate, recount
from slm4ie.data.curate.stages import CORPUS_STAGES
from slm4ie.utils.cli import add_selection, validate_selection


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the arguments both subcommands share.

    Args:
        parser: Subcommand parser to extend.
    """
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the curation config, e.g. configs/data/curate.yaml.",
    )
    parser.add_argument(
        "--extract-config",
        type=Path,
        default=None,
        help="Path to extract.yaml (default: configs/data/extract.yaml).",
    )
    parser.add_argument("--input-dir", type=Path, default=None, help="Override the config's input_dir.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override the config's output_dir.")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to `sys.argv`).

    Returns:
        Parsed namespace, including the selected subcommand in `command`.

    Raises:
        SystemExit: If the selection is missing, or a corpus-wide stage is
            requested for a subset of datasets.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Build the SLM4IE pretraining corpus stage-by-stage. Each stage writes a durable "
            "artifact under <output_dir>/ and a .complete sentinel; on rerun, stale stages "
            "auto-invalidate."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Build (or resume) the corpus.")
    add_selection(run_parser, "datasets", "Dataset keys to process.", "Process every dataset.")
    _add_common_arguments(run_parser)
    run_parser.add_argument(
        "--stage",
        choices=ALL_STAGE_NAMES,
        default="all",
        help="Stage to run. Default: all (skips finished stages).",
    )
    run_parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Force re-run. With --stage X: drop X's sentinel and all downstream sentinels. "
            "Without --stage: nuke <output_dir>."
        ),
    )
    run_parser.add_argument(
        "--max-workers",
        "--tasks",
        dest="workers",
        type=int,
        default=1,
        help="Parallel workers. 1=serial (default), 0=cpu_count//2, N=N.",
    )
    run_parser.add_argument(
        "--mlflow",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable/disable MLflow pretrain-build tracking (only on --all builds), overriding "
            "the config's mlflow.enabled. Default: defer to config."
        ),
    )

    recount_parser = subparsers.add_parser(
        "recount",
        help="Backfill per-source record counts onto existing scoped sentinels.",
    )
    _add_common_arguments(recount_parser)

    diagnose_parser = subparsers.add_parser(
        "diagnose",
        help="Report foreign-language leakage in the finished corpus (read-only).",
    )
    diagnose_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the curation config, e.g. configs/data/curate.yaml.",
    )
    diagnose_parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Corpus root to sample (default: the config's output_dir + final stage folder).",
    )
    diagnose_parser.add_argument("--per-dataset", type=int, default=2000, help="Max docs sampled per dataset.")
    diagnose_parser.add_argument(
        "--max-shards-per-dataset", type=int, default=2, help="Max shards scanned per dataset."
    )
    diagnose_parser.add_argument(
        "--min-unit-chars", type=int, default=50, help="Paragraphs shorter than this are not classified."
    )
    diagnose_parser.add_argument(
        "--whole-doc-chars", type=int, default=4000, help="Chars used for the whole-document classification."
    )
    diagnose_parser.add_argument(
        "--max-paragraphs", type=int, default=40, help="Classify at most this many paragraphs per document."
    )
    diagnose_parser.add_argument(
        "--candidates",
        type=str,
        default=None,
        help=(
            "Comma-separated ISO 639-1 candidate override (e.g. 'sl,en,de,hr,sr,it,fr'). Defaults to the "
            "config's full set; a smaller focused set is much faster and still separates sl from English."
        ),
    )

    args = parser.parse_args(argv)
    if args.command == "run":
        validate_selection(parser, args, "datasets")
        if args.datasets and args.stage in CORPUS_STAGES:
            parser.error(f"--stage {args.stage} is corpus-wide; run it with --all, not with positional dataset keys.")
    return args


def main() -> None:
    """Dispatch the selected subcommand."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()

    if args.command == "diagnose":
        print(
            diagnose_language_leakage(
                pretrain_config=args.config,
                base_dir=args.base_dir,
                per_dataset=args.per_dataset,
                max_shards=args.max_shards_per_dataset,
                min_unit_chars=args.min_unit_chars,
                whole_doc_chars=args.whole_doc_chars,
                max_paragraphs=args.max_paragraphs,
                candidates=args.candidates.split(",") if args.candidates else None,
            )
        )
        return

    if args.command == "recount":
        recount(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            pretrain_config=args.config,
            extract_config=args.extract_config,
        )
        return

    curate(
        datasets=args.datasets,
        run_all=args.all,
        stage=args.stage,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        force=args.force,
        workers=args.workers,
        pretrain_config=args.config,
        extract_config=args.extract_config,
        mlflow_enabled=args.mlflow,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
