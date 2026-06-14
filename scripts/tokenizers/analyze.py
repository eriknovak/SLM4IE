"""Evaluate trained tokenizers and emit the comparison report.

Thin CLI wrapper around `slm4ie.tokenizers.analysis`. It loads
`configs/tokenizers/tokenizers.yaml`, ensures the held-out evaluation sample and
the Sloleks-derived gold lexicon exist, runs the six metrics over every trained
artifact, writes `report.md`/`report.json`, and logs the sweep to MLflow.

Examples:
    Evaluate the whole sweep:

        uv run python scripts/tokenizers/analyze.py --all

    Evaluate specific runs:

        uv run python scripts/tokenizers/analyze.py bpe-16000 morphpiece-32000
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from slm4ie.data.io_utils import find_project_root
from slm4ie.data.parallel import (
    configure_script_logging,
    io_default,
    resolve_workers,
    run_parallel,
)
from slm4ie.tokenizers.analysis import evaluate_artifact, log_results_to_mlflow, write_report
from slm4ie.tokenizers.corpus import iter_sample_cache, sample_corpus, write_sample_cache
from slm4ie.tokenizers.metrics import iter_words
from slm4ie.tokenizers.morphology import build_morph_lexicon, load_lexicon, save_lexicon
from slm4ie.tokenizers.train import select_runs
from slm4ie.utils.config import TokenizerSweepConfig, load_tokenizer_config

logger = logging.getLogger(__name__)

#: Default location of the tokenizer config relative to the project root.
DEFAULT_CONFIG_RELPATH = Path("configs") / "tokenizers" / "tokenizers.yaml"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv (Optional[List[str]]): Argument list (defaults to `sys.argv`).

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate trained tokenizers and build the comparison report.")
    parser.add_argument(
        "run_keys",
        nargs="*",
        help="Run keys '<name>-<vocab>' to evaluate. Mutually exclusive with --all.",
    )
    parser.add_argument("--all", action="store_true", help="Evaluate every trained run.")
    parser.add_argument("--tokenizer", nargs="+", default=None, help="Restrict to these names.")
    parser.add_argument("--vocab", nargs="+", type=int, default=None, help="Restrict to these sizes.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to tokenizers.yaml (default: configs/tokenizers/tokenizers.yaml).",
    )
    parser.add_argument(
        "--max-forms",
        type=int,
        default=None,
        help="Cap on lexicon forms used by the morph metrics (default: all).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the evaluation sample and morpheme lexicon.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=0,
        help="Parallel evaluations (thread pool). 0=auto, 1=serial, N=N workers.",
    )
    return parser.parse_args(argv)


def _prepare_eval_inputs(cfg: TokenizerSweepConfig, force: bool):
    """Materialize the eval word list and the gold morpheme lexicon.

    Args:
        cfg (TokenizerSweepConfig): The resolved sweep configuration.
        force (bool): Rebuild the eval sample and lexicon even if present.

    Returns:
        Tuple[List[str], MorphLexicon]: The eval word tokens and the lexicon.
    """
    eval_path = cfg.eval_sample_path
    if force or not eval_path.exists():
        logger.info("Sampling held-out evaluation corpus -> %s", eval_path)
        write_sample_cache(sample_corpus(cfg.corpus_root, cfg.eval_budget), eval_path)
    eval_words = [word for line in iter_sample_cache(eval_path) for word in iter_words(line)]
    logger.info("Evaluation sample: %d word tokens", len(eval_words))

    lexicon_path = cfg.lexicon_path
    if force or not lexicon_path.exists():
        logger.info("Deriving morpheme lexicon from %s", cfg.sloleks_path)
        save_lexicon(build_morph_lexicon(cfg.sloleks_path, min_stem_len=cfg.min_stem_len), lexicon_path)
    lexicon = load_lexicon(lexicon_path)
    logger.info("Gold lexicon: %d forms", len(lexicon.by_form))
    return eval_words, lexicon


def main() -> None:
    """Run tokenizer evaluation and reporting from CLI arguments."""
    args = parse_args()
    project_root = find_project_root()
    config_path = args.config if args.config else project_root / DEFAULT_CONFIG_RELPATH
    cfg = load_tokenizer_config(config_path)

    candidates = select_runs(
        cfg,
        run_keys=args.run_keys or None,
        tokenizers=args.tokenizer,
        vocab_sizes=args.vocab,
    )
    keys = [k for k in candidates if (cfg.output_root / k / "metadata.json").exists()]
    if not keys:
        logger.error("No trained artifacts found under %s. Run train.py first.", cfg.output_root)
        sys.exit(1)

    workers = resolve_workers(args.max_workers, len(keys), io_default(len(keys)))
    configure_script_logging(parallel=workers > 1, console_level=logging.INFO)

    eval_words, lexicon = _prepare_eval_inputs(cfg, force=args.force)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = project_root / "logs" / Path(__file__).stem / stamp

    def kwargs_for(_key: str) -> Dict[str, Any]:
        """Return per-run kwargs for `evaluate_artifact`.

        Args:
            _key (str): Run key (unused; kwargs are shared by reference).

        Returns:
            Dict[str, Any]: Keyword arguments for `evaluate_artifact`.
        """
        return {
            "output_root": cfg.output_root,
            "lexicon": lexicon,
            "eval_words": eval_words,
            "alpha": cfg.renyi_alpha,
            "max_forms": args.max_forms,
        }

    # Thread pool so the lexicon and eval words are shared in-process rather
    # than pickled to each worker.
    results, failures = run_parallel(
        evaluate_artifact,
        keys,
        max_workers=workers,
        desc="tokenizer-analyze",
        pool="thread",
        kwargs_for=kwargs_for,
        log_dir=log_dir,
    )

    records = [r for r in results.values() if r is not None]
    if records:
        md_path, json_path = write_report(records, cfg.report_dir)
        logger.info("Wrote report: %s", md_path)
        log_results_to_mlflow(records, cfg, (md_path, json_path))

    logger.info(
        "Done. Evaluated %d, failed %s.",
        len(records),
        [k for k, _ in failures] or "none",
    )
    if failures:
        sys.exit(2)


if __name__ == "__main__":
    main()
