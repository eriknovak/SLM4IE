"""Evaluate trained tokenizers and emit the comparison report.

Thin CLI wrapper around `slm4ie.tokenizers.analysis`. It loads
`configs/tokenizers/tokenizers.yaml`, ensures the held-out evaluation sample and
the Sloleks-derived gold lexicon exist, runs the six metrics over every trained
artifact, writes `report.md`/`report.json`, and logs the sweep to MLflow.

Examples:
    Evaluate the whole sweep:

        uv run python scripts/tokenizers/analyze.py --all

    Evaluate one tokenizer (optionally one vocab size):

        uv run python scripts/tokenizers/analyze.py --tokenizer bpe
        uv run python scripts/tokenizers/analyze.py --tokenizer bpe --vocab-size 16000
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
    cpu_default,
    resolve_workers,
    run_parallel,
)
from slm4ie.tokenizers.analysis import (
    augment_with_statistics,
    evaluate_artifact,
    log_results_to_mlflow,
    write_report,
)
from slm4ie.tokenizers.corpus import iter_sample_cache, sample_corpus, write_sample_cache
from slm4ie.tokenizers.metrics import iter_words
from slm4ie.tokenizers.morphology import (
    MorphemeSegmentation,
    build_derivational_lexicon,
    build_morph_lexicon,
    load_lexicon,
    sample_segmentations,
    save_lexicon,
)
from slm4ie.tokenizers.train import resolve_run_selection
from slm4ie.utils.config import TokenizerSweepConfig, load_tokenizer_config

logger = logging.getLogger(__name__)

#: Default location of the tokenizer config relative to the project root.
DEFAULT_CONFIG_RELPATH = Path("configs") / "tokenizers" / "tokenizers.yaml"

#: Conservative upper bound on parallel eval workers so a large box is not
#: saturated by default; raise it explicitly with --max-workers.
DEFAULT_MAX_WORKERS = 8

#: Heavy, read-only eval inputs shared with process-pool workers via fork
#: inheritance (copy-on-write) instead of being pickled per task. Populated in
#: `main` before the pool is created; read by `_evaluate_worker` in each child.
_EVAL_DOCS: List[List[str]] = []
_MORPH_SAMPLE: List[MorphemeSegmentation] = []
_DERIV_SAMPLE: List[MorphemeSegmentation] = []


def _evaluate_worker(
    key: str,
    *,
    output_root: Path,
    alpha: float,
) -> Optional[Dict[str, Any]]:
    """Evaluate one run, reading the shared eval docs and morph sample globals.

    The grouped evaluation documents and the shared morph and derivational form
    samples live in module globals so a forked process-pool worker inherits them
    copy-on-write, rather than receiving them through pickled keyword arguments
    (which would copy the multi-million-form inputs once per task).

    Args:
        key (str): Run key (`<name>-<vocab>`).
        output_root (Path): Directory holding per-run artifact subdirs.
        alpha (float): Renyi order.

    Returns:
        Optional[Dict[str, Any]]: The metrics record, or None when skipped.
    """
    return evaluate_artifact(
        key,
        output_root=output_root,
        eval_docs=_EVAL_DOCS,
        morph_sample=_MORPH_SAMPLE,
        deriv_sample=_DERIV_SAMPLE,
        alpha=alpha,
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv (Optional[List[str]]): Argument list (defaults to `sys.argv`).

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate trained tokenizers and build the comparison report.")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Evaluate this one tokenizer (all its vocab sizes unless --vocab-size narrows it).",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Narrow --tokenizer to this single vocab size.",
    )
    parser.add_argument("--all", action="store_true", help="Evaluate every trained run.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to tokenizers.yaml (default: configs/tokenizers/tokenizers.yaml).",
    )
    parser.add_argument(
        "--morph-form-sample",
        type=int,
        default=None,
        help="Override the morph form-sample size (config stats.morph_form_sample). Useful for quick validation.",
    )
    parser.add_argument(
        "--n-resamples",
        type=int,
        default=None,
        help="Override the bootstrap resample count B (config stats.n_resamples). Useful for quick validation.",
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


def _prepare_eval_inputs(cfg: TokenizerSweepConfig, force: bool, morph_form_sample: int):
    """Materialize the grouped eval docs and the shared morph form sample.

    The evaluation documents are kept grouped (one word list per document) so the
    corpus metrics can resample documents. The morph sample is a deterministic
    draw from the Sloleks-derived lexicon, identical across runs so per-form morph
    statistics align by index for the paired tests.

    Args:
        cfg (TokenizerSweepConfig): The resolved sweep configuration.
        force (bool): Rebuild the eval sample and lexicon even if present.
        morph_form_sample (int): Size of the shared morph form sample.

    Returns:
        Tuple[List[List[str]], List[MorphemeSegmentation], List[MorphemeSegmentation]]:
            The grouped eval documents, the shared inflectional morph form
            sample, and the derivational form sample (empty when no derivational
            gold is configured).
    """
    eval_path = cfg.eval_sample_path
    if force or not eval_path.exists():
        logger.info("Sampling held-out evaluation corpus -> %s", eval_path)
        write_sample_cache(sample_corpus(cfg.corpus_root, cfg.eval_budget), eval_path)
    eval_docs = [iter_words(line) for line in iter_sample_cache(eval_path)]
    logger.info("Evaluation sample: %d documents, %d word tokens", len(eval_docs), sum(len(d) for d in eval_docs))

    infl_path = cfg.infl_lexicon_path
    if force or not infl_path.exists():
        logger.info("Deriving inflectional gold lexicon from %s", cfg.sloleks_path)
        save_lexicon(build_morph_lexicon(cfg.sloleks_path, min_stem_len=cfg.min_stem_len), infl_path)
    lexicon = load_lexicon(infl_path)
    morph_sample = sample_segmentations(lexicon, morph_form_sample, cfg.stats_seed)
    logger.info("Inflectional gold: %d forms; morph sample: %d forms", len(lexicon.by_form), len(morph_sample))

    deriv_sample: List[MorphemeSegmentation] = []
    if cfg.needs_derivational():
        deriv_path = cfg.deriv_lexicon_path
        if force or not deriv_path.exists():
            logger.info("Deriving derivational gold lexicon from %s", cfg.sloleks_relations_path)
            save_lexicon(build_derivational_lexicon(cfg.sloleks_relations_path), deriv_path)
        deriv_lexicon = load_lexicon(deriv_path)
        deriv_sample = sample_segmentations(deriv_lexicon, morph_form_sample, cfg.stats_seed)
        logger.info(
            "Derivational gold: %d forms; deriv sample: %d forms", len(deriv_lexicon.by_form), len(deriv_sample)
        )

    return eval_docs, morph_sample, deriv_sample


def main() -> None:
    """Run tokenizer evaluation and reporting from CLI arguments."""
    args = parse_args()
    project_root = find_project_root()
    config_path = args.config if args.config else project_root / DEFAULT_CONFIG_RELPATH
    cfg = load_tokenizer_config(config_path)

    try:
        candidates = resolve_run_selection(cfg, all_runs=args.all, tokenizer=args.tokenizer, vocab_size=args.vocab_size)
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(2)
    keys = [k for k in candidates if (cfg.output_root / k / "metadata.json").exists()]
    if not keys:
        logger.error("No trained artifacts found under %s. Run train.py first.", cfg.output_root)
        sys.exit(1)

    # CPU-bound metrics: cap the auto default conservatively so a many-core box
    # is not saturated by default (override with --max-workers).
    workers = resolve_workers(args.max_workers, len(keys), min(DEFAULT_MAX_WORKERS, cpu_default(len(keys))))
    configure_script_logging(parallel=workers > 1, console_level=logging.INFO)

    morph_form_sample = args.morph_form_sample if args.morph_form_sample is not None else cfg.stats_morph_form_sample
    n_resamples = args.n_resamples if args.n_resamples is not None else cfg.stats_n_resamples

    # Publish the heavy eval inputs to module globals BEFORE the process pool
    # forks, so workers inherit them copy-on-write rather than via pickling.
    global _EVAL_DOCS, _MORPH_SAMPLE, _DERIV_SAMPLE
    _EVAL_DOCS, _MORPH_SAMPLE, _DERIV_SAMPLE = _prepare_eval_inputs(
        cfg, force=args.force, morph_form_sample=morph_form_sample
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = project_root / "logs" / Path(__file__).stem / stamp

    def kwargs_for(_key: str) -> Dict[str, Any]:
        """Return per-run kwargs for `_evaluate_worker`.

        Only small, picklable values go here; the eval docs and morph sample are
        shared via module globals (see `_evaluate_worker`).

        Args:
            _key (str): Run key (unused; kwargs are identical per run).

        Returns:
            Dict[str, Any]: Keyword arguments for `_evaluate_worker`.
        """
        return {"output_root": cfg.output_root, "alpha": cfg.renyi_alpha}

    # Process pool for real CPU parallelism (a thread pool would serialize on the
    # GIL). The eval docs and morph sample are not pickled per task; forked
    # workers inherit them from the globals set above.
    results, failures = run_parallel(
        _evaluate_worker,
        keys,
        max_workers=workers,
        desc="tokenizer-analyze",
        pool="process",
        kwargs_for=kwargs_for,
        log_dir=log_dir,
    )

    records = [r for r in results.values() if r is not None]
    if records:
        logger.info("Computing bootstrap CIs + paired significance (B=%d) ...", n_resamples)
        significance, stats_config = augment_with_statistics(
            records,
            cfg.output_root,
            n_resamples=n_resamples,
            ci_level=cfg.stats_ci_level,
            seed=cfg.stats_seed,
            morph_form_sample=morph_form_sample,
        )
        md_path, json_path = write_report(records, cfg.report_dir, significance, stats_config)
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
