"""Train tokenizers for the Slovenian comparison sweep.

Thin CLI wrapper around `slm4ie.tokenizers.train`. It loads
`configs/tokenizers/tokenizers.yaml`, prepares the shared training sample (and
the derived morpheme lexicon when a morphological backend is requested), then
trains one tokenizer (or the whole sweep) and logs the runs to MLflow. The
selection is deliberately one-or-all: `--all`, or `--tokenizer` (optionally
narrowed to one `--vocab-size`).

Examples:
    Train the whole sweep:

        uv run python scripts/tokenizers/train.py --all

    Train one tokenizer across all its vocab sizes:

        uv run python scripts/tokenizers/train.py --tokenizer bpe

    Train one specific run:

        uv run python scripts/tokenizers/train.py --tokenizer bpe --vocab-size 16000
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
from slm4ie.tokenizers.train import (
    log_training_to_mlflow,
    prepare_inputs,
    select_runs,
    train_one,
)
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
    parser = argparse.ArgumentParser(
        description=("Train the tokenizer comparison sweep declared in configs/tokenizers/tokenizers.yaml.")
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Train this one tokenizer (all its vocab sizes unless --vocab-size narrows it).",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Narrow --tokenizer to this single vocab size.",
    )
    parser.add_argument("--all", action="store_true", help="Train every run in the sweep.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to tokenizers.yaml (default: configs/tokenizers/tokenizers.yaml).",
    )
    parser.add_argument("--force", action="store_true", help="Retrain existing artifacts.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=0,
        help="Parallel runs. 0=auto (cpu_count // 2), 1=serial, N=N workers.",
    )
    return parser.parse_args(argv)


def _resolve_selection(cfg: TokenizerSweepConfig, args: argparse.Namespace) -> List[str]:
    """Resolve the selected run keys from the one-or-all CLI flags.

    Either `--all` (the whole sweep) or `--tokenizer` (one tokenizer, optionally
    narrowed to a single `--vocab-size`). The two modes are mutually exclusive.

    Args:
        cfg (TokenizerSweepConfig): The resolved sweep configuration.
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        List[str]: The selected run keys.

    Raises:
        ValueError: If the flags are missing, combined illegally, or name a
            tokenizer/vocab size that is not in the configured sweep.
    """
    if args.all:
        if args.tokenizer or args.vocab_size is not None:
            raise ValueError("--all takes no other selector; drop --tokenizer/--vocab-size.")
        return select_runs(cfg)
    if not args.tokenizer:
        raise ValueError("Specify --tokenizer NAME (optionally --vocab-size N), or --all.")
    if args.tokenizer not in cfg.tokenizers:
        raise ValueError(f"Unknown tokenizer {args.tokenizer!r}. Configured: {', '.join(cfg.tokenizers)}.")
    if args.vocab_size is not None and args.vocab_size not in cfg.vocab_sizes:
        raise ValueError(f"Unknown vocab size {args.vocab_size}. Configured: {cfg.vocab_sizes}.")
    vocab_sizes = [args.vocab_size] if args.vocab_size is not None else None
    return select_runs(cfg, tokenizers=[args.tokenizer], vocab_sizes=vocab_sizes)


def main() -> None:
    """Run the tokenizer training sweep from CLI arguments."""
    args = parse_args()
    project_root = find_project_root()
    config_path = args.config if args.config else project_root / DEFAULT_CONFIG_RELPATH
    cfg = load_tokenizer_config(config_path)

    try:
        keys = _resolve_selection(cfg, args)
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(2)
    if not keys:
        logger.warning("No runs selected; nothing to do.")
        return

    cfg.output_root.mkdir(parents=True, exist_ok=True)
    workers = resolve_workers(args.max_workers, len(keys), cpu_default(len(keys)))
    configure_script_logging(parallel=workers > 1, console_level=logging.INFO)

    sample_path, lexicon_path = prepare_inputs(cfg, force=args.force)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = project_root / "logs" / Path(__file__).stem / stamp

    def kwargs_for(_key: str) -> Dict[str, Any]:
        """Return per-run kwargs for `train_one`.

        Args:
            _key (str): Run key (unused; kwargs are identical per run).

        Returns:
            Dict[str, Any]: Keyword arguments for `train_one`.
        """
        return {
            "cfg": cfg,
            "sample_path": sample_path,
            "lexicon_path": lexicon_path,
            "force": args.force,
        }

    results, failures = run_parallel(
        train_one,
        keys,
        max_workers=workers,
        desc="tokenizer-train",
        pool="process",
        kwargs_for=kwargs_for,
        log_dir=log_dir,
    )

    skipped = [k for k, v in results.items() if v is None]
    trained_keys = [k for k, v in results.items() if v is not None]
    logger.info(
        "Done. Trained %d, skipped %d, failed %s.",
        len(trained_keys),
        len(skipped),
        [k for k, _ in failures] or "none",
    )

    if trained_keys:
        log_training_to_mlflow(trained_keys, cfg)

    if failures:
        sys.exit(2)


if __name__ == "__main__":
    main()
