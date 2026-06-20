"""Train tokenizers for the Slovenian comparison sweep.

Thin CLI wrapper around `slm4ie.tokenizers.train`. It loads
`configs/tokenizers/tokenizers.yaml`, prepares the shared training sample (and
the derived morpheme lexicon when a morphological backend is requested), then
trains one tokenizer (or the whole sweep) and logs the runs to MLflow. The
selection is deliberately one-or-all: pass a single target or `--all`.

Examples:
    Train the whole sweep:

        uv run python scripts/tokenizers/train.py --all

    Train one tokenizer across all its vocab sizes:

        uv run python scripts/tokenizers/train.py bpe

    Train one specific run:

        uv run python scripts/tokenizers/train.py bpe-16000
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
    parse_run_key,
    prepare_inputs,
    select_runs,
    train_one,
)
from slm4ie.utils.config import load_tokenizer_config

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
        "target",
        nargs="?",
        default=None,
        help=(
            "One tokenizer to train: a tokenizer name (e.g. 'bpe' -> all its "
            "vocab sizes) or a single run key (e.g. 'bpe-16000'). Mutually "
            "exclusive with --all."
        ),
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


def _resolve_target_keys(cfg, target: str) -> List[str]:
    """Resolve a single train target into run keys.

    A bare tokenizer name expands to every configured vocab size for that
    tokenizer; a full `<name>-<vocab>` key selects exactly that run.

    Args:
        cfg (TokenizerSweepConfig): The resolved sweep configuration.
        target (str): A configured tokenizer name or a run key.

    Returns:
        List[str]: The selected run key(s).

    Raises:
        ValueError: If `target` is neither a configured tokenizer name nor a
            well-formed run key.
    """
    if target in cfg.tokenizers:
        return select_runs(cfg, tokenizers=[target])
    parse_run_key(target)  # validates the '<name>-<vocab>' shape, else ValueError
    return select_runs(cfg, run_keys=[target])


def main() -> None:
    """Run the tokenizer training sweep from CLI arguments."""
    args = parse_args()
    project_root = find_project_root()
    config_path = args.config if args.config else project_root / DEFAULT_CONFIG_RELPATH
    cfg = load_tokenizer_config(config_path)

    if args.all and args.target:
        logger.error("Pass either a single target or --all, not both.")
        sys.exit(2)
    if not (args.all or args.target):
        logger.error("Specify a single tokenizer/run-key target, or --all.")
        sys.exit(2)

    if args.all:
        keys = select_runs(cfg)
    else:
        try:
            keys = _resolve_target_keys(cfg, args.target)
        except ValueError:
            logger.error(
                "Unknown target %r. Use a configured tokenizer name (%s) or a '<name>-<vocab>' run key.",
                args.target,
                ", ".join(cfg.tokenizers),
            )
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
