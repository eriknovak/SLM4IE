"""Export trained tokenizers as HuggingFace tokenizer directories.

Thin CLI wrapper around `slm4ie.tokenizers.hf_export`. For each selected trained
artifact it writes the HuggingFace tokenizer files into the artifact directory,
so the fast backends load via `AutoTokenizer.from_pretrained(<dir>)` and
MorphPiece loads via `slm4ie.tokenizers.hf_export.load_pretrained(<dir>)`. This
is the bridge to the LM-pretraining phase.

Examples:
    Export the whole sweep:

        uv run python scripts/tokenizers/export.py --all

    Export one tokenizer (optionally one vocab size):

        uv run python scripts/tokenizers/export.py --tokenizer bpe
        uv run python scripts/tokenizers/export.py --tokenizer bpe --vocab-size 32000
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from slm4ie.data.io_utils import find_project_root
from slm4ie.data.parallel import configure_script_logging
from slm4ie.tokenizers.hf_export import save_pretrained_dir
from slm4ie.tokenizers.train import resolve_run_selection
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
    parser = argparse.ArgumentParser(description="Export trained tokenizers as HuggingFace tokenizer directories.")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Export this one tokenizer (all its vocab sizes unless --vocab-size narrows it).",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Narrow --tokenizer to this single vocab size.",
    )
    parser.add_argument("--all", action="store_true", help="Export every trained run.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to tokenizers.yaml (default: configs/tokenizers/tokenizers.yaml).",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Export the selected trained tokenizers from CLI arguments."""
    args = parse_args()
    configure_script_logging(parallel=False, console_level=logging.INFO)
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

    exported = 0
    failures: List[str] = []
    for key in keys:
        artifact_dir = cfg.output_root / key
        try:
            save_pretrained_dir(artifact_dir)
            logger.info("Exported %s -> %s", key, artifact_dir)
            exported += 1
        except Exception:
            logger.exception("Failed exporting %s", key)
            failures.append(key)

    logger.info("Done. Exported %d, failed %s.", exported, failures or "none")
    if failures:
        sys.exit(2)


if __name__ == "__main__":
    main()
