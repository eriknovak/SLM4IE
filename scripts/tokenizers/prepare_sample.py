"""Materialize the persistent tokenizer training sample (and morph lexicon).

Thin CLI wrapper around `slm4ie.tokenizers.train.prepare_inputs`. It draws the
shared, seeded training sample from the deduplicated corpus once and writes it
to the persistent cache path declared in `configs/tokenizers/tokenizers.yaml`
(`output.root/corpus_sample.txt.gz`), plus the Sloleks-derived morpheme lexicon
when a morphological backend is in the sweep.

Running this first decouples the (expensive) sampling from training: every
`scripts/tokenizers/train.py` run then reuses the cached sample instead of
re-drawing it, so repeated or parallel-capped sweeps are reproducible and cheap.
Both artifacts are kept unless `--force` is passed. `train.py` reuses whatever
this writes, so the sample is identical across every run of the sweep.

Examples:
    Materialize the sample and lexicon (no-op if they already exist):

        uv run python scripts/tokenizers/prepare_sample.py

    Rebuild them from scratch:

        uv run python scripts/tokenizers/prepare_sample.py --force
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from slm4ie.data.io_utils import find_project_root
from slm4ie.data.parallel import configure_script_logging
from slm4ie.tokenizers.train import prepare_inputs
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
        description=(
            "Materialize the persistent tokenizer training sample (and morpheme "
            "lexicon) declared in configs/tokenizers/tokenizers.yaml, so train.py "
            "reuses them instead of re-sampling."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to tokenizers.yaml (default: configs/tokenizers/tokenizers.yaml).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the sample and lexicon even when they already exist.",
    )
    return parser.parse_args(argv)


def _describe(path: Path) -> str:
    """Return a human-readable `path (size)` string for logging.

    Args:
        path (Path): An existing file path.

    Returns:
        str: The path followed by its size in MiB, or just the path when its
            size cannot be read.
    """
    try:
        mib = path.stat().st_size / (1024 * 1024)
    except OSError:
        return str(path)
    return f"{path} ({mib:.1f} MiB)"


def main() -> None:
    """Materialize the persistent training sample and lexicon from CLI args."""
    args = parse_args()
    configure_script_logging(parallel=False, console_level=logging.INFO)

    project_root = find_project_root()
    config_path = args.config if args.config else project_root / DEFAULT_CONFIG_RELPATH
    cfg = load_tokenizer_config(config_path)

    cfg.output_root.mkdir(parents=True, exist_ok=True)
    sample_path, lexicon_path = prepare_inputs(cfg, force=args.force)

    logger.info("Training sample ready: %s", _describe(sample_path))
    if lexicon_path is not None:
        logger.info("Morpheme lexicon ready: %s", _describe(lexicon_path))
    else:
        logger.info("No morphological backend in the sweep; lexicon not built.")

    if not sample_path.exists():
        logger.error("Sample was not materialized: %s", sample_path)
        sys.exit(1)


if __name__ == "__main__":
    main()
