"""Tokenizer training orchestration.

Plans the tokenizer x vocab-size sweep, prepares the shared inputs (one cached
training sample and, when a morphological backend is requested, the derived
morpheme lexicon), and trains a single run. The thin CLI wrapper in
`scripts/tokenizers/train.py` dispatches `train_one` across runs with
`run_parallel`. Inputs are prepared once in the parent so every run trains on
identical text and the corpus is never re-streamed per run.
"""

from __future__ import annotations

import logging
from itertools import product
from pathlib import Path
from typing import List, Optional, Tuple

import slm4ie.tokenizers.backends  # noqa: F401  (registers backends on import)
from slm4ie.tokenizers.base import TrainContext
from slm4ie.tokenizers.corpus import iter_sample_cache, sample_corpus, write_sample_cache
from slm4ie.tokenizers.morphology import build_morph_lexicon, load_lexicon, save_lexicon
from slm4ie.tokenizers.registry import get_tokenizer
from slm4ie.utils.config import TokenizerSweepConfig

logger = logging.getLogger(__name__)


def run_key(name: str, vocab_size: int) -> str:
    """Return the canonical run key for a tokenizer and vocab size.

    Args:
        name (str): Tokenizer registry key.
        vocab_size (int): Vocabulary size.

    Returns:
        str: `"<name>-<vocab_size>"`.
    """
    return f"{name}-{vocab_size}"


def parse_run_key(key: str) -> Tuple[str, int]:
    """Split a run key back into its tokenizer name and vocab size.

    Args:
        key (str): A key produced by `run_key`.

    Returns:
        Tuple[str, int]: The tokenizer name and vocab size.

    Raises:
        ValueError: If `key` is not of the form `<name>-<int>`.
    """
    name, _, vocab = key.rpartition("-")
    if not name or not vocab.isdigit():
        raise ValueError(f"Malformed run key: {key!r} (expected '<name>-<vocab>').")
    return name, int(vocab)


def plan_runs(cfg: TokenizerSweepConfig) -> List[str]:
    """Enumerate every tokenizer x vocab-size run key in the sweep.

    Args:
        cfg (TokenizerSweepConfig): The resolved sweep configuration.

    Returns:
        List[str]: Run keys in tokenizer-major, vocab-minor order.
    """
    return [run_key(name, vocab) for name, vocab in product(cfg.tokenizers, cfg.vocab_sizes)]


def prepare_inputs(cfg: TokenizerSweepConfig, force: bool = False) -> Tuple[Path, Optional[Path]]:
    """Materialize the shared training sample and morpheme lexicon.

    Both artifacts are reused across every run. Existing artifacts are kept
    unless `force` is set.

    Args:
        cfg (TokenizerSweepConfig): The resolved sweep configuration.
        force (bool): Rebuild even when the artifacts already exist.

    Returns:
        Tuple[Path, Optional[Path]]: The training-sample path and the morpheme
            lexicon path (None when no morphological backend is requested).
    """
    sample_path = cfg.corpus_sample_path
    if force or not sample_path.exists():
        logger.info("Sampling training corpus -> %s", sample_path)
        write_sample_cache(sample_corpus(cfg.corpus_root, cfg.train_budget), sample_path)
    else:
        logger.info("Reusing cached training sample: %s", sample_path)

    lexicon_path: Optional[Path] = None
    if cfg.needs_morphology():
        lexicon_path = cfg.lexicon_path
        if force or not lexicon_path.exists():
            logger.info("Deriving morpheme lexicon from %s", cfg.sloleks_path)
            lexicon = build_morph_lexicon(cfg.sloleks_path, min_stem_len=cfg.min_stem_len)
            save_lexicon(lexicon, lexicon_path)
        else:
            logger.info("Reusing cached morpheme lexicon: %s", lexicon_path)
    return sample_path, lexicon_path


def train_one(
    key: str,
    *,
    cfg: TokenizerSweepConfig,
    sample_path: Path,
    lexicon_path: Optional[Path],
    force: bool = False,
) -> Optional[Path]:
    """Train and persist one tokenizer x vocab-size run.

    Args:
        key (str): Run key (`<name>-<vocab>`).
        cfg (TokenizerSweepConfig): The resolved sweep configuration.
        sample_path (Path): Cached training-sample path.
        lexicon_path (Optional[Path]): Morpheme lexicon path for morphological
            backends, or None.
        force (bool): Retrain even when the artifact already exists.

    Returns:
        Optional[Path]: The artifact directory, or None when the run was
            skipped because its output already exists.

    Raises:
        ValueError: If a morphological run is requested without a lexicon.
    """
    name, vocab_size = parse_run_key(key)
    out_dir = cfg.output_root / key
    if (out_dir / "metadata.json").exists() and not force:
        logger.info("Skipping %s; artifact exists (use --force to retrain).", key)
        return None

    lexicon = None
    if name.startswith("morph"):
        if lexicon_path is None:
            raise ValueError(f"{name} requires a morpheme lexicon but none was prepared.")
        lexicon = load_lexicon(lexicon_path)

    context = TrainContext(
        special_tokens=list(cfg.special_tokens),
        seed=cfg.train_budget.seed,
        lexicon=lexicon,
    )
    tokenizer = get_tokenizer(name)()
    logger.info("Training %s (vocab=%d)", name, vocab_size)
    tokenizer.train(iter_sample_cache(sample_path), vocab_size, config=context)
    tokenizer.save(out_dir)
    logger.info("Saved %s -> %s", key, out_dir)
    return out_dir


def select_runs(
    cfg: TokenizerSweepConfig,
    *,
    run_keys: Optional[List[str]] = None,
    tokenizers: Optional[List[str]] = None,
    vocab_sizes: Optional[List[int]] = None,
) -> List[str]:
    """Resolve which run keys to execute from explicit keys or filters.

    Args:
        cfg (TokenizerSweepConfig): The resolved sweep configuration.
        run_keys (Optional[List[str]]): Explicit run keys; takes precedence.
        tokenizers (Optional[List[str]]): Restrict to these tokenizer names.
        vocab_sizes (Optional[List[int]]): Restrict to these vocab sizes.

    Returns:
        List[str]: The selected run keys, preserving sweep order.
    """
    planned = plan_runs(cfg)
    if run_keys:
        planned_set = set(planned)
        unknown = [k for k in run_keys if k not in planned_set]
        if unknown:
            logger.warning("Run key(s) not in the configured sweep: %s", unknown)
        return list(run_keys)

    selected = planned
    if tokenizers:
        wanted = set(tokenizers)
        selected = [k for k in selected if parse_run_key(k)[0] in wanted]
    if vocab_sizes:
        wanted_vocab = set(vocab_sizes)
        selected = [k for k in selected if parse_run_key(k)[1] in wanted_vocab]
    return selected
