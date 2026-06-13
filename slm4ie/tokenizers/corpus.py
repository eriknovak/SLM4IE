"""Sample training and evaluation text from the deduplicated corpus.

The pretraining corpus (`pretrain/05_2_dedup/<dataset>/*.jsonl.gz`, datatrove
`{text, id, metadata}` shape) is far too large to feed whole into a tokenizer
trainer. This module streams it, draws a budgeted, optionally source-weighted
sample, and materializes that sample once to a cache file so every tokenizer in
the sweep trains on identical input and no run re-streams billions of tokens.
"""

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from slm4ie.data.io_utils import open_output, open_text_stream

logger = logging.getLogger(__name__)


@dataclass
class SampleBudget:
    """Constraints controlling how much text `sample_corpus` draws.

    Attributes:
        max_bytes (Optional[int]): Total UTF-8 byte budget across the whole
            sample, or None for no byte cap.
        max_docs (Optional[int]): Total document budget, or None for no cap.
        seed (int): Seed for the deterministic shard shuffle.
        weight_key (str): Metadata field used to group documents into sources
            for weighting (typically `dataset` or `domain`).
        source_weights (Dict[str, float]): Relative weights per source value.
            Sources absent from the map default to weight 1.0; a weight of 0
            excludes a source. An empty map samples all sources uniformly.
    """

    max_bytes: Optional[int] = None
    max_docs: Optional[int] = None
    seed: int = 0
    weight_key: str = "dataset"
    source_weights: Dict[str, float] = field(default_factory=dict)


def iter_dedup_documents(
    root: Path,
    datasets: Optional[List[str]] = None,
) -> Iterator[Dict[str, Any]]:
    """Stream datatrove documents from the deduplicated corpus tree.

    Reads `root/<dataset>/*.jsonl.gz` in sorted, deterministic order. Hidden
    and underscore-prefixed subdirectories (e.g. dedup state) are skipped.

    Args:
        root (Path): Corpus root, e.g. `pretrain/05_2_dedup`.
        datasets (Optional[List[str]]): Restrict to these dataset subdirs;
            None reads every subdir.

    Yields:
        Dict[str, Any]: One parsed `{text, id, metadata}` record per line.
    """
    for _source, shard in _iter_shards(Path(root), datasets):
        with open_text_stream(shard) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def sample_corpus(root: Path, budget: SampleBudget) -> Iterator[str]:
    """Yield sampled document texts honoring the budget and source weights.

    Per-source byte/document quotas are computed from the normalized weights,
    then each source's shards are consumed in a seeded shuffled order until its
    quota is met. The traversal is fully deterministic for a fixed seed.

    Args:
        root (Path): Corpus root, e.g. `pretrain/05_2_dedup`.
        budget (SampleBudget): Sampling constraints.

    Yields:
        str: The `text` field of each sampled document.
    """
    root = Path(root)
    sources = _group_shards_by_source(root, budget.weight_key)
    weights = _resolve_weights(sources.keys(), budget.source_weights)
    total_weight = sum(weights.values()) or 1.0
    rng = random.Random(budget.seed)

    for source in sorted(weights):
        weight = weights[source]
        share = weight / total_weight
        quota_bytes = None if budget.max_bytes is None else int(budget.max_bytes * share)
        quota_docs = None if budget.max_docs is None else max(1, int(budget.max_docs * share))

        shards = list(sources[source])
        rng.shuffle(shards)
        used_bytes = 0
        used_docs = 0
        for shard in shards:
            if _quota_met(used_bytes, used_docs, quota_bytes, quota_docs):
                break
            with open_text_stream(shard) as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    text = json.loads(line).get("text") or ""
                    if not text:
                        continue
                    yield text
                    used_bytes += len(text.encode("utf-8"))
                    used_docs += 1
                    if _quota_met(used_bytes, used_docs, quota_bytes, quota_docs):
                        break


def write_sample_cache(texts: Iterator[str], cache_path: Path) -> Path:
    """Materialize sampled texts to a one-document-per-line cache file.

    Internal newlines are collapsed to spaces so each document occupies a
    single line, which the line-oriented tokenizer trainers expect. The output
    is gzip-compressed when `cache_path` ends in `.gz`.

    Args:
        texts (Iterator[str]): Sampled document texts.
        cache_path (Path): Destination file (`.txt` or `.txt.gz`).

    Returns:
        Path: `cache_path`, for convenient chaining.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open_output(cache_path) as out:
        for text in texts:
            flattened = " ".join(text.split())
            if not flattened:
                continue
            out.write(flattened)
            out.write("\n")
    return cache_path


def iter_sample_cache(cache_path: Path) -> Iterator[str]:
    """Yield one text per line from a cache written by `write_sample_cache`.

    Args:
        cache_path (Path): Path to the `.txt`/`.txt.gz` cache file.

    Yields:
        str: One cached document per line, trailing newline stripped.
    """
    with open_text_stream(Path(cache_path)) as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line:
                yield line


def _iter_shards(
    root: Path,
    datasets: Optional[List[str]],
) -> Iterator[Tuple[str, Path]]:
    """Yield `(dataset, shard_path)` pairs across the corpus tree.

    Args:
        root (Path): Corpus root.
        datasets (Optional[List[str]]): Restrict to these subdirs, or None.

    Yields:
        Tuple[str, Path]: Dataset subdir name and one shard path.
    """
    if not root.exists():
        logger.warning("Corpus root does not exist: %s", root)
        return
    wanted = set(datasets) if datasets else None
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        if sub.name.startswith((".", "_")):
            continue
        if wanted is not None and sub.name not in wanted:
            continue
        for shard in sorted(sub.glob("*.jsonl.gz")):
            yield sub.name, shard


def _group_shards_by_source(root: Path, weight_key: str) -> Dict[str, List[Path]]:
    """Group corpus shards by their source value.

    The source value is the directory name, which equals the dataset key for
    the standard layout. `weight_key` is accepted for symmetry with
    `SampleBudget`; per-document grouping on `domain` would require scanning
    every record, so directory-level grouping (dataset) is used.

    Args:
        root (Path): Corpus root.
        weight_key (str): Metadata field the caller intends to weight on.

    Returns:
        Dict[str, List[Path]]: Shard paths grouped by source name.
    """
    del weight_key  # Directory-level grouping; documented above.
    grouped: Dict[str, List[Path]] = {}
    for source, shard in _iter_shards(root, None):
        grouped.setdefault(source, []).append(shard)
    return grouped


def _resolve_weights(
    sources: Any,
    source_weights: Dict[str, float],
) -> Dict[str, float]:
    """Assign a positive weight to each available source.

    Args:
        sources (Any): Iterable of available source names.
        source_weights (Dict[str, float]): User-supplied weights; missing
            sources default to 1.0, non-positive weights are dropped.

    Returns:
        Dict[str, float]: Source name to positive weight.
    """
    resolved: Dict[str, float] = {}
    for source in sources:
        weight = source_weights.get(source, 1.0)
        if weight > 0:
            resolved[source] = float(weight)
    return resolved


def _quota_met(
    used_bytes: int,
    used_docs: int,
    quota_bytes: Optional[int],
    quota_docs: Optional[int],
) -> bool:
    """Return True when any active quota has been reached.

    Args:
        used_bytes (int): Bytes consumed so far for the source.
        used_docs (int): Documents consumed so far for the source.
        quota_bytes (Optional[int]): Byte quota, or None.
        quota_docs (Optional[int]): Document quota, or None.

    Returns:
        bool: True if a set quota is met or exceeded.
    """
    if quota_bytes is not None and used_bytes >= quota_bytes:
        return True
    if quota_docs is not None and used_docs >= quota_docs:
        return True
    return False
