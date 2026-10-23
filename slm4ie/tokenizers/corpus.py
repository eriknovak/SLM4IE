"""Sample training and evaluation text from the deduplicated corpus.

The pretraining corpus (`pretrain/06_sentence_dedup/<dataset>/*.jsonl.gz`, datatrove
`{text, id, metadata}` shape) is far too large to feed whole into a tokenizer
trainer. This module streams it, draws a budgeted, optionally source-weighted
sample, and materializes that sample once to a cache file so every tokenizer in
the sweep trains on identical input and no run re-streams billions of tokens.
"""

import hashlib
import json
import logging
import random
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from slm4ie.utils.io import open_output, open_text_stream
from slm4ie.utils.parallel import run_parallel

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
        root (Path): Corpus root, e.g. `pretrain/06_sentence_dedup`.
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


def iter_sample_documents(
    root: Path,
    budget: SampleBudget,
    split: Optional["CorpusSplit"] = None,
    side: Optional[str] = None,
) -> Iterator[Tuple[str, str]]:
    """Yield `(source, text)` for each sampled document, honoring the budget.

    Per-source byte/document quotas are computed from the normalized weights,
    then each source's shards are consumed in a seeded shuffled order until its
    quota is met. The traversal is fully deterministic for a fixed seed. With a
    `split`, only documents on `side` are drawn, and when the budget names no
    weights each source weighs as much as its bytes on that side, so the
    sample keeps the side's source mix.

    Args:
        root (Path): Corpus root, e.g. `pretrain/06_sentence_dedup`.
        budget (SampleBudget): Sampling constraints.
        split (Optional[CorpusSplit]): The corpus split to draw from, or None
            to draw from every document.
        side (Optional[str]): `train` or `eval`; required with `split`.

    Yields:
        Tuple[str, str]: The source name and the `text` of each sampled document.

    Raises:
        ValueError: If `split` is given without a valid `side`.
    """
    if split is not None and side not in SPLIT_SIDES:
        raise ValueError(f"side must be one of {SPLIT_SIDES} with a split, got {side!r}")
    root = Path(root)
    sources = _group_shards_by_source(root, budget.weight_key)
    source_weights = budget.source_weights
    if split is not None and not source_weights:
        source_weights = {name: float(split.counts[name][side]["bytes"]) for name in split.counts}
    weights = _resolve_weights(sources.keys(), source_weights)
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
                    record = json.loads(line)
                    text = record.get("text") or ""
                    if not text:
                        continue
                    if split is not None and split.side_of(source, str(record.get("id"))) != side:
                        continue
                    yield source, text
                    used_bytes += len(text.encode("utf-8"))
                    used_docs += 1
                    if _quota_met(used_bytes, used_docs, quota_bytes, quota_docs):
                        break


def sample_corpus(
    root: Path,
    budget: SampleBudget,
    split: Optional["CorpusSplit"] = None,
    side: Optional[str] = None,
) -> Iterator[str]:
    """Yield sampled document texts honoring the budget and source weights.

    Args:
        root (Path): Corpus root, e.g. `pretrain/06_sentence_dedup`.
        budget (SampleBudget): Sampling constraints.
        split (Optional[CorpusSplit]): The corpus split to draw from, or None.
        side (Optional[str]): `train` or `eval`; required with `split`.

    Yields:
        str: The `text` field of each sampled document.
    """
    for _source, text in iter_sample_documents(root, budget, split, side):
        yield text


def write_sample(texts: Iterator[str], cache_path: Path) -> Path:
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


def materialize_sample(
    root: Path,
    budget: SampleBudget,
    cache_path: Path,
    split: Optional["CorpusSplit"] = None,
    side: Optional[str] = None,
) -> Path:
    """Draw a sample, write it with `write_sample`, and record its source mix.

    The documents, words and bytes drawn per source are written beside the
    cache as `<cache name>.sources.json`, so the sample's composition can be
    tabled without re-reading it.

    Args:
        root (Path): Corpus root.
        budget (SampleBudget): Sampling constraints.
        cache_path (Path): Destination of the sample (`.txt` or `.txt.gz`).
        split (Optional[CorpusSplit]): The corpus split to draw from, or None.
        side (Optional[str]): `train` or `eval`; required with `split`.

    Returns:
        Path: `cache_path`, for convenient chaining.
    """
    counts: Dict[str, Dict[str, int]] = {}

    def tallied() -> Iterator[str]:
        for source, text in iter_sample_documents(root, budget, split, side):
            tally = counts.setdefault(source, {"documents": 0, "words": 0, "bytes": 0})
            tally["documents"] += 1
            tally["words"] += len(text.split())
            tally["bytes"] += len(text.encode("utf-8"))
            yield text

    write_sample(tallied(), cache_path)
    sources_path = Path(cache_path).with_name(Path(cache_path).name + ".sources.json")
    sources_path.write_text(json.dumps({"side": side, "sources": counts}, indent=2, sort_keys=True), encoding="utf-8")
    return Path(cache_path)


#: The two sides of a corpus split.
SPLIT_SIDES = ("train", "eval")

#: Rank value above every 64-bit hash, so a source can put all its documents on one side.
_RANK_CEILING = 2**64


def stable_rank(key: str) -> int:
    """Return a reproducible 64-bit rank for `key`, uniform over keys.

    Args:
        key (str): Any identifier, such as `<source>/<document id>`.

    Returns:
        int: The first eight bytes of the key's BLAKE2b digest, as an integer.
    """
    return int.from_bytes(hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest(), "big")


@dataclass(frozen=True)
class CorpusSplit:
    """A fixed split of every corpus document into a training and an evaluation side.

    Within each source, documents are ordered by `stable_rank` of
    `<source>/<id>`, and the lowest `train_share` of them form the training
    side, so every source keeps the same proportion on both sides.

    Attributes:
        train_share (float): Share of each source's documents on the training side.
        thresholds (Dict[str, int]): Per source, the rank below which a document
            is on the training side.
        counts (Dict[str, Dict[str, Dict[str, int]]]): Per source and side, the
            `documents`, `words` and `bytes` the side holds.
        corpus_fingerprint (str): `corpus_fingerprint` of the corpus the split
            was fitted on.
    """

    train_share: float
    thresholds: Dict[str, int]
    counts: Dict[str, Dict[str, Dict[str, int]]]
    corpus_fingerprint: str

    def side_of(self, source: str, doc_id: str) -> str:
        """Return the side a document belongs to.

        Args:
            source (str): The document's source (its dataset subdir).
            doc_id (str): The document's `id`.

        Returns:
            str: `train` or `eval`.

        Raises:
            KeyError: If `source` was not part of the corpus the split was fitted on.
        """
        return "train" if stable_rank(f"{source}/{doc_id}") < self.thresholds[source] else "eval"


def _rank_source(source: str, *, root: Path, train_share: float) -> Dict[str, Any]:
    """Rank one source's documents and find its training-side threshold.

    Top-level so it pickles for the process pool.

    Args:
        source (str): Dataset subdir under `root`.
        root (Path): Corpus root.
        train_share (float): Share of documents on the training side.

    Returns:
        Dict[str, Any]: The source's `threshold` and per-side `counts`.
    """
    ranks: List[int] = []
    words: List[int] = []
    sizes: List[int] = []
    for _name, shard in _iter_shards(root, [source]):
        with open_text_stream(shard) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                text = record.get("text") or ""
                ranks.append(stable_rank(f"{source}/{record.get('id')}"))
                words.append(len(text.split()))
                sizes.append(len(text.encode("utf-8")))
    order = sorted(range(len(ranks)), key=ranks.__getitem__)
    n_train = round(len(order) * train_share)
    threshold = ranks[order[n_train]] if n_train < len(order) else _RANK_CEILING
    counts = {side: {"documents": 0, "words": 0, "bytes": 0} for side in SPLIT_SIDES}
    for position, index in enumerate(order):
        tally = counts["train" if position < n_train else "eval"]
        tally["documents"] += 1
        tally["words"] += words[index]
        tally["bytes"] += sizes[index]
    return {"threshold": threshold, "counts": counts}


def fit_corpus_split(root: Path, train_share: float, max_workers: int) -> CorpusSplit:
    """Split every document of the corpus into a training and an evaluation side.

    One streaming pass per source, run in parallel across sources.

    Args:
        root (Path): Corpus root, e.g. `pretrain/06_sentence_dedup`.
        train_share (float): Share of each source's documents on the training side.
        max_workers (int): Sources ranked at once.

    Returns:
        CorpusSplit: The thresholds and the per-side counts of every source.

    Raises:
        ValueError: If `train_share` is not strictly between 0 and 1.
        RuntimeError: If any source fails to rank.
    """
    if not 0 < train_share < 1:
        raise ValueError(f"train_share must be in (0, 1), got {train_share}")
    sources = sorted(_group_shards_by_source(Path(root), "dataset"))
    results, failures = run_parallel(
        _rank_source,
        sources,
        max_workers=max_workers,
        desc="split",
        kwargs_for=lambda _source: {"root": Path(root), "train_share": train_share},
    )
    if failures:
        raise RuntimeError(f"Corpus split failed for sources {[name for name, _exc in failures]}")
    return CorpusSplit(
        train_share=train_share,
        thresholds={name: results[name]["threshold"] for name in sources},
        counts={name: results[name]["counts"] for name in sources},
        corpus_fingerprint=corpus_fingerprint(Path(root)),
    )


def save_corpus_split(split: CorpusSplit, path: Path) -> Path:
    """Write a corpus split to JSON.

    Args:
        split (CorpusSplit): The split to persist.
        path (Path): Destination `.json` file.

    Returns:
        Path: `path`, for convenient chaining.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train_share": split.train_share,
        "thresholds": split.thresholds,
        "counts": split.counts,
        "corpus_fingerprint": split.corpus_fingerprint,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_corpus_split(path: Path) -> CorpusSplit:
    """Read a corpus split written by `save_corpus_split`.

    Args:
        path (Path): The split's `.json` file.

    Returns:
        CorpusSplit: The persisted split.

    Raises:
        FileNotFoundError: If `path` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus split not found: {path}; run `sweep_tokenizers.py sample` first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return CorpusSplit(
        train_share=float(payload["train_share"]),
        thresholds={name: int(value) for name, value in payload["thresholds"].items()},
        counts=payload["counts"],
        corpus_fingerprint=payload["corpus_fingerprint"],
    )


def corpus_fingerprint(root: Path) -> str:
    """Return a digest of the corpus shard set: each shard's path, size and modification time.

    A rebuilt stage rewrites its shards, so the digest changes and a split
    fitted on the old corpus is refused rather than reused.

    Args:
        root (Path): Corpus root.

    Returns:
        str: The SHA-256 hex digest over every shard's relative path, size and mtime.
    """
    digest = hashlib.sha256()
    for _source, shard in _iter_shards(Path(root), None):
        stat = shard.stat()
        digest.update(f"{shard.relative_to(root)}\t{stat.st_size}\t{stat.st_mtime_ns}\n".encode("utf-8"))
    return digest.hexdigest()


def check_sample_side(cache_path: Path, side: str) -> None:
    """Refuse a cached sample that was not drawn from `side` of a corpus split.

    Args:
        cache_path (Path): A sample written by `materialize_sample`.
        side (str): The side the caller needs, `train` or `eval`.

    Raises:
        ValueError: If the sample has no source sidecar or was drawn from another side.
    """
    sources_path = Path(cache_path).with_name(Path(cache_path).name + ".sources.json")
    drawn = json.loads(sources_path.read_text(encoding="utf-8")).get("side") if sources_path.exists() else None
    if drawn != side:
        raise ValueError(
            f"Cached sample {cache_path} was drawn from side {drawn!r}, not {side!r}; rerun `sample --force`"
        )


def iter_sample(cache_path: Path) -> Iterator[str]:
    """Yield one text per line from a cache written by `write_sample`.

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


def count_sample_characters(cache_path: Path) -> Counter:
    """Count the non-space characters of a cached sample, after NFC.

    The character-level backends normalize to NFC and never keep whitespace as
    a symbol, so these counts are the ones their alphabets are drawn from.

    Args:
        cache_path (Path): Path to the `.txt`/`.txt.gz` cache file.

    Returns:
        Counter: Occurrences of each character.
    """
    counts: Counter = Counter()
    for line in iter_sample(cache_path):
        counts.update(unicodedata.normalize("NFC", line))
    for char in [c for c in counts if c.isspace()]:
        del counts[char]
    return counts


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
