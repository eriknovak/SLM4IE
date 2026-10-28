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
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from slm4ie.utils.io import open_output, open_text_stream
from slm4ie.utils.parallel import run_parallel

# ===============================
# Logging
# ===============================

logger = logging.getLogger(__name__)

# ===============================
# Defaults
# ===============================

#: The two sides of a corpus split.
SPLIT_SIDES = ("train", "eval")

#: Rank value above every 64-bit hash, so a source can put all its documents on one side.
_RANK_CEILING = 2**64

# ===============================
# Sampling
# ===============================


@dataclass
class SampleBudget:
    """Constraints controlling how much text `sample_corpus` draws.

    Attributes:
        max_bytes (int | None): Total UTF-8 byte budget across the whole
            sample, or None for no byte cap.
        max_docs (int | None): Total document budget, or None for no cap.
        seed (int): Seed for the deterministic shard shuffle.
        weight_key (str): Metadata field used to group documents into sources
            for weighting (typically `dataset` or `domain`).
        source_weights (dict[str, float]): Relative weights per source value.
            Sources absent from the map default to weight 1.0; a weight of 0
            excludes a source. An empty map samples all sources uniformly.
    """

    max_bytes: int | None = None
    max_docs: int | None = None
    seed: int = 0
    weight_key: str = "dataset"
    source_weights: dict[str, float] = field(default_factory=dict)


def iter_dedup_documents(
    root: Path,
    datasets: list[str] | None = None,
) -> Iterator[dict[str, Any]]:
    """Stream datatrove documents from the deduplicated corpus tree.

    Reads `root/<dataset>/*.jsonl.gz` in sorted, deterministic order. Hidden
    and underscore-prefixed subdirectories (e.g. dedup state) are skipped.

    Args:
        root: Corpus root, e.g. `pretrain/06_sentence_dedup`.
        datasets: Restrict to these dataset subdirs;
            None reads every subdir.

    Yields:
        One parsed `{text, id, metadata}` record per line.
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
    split: "CorpusSplit | None" = None,
    side: str | None = None,
) -> Iterator[tuple[str, str]]:
    """Yield `(source, text)` for each sampled document, honoring the budget.

    Per-source byte/document quotas are computed from the normalized weights,
    then each source's shards are consumed in a seeded shuffled order until its
    quota is met. The traversal is fully deterministic for a fixed seed. With a
    `split`, only documents on `side` are drawn, and when the budget names no
    weights each source weighs as much as its bytes on that side, so the
    sample keeps the side's source mix.

    Args:
        root: Corpus root, e.g. `pretrain/06_sentence_dedup`.
        budget: Sampling constraints.
        split: The corpus split to draw from, or None
            to draw from every document.
        side: `train` or `eval`; required with `split`.

    Yields:
        The source name and the `text` of each sampled document.

    Raises:
        ValueError: If `split` is given without a valid `side`.
    """
    if split is not None and side not in SPLIT_SIDES:
        raise ValueError(f"side must be one of {SPLIT_SIDES} with a split, got {side!r}")

    # resolve the per-source weights
    root = Path(root)
    sources = _group_shards_by_source(root, budget.weight_key)
    source_weights = budget.source_weights
    if split is not None and not source_weights:
        source_weights = {name: float(split.counts[name][side]["bytes"]) for name in split.counts}
    weights = _resolve_weights(sources.keys(), source_weights)
    total_weight = sum(weights.values()) or 1.0
    rng = random.Random(budget.seed)

    # draw each source up to its quota
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
    split: "CorpusSplit | None" = None,
    side: str | None = None,
) -> Iterator[str]:
    """Yield sampled document texts honoring the budget and source weights.

    Args:
        root: Corpus root, e.g. `pretrain/06_sentence_dedup`.
        budget: Sampling constraints.
        split: The corpus split to draw from, or None.
        side: `train` or `eval`; required with `split`.

    Yields:
        The `text` field of each sampled document.
    """
    for _source, text in iter_sample_documents(root, budget, split, side):
        yield text


def write_sample(texts: Iterator[str], cache_path: Path) -> Path:
    """Materialize sampled texts to a one-document-per-line cache file.

    Internal newlines are collapsed to spaces so each document occupies a
    single line, which the line-oriented tokenizer trainers expect. The output
    is gzip-compressed when `cache_path` ends in `.gz`.

    Args:
        texts: Sampled document texts.
        cache_path: Destination file (`.txt` or `.txt.gz`).

    Returns:
        `cache_path`, for convenient chaining.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # write one flattened document per line
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
    split: "CorpusSplit | None" = None,
    side: str | None = None,
) -> Path:
    """Draw a sample, write it with `write_sample`, and record its source mix.

    The documents, words and bytes drawn per source are written beside the
    cache as `<cache name>.sources.json`, so the sample's composition can be
    tabled without re-reading it.

    Args:
        root: Corpus root.
        budget: Sampling constraints.
        cache_path: Destination of the sample (`.txt` or `.txt.gz`).
        split: The corpus split to draw from, or None.
        side: `train` or `eval`; required with `split`.

    Returns:
        `cache_path`, for convenient chaining.
    """
    counts: dict[str, dict[str, int]] = {}

    def tallied() -> Iterator[str]:
        for source, text in iter_sample_documents(root, budget, split, side):
            tally = counts.setdefault(source, {"documents": 0, "words": 0, "bytes": 0})
            tally["documents"] += 1
            tally["words"] += len(text.split())
            tally["bytes"] += len(text.encode("utf-8"))
            yield text

    # write the sample
    write_sample(tallied(), cache_path)

    # write the source mix
    sources_path = Path(cache_path).with_name(Path(cache_path).name + ".sources.json")
    sources_path.write_text(json.dumps({"side": side, "sources": counts}, indent=2, sort_keys=True), encoding="utf-8")
    return Path(cache_path)


# ===============================
# Corpus split
# ===============================


def stable_rank(key: str) -> int:
    """Return a reproducible 64-bit rank for `key`, uniform over keys.

    Args:
        key: Any identifier, such as `<source>/<document id>`.

    Returns:
        The first eight bytes of the key's BLAKE2b digest, as an integer.
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
        thresholds (dict[str, int]): Per source, the rank below which a document
            is on the training side.
        counts (dict[str, dict[str, dict[str, int]]]): Per source and side, the
            `documents`, `words` and `bytes` the side holds.
        corpus_fingerprint (str): `corpus_fingerprint` of the corpus the split
            was fitted on.
    """

    train_share: float
    thresholds: dict[str, int]
    counts: dict[str, dict[str, dict[str, int]]]
    corpus_fingerprint: str

    def side_of(self, source: str, doc_id: str) -> str:
        """Return the side a document belongs to.

        Args:
            source: The document's source (its dataset subdir).
            doc_id: The document's `id`.

        Returns:
            `train` or `eval`.

        Raises:
            KeyError: If `source` was not part of the corpus the split was fitted on.
        """
        return "train" if stable_rank(f"{source}/{doc_id}") < self.thresholds[source] else "eval"


def _rank_source(source: str, *, root: Path, train_share: float) -> dict[str, Any]:
    """Rank one source's documents and find its training-side threshold.

    Top-level so it pickles for the process pool.

    Args:
        source: Dataset subdir under `root`.
        root: Corpus root.
        train_share: Share of documents on the training side.

    Returns:
        The source's `threshold` and per-side `counts`.
    """
    # rank every document
    ranks: list[int] = []
    words: list[int] = []
    sizes: list[int] = []
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

    # find the training-side threshold
    order = sorted(range(len(ranks)), key=ranks.__getitem__)
    n_train = round(len(order) * train_share)
    threshold = ranks[order[n_train]] if n_train < len(order) else _RANK_CEILING

    # count each side
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
        root: Corpus root, e.g. `pretrain/06_sentence_dedup`.
        train_share: Share of each source's documents on the training side.
        max_workers: Sources ranked at once.

    Returns:
        The thresholds and the per-side counts of every source.

    Raises:
        ValueError: If `train_share` is not strictly between 0 and 1.
        RuntimeError: If any source fails to rank.
    """
    if not 0 < train_share < 1:
        raise ValueError(f"train_share must be in (0, 1), got {train_share}")

    # rank the sources in parallel
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
        split: The split to persist.
        path: Destination `.json` file.

    Returns:
        `path`, for convenient chaining.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # write the split
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
        path: The split's `.json` file.

    Returns:
        The persisted split.

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
        root: Corpus root.

    Returns:
        The SHA-256 hex digest over every shard's relative path, size and mtime.
    """
    # hash each shard's path, size and mtime
    digest = hashlib.sha256()
    for _source, shard in _iter_shards(Path(root), None):
        stat = shard.stat()
        digest.update(f"{shard.relative_to(root)}\t{stat.st_size}\t{stat.st_mtime_ns}\n".encode("utf-8"))
    return digest.hexdigest()


# ===============================
# Cached samples
# ===============================


def check_sample_side(cache_path: Path, side: str) -> None:
    """Refuse a cached sample that was not drawn from `side` of a corpus split.

    Args:
        cache_path: A sample written by `materialize_sample`.
        side: The side the caller needs, `train` or `eval`.

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
        cache_path: Path to the `.txt`/`.txt.gz` cache file.

    Yields:
        One cached document per line, trailing newline stripped.
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
        cache_path: Path to the `.txt`/`.txt.gz` cache file.

    Returns:
        Occurrences of each character.
    """
    # count the characters
    counts: Counter = Counter()
    for line in iter_sample(cache_path):
        counts.update(unicodedata.normalize("NFC", line))

    # drop whitespace
    for char in [c for c in counts if c.isspace()]:
        del counts[char]
    return counts


# ===============================
# Private helpers
# ===============================


def _iter_shards(
    root: Path,
    datasets: list[str] | None,
) -> Iterator[tuple[str, Path]]:
    """Yield `(dataset, shard_path)` pairs across the corpus tree.

    Args:
        root: Corpus root.
        datasets: Restrict to these subdirs, or None.

    Yields:
        Dataset subdir name and one shard path.
    """
    if not root.exists():
        logger.warning("Corpus root does not exist: %s", root)
        return
    # walk the dataset subdirs in name order
    wanted = set(datasets) if datasets else None
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        if sub.name.startswith((".", "_")):
            continue
        if wanted is not None and sub.name not in wanted:
            continue
        for shard in sorted(sub.glob("*.jsonl.gz")):
            yield sub.name, shard


def _group_shards_by_source(root: Path, weight_key: str) -> dict[str, list[Path]]:
    """Group corpus shards by their source value.

    The source value is the directory name, which equals the dataset key for
    the standard layout. `weight_key` is accepted for symmetry with
    `SampleBudget`; per-document grouping on `domain` would require scanning
    every record, so directory-level grouping (dataset) is used.

    Args:
        root: Corpus root.
        weight_key: Metadata field the caller intends to weight on.

    Returns:
        Shard paths grouped by source name.
    """
    del weight_key  # directory-level grouping, as documented above
    grouped: dict[str, list[Path]] = {}
    for source, shard in _iter_shards(root, None):
        grouped.setdefault(source, []).append(shard)
    return grouped


def _resolve_weights(
    sources: Any,
    source_weights: dict[str, float],
) -> dict[str, float]:
    """Assign a positive weight to each available source.

    Args:
        sources: Iterable of available source names.
        source_weights: User-supplied weights; missing
            sources default to 1.0, non-positive weights are dropped.

    Returns:
        Source name to positive weight.
    """
    resolved: dict[str, float] = {}
    for source in sources:
        weight = source_weights.get(source, 1.0)
        if weight > 0:
            resolved[source] = float(weight)
    return resolved


def _quota_met(
    used_bytes: int,
    used_docs: int,
    quota_bytes: int | None,
    quota_docs: int | None,
) -> bool:
    """Return True when any active quota has been reached.

    Args:
        used_bytes: Bytes consumed so far for the source.
        used_docs: Documents consumed so far for the source.
        quota_bytes: Byte quota, or None.
        quota_docs: Document quota, or None.

    Returns:
        True if a set quota is met or exceeded.
    """
    if quota_bytes is not None and used_bytes >= quota_bytes:
        return True
    if quota_docs is not None and used_docs >= quota_docs:
        return True
    return False
