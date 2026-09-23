"""Stratified sampling of curation decisions from an already-built corpus.

Every filtering stage writes only the documents it keeps, so what a stage
*dropped* exists on disk only as the difference between its output and its
input. This module reconstructs that difference and draws a balanced sample
from it, so a judge can score a stage's drops instead of its survivors alone.

The strata are `dataset x stage x decision`, where the decision is `kept`
(the document is in the stage's output) or `dropped` (it is in the stage's
input but not its output). Cells are sampled independently with a fixed seed,
then merged on document id: one row per document, carrying every cell it was
drawn into.

A stage does not keep its documents in the shard they arrived in, so no output
shard can be set against an input shard opposite it. Deciding what a stage
dropped therefore needs the stage's *whole* output: the cell reads all of it
once, into a kept sample and an index of surviving ids, and then reads a
bounded number of input shards, where every document the index does not know
was dropped. `shards_per_cell` bounds only that second read.

Text is truncated to `max_chars` characters. Kept text comes from the stage's
output and dropped text from its input, so each row shows what the stage
produced or what it discarded.

One caveat rides on the ids: a source whose ids are not unique reports fewer
drops than it made, because a dropped document sharing an id with a surviving
one reads as a survivor.
"""

import gzip
import hashlib
import json
import logging
import random
from array import array
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from slm4ie.data.curate.stages import STAGE_DIRS, final_corpus_dir, upstream_stage
from slm4ie.data.io_utils import resolve_project_path

logger = logging.getLogger(__name__)

#: Stages that make a keep-or-drop decision per document. `convert` only
#: reshapes records and `statistics` writes no corpus, so neither is judged.
JUDGED_STAGES: Tuple[str, ...] = (
    "language",
    "spam",
    "quality",
    "repetition",
    "exact_dedup",
    "sentence_dedup",
)

#: The two decisions a judged stage can make about a document.
DECISIONS: Tuple[str, str] = ("kept", "dropped")

#: Bytes read from disk at a time. The corpus lives on a spinning disk, where
#: large reads keep concurrent shard scans from turning into seeks.
READ_BUFFER: int = 1 << 23


def _cell_rng(seed: int, *key_parts: str) -> random.Random:
    """Build a reproducible generator for one sampling cell.

    Args:
        seed: Run-wide seed.
        key_parts: Strings identifying the cell, e.g. dataset and stage.

    Returns:
        A `random.Random` whose stream depends only on *seed* and *key_parts*,
        so cells are independent of iteration order and worker count.
    """
    digest = hashlib.blake2b("\x1f".join((str(seed), *key_parts)).encode("utf-8"), digest_size=8).digest()
    return random.Random(int.from_bytes(digest, "big"))


def _iter_shard(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield every document in a gzipped JSONL shard.

    Args:
        path: Path to a `*.jsonl.gz` shard.

    Yields:
        One decoded document per line.
    """
    with open(path, "rb", buffering=READ_BUFFER) as raw, gzip.open(raw, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _field(doc: Dict[str, Any], key: str) -> Optional[str]:
    """Read a field that may sit at the top level or inside `metadata`.

    The convert stage writes `dataset` and `domain` as top-level keys; every
    later stage carries them under `metadata`.

    Args:
        doc: A decoded document.
        key: Field name to read.

    Returns:
        The field's value as a string, or None when it is absent.
    """
    value = doc.get(key)
    if value is None:
        value = (doc.get("metadata") or {}).get(key)
    return None if value is None else str(value)


def _reservoir_add(reservoir: List[Any], item: Any, seen: int, size: int, rng: random.Random) -> None:
    """Offer one item to a reservoir sample of fixed size.

    Args:
        reservoir: Items kept so far; mutated in place.
        item: The candidate item.
        seen: Number of candidates offered before this one.
        size: Target sample size.
        rng: Generator driving the replacement decision.
    """
    if len(reservoir) < size:
        reservoir.append(item)
        return
    index = rng.randrange(seen + 1)
    if index < size:
        reservoir[index] = item


def resolve_output_dir(config_path: Path, override: Optional[Path] = None) -> Path:
    """Resolve the folder holding the stage outputs.

    Args:
        config_path: Path to the curation config, read for its `output_dir`.
        override: Explicit folder, which wins over the config.

    Returns:
        Path to the folder holding `00_convert/`, `01_language/` and the rest.

    Raises:
        FileNotFoundError: If the config cannot be read, or neither *override*
            nor its `output_dir` is set.
    """
    if override is not None:
        return override
    try:
        with config_path.open() as fh:
            cfg = yaml.safe_load(fh) or {}
    except OSError as exc:
        raise FileNotFoundError(f"curation config not readable: {config_path}") from exc
    output_dir = cfg.get("output_dir")
    if output_dir is None:
        raise FileNotFoundError(f"no corpus root: pass an override or set output_dir in {config_path}.")
    return resolve_project_path(output_dir)


def roster(output_dir: Path) -> List[str]:
    """List the datasets that reached the final corpus.

    Args:
        output_dir: The curation `output_dir`, holding the stage folders.

    Returns:
        Sorted dataset keys present in the final corpus folder. Earlier
        stages may hold more (excluded benchmark corpora, sources that were
        dropped entirely), and those are not sampled.
    """
    final_dir = output_dir / final_corpus_dir()
    return sorted(child.name for child in final_dir.iterdir() if child.is_dir())


def _shards(output_dir: Path, stage: str, dataset: str) -> List[Path]:
    """List a stage's shards for one dataset, in sorted order.

    Args:
        output_dir: The curation `output_dir`.
        stage: Stage name.
        dataset: Dataset key.

    Returns:
        Sorted shard paths; empty when the stage never wrote this dataset.
    """
    stage_dir = output_dir / STAGE_DIRS[stage] / dataset
    return sorted(stage_dir.glob("*.jsonl.gz")) if stage_dir.is_dir() else []


class SurvivorIndex:
    """The ids a stage kept, held as sorted 64-bit hashes.

    A stage redistributes its documents freely across output shards, so
    deciding whether one input document survived means consulting the stage's
    whole output, not a shard opposite it. Storing a hash per id keeps that
    index to 8 bytes per document, which fits a corpus of tens of millions in
    memory while a plain set of the id strings would not.

    The hashes come from the interpreter's own `hash`, which is seeded per
    process, so an index is only ever valid inside the process that built it.
    """

    def __init__(self) -> None:
        """Start an empty index."""
        self._hashes = array("Q")
        self._sorted: Optional[Any] = None

    def add(self, document_id: str) -> None:
        """Record one surviving id.

        Args:
            document_id: The document's id.
        """
        self._hashes.append(hash(document_id) & 0xFFFFFFFFFFFFFFFF)

    def freeze(self) -> None:
        """Sort the recorded hashes so the index can be queried."""
        self._sorted = np.sort(np.frombuffer(self._hashes, dtype=np.uint64))

    def __contains__(self, document_id: str) -> bool:
        """Report whether an id survived the stage.

        Args:
            document_id: The document's id.

        Returns:
            True if the id is in the index. Distinct ids collide with
            probability around 1e-5 over a corpus this size, so a rare drop
            may read as a survivor; none is ever invented the other way.
        """
        needle = hash(document_id) & 0xFFFFFFFFFFFFFFFF
        position = int(np.searchsorted(self._sorted, needle))
        return position < len(self._sorted) and bool(self._sorted[position] == needle)

    def __len__(self) -> int:
        """Return how many ids were recorded.

        Returns:
            The number of documents the stage kept.
        """
        return len(self._hashes)


def _row(doc: Dict[str, Any], dataset: str, stage: str, decision: str, shard: Path, max_chars: int) -> Dict[str, Any]:
    """Build one sample row from a document.

    Args:
        doc: The decoded document.
        dataset: Dataset key.
        stage: Stage the decision belongs to.
        decision: `kept` or `dropped`.
        shard: Shard the document was read from.
        max_chars: Characters of text to keep.

    Returns:
        A row carrying the document's identity, its single cell, and its
        truncated text.
    """
    text = doc.get("text") or ""
    return {
        "id": doc["id"],
        "dataset": dataset,
        "domain": _field(doc, "domain"),
        "cells": [{"stage": stage, "decision": decision, "shard": shard.name}],
        "chars": len(text),
        "truncated": len(text) > max_chars,
        "text": text[:max_chars],
    }


def sample_cell(
    output_dir: Path,
    dataset: str,
    stage: str,
    per_cell: int,
    shards_per_cell: int,
    max_chars: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Draw the kept and dropped samples for one dataset and stage.

    Reads the stage's whole output for this dataset, which gives both the kept
    sample and the index of surviving ids, then reads a bounded number of its
    input shards, where any document the index does not know was dropped.

    Args:
        output_dir: The curation `output_dir`.
        dataset: Dataset key.
        stage: A stage from `JUDGED_STAGES`.
        per_cell: Documents to draw per decision.
        shards_per_cell: Input shards to read for the dropped sample, or 0 for
            all of them.
        max_chars: Characters of text to keep per document.
        seed: Run-wide seed.

    Returns:
        Sample rows, each with exactly one cell.

    Raises:
        ValueError: If *stage* has no upstream stage, so it makes no
            keep-or-drop decision.
    """
    previous = upstream_stage(stage)
    if previous is None:
        raise ValueError(f"{stage} has no upstream stage, so it makes no keep-or-drop decision")
    shards = _shards(output_dir, stage, dataset)
    if not shards:
        logger.warning("[%s/%s] no shards; skipped", dataset, stage)
        return []

    kept: List[Dict[str, Any]] = []
    kept_rng = _cell_rng(seed, dataset, stage, "kept")
    kept_seen = 0
    survivors = SurvivorIndex()
    for shard in shards:
        for doc in _iter_shard(shard):
            _reservoir_add(kept, _row(doc, dataset, stage, "kept", shard, max_chars), kept_seen, per_cell, kept_rng)
            kept_seen += 1
            survivors.add(doc["id"])
    survivors.freeze()

    inputs = _shards(output_dir, previous, dataset)
    chooser = _cell_rng(seed, dataset, stage, "shards")
    if 0 < shards_per_cell < len(inputs):
        inputs = sorted(chooser.sample(inputs, shards_per_cell))

    dropped: List[Dict[str, Any]] = []
    dropped_rng = _cell_rng(seed, dataset, stage, "dropped")
    dropped_seen = read = 0
    for shard in inputs:
        for doc in _iter_shard(shard):
            read += 1
            if doc["id"] in survivors:
                continue
            row = _row(doc, dataset, stage, "dropped", shard, max_chars)
            _reservoir_add(dropped, row, dropped_seen, per_cell, dropped_rng)
            dropped_seen += 1

    logger.info(
        "[%s/%s] kept %d of %d (whole stage), dropped %d of %d in %d of %d input shard(s)",
        dataset,
        stage,
        len(kept),
        kept_seen,
        len(dropped),
        dropped_seen,
        len(inputs),
        len(_shards(output_dir, previous, dataset)),
    )
    return kept + dropped


def _merge(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse rows that describe the same document.

    A document kept by one stage is usually kept by the next, so the same
    text can be drawn into several cells. Judging it once and attaching every
    cell keeps the judge's cost down without losing a stratum.

    Args:
        rows: Single-cell rows from every sampled cell.

    Returns:
        One row per document id, sorted by id, with the cells merged and
        sorted.
    """
    merged: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        existing = merged.get(row["id"])
        if existing is None:
            merged[row["id"]] = row
            continue
        existing["cells"].extend(row["cells"])
    for row in merged.values():
        row["cells"].sort(key=lambda cell: (cell["stage"], cell["decision"]))
    return [merged[key] for key in sorted(merged)]


def _cache_dir(destination: Path, per_cell: int, shards_per_cell: int, max_chars: int, seed: int) -> Path:
    """Name the folder holding each cell's drawn rows.

    A cell costs a pass over a stage, so a run that stops partway should not
    start over. The settings that change what a cell contains are folded into
    the folder name, so a cache is never reused for a different draw.

    Args:
        destination: The JSONL file the sample is written to.
        per_cell: Documents per cell.
        shards_per_cell: Input shards searched for drops per cell.
        max_chars: Characters of text kept per document.
        seed: Run-wide seed.

    Returns:
        Path to the cache folder beside *destination*.
    """
    settings = f"{per_cell}-{shards_per_cell}-{max_chars}-{seed}"
    digest = hashlib.blake2b(settings.encode("utf-8"), digest_size=4).hexdigest()
    return destination.parent / f"{destination.stem}.cells-{digest}"


def _cell_file(cache: Path, dataset: str, stage: str) -> Path:
    """Name one cell's file inside the cache.

    Args:
        cache: The cache folder.
        dataset: Dataset key.
        stage: Stage name.

    Returns:
        Path to the cell's JSONL file.
    """
    return cache / f"{dataset}__{stage}.jsonl"


def _read_cell(cache: Path, dataset: str, stage: str) -> List[Dict[str, Any]]:
    """Read a cell's drawn rows back from the cache.

    Args:
        cache: The cache folder.
        dataset: Dataset key.
        stage: Stage name.

    Returns:
        The rows the cell drew, empty when the cell was never drawn.
    """
    path = _cell_file(cache, dataset, stage)
    if not path.is_file():
        return []
    # split("\n"), not splitlines(): document text carries \u2028 and friends,
    # which splitlines() would treat as line breaks and tear a JSON row in half.
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def _draw_cell(argument: Tuple[Path, str, str, int, int, int, int, Path]) -> None:
    """Draw one cell and write it to the cache.

    Args:
        argument: The cell's arguments for `sample_cell`, with the cache folder
            last. Packed into one tuple so the call maps over a process pool.
    """
    output_dir, dataset, stage, per_cell, shards_per_cell, max_chars, seed, cache = argument
    rows = sample_cell(output_dir, dataset, stage, per_cell, shards_per_cell, max_chars, seed)
    path = _cell_file(cache, dataset, stage)
    # Written aside, then moved: a cell file exists only once it is complete.
    partial = path.with_suffix(".partial")
    partial.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    partial.replace(path)


def draw_stratified_sample(
    output_dir: Path,
    destination: Path,
    datasets: Optional[List[str]] = None,
    stages: Sequence[str] = JUDGED_STAGES,
    per_cell: int = 40,
    shards_per_cell: int = 3,
    max_chars: int = 2000,
    seed: int = 20260916,
    workers: int = 1,
) -> Dict[str, int]:
    """Draw a `dataset x stage x decision` sample and write it as JSONL.

    Args:
        output_dir: The curation `output_dir` holding the stage folders.
        destination: JSONL file to write; its parent is created.
        datasets: Dataset keys to sample, or None for every dataset in the
            final corpus.
        stages: Stages to sample, defaulting to every judged stage.
        per_cell: Documents per cell.
        shards_per_cell: Input shards searched for drops per cell, or 0 for
            all of them.
        max_chars: Characters of text kept per document.
        seed: Run-wide seed; the same seed redraws the same sample.
        workers: Cells sampled in parallel. 1 runs in this process. A cell
            is bound by how fast its stage can be read off disk, so the
            useful number is whatever keeps that disk busy.

    Each cell's rows are cached beside *destination*, so a run that is
    interrupted or restarted redraws only the cells it never finished.

    Returns:
        Counts keyed `documents`, `cells`, `kept` and `dropped`.

    Raises:
        ValueError: If a requested stage is not a judged stage.
    """
    unknown = [stage for stage in stages if stage not in JUDGED_STAGES]
    if unknown:
        raise ValueError(f"not judged stages: {', '.join(unknown)}; expected {', '.join(JUDGED_STAGES)}")

    keys = datasets or roster(output_dir)
    cells = [(dataset, stage) for dataset in sorted(keys) for stage in stages]
    logger.info("sampling %d cells over %d dataset(s) with %d worker(s)", len(cells), len(keys), workers)

    cache = _cache_dir(destination, per_cell, shards_per_cell, max_chars, seed)
    cache.mkdir(parents=True, exist_ok=True)
    pending = [cell for cell in cells if not _cell_file(cache, *cell).is_file()]
    if len(pending) < len(cells):
        logger.info("%d cell(s) already drawn in %s; %d to go", len(cells) - len(pending), cache, len(pending))

    arguments = [
        (output_dir, dataset, stage, per_cell, shards_per_cell, max_chars, seed, cache) for dataset, stage in pending
    ]
    if workers > 1 and len(arguments) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_draw_cell, arguments))
    else:
        for argument in arguments:
            _draw_cell(argument)

    rows = [row for cell in cells for row in _read_cell(cache, *cell)]
    merged = _merge(rows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as fh:
        for row in merged:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = {
        "documents": len(merged),
        "cells": len(rows),
        "kept": sum(1 for row in rows if row["cells"][0]["decision"] == "kept"),
        "dropped": sum(1 for row in rows if row["cells"][0]["decision"] == "dropped"),
    }
    logger.info(
        "wrote %d documents (%d cells: %d kept, %d dropped) to %s",
        counts["documents"],
        counts["cells"],
        counts["kept"],
        counts["dropped"],
        destination,
    )
    return counts
