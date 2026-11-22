"""Tokenize a corpus stage into a pretokenized output: the argv-free entry point.

One input shard (a `*.jsonl.gz` file of the corpus stage) becomes one part per
output format and split. `convert_shard` tokenizes its documents, cuts them into
chunks at sentence boundaries, wraps each chunk with the tokenizer's template,
and writes every sample to every configured format. A part is written under
`_partial/` and moved into place only when complete, with a marker under
`_parts/` holding its counts, so a rerun skips finished shards.
"""

from __future__ import annotations

import gzip
import json
import logging
import shutil
import time
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from slm4ie.data.pretokenize.chunking import chunk_document, split_sentences
from slm4ie.data.pretokenize.config import PretokenizeConfig, PretokenizeEntry
from slm4ie.data.pretokenize.formats import PartWriter, Sample, get_format
from slm4ie.data.pretokenize.splits import SPLITS, resolve_validation_share, split_of
from slm4ie.utils.cli import stamped_log_dir
from slm4ie.utils.parallel import choose_workers, run_parallel
from slm4ie.utils.versioning import config_hash, corpus_digest, file_sha256, shard_listing

# ===============================
# Logging
# ===============================

logger = logging.getLogger(__name__)

# ===============================
# Constants
# ===============================

PARTIAL_DIR = "_partial"
PARTS_DIR = "_parts"
MANIFEST_FILE = "manifest.json"
# the manifest section that changes with every run of the same output
LAST_RUN_KEY = "last_run"
SETTINGS_FILE = "_settings.json"
INPUTS_FILE = "_inputs.json"
# a corpus stage holds <dataset>/*.jsonl.gz
INPUT_GLOBS = ("*.jsonl.gz",)
# documents tokenized in one batch call
BATCH_DOCS = 1000
# characters per batch call: worker memory grows with the text in a batch, not its count
BATCH_CHARS = 4_000_000
COUNT_KEYS = ("documents", "samples", "tokens", "dropped_chunks", "dropped_tokens")
# a text whose ids show where a template puts its special tokens
WRAP_PROBE = "a"
# the run's closing line, beside the per-shard logs
SUMMARY_LOG = "summary.log"

# ===============================
# Helper functions
# ===============================


@lru_cache(maxsize=4)
def load_tokenizer(tokenizer_dir: str) -> Any:
    """Load an exported tokenizer once per process.

    Args:
        tokenizer_dir: The exported HuggingFace tokenizer folder.

    Returns:
        The tokenizer; MorphPiece exports load their own code.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)


def template_wrap(tokenizer: Any) -> tuple[list[int], list[int]]:
    """Return the special-token ids the tokenizer's single template adds around a text.

    A fast tokenizer applies its template only inside its own encode call, not
    in `build_inputs_with_special_tokens`, so the wrap is read off by encoding
    a probe text with and without special tokens.

    Args:
        tokenizer: The exported tokenizer.

    Returns:
        The ids added before and after the text.

    Raises:
        ValueError: If the plain encoding is not found inside the wrapped one.
    """
    plain = tokenizer(WRAP_PROBE, add_special_tokens=False)["input_ids"]
    wrapped = tokenizer(WRAP_PROBE)["input_ids"]
    for start in range(len(wrapped) - len(plain) + 1):
        if wrapped[start : start + len(plain)] == plain:
            return wrapped[:start], wrapped[start + len(plain) :]
    raise ValueError(f"Template of the tokenizer does not keep the text's ids: {plain} not in {wrapped}")


def _encode_with_offsets(tokenizer: Any, texts: list[str]) -> list[tuple[list[int], list[tuple[int, int]]]]:
    """Tokenize texts without special tokens, keeping each token's character span.

    Args:
        tokenizer: A fast tokenizer, or the slow MorphPiece one with `encode_with_offsets`.
        texts: The document texts.

    Returns:
        The ids and offsets of each text.
    """
    if tokenizer.is_fast:
        encoded = tokenizer(texts, add_special_tokens=False, return_offsets_mapping=True)
        return list(zip(encoded["input_ids"], encoded["offset_mapping"], strict=True))
    return [(row["input_ids"], row["offset_mapping"]) for row in map(tokenizer.encode_with_offsets, texts)]


def _iter_documents(shard_path: Path) -> Iterator[tuple[str, str, str]]:
    """Yield each document of an input shard with its id and source.

    Args:
        shard_path: A `*.jsonl.gz` file of the corpus stage.

    Yields:
        `(doc_id, source, text)`; the source is the record's dataset, or the
        shard's folder name without one.
    """
    with gzip.open(shard_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                source = (record.get("metadata") or {}).get("dataset") or shard_path.parent.name
                yield str(record["id"]), str(source), record.get("text") or ""


def _batches(
    documents: Iterator[tuple[str, str, str]], max_docs: int, max_chars: int
) -> Iterator[list[tuple[str, str, str]]]:
    """Group documents into batches of at most `max_docs` documents and `max_chars` characters.

    A document longer than `max_chars` forms a batch of its own.

    Args:
        documents: The `(doc_id, source, text)` documents.
        max_docs: Most documents in one batch.
        max_chars: Most characters of text in one batch.

    Yields:
        The batches, in order.
    """
    batch: list[tuple[str, str, str]] = []
    chars = 0
    for document in documents:
        length = len(document[2])
        if batch and chars + length > max_chars:
            yield batch
            batch, chars = [], 0
        batch.append(document)
        chars += length
        if len(batch) >= max_docs:
            yield batch
            batch, chars = [], 0
    if batch:
        yield batch


def _empty_counts() -> dict[str, int]:
    """Return zeroed counts for one source and split.

    Returns:
        Each of `COUNT_KEYS` mapped to 0.
    """
    return dict.fromkeys(COUNT_KEYS, 0)


def _counts_by_source() -> defaultdict[str, dict[str, dict[str, int]]]:
    """Return a mapping that starts every new source with zeroed counts per split.

    Returns:
        Source to split to counts, filled on first access.
    """
    return defaultdict(lambda: {split: _empty_counts() for split in SPLITS})


def _move_parts_into_place(entry: PretokenizeEntry, partial: Path, part: str) -> None:
    """Move a closed part of every format and split from `_partial/` to its split folder.

    Args:
        entry: The conversion settings.
        partial: The part's `_partial/<part>` folder.
        part: The part name.
    """
    for name in entry.formats:
        output_format = get_format(name)
        for split in SPLITS:
            source = output_format.part_path(partial / name / split, part)
            target = output_format.part_path(entry.output_dir / name / split, part)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink(missing_ok=True)
            source.rename(target)


def _directory_digest(folder: Path) -> str:
    """Hash the name and bytes of every file directly in a folder.

    Args:
        folder: The folder, such as an exported tokenizer.

    Returns:
        A `sha256:`-prefixed digest over each file's name and SHA-256.
    """
    return config_hash({path.name: file_sha256(path) for path in sorted(folder.iterdir()) if path.is_file()})


def _read_json(path: Path) -> Any:
    """Read a JSON file, or return None when it does not exist.

    Args:
        path: The file.

    Returns:
        The parsed content, or None.
    """
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _write_json(path: Path, payload: Any) -> None:
    """Write JSON through a temporary file, so a reader never sees half of it.

    Args:
        path: The file.
        payload: The content.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.rename(path)


def part_marker(output_dir: Path, part: str) -> Path:
    """Return the marker whose presence says a part is complete.

    Args:
        output_dir: The entry's output folder.
        part: The part name.

    Returns:
        `<output_dir>/_parts/<part>.json`.
    """
    return output_dir / PARTS_DIR / f"{part}.json"


# ===============================
# Shard conversion
# ===============================


def convert_shard(shard_path: Path, part: str, entry: PretokenizeEntry, share: float) -> dict[str, Any]:
    """Tokenize one input shard into one part per format and split.

    A complete part (its marker exists) is skipped and its counts returned. A
    part is written under `<output>/_partial/<part>/`, moved into each format's
    split folder when every writer has closed, and only then marked complete.

    Args:
        shard_path: A `*.jsonl.gz` file of the corpus stage.
        part: The part name; parts sort in input-shard order.
        entry: The conversion settings.
        share: The validation share, as `resolve_validation_share` returns it.

    Returns:
        `{"part", "shard", "counts"}`, where `counts` maps source to split to
        the counts of documents, samples, tokens and dropped chunks and tokens.
    """
    marker = part_marker(entry.output_dir, part)
    if marker.exists():
        return _read_json(marker)

    # load the tokenizer and open one writer per format and split
    tokenizer = load_tokenizer(str(entry.tokenizer_dir))
    prefix, suffix = template_wrap(tokenizer)
    body_len = entry.max_seq_len - len(prefix) - len(suffix)
    partial = entry.output_dir / PARTIAL_DIR / part
    shutil.rmtree(partial, ignore_errors=True)
    writers: dict[tuple[str, str], PartWriter] = {}
    for name in entry.formats:
        output_format = get_format(name)
        for split in SPLITS:
            path = output_format.part_path(partial / name / split, part)
            writers[name, split] = output_format.open_part(path, len(tokenizer), entry.shard_size_bytes)

    # tokenize, chunk, wrap and write each document
    counts = _counts_by_source()
    for batch in _batches(_iter_documents(shard_path), BATCH_DOCS, BATCH_CHARS):
        encoded = _encode_with_offsets(tokenizer, [text for _, _, text in batch])
        for (doc_id, source, text), (ids, offsets) in zip(batch, encoded, strict=True):
            split = split_of(doc_id, entry.seed, share)
            tally = counts[source][split]
            tally["documents"] += 1
            for index, chunk in enumerate(chunk_document(split_sentences(ids, offsets, text), body_len)):
                wrapped = [*prefix, *chunk, *suffix]
                if len(wrapped) < entry.min_chunk_tokens:
                    tally["dropped_chunks"] += 1
                    tally["dropped_tokens"] += len(chunk)
                    continue
                sample = Sample(input_ids=tuple(wrapped), id=f"{doc_id}#{index}")
                for name in entry.formats:
                    writers[name, split].write(sample)
                tally["samples"] += 1
                tally["tokens"] += sample.len

    # close the writers, move the parts into place, then mark the part complete
    for writer in writers.values():
        writer.close()
    _move_parts_into_place(entry, partial, part)
    shutil.rmtree(partial, ignore_errors=True)
    result = {
        "part": part,
        "shard": str(shard_path),
        "counts": {source: dict(splits) for source, splits in counts.items()},
    }
    _write_json(marker, result)
    logger.info("Converted %s -> part %s", shard_path, part)
    return result


# ===============================
# Entry conversion
# ===============================


@dataclass
class PretokenizeSummary:
    """Outcome of a pretokenize run, by entry key.

    Attributes:
        written: Entries converted in full, with their manifest written.
        skipped: Entries whose manifest already matched their settings and inputs.
        failed: Entries with at least one failed input shard, mapped to the failed parts.
    """

    written: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: dict[str, list[str]] = field(default_factory=dict)


def _settings(entry: PretokenizeEntry) -> dict[str, Any]:
    """Return the settings an output was built with, for the manifest and resume check.

    Args:
        entry: The conversion settings.

    Returns:
        The corpus, tokenizer and its digest, lengths, split and formats.
    """
    return {
        "corpus": str(entry.corpus_dir),
        "tokenizer": str(entry.tokenizer_dir),
        "tokenizer_digest": _directory_digest(entry.tokenizer_dir),
        "max_seq_len": entry.max_seq_len,
        "min_chunk_tokens": entry.min_chunk_tokens,
        "formats": list(entry.formats),
        "validation_share": entry.validation_share,
        "min_validation_docs": entry.min_validation_docs,
        "seed": entry.seed,
    }


def _list_inputs(entry: PretokenizeEntry) -> list[tuple[str, int, int]]:
    """List the input shards with their sizes and document counts.

    Counting documents decompresses every shard, so the counts are cached in
    the output folder and reused while the shard paths and sizes match.

    Args:
        entry: The conversion settings.

    Returns:
        `(relative path, size, documents)` per shard, sorted by path.

    Raises:
        FileNotFoundError: If the corpus stage holds no shard.
    """
    listing = shard_listing(entry.corpus_dir, globs=INPUT_GLOBS)
    if not listing:
        raise FileNotFoundError(f"No input shards matching {INPUT_GLOBS} under {entry.corpus_dir}")
    cache_path = entry.output_dir / INPUTS_FILE
    cached = _read_json(cache_path)
    if cached and [(path, size) for path, size, _ in cached] == [(path, size) for path, size, _ in listing]:
        return [tuple(row) for row in cached]
    counted = shard_listing(entry.corpus_dir, globs=INPUT_GLOBS, with_rows=True)
    _write_json(cache_path, counted)
    return counted


def _convert_part(part: str, *, shard_path: Path, entry: PretokenizeEntry, share: float) -> dict[str, Any]:
    """Run `convert_shard` with the part name first, as `run_parallel` calls it.

    Args:
        part: The part name.
        shard_path: The input shard.
        entry: The conversion settings.
        share: The validation share.

    Returns:
        The part's result, as `convert_shard` returns it.
    """
    return convert_shard(shard_path, part, entry, share)


def _sum_counts(results: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, int]]]:
    """Add the parts' counts per source and split.

    Args:
        results: Part results, as `convert_shard` returns them.

    Returns:
        Source to split to the summed counts.
    """
    totals = _counts_by_source()
    for result in results:
        for source, splits in result["counts"].items():
            for split, counts in splits.items():
                for key, value in counts.items():
                    totals[source][split][key] += value
    return {source: dict(splits) for source, splits in sorted(totals.items())}


def _build_manifest(
    entry: PretokenizeEntry,
    settings: dict[str, Any],
    share: float,
    inputs_digest: str,
    inputs: list[tuple[str, int, int]],
    results: dict[str, dict[str, Any]],
    converted: list[dict[str, Any]],
    start: float,
) -> dict[str, Any]:
    """Assemble the manifest of a finished output.

    Args:
        entry: The conversion settings.
        settings: The settings the output was built with.
        share: The validation share used.
        inputs_digest: Digest of the input shard set.
        inputs: `(relative path, size, documents)` per input shard.
        results: Part results by part name, in input-shard order.
        converted: Results of the parts this run converted, for the throughput.
        start: `time.perf_counter()` when this run began.

    Returns:
        The manifest, with counts per source and split, totals and the last run.
    """
    counts = _sum_counts(list(results.values()))
    elapsed = time.perf_counter() - start
    tokens = sum(split["tokens"] for splits in _sum_counts(converted).values() for split in splits.values())
    shards = [{"part": part, "path": path, "documents": rows} for part, (path, _, rows) in zip(results, inputs)]
    totals = {split: {key: sum(c[split][key] for c in counts.values()) for key in COUNT_KEYS} for split in SPLITS}
    return {
        "key": entry.key,
        "settings": settings,
        "validation_share_used": share,
        "inputs": {"digest": inputs_digest, "shards": shards},
        "counts": counts,
        "totals": totals,
        LAST_RUN_KEY: {
            "seconds": round(elapsed, 1),
            "parts_converted": len(converted),
            "parts_resumed": len(results) - len(converted),
            "tokens_converted": tokens,
            "tokens_per_second": round(tokens / elapsed) if elapsed else None,
        },
    }


def _log_summary(entry: PretokenizeEntry, manifest: dict[str, Any], log_dir: Path) -> None:
    """Log the run's closing line and write it to the run's log folder.

    Args:
        entry: The conversion settings.
        manifest: The manifest just written.
        log_dir: The run's log folder, beside the per-shard logs.
    """
    last_run, totals = manifest[LAST_RUN_KEY], manifest["totals"]
    dropped = sum(totals[split]["dropped_chunks"] for split in SPLITS)
    line = (
        f"{entry.key}: {last_run['parts_converted']} parts converted, {last_run['parts_resumed']} resumed; "
        f"{last_run['tokens_converted']} tokens in {last_run['seconds']} s ({last_run['tokens_per_second']} tokens/s); "
        f"{dropped} chunks dropped below {entry.min_chunk_tokens} tokens"
    )
    logger.info("%s", line)
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / SUMMARY_LOG).open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def convert_entry(entry: PretokenizeEntry, *, force: bool = False, max_workers: int = 0) -> tuple[str, list[str]]:
    """Convert every input shard of one entry, merge the parts and write the manifest.

    The manifest is written last and marks the output complete. An output whose
    manifest matches the current settings and inputs is skipped unless `force`.
    Parts finished by an interrupted run with the same settings are reused.

    Args:
        entry: The conversion settings.
        force: Delete the output and convert from scratch.
        max_workers: Parallel input shards; 0 sizes by cores, shards and memory.

    Returns:
        The outcome (`written`, `skipped` or `failed`) and the failed parts.

    Raises:
        ValueError: If unfinished parts were built with other settings and
            `force` is not set.
    """
    start = time.perf_counter()
    if force:
        shutil.rmtree(entry.output_dir, ignore_errors=True)

    # skip a complete output; refuse parts built from other settings or inputs
    settings = _settings(entry)
    inputs_digest = corpus_digest(entry.corpus_dir, globs=INPUT_GLOBS)
    manifest = _read_json(entry.output_dir / MANIFEST_FILE)
    if manifest and manifest["settings"] == settings and manifest["inputs"]["digest"] == inputs_digest:
        logger.info("Skipped %s: output matches its settings and inputs", entry.key)
        return "skipped", []
    built_from = {"settings": settings, "inputs_digest": inputs_digest}
    previous = _read_json(entry.output_dir / SETTINGS_FILE)
    if previous is not None and previous != built_from:
        raise ValueError(f"{entry.output_dir} holds parts built from other settings or inputs; rerun with --force")
    (entry.output_dir / MANIFEST_FILE).unlink(missing_ok=True)
    _write_json(entry.output_dir / SETTINGS_FILE, built_from)

    # list the inputs and set the validation share from the document count
    inputs = _list_inputs(entry)
    total_docs = sum(rows for _, _, rows in inputs)
    share = resolve_validation_share(entry.validation_share, entry.min_validation_docs, total_docs)
    parts = {f"{index:05d}": entry.corpus_dir / path for index, (path, _, _) in enumerate(inputs)}

    # convert the shards in parallel, one part each; parts done earlier are reused
    log_dir = stamped_log_dir("pretokenize")
    resumed = {part for part in parts if part_marker(entry.output_dir, part).exists()}
    workers = choose_workers(
        max_workers, len(parts), entry.workers.memory_per_worker_gb, label=f"pretokenize {entry.key}"
    )
    results, failures = run_parallel(
        _convert_part,
        list(parts),
        max_workers=workers,
        desc=f"pretokenize {entry.key}",
        kwargs_for=lambda part: {"shard_path": parts[part], "entry": entry, "share": share},
        log_dir=log_dir,
        max_items_per_worker=entry.workers.max_items_per_worker,
    )
    if failures:
        return "failed", sorted(part for part, _ in failures)
    # parts finish in any order; the manifest pairs them with inputs in shard order
    results = {part: results[part] for part in parts}

    # merge each split's parts in shard order, then write the manifest last
    for name in entry.formats:
        for split in SPLITS:
            get_format(name).merge_parts(entry.output_dir / name / split, list(parts))
    converted = [results[part] for part in parts if part not in resumed]
    manifest = _build_manifest(
        entry,
        settings=settings,
        share=share,
        inputs_digest=inputs_digest,
        inputs=inputs,
        results=results,
        converted=converted,
        start=start,
    )
    _write_json(entry.output_dir / MANIFEST_FILE, manifest)
    shutil.rmtree(entry.output_dir / PARTIAL_DIR, ignore_errors=True)
    _log_summary(entry, manifest, log_dir)
    return "written", []


def run_pretokenize(
    config: PretokenizeConfig, keys: list[str], *, force: bool = False, max_workers: int = 0
) -> PretokenizeSummary:
    """Convert the selected registry entries and log each finished output's lineage.

    Lineage goes to MLflow as the registry's `mlflow` section says (off when it
    sets no `enabled: true`), once per written or skipped output.

    Args:
        config: The loaded pretokenize registry.
        keys: Entry keys to convert.
        force: Convert entries again from scratch, and replace their MLflow runs.
        max_workers: Parallel input shards per entry; 0 sizes by cores, shards and memory.

    Returns:
        The written, skipped and failed entries.

    Raises:
        KeyError: If a key is not in the registry.
    """
    from slm4ie.data.pretokenize.tracking import DEFAULT_EXPERIMENT, log_pretokenized

    unknown = sorted(set(keys) - set(config.entries))
    if unknown:
        raise KeyError(f"Unknown pretokenize entries {unknown}; declared: {sorted(config.entries)}")
    summary = PretokenizeSummary()
    for key in keys:
        entry = config.entries[key]
        outcome, failed_parts = convert_entry(entry, force=force, max_workers=max_workers)
        if outcome == "failed":
            summary.failed[key] = failed_parts
            continue
        getattr(summary, outcome).append(key)
        log_pretokenized(
            entry,
            enabled=bool(config.mlflow.get("enabled", False)),
            experiment=config.mlflow.get("experiment") or DEFAULT_EXPERIMENT,
            tracking_uri=config.mlflow.get("tracking_uri"),
            force=force,
        )
    return summary
