"""Helpers for the datatrove dedup blocks used by the curation pipeline.

The actual block instantiation lives in `pipeline.py`. This module
exposes `doc_text` (the content getter for whole-document exact dedup),
two `ExactDedupConfig` factories (`make_exact_config` for the
parameterized form driven by `curate.yaml::exact_dedup`, and
`default_exact_config` as a zero-arg alias kept for back-compat), and
`CompactSentenceDedupSignature`, a memory-lean drop-in for datatrove's
sentence-dedup signature step.
"""

from typing import Literal

import numpy as np
from datatrove.data import Document, DocumentsPipeline
from datatrove.pipeline.dedup import ExactDedupConfig, SentenceDedupSignature
from datatrove.utils.hashing import HashConfig
from datatrove.utils.typeshelper import StatHints

#: Signatures buffered as Python tuples before packing into a numpy chunk.
SIGNATURE_FLUSH_EVERY: int = 1_000_000


def doc_text(doc: Document) -> str:
    """Return the text payload to hash for whole-document exact dedup.

    Args:
        doc: A datatrove Document.

    Returns:
        The document's text. `ExactDedupConfig` requires the getter
        to return `bytes` or `str`; we hash text as-is so two docs
        with byte-identical bodies are treated as duplicates regardless
        of their metadata.
    """
    return doc.text


def make_exact_config(
    *,
    precision: Literal[32, 64] = 64,
    hash_fc: Literal["sha1", "xxhash"] = "xxhash",
    only_dedup_in_index: bool = True,
) -> ExactDedupConfig:
    """Build an `ExactDedupConfig` parameterized for the curate pipeline.

    Wraps datatrove's `ExactDedupConfig` so the CLI can pass through the
    output-affecting knobs declared under `exact_dedup:` in `curate.yaml`.
    The content getter is always `doc_text` — exact dedup operates on
    document text, never on metadata.

    Args:
        precision: Hash width in bits. Choose `32` only for very small
            corpora (collision risk grows past ~10M docs); `64` is the
            collision-safe default up to ~10B docs.
        hash_fc: Hash function. `"xxhash"` is faster; `"sha1"` is
            cryptographically strong but unnecessary for dedup.
        only_dedup_in_index: When True, only deduplicate within the
            current run's index (datatrove default). Set to False when
            extending an existing dedup index across runs.

    Returns:
        A configured `ExactDedupConfig` with `doc_text` as the content
        getter.
    """
    return ExactDedupConfig(
        content_getter=doc_text,
        hash_config=HashConfig(precision=precision, hash_fc=hash_fc),
        only_dedup_in_index=only_dedup_in_index,
    )


def default_exact_config() -> ExactDedupConfig:
    """Build the default `ExactDedupConfig` used by the curation pipeline.

    Thin alias for `make_exact_config()` — kept for back-compat with
    test fixtures and the existing pipeline call sites. New code should
    call `make_exact_config(...)` directly so any non-default knob is
    explicit at the call site.

    Returns:
        An `ExactDedupConfig` whose `content_getter` is `doc_text` and
        whose hashing knobs match `make_exact_config`'s defaults (64-bit
        xxhash, `only_dedup_in_index=True`).
    """
    return make_exact_config()


class CompactSentenceDedupSignature(SentenceDedupSignature):
    """Sentence-dedup signature step that buffers hashes as packed numpy chunks.

    datatrove's `SentenceDedupSignature.run` keeps every `(hash, doc, sent)`
    triple of a task in one Python list until the task ends, at roughly 150
    bytes per sentence window; a multi-gigabyte shard then needs several
    gigabytes of RAM per worker. This subclass packs the triples into the
    same 14-byte structured dtype every `SIGNATURE_FLUSH_EVERY` entries, so a
    task holds about a tenth of the memory. Hashing, sorting and the files
    written are unchanged, so the find and filter steps read them as usual.
    """

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> None:
        """Hash every sentence window of the task's documents and save the signatures.

        Args:
            data: Documents of this task.
            rank: Task rank; names the signature files.
            world_size: Total number of tasks (unused, part of the step API).
        """
        dtype = np.dtype([("hash", self.config.hash_config.np_descr), ("doc", "<u4"), ("sent", "<u2")])
        chunks = []
        buffer = []
        for doc_idx, doc in enumerate(data):
            with self.stats.time_stats:
                self.stat_update(StatHints.total)
                buffer.extend(self.get_hashes(doc, doc_idx))
                if len(buffer) >= SIGNATURE_FLUSH_EVERY:
                    chunks.append(np.array(buffer, dtype=dtype))
                    buffer = []
        chunks.append(np.array(buffer, dtype=dtype))
        signatures = np.concatenate(chunks)
        del chunks, buffer
        self.save_hashes(rank, signatures)
