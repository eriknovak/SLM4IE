"""Helpers for the datatrove dedup blocks used by the curation pipeline.

The actual block instantiation lives in `pipeline.py`. This module
exposes the small bits the pipeline builder needs: the
`ExactDedupConfig` factory and an `ExactDedupConfig.content_getter`
implementation that hashes `doc.text`.
"""

from typing import Literal

from datatrove.data import Document
from datatrove.pipeline.dedup import ExactDedupConfig
from datatrove.utils.hashing import HashConfig


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


def default_exact_config() -> ExactDedupConfig:
    """Build the default `ExactDedupConfig` used by the curation pipeline.

    Returns:
        An `ExactDedupConfig` whose `content_getter` is `doc_text` and
        whose `hash_config` and `document_priority` keep the upstream
        datatrove defaults.
    """
    return ExactDedupConfig(content_getter=doc_text)


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
