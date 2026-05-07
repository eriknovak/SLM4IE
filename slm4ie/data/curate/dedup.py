"""Helpers for the datatrove dedup blocks used by the curation pipeline.

The actual block instantiation lives in `pipeline.py`. This module
exposes the small bits the pipeline builder needs: the
`ExactDedupConfig` factory and an `ExactDedupConfig.content_getter`
implementation that hashes `doc.text`.
"""

from datatrove.data import Document
from datatrove.pipeline.dedup import ExactDedupConfig


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
