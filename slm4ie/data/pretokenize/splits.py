"""Assign each corpus document to the train or the validation split.

The split is drawn by document, never by chunk, so every chunk of a document
lands in one split. A seeded hash of the document id decides it, which is
reproducible, needs no shared state between parallel workers, and applies the
same share in every source, so the validation split mirrors the sources.
"""

from __future__ import annotations

import hashlib

# ===============================
# Constants
# ===============================

TRAIN = "train"
VALIDATION = "validation"
SPLITS = (TRAIN, VALIDATION)

# the hash prefix read as a fraction of 2**64
_HASH_BYTES = 8
_HASH_RANGE = 2 ** (8 * _HASH_BYTES)

# ===============================
# Split functions
# ===============================


def resolve_validation_share(share: float, min_docs: int, total_docs: int) -> float:
    """Return the share that gives about `min_docs` validation documents or more.

    The hash split meets the share on average, so a share raised to the
    minimum gives `min_docs` documents give or take sampling noise (about
    `sqrt(min_docs)`); the manifest records the count actually drawn.

    Args:
        share: The configured validation share.
        min_docs: The validation documents wanted at least, on average.
        total_docs: Documents in the corpus.

    Returns:
        `share`, raised to `min_docs / total_docs` when that is larger, and at
        most 1.

    Raises:
        ValueError: If `total_docs` is not positive.
    """
    if total_docs <= 0:
        raise ValueError(f"total_docs must be positive, got {total_docs}")
    return min(1.0, max(share, min_docs / total_docs))


def is_validation(doc_id: str, seed: int, share: float) -> bool:
    """Return True when a document belongs to the validation split.

    Args:
        doc_id: The document's id.
        seed: The split seed.
        share: The validation share, as `resolve_validation_share` returns it.

    Returns:
        True when the seeded hash of `doc_id`, read as a fraction, is below `share`.
    """
    digest = hashlib.sha256(f"{seed}:{doc_id}".encode()).digest()
    return int.from_bytes(digest[:_HASH_BYTES], "big") / _HASH_RANGE < share


def split_of(doc_id: str, seed: int, share: float) -> str:
    """Return the split name of a document.

    Args:
        doc_id: The document's id.
        seed: The split seed.
        share: The validation share.

    Returns:
        `VALIDATION` or `TRAIN`.
    """
    return VALIDATION if is_validation(doc_id, seed, share) else TRAIN
