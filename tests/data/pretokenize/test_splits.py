"""Tests for slm4ie/data/pretokenize/splits.py (the document split)."""

import pytest

from slm4ie.data.pretokenize.splits import TRAIN, VALIDATION, is_validation, resolve_validation_share, split_of


def test_split_is_deterministic_for_a_seed():
    """The same id and seed always give the same split."""
    assert [split_of(f"doc-{i}", 13, 0.1) for i in range(200)] == [split_of(f"doc-{i}", 13, 0.1) for i in range(200)]


def test_seed_changes_the_split():
    """Another seed puts other documents in validation."""
    first = {i for i in range(2000) if is_validation(f"doc-{i}", 13, 0.1)}
    second = {i for i in range(2000) if is_validation(f"doc-{i}", 14, 0.1)}
    assert first != second


def test_share_is_met_within_each_source():
    """Every source sends about the configured share of its documents to validation."""
    for source in ("c4", "kas", "gigafida"):
        hits = sum(is_validation(f"{source}:idx-{i:08d}", 13, 0.05) for i in range(20_000))
        assert 800 < hits < 1200


def test_split_names_are_train_and_validation():
    """A share of 1 sends every document to validation, a tiny share none."""
    assert split_of("doc", 13, 1.0) == VALIDATION
    assert split_of("doc", 13, 1e-12) == TRAIN


def test_share_is_raised_to_reach_the_minimum_documents():
    """With few documents the share rises so validation holds the minimum."""
    assert resolve_validation_share(0.001, 10_000, 1_000_000) == 0.01


def test_share_is_kept_when_it_already_gives_enough_documents():
    """With many documents the configured share stands."""
    assert resolve_validation_share(0.001, 10_000, 22_806_079) == 0.001


def test_share_never_exceeds_one():
    """A corpus smaller than the minimum sends every document to validation."""
    assert resolve_validation_share(0.001, 10_000, 500) == 1.0


def test_empty_corpus_rejected():
    """A corpus with no documents fails."""
    with pytest.raises(ValueError, match="total_docs"):
        resolve_validation_share(0.001, 10_000, 0)
