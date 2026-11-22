"""Tests for `slm4ie.utils.parallel`."""

import os

import pytest

from slm4ie.utils.parallel import (
    WorkerSettings,
    choose_workers,
    parse_worker_settings,
    run_parallel,
    size_workers,
)


def _worker_pid(_key: str) -> int:
    """Return the id of the process that ran the item."""
    return os.getpid()


def _tokenizer_threads(_key: str) -> str | None:
    """Return the tokenizer thread count the worker sees."""
    return os.environ.get("RAYON_NUM_THREADS")


@pytest.mark.parametrize(
    ("cores", "n_items", "available_gb", "cap", "expected"),
    [
        (8, 100, 1000.0, None, (4, "cores")),
        (40, 3, 1000.0, None, (3, "items")),
        (40, 100, 50.0, None, (10, "memory")),
        (40, 100, 1000.0, 2, (2, "cap")),
    ],
)
def test_tightest_limit_sets_the_count(
    cores: int, n_items: int, available_gb: float, cap: int | None, expected: tuple[int, str]
) -> None:
    """Half the cores, the items, 80% of memory or the cap: the smallest wins."""
    count = size_workers(cores, n_items, available_gb, memory_per_worker_gb=4.0, cap=cap)
    assert (count.workers, count.limit) == expected


def test_undeclared_memory_assumes_eight_gb() -> None:
    """No memory per worker sizes the pool for 8 GB and flags the assumption."""
    count = size_workers(cores=40, n_items=100, available_gb=80.0, memory_per_worker_gb=None)
    assert (count.workers, count.limit) == (8, "memory")
    assert count.is_memory_assumed


def test_too_little_memory_still_gives_one_worker() -> None:
    """A worker larger than the memory budget still runs alone."""
    count = size_workers(cores=40, n_items=100, available_gb=4.0, memory_per_worker_gb=8.0)
    assert (count.workers, count.limit) == (1, "memory")


def test_explicit_worker_count_overrides_the_default() -> None:
    """A `--max-workers` value is used unchanged, whatever the memory."""
    assert choose_workers(30, n_items=100, memory_per_worker_gb=1000.0, label="test") == 30


def test_workers_are_replaced_after_the_item_limit() -> None:
    """With one item per worker, no worker process handles two items."""
    keys = [f"item-{i}" for i in range(4)]
    results, failures = run_parallel(_worker_pid, keys, max_workers=2, desc="test", max_items_per_worker=1)
    assert not failures
    assert len(set(results.values())) == len(keys)


def test_workers_see_their_share_of_tokenizer_threads() -> None:
    """Each worker's tokenizer may use the cores divided by the workers."""
    results, failures = run_parallel(_tokenizer_threads, ["a", "b"], max_workers=2, desc="test")
    assert not failures
    expected = str(max(1, (os.process_cpu_count() or 2) // 2))
    assert set(results.values()) == {expected}


def test_thread_pool_rejects_worker_replacement() -> None:
    """Worker replacement in a thread pool raises, even for a serial run."""
    with pytest.raises(ValueError, match="max_items_per_worker"):
        run_parallel(_worker_pid, ["a"], max_workers=1, desc="test", pool="thread", max_items_per_worker=1)


def test_absent_workers_section_gives_defaults() -> None:
    """A config without a `workers:` mapping declares nothing."""
    assert parse_worker_settings(None, "test") == WorkerSettings()


def test_workers_section_values_are_read() -> None:
    """Declared memory and item limit are returned, memory as a float."""
    settings = parse_worker_settings({"memory_per_worker_gb": 3, "max_items_per_worker": 50}, "test")
    assert settings == WorkerSettings(memory_per_worker_gb=3.0, max_items_per_worker=50)


@pytest.mark.parametrize("memory", [0, -2.5])
def test_non_positive_memory_is_rejected(memory: float) -> None:
    """Zero or negative memory per worker raises, naming the key."""
    with pytest.raises(ValueError, match="memory_per_worker_gb"):
        parse_worker_settings({"memory_per_worker_gb": memory}, "test")


@pytest.mark.parametrize("max_items", [2.5, 0, -1])
def test_bad_item_limit_is_rejected(max_items: float) -> None:
    """A non-integer or non-positive item limit raises, naming the key."""
    with pytest.raises(ValueError, match="max_items_per_worker"):
        parse_worker_settings({"max_items_per_worker": max_items}, "test")


def test_unknown_workers_key_is_rejected() -> None:
    """A key the section does not define raises, naming the key."""
    with pytest.raises(ValueError, match="threads"):
        parse_worker_settings({"threads": 4}, "test")
