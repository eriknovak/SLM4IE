"""Bounded-pool helpers for per-dataset parallel execution.

Provides a small dispatcher that runs a worker function over a list of
dataset keys with a bounded process or thread pool. Per-key exceptions
are isolated so a single failure does not abort the whole run.

Sizes pools by cores, items and free memory, so a pool of memory-hungry
workers does not fill the machine's RAM. Process pools can replace workers
after a set number of items, and give each worker's tokenizer its share
of the cores.

Also provides per-task file logging and a single self-updating tqdm
bar in parallel mode, so that worker output does not interleave on the
console.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

# ===============================
# Logging
# ===============================

logger = logging.getLogger(__name__)

# ===============================
# Defaults
# ===============================

#: Env var set by `run_parallel` while a parallel pool is active. Worker
#: code can call `workers_quiet()` to detect it and silence per-task
#: progress bars that would otherwise interleave on the console.
_PARALLEL_ENV_VAR = "SLM4IE_DATA_PARALLEL"

#: Shared formatter for per-dataset log files. Matches the console
#: format used by `configure_script_logging` so log files look familiar.
_FILE_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

#: Third-party loggers whose INFO output is pure chatter for our scripts.
#: Bumped to WARNING so neither the console nor the per-dataset log files
#: are flooded by shard progress, "Generating split…" lines, retry warnings,
#: etc. Per-dataset detail we care about comes from `slm4ie.*` loggers.
_NOISY_THIRD_PARTY_LOGGERS = (
    "datasets",
    "huggingface_hub",
    "urllib3",
    "filelock",
    "fsspec",
)

#: Share of the machine's available RAM a worker pool may fill.
MEMORY_BUDGET_SHARE = 0.8

#: Memory per worker (GB) assumed when a step declares none.
DEFAULT_MEMORY_PER_WORKER_GB = 8.0

#: Keys a step's `workers:` mapping may hold.
_WORKER_SETTINGS_KEYS = frozenset({"memory_per_worker_gb", "max_items_per_worker"})

#: Linux file holding the `MemAvailable` reading.
_MEMINFO_PATH = Path("/proc/meminfo")

_KIB_PER_GB = 1024**2

#: Env var that sets the HuggingFace fast tokenizers' thread count.
_TOKENIZER_THREADS_ENV_VAR = "RAYON_NUM_THREADS"

# ===============================
# Worker settings
# ===============================


@dataclass(frozen=True)
class WorkerSettings:
    """What a step declares about one worker of its pool.

    Attributes:
        memory_per_worker_gb: Peak memory of one worker in GB, or None when the
            step declares none and `DEFAULT_MEMORY_PER_WORKER_GB` applies.
        max_items_per_worker: Items one worker handles before it is replaced,
            or None to keep workers for the whole run.
    """

    memory_per_worker_gb: float | None = None
    max_items_per_worker: int | None = None


def parse_worker_settings(raw: Any, where: str) -> WorkerSettings:
    """Read and validate the `workers:` mapping of a step's YAML registry.

    Args:
        raw: The mapping as loaded from YAML; None gives the defaults.
        where: The config location, named in errors.

    Returns:
        The validated settings.

    Raises:
        ValueError: If the section is not a mapping, holds an unknown key,
            a memory that is not a positive number, or an item limit that is
            not a positive integer.
    """
    # read the section
    if raw is None:
        return WorkerSettings()
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: workers must be a mapping, got {raw!r}")
    unknown = sorted(set(raw) - _WORKER_SETTINGS_KEYS)
    if unknown:
        raise ValueError(f"{where}: workers has unknown keys {unknown}; allowed: {sorted(_WORKER_SETTINGS_KEYS)}")

    # validate the memory per worker
    memory = raw.get("memory_per_worker_gb")
    if memory is not None and (isinstance(memory, bool) or not isinstance(memory, (int, float)) or memory <= 0):
        raise ValueError(f"{where}: workers.memory_per_worker_gb must be a positive number, got {memory!r}")

    # validate the item limit
    max_items = raw.get("max_items_per_worker")
    if max_items is not None and (isinstance(max_items, bool) or not isinstance(max_items, int) or max_items <= 0):
        raise ValueError(f"{where}: workers.max_items_per_worker must be a positive integer, got {max_items!r}")

    return WorkerSettings(
        memory_per_worker_gb=None if memory is None else float(memory),
        max_items_per_worker=max_items,
    )


# ===============================
# Worker counts
# ===============================


def resolve_workers(requested: int, n_items: int, default: int) -> int:
    """Resolve effective worker count from a CLI flag.

    Args:
        requested: Value passed by the user (0 means "auto").
        n_items: Number of items to process; the result is capped here.
        default: Fallback when `requested` is falsy.

    Returns:
        Worker count in `[1, n_items]`.
    """
    effective = default if not requested else requested
    return max(1, min(effective, max(1, n_items)))


def cpu_default(n_items: int) -> int:
    """Return a CPU-bound default worker count.

    Args:
        n_items: Number of items to process.

    Returns:
        `min(cpu_count() // 2, n_items)`, never below 1.
    """
    cores = os.cpu_count() or 2
    return max(1, min(cores // 2, max(1, n_items)))


@dataclass(frozen=True)
class WorkerCount:
    """A pool size and the limit that set it.

    Attributes:
        workers: Number of pool workers, at least 1.
        limit: The binding limit: `"cores"`, `"items"`, `"memory"` or `"cap"`.
        memory_per_worker_gb: Memory per worker the count was sized for.
        is_memory_assumed: True when the step declared no memory per worker
            and `DEFAULT_MEMORY_PER_WORKER_GB` was used.
    """

    workers: int
    limit: str
    memory_per_worker_gb: float
    is_memory_assumed: bool


def size_workers(
    cores: int,
    n_items: int,
    available_gb: float,
    memory_per_worker_gb: float | None,
    cap: int | None = None,
) -> WorkerCount:
    """Size a pool by cores, items and available memory.

    The count is the minimum of half the cores, the item count, the workers
    that fit in `MEMORY_BUDGET_SHARE` of the available memory, and `cap`
    when given; never below 1. On a tie the limit named first in that order
    is reported.

    Args:
        cores: CPU cores the process may use.
        n_items: Number of items the pool will process.
        available_gb: Memory available on the machine, in GB.
        memory_per_worker_gb: Peak memory of one worker, in GB, as the step
            declares it; None assumes `DEFAULT_MEMORY_PER_WORKER_GB`.
        cap: Optional step-specific upper bound on the count.

    Returns:
        The worker count, its binding limit and the memory it assumed.
    """
    # fall back to the default memory per worker
    is_memory_assumed = memory_per_worker_gb is None
    per_worker = DEFAULT_MEMORY_PER_WORKER_GB if memory_per_worker_gb is None else memory_per_worker_gb

    # take the tightest limit
    limits = [
        ("cores", cores // 2),
        ("items", n_items),
        ("memory", int(MEMORY_BUDGET_SHARE * available_gb // per_worker)),
    ]
    if cap is not None:
        limits.append(("cap", cap))
    limit, workers = min(limits, key=lambda pair: pair[1])

    return WorkerCount(
        workers=max(1, workers),
        limit=limit,
        memory_per_worker_gb=per_worker,
        is_memory_assumed=is_memory_assumed,
    )


def available_memory_gb(meminfo: Path = _MEMINFO_PATH) -> float:
    """Read the machine's available memory from the kernel.

    Args:
        meminfo: Path to the Linux `meminfo` file.

    Returns:
        The `MemAvailable` reading in GB.

    Raises:
        OSError: If `meminfo` cannot be read.
        ValueError: If `meminfo` has no `MemAvailable` line.
    """
    for line in meminfo.read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / _KIB_PER_GB
    raise ValueError(f"no 'MemAvailable' line in {meminfo}")


def choose_workers(
    requested: int,
    n_items: int,
    memory_per_worker_gb: float | None,
    *,
    label: str,
    cap: int | None = None,
) -> int:
    """Choose a pool size from the CLI flag or the machine, and log it.

    An explicit `requested` count wins, capped only by the item count, as
    `resolve_workers` does. Otherwise the count comes from `size_workers`
    over this process's cores and the machine's available memory.

    Args:
        requested: Value passed by the user (0 means "auto").
        n_items: Number of items the pool will process.
        memory_per_worker_gb: Peak memory of one worker, in GB, from the
            step's config; None assumes `DEFAULT_MEMORY_PER_WORKER_GB`.
        label: Step name used in the log lines.
        cap: Optional step-specific upper bound on the automatic count.

    Returns:
        The worker count, at least 1.
    """
    # honour an explicit worker count
    if requested:
        workers = resolve_workers(requested, n_items, requested)
        logger.info("%s: %d workers, set by --max-workers", label, workers)
        return workers

    # size the pool by the machine
    cores = os.process_cpu_count() or 2
    available_gb = available_memory_gb()
    count = size_workers(cores, n_items, available_gb, memory_per_worker_gb, cap)

    # log the count and the limit that set it
    if count.is_memory_assumed:
        logger.info(
            "%s: no memory per worker declared, assuming %.1f GB",
            label,
            DEFAULT_MEMORY_PER_WORKER_GB,
        )
    logger.info(
        "%s: %d workers, limited by %s (cores=%d items=%d available=%.1f GB per worker=%.1f GB)",
        label,
        count.workers,
        count.limit,
        cores,
        n_items,
        available_gb,
        count.memory_per_worker_gb,
    )
    return count.workers


def io_default(n_items: int, cap: int = 4) -> int:
    """Return an I/O-bound default worker count.

    Args:
        n_items: Number of items to process.
        cap: Hard upper bound to stay polite to remote servers.

    Returns:
        `min(cap, n_items)`, never below 1.
    """
    return max(1, min(cap, max(1, n_items)))


def workers_quiet() -> bool:
    """Return True when the caller is running inside a parallel pool.

    Worker code uses this to suppress noisy progress bars and other
    console output that would interleave when many workers run at once.
    Set by `run_parallel` for the lifetime of the executor.

    Returns:
        True when the parallel-mode env var is set, False otherwise.
    """
    return os.environ.get(_PARALLEL_ENV_VAR) == "1"


# ===============================
# Logging setup
# ===============================


def configure_script_logging(
    parallel: bool,
    console_level: int | None = None,
) -> None:
    """Configure root logging for a data script.

    Always installs the project's standard console format at INFO level,
    quiets noisy third-party loggers (`datasets`, `huggingface_hub`,
    `urllib3`, `filelock`, `fsspec`) to WARNING, and disables HuggingFace
    progress bars so they do not interleave with our own output.

    When `parallel` is True, the console handler is bumped to WARNING
    so routine worker output does not spam stderr (per-dataset INFO/DEBUG
    still reaches the per-key log files set up by `run_parallel`).

    When `console_level` is given, the console handler's level is set
    to that value after the parallel adjustment, so callers can force a
    quiet console even in serial mode while letting INFO records reach
    the per-dataset log files.

    Args:
        parallel: True when `--max-workers` will be greater than 1.
        console_level: Optional explicit level for the console
            StreamHandler. When None, defaults to WARNING in parallel
            mode and INFO in serial mode.
    """
    # install the console format and quiet third-party loggers
    logging.basicConfig(
        level=logging.INFO,
        format=_FILE_LOG_FORMAT,
    )
    for name in _NOISY_THIRD_PARTY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    # disable the HuggingFace progress bars
    try:
        from datasets.utils.logging import disable_progress_bar

        disable_progress_bar()
    except ImportError:
        pass
    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except ImportError:
        pass

    # set the console handler level
    if parallel or console_level is not None:
        effective_level = console_level if console_level is not None else logging.WARNING
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.setLevel(effective_level)


# ===============================
# Pool dispatch
# ===============================


class _ThreadFilter(logging.Filter):
    """Pass only records emitted by the configured thread.

    Used in thread-pool mode so each per-dataset file handler captures
    only its own task's records.
    """

    def __init__(self, ident: int) -> None:
        """Initialize with the thread identifier to match.

        Args:
            ident: Result of `threading.get_ident()` from the worker.
        """
        super().__init__()
        self._ident = ident

    def filter(self, record: logging.LogRecord) -> bool:
        """Return True when `record.thread` matches the bound ident."""
        return record.thread == self._ident


def _limit_tokenizer_threads(threads: int) -> None:
    """Cap the HuggingFace fast tokenizers' threads in a pool worker.

    Top-level so it pickles cleanly as a `ProcessPoolExecutor` initializer,
    which runs before the worker builds its first tokenizer.

    Args:
        threads: Thread count the worker's tokenizers may use.
    """
    os.environ[_TOKENIZER_THREADS_ENV_VAR] = str(threads)


def _worker_with_log(
    func: Callable[..., Any],
    key: str,
    kwargs: dict[str, Any],
    log_path: Path | None,
    use_thread_filter: bool,
) -> Any:
    """Run `func(key, **kwargs)` with an optional per-key file handler.

    Top-level so it pickles cleanly for `ProcessPoolExecutor`.

    Args:
        func: Worker callable. Must itself be top-level for pickling.
        key: Dataset key (used for log filename + as the first arg).
        kwargs: Keyword args forwarded to `func`.
        log_path: When set, attach a `FileHandler` writing to this path
            for the duration of the call. Parents are created.
        use_thread_filter: When True, the handler also gets a
            `_ThreadFilter` bound to the calling thread's ident so it
            only captures records from this task (needed for thread
            pool, harmless for process pool).

    Returns:
        Whatever `func` returns.
    """
    # attach the per-key file handler
    handler: logging.FileHandler | None = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(str(log_path), encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(_FILE_LOG_FORMAT))
        if use_thread_filter:
            handler.addFilter(_ThreadFilter(threading.get_ident()))
        root = logging.getLogger()
        root.addHandler(handler)
        # let INFO records through to the handlers
        if root.level > logging.INFO or root.level == logging.NOTSET:
            root.setLevel(logging.INFO)

    # run the worker and detach the handler
    try:
        return func(key, **kwargs)
    finally:
        if handler is not None:
            logging.getLogger().removeHandler(handler)
            handler.close()


def run_parallel(
    func: Callable[..., Any],
    keys: Iterable[str],
    *,
    max_workers: int,
    desc: str,
    pool: str = "process",
    kwargs_for: Callable[[str], dict[str, Any]] = lambda _k: {},
    log_dir: Path | None = None,
    max_items_per_worker: int | None = None,
) -> tuple[dict[str, Any], list[tuple[str, BaseException]]]:
    """Run `func(key, **kwargs_for(key))` over `keys` with bounded concurrency.

    When `max_workers <= 1` or there is at most one key, runs serially
    without an executor so tracebacks stay unwrapped for debugging. When
    parallel, sets `SLM4IE_DATA_PARALLEL=1` for the duration of the
    pool so workers can call `workers_quiet()` to suppress their own
    progress bars, and drives a single self-updating tqdm bar on stderr
    with a postfix showing running/done/skipped/failed/waiting counts.
    The start timestamp is logged once before the bar opens; elapsed
    time is shown inside the bar itself.

    In a process pool, each worker's HuggingFace fast tokenizers get
    `max(1, cores // max_workers)` threads, so the workers together do not
    oversubscribe the cores.

    Args:
        func: Worker callable. Must be picklable when `pool="process"`.
        keys: Dataset keys (or any string identifiers) to process.
        max_workers: Effective worker count (use `resolve_workers`).
        desc: Short label used as the tqdm bar description.
        pool: Either `"process"` (CPU-bound) or `"thread"` (I/O-bound).
        kwargs_for: Builds per-key keyword args. Resolve config / paths
            here in the parent so workers stay pickle-clean and config
            files are not re-read N times.
        log_dir: When set, every task's `logger` output is captured into
            `log_dir / f"{key}.log"` for the duration of the call.
            Applies to both serial and parallel paths.
        max_items_per_worker: When set, each process-pool worker is
            replaced by a fresh one after this many items, so memory does
            not grow across items. Fresh workers are spawned rather than
            forked, so `func` and its arguments must be importable. None
            keeps each worker for the whole run.

    Returns:
        `(results_by_key, failures)`. `results_by_key` maps each key
        to whatever `func` returned. `failures` is a list of
        `(key, exception)` pairs for keys that raised; callers decide
        the exit code from `len(failures)`.

    Raises:
        ValueError: If `pool` is not `"process"` or `"thread"`, or if
            `max_items_per_worker` is set with a thread pool.
    """
    # reject worker replacement in a thread pool
    if pool == "thread" and max_items_per_worker is not None:
        raise ValueError(f"max_items_per_worker={max_items_per_worker} needs a process pool, not a thread pool")

    # prepare the results and the log folder
    keys_list = list(keys)
    results: dict[str, Any] = {}
    failures: list[tuple[str, BaseException]] = []

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)

    def _log_path_for(key: str) -> Path | None:
        return None if log_dir is None else log_dir / f"{key}.log"

    # run serially so tracebacks stay unwrapped
    serial = max_workers <= 1 or len(keys_list) <= 1

    if serial:
        for key in tqdm(keys_list, desc=desc):
            try:
                results[key] = _worker_with_log(
                    func,
                    key,
                    kwargs_for(key),
                    _log_path_for(key),
                    use_thread_filter=False,
                )
            except Exception as exc:
                logger.exception("Failed processing '%s'", key)
                failures.append((key, exc))
        return results, failures

    # pick the pool kind
    if pool == "process":
        cores = os.process_cpu_count() or 2
        tokenizer_threads = max(1, cores // max_workers)

        def _make_executor() -> ProcessPoolExecutor:
            return ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_limit_tokenizer_threads,
                initargs=(tokenizer_threads,),
                max_tasks_per_child=max_items_per_worker,
            )

        use_thread_filter = False
    elif pool == "thread":

        def _make_executor() -> ThreadPoolExecutor:
            return ThreadPoolExecutor(max_workers=max_workers)

        use_thread_filter = True
    else:
        raise ValueError(f"Unknown pool kind: {pool!r}")

    # run the pool with a single progress bar
    total = len(keys_list)
    done = 0
    skipped = 0
    failed = 0

    prev_env = os.environ.get(_PARALLEL_ENV_VAR)
    os.environ[_PARALLEL_ENV_VAR] = "1"
    ticker_stop = threading.Event()
    ticker_thread: threading.Thread | None = None
    try:
        with (
            _make_executor() as executor,
            tqdm(
                total=total,
                desc=desc,
                unit="task",
                dynamic_ncols=True,
            ) as bar,
        ):
            future_to_key = {
                executor.submit(
                    _worker_with_log,
                    func,
                    key,
                    kwargs_for(key),
                    _log_path_for(key),
                    use_thread_filter,
                ): key
                for key in keys_list
            }

            def _refresh_postfix() -> None:
                completed = done + skipped + failed
                running = max(0, min(max_workers, total - completed))
                waiting = max(0, total - completed - running)
                bar.set_postfix(
                    running=running,
                    done=done,
                    skipped=skipped,
                    failed=failed,
                    waiting=waiting,
                    refresh=False,
                )

            def _tick() -> None:
                while not ticker_stop.wait(1.0):
                    bar.refresh()

            _refresh_postfix()
            bar.refresh()
            ticker_thread = threading.Thread(target=_tick, daemon=True)
            ticker_thread.start()
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    value = future.result()
                except Exception as exc:
                    logger.exception("Failed processing '%s'", key)
                    failures.append((key, exc))
                    failed += 1
                else:
                    results[key] = value
                    if value is None:
                        skipped += 1
                    else:
                        done += 1
                _refresh_postfix()
                bar.update(1)
    finally:
        ticker_stop.set()
        if ticker_thread is not None:
            ticker_thread.join(timeout=2.0)
        if prev_env is None:
            os.environ.pop(_PARALLEL_ENV_VAR, None)
        else:
            os.environ[_PARALLEL_ENV_VAR] = prev_env

    # log the totals
    completed = done + skipped + failed
    logger.info(
        "%s: finished %d/%d (done=%d skipped=%d failed=%d)",
        desc,
        completed,
        total,
        done,
        skipped,
        failed,
    )

    return results, failures
