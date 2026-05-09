"""Bounded-pool helpers for per-dataset parallel execution.

Provides a small dispatcher that runs a worker function over a list of
dataset keys with a bounded process or thread pool. Per-key exceptions
are isolated so a single failure does not abort the whole run.

Also provides per-task file logging and a periodic console summary
when running in parallel mode, so that worker output does not interleave
on the console.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

logger = logging.getLogger(__name__)

#: Env var set by `run_parallel` while a parallel pool is active. Worker
#: code can call `workers_quiet()` to detect it and silence per-task
#: progress bars that would otherwise interleave on the console.
_PARALLEL_ENV_VAR = "SLM4IE_DATA_PARALLEL"

#: Shared formatter for per-dataset log files. Matches the console
#: format used by `configure_script_logging` so log files look familiar.
_FILE_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def resolve_workers(requested: int, n_items: int, default: int) -> int:
    """Resolve effective worker count from a CLI flag.

    Args:
        requested: Value passed by the user (0 means "auto").
        n_items: Number of items to process; the result is capped here.
        default: Fallback when ``requested`` is falsy.

    Returns:
        Worker count in ``[1, n_items]``.
    """
    effective = default if not requested else requested
    return max(1, min(effective, max(1, n_items)))


def cpu_default(n_items: int) -> int:
    """Return a CPU-bound default worker count.

    Args:
        n_items: Number of items to process.

    Returns:
        ``min(cpu_count() // 2, n_items)``, never below 1.
    """
    cores = os.cpu_count() or 2
    return max(1, min(cores // 2, max(1, n_items)))


def io_default(n_items: int, cap: int = 4) -> int:
    """Return an I/O-bound default worker count.

    Args:
        n_items: Number of items to process.
        cap: Hard upper bound to stay polite to remote servers.

    Returns:
        ``min(cap, n_items)``, never below 1.
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


def configure_script_logging(parallel: bool) -> None:
    """Configure root logging for a data script.

    Always installs the project's standard console format at INFO level.
    When ``parallel`` is True, the console handler is bumped to WARNING
    so routine worker output does not spam stderr (per-dataset INFO/DEBUG
    still reaches the per-key log files set up by `run_parallel`).

    Args:
        parallel: True when ``--max-workers`` will be greater than 1.
    """
    logging.basicConfig(
        level=logging.INFO,
        format=_FILE_LOG_FORMAT,
    )
    if parallel:
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(
                handler, logging.FileHandler
            ):
                handler.setLevel(logging.WARNING)


@dataclass
class _Counts:
    """Mutable counters shared between main thread and summary thread."""

    total: int
    done: int = 0
    skipped: int = 0
    failed: int = 0


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
        """Return True when ``record.thread`` matches the bound ident."""
        return record.thread == self._ident


def _worker_with_log(
    func: Callable[..., Any],
    key: str,
    kwargs: Dict[str, Any],
    log_path: Optional[Path],
    use_thread_filter: bool,
) -> Any:
    """Run ``func(key, **kwargs)`` with an optional per-key file handler.

    Top-level so it pickles cleanly for ``ProcessPoolExecutor``.

    Args:
        func: Worker callable. Must itself be top-level for pickling.
        key: Dataset key (used for log filename + as the first arg).
        kwargs: Keyword args forwarded to ``func``.
        log_path: When set, attach a `FileHandler` writing to this path
            for the duration of the call. Parents are created.
        use_thread_filter: When True, the handler also gets a
            `_ThreadFilter` bound to the calling thread's ident so it
            only captures records from this task (needed for thread
            pool, harmless for process pool).

    Returns:
        Whatever ``func`` returns.
    """
    handler: Optional[logging.FileHandler] = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(str(log_path), encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(_FILE_LOG_FORMAT))
        if use_thread_filter:
            handler.addFilter(_ThreadFilter(threading.get_ident()))
        root = logging.getLogger()
        root.addHandler(handler)
        # Ensure the root threshold lets INFO records through to handlers.
        if root.level > logging.INFO or root.level == logging.NOTSET:
            root.setLevel(logging.INFO)

    try:
        return func(key, **kwargs)
    finally:
        if handler is not None:
            logging.getLogger().removeHandler(handler)
            handler.close()


def _summary_loop(
    desc: str,
    counts: _Counts,
    max_workers: int,
    interval: float,
    lock: threading.Lock,
    stop: threading.Event,
) -> None:
    """Print a periodic summary line until ``stop`` is set.

    Args:
        desc: Short label used in the log line.
        counts: Shared counters updated by the main thread.
        max_workers: Pool size; bounds the reported "running" value.
        interval: Seconds between summary lines.
        lock: Guards reads of ``counts``.
        stop: Event the main thread sets when the run is finishing.
    """
    while not stop.wait(interval):
        with lock:
            completed = counts.done + counts.skipped + counts.failed
            running = max(0, min(max_workers, counts.total - completed))
            waiting = max(0, counts.total - completed - running)
            line = (
                f"{desc}: running={running} done={counts.done} "
                f"skipped={counts.skipped} failed={counts.failed} "
                f"waiting={waiting} ({completed}/{counts.total})"
            )
        print(line, file=sys.stderr, flush=True)


def run_parallel(
    func: Callable[..., Any],
    keys: Iterable[str],
    *,
    max_workers: int,
    desc: str,
    pool: str = "process",
    kwargs_for: Callable[[str], Dict[str, Any]] = lambda _k: {},
    log_dir: Optional[Path] = None,
    summary_interval: float = 30.0,
) -> Tuple[Dict[str, Any], List[Tuple[str, BaseException]]]:
    """Run ``func(key, **kwargs_for(key))`` over ``keys`` with bounded concurrency.

    When ``max_workers <= 1`` or there is at most one key, runs serially
    without an executor so tracebacks stay unwrapped for debugging. When
    parallel, sets ``SLM4IE_DATA_PARALLEL=1`` for the duration of the
    pool so workers can call `workers_quiet()` to suppress their own
    progress bars, and emits a periodic plain-text summary line to
    stderr in place of an outer tqdm bar.

    Args:
        func: Worker callable. Must be picklable when ``pool="process"``.
        keys: Dataset keys (or any string identifiers) to process.
        max_workers: Effective worker count (use ``resolve_workers``).
        desc: Short label used in the summary line / outer tqdm bar.
        pool: Either ``"process"`` (CPU-bound) or ``"thread"`` (I/O-bound).
        kwargs_for: Builds per-key keyword args. Resolve config / paths
            here in the parent so workers stay pickle-clean and config
            files are not re-read N times.
        log_dir: When set, every task's `logger` output is captured into
            ``log_dir / f"{key}.log"`` for the duration of the call.
            Applies to both serial and parallel paths.
        summary_interval: Seconds between periodic console summary lines
            in parallel mode. Ignored when serial.

    Returns:
        ``(results_by_key, failures)``. ``results_by_key`` maps each key
        to whatever ``func`` returned. ``failures`` is a list of
        ``(key, exception)`` pairs for keys that raised; callers decide
        the exit code from ``len(failures)``.

    Raises:
        ValueError: If ``pool`` is not ``"process"`` or ``"thread"``.
    """
    keys_list = list(keys)
    results: Dict[str, Any] = {}
    failures: List[Tuple[str, BaseException]] = []

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)

    def _log_path_for(key: str) -> Optional[Path]:
        return None if log_dir is None else log_dir / f"{key}.log"

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

    if pool == "process":
        executor_cls = ProcessPoolExecutor
        use_thread_filter = False
    elif pool == "thread":
        executor_cls = ThreadPoolExecutor
        use_thread_filter = True
    else:
        raise ValueError(f"Unknown pool kind: {pool!r}")

    counts = _Counts(total=len(keys_list))
    lock = threading.Lock()
    stop = threading.Event()
    summary_thread = threading.Thread(
        target=_summary_loop,
        args=(desc, counts, max_workers, summary_interval, lock, stop),
        daemon=True,
    )

    prev_env = os.environ.get(_PARALLEL_ENV_VAR)
    os.environ[_PARALLEL_ENV_VAR] = "1"
    summary_thread.start()
    try:
        with executor_cls(max_workers=max_workers) as executor:
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
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    value = future.result()
                except Exception as exc:
                    logger.exception("Failed processing '%s'", key)
                    failures.append((key, exc))
                    with lock:
                        counts.failed += 1
                else:
                    results[key] = value
                    with lock:
                        if value is None:
                            counts.skipped += 1
                        else:
                            counts.done += 1
    finally:
        stop.set()
        summary_thread.join(timeout=summary_interval + 1.0)
        if prev_env is None:
            os.environ.pop(_PARALLEL_ENV_VAR, None)
        else:
            os.environ[_PARALLEL_ENV_VAR] = prev_env

    completed = counts.done + counts.skipped + counts.failed
    logger.info(
        "%s: finished %d/%d (done=%d skipped=%d failed=%d)",
        desc,
        completed,
        counts.total,
        counts.done,
        counts.skipped,
        counts.failed,
    )

    return results, failures
