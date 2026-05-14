"""Dataset download orchestration and per-source dispatch."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from slm4ie.data.catalog import ConfigError, DatasetConfig, load_config
from slm4ie.data.parallel import io_default, resolve_workers, run_parallel

logger = logging.getLogger(__name__)


@dataclass
class DownloaderResult:
    """Outcome of a per-source download call.

    Attributes:
        completed: Sub-unit identifiers that finished successfully.
        failed: Sub-unit identifiers paired with the exception that
            ended their attempt.
    """

    completed: List[str] = field(default_factory=list)
    failed: List[Tuple[str, BaseException]] = field(default_factory=list)


class DatasetDownloadError(Exception):
    """Raised when one or more sub-units of a dataset failed.

    Attributes:
        dataset_key: The dataset's registry key.
        failed: Sub-unit identifier paired with the failure exception.
        n_completed: Number of sub-units that succeeded.
        n_total: Total number of sub-units attempted.
    """

    def __init__(
        self,
        dataset_key: str,
        failed: List[Tuple[str, BaseException]],
        n_completed: int,
        n_total: int,
    ):
        """Build a per-dataset aggregate failure error.

        Args:
            dataset_key: The dataset's registry key.
            failed: Sub-unit identifier paired with the exception.
            n_completed: Number of sub-units that succeeded.
            n_total: Total number of sub-units attempted.
        """
        self.dataset_key = dataset_key
        self.failed = list(failed)
        self.n_completed = n_completed
        self.n_total = n_total
        unit_summary = ", ".join(
            f"{unit}: {exc.__class__.__name__}: {str(exc).splitlines()[0]}"
            for unit, exc in self.failed
        )
        super().__init__(
            f"{dataset_key}: {len(self.failed)}/{n_total} sub-units failed"
            f" ({unit_summary})"
        )


def _augment_with_note(
    exc: BaseException, note: Optional[str]
) -> BaseException:
    """Return an exception whose message embeds the dataset note.

    Args:
        exc: Original exception raised by the source downloader.
        note: Optional informational note from the dataset config.

    Returns:
        BaseException: The original exception when `note` is empty,
            otherwise a `RuntimeError` whose message wraps the original
            class name, message, and the note.
    """
    if not note:
        return exc
    return RuntimeError(
        f"{exc.__class__.__name__}: {exc} — {note.strip()}"
    )


# Source modules are imported AFTER DownloaderResult / _augment_with_note
# are defined so they can import them from this module without triggering
# a circular import.
from slm4ie.data.sources import http as http_source  # noqa: E402
from slm4ie.data.sources import huggingface as hf_source  # noqa: E402


def _http_download(
    config: DatasetConfig, output_path: Path, force: bool
) -> DownloaderResult:
    """Dispatch to the http source downloader via module attribute lookup."""
    return http_source.download(config, output_path, force)


def _hf_download(
    config: DatasetConfig, output_path: Path, force: bool
) -> DownloaderResult:
    """Dispatch to the huggingface source downloader via module attribute lookup."""
    return hf_source.download(config, output_path, force)


_SOURCES: Dict[str, Callable[[DatasetConfig, Path, bool], DownloaderResult]] = {
    "http": _http_download,
    "huggingface": _hf_download,
}

#: Frozen set of known source names. Used for fail-fast validation
#: without depending on the patchable dispatch-time wrappers.
_SOURCE_NAMES = frozenset(_SOURCES.keys())

__all__ = [
    "ConfigError",
    "DatasetConfig",
    "DatasetDownloadError",
    "DownloaderResult",
    "download_datasets",
    "load_config",
]


def _note_suffix(note: Optional[str]) -> str:
    """Render a dataset note as a human-readable trailing suffix.

    Args:
        note: Optional informational note from the dataset config.

    Returns:
        str: An empty string when `note` is falsy, otherwise the note
        prefixed with ` — ` for inline embedding in error messages.
    """
    if not note:
        return ""
    return f" — {note.strip()}"


def _validate_selection(
    selected: Dict[str, DatasetConfig],
    base_output_dir: str,
    explicit: bool,
) -> None:
    """Raise ConfigError if any selected dataset cannot be processed.

    Always raises on unknown source. Additionally, when `explicit` is
    True, escalates `enabled: false` and missing-manual datasets into
    errors (so a user who explicitly named a key gets a hard failure
    instead of a silent skip).

    Args:
        selected: Datasets chosen for this run.
        base_output_dir: Base directory where dataset subdirs live.
        explicit: True when the user passed `dataset_keys` rather than
            relying on the enabled-by-default selection.

    Raises:
        ConfigError: When one or more selected datasets violate the
            preconditions above. All problems are aggregated into one
            error so the user sees them in a single round.
    """
    problems: List[str] = []
    for key, config in selected.items():
        if explicit and not config.enabled:
            problems.append(
                f"{key}: explicitly selected but disabled in config"
                + _note_suffix(config.note)
            )
            continue
        if config.source not in _SOURCE_NAMES:
            problems.append(
                f"{key}: unknown source '{config.source}'"
                + _note_suffix(config.note)
            )
            continue
        if (
            explicit
            and config.manual
            and not _dir_has_files(Path(base_output_dir) / config.output_dir)
        ):
            problems.append(
                f"{key}: explicitly selected but requires manual download"
                + _note_suffix(config.note)
            )

    if problems:
        raise ConfigError(problems)


def _download_one(
    key: str,
    config: "DatasetConfig",
    base_output_dir: str,
    force: bool,
) -> Optional[str]:
    """Download a single dataset and return the output path string.

    Args:
        key: Dataset key (used for log messages).
        config: Per-dataset configuration loaded from the YAML.
        base_output_dir: Base directory under which the dataset's
            `output_dir` subdirectory will live.
        force: Re-download even if the destination already has files.

    Returns:
        Optional[str]: String form of the output path on success, or
            None for skipped/disabled/manual datasets.
    """
    output_path = Path(base_output_dir) / config.output_dir

    if config.manual:
        if _dir_has_files(output_path):
            logger.info(
                "Manual dataset '%s' found at %s", key, output_path,
            )
        else:
            note = config.note or "Manual download required."
            logger.warning(
                "Dataset '%s' requires manual download: %s", key, note,
            )
        return None

    logger.info(
        "Downloading '%s' (%s) from %s",
        config.name,
        key,
        config.source,
    )

    result = _SOURCES[config.source](config, output_path, force)
    if result.failed:
        raise DatasetDownloadError(
            key,
            result.failed,
            n_completed=len(result.completed),
            n_total=len(result.completed) + len(result.failed),
        )

    return str(output_path)


def download_datasets(
    config_path: Path,
    dataset_keys: Optional[List[str]] = None,
    force: bool = False,
    output_dir_override: Optional[str] = None,
    only_benchmarks: bool = False,
    exclude_benchmarks: bool = False,
    max_workers: int = 0,
    log_dir: Optional[Path] = None,
) -> None:
    """Download datasets according to configuration.

    Args:
        config_path: Path to the YAML config file.
        dataset_keys: Specific dataset keys to download.
            If None, downloads all enabled datasets.
        force: Re-download even if output exists.
        output_dir_override: Override base output directory.
        only_benchmarks: When True, restrict the default selection to
            datasets with `benchmark: true`. Ignored when
            `dataset_keys` is provided.
        exclude_benchmarks: When True, drop benchmark datasets from the
            default selection. Ignored when `dataset_keys` is
            provided. Mutually exclusive with `only_benchmarks`.
        max_workers: Number of datasets to download in parallel.
            `0` (default) picks `min(4, n_datasets)` to stay polite
            to remote servers; `1` runs serially; `N > 1` uses that
            many threads (capped at the number of selected datasets).
        log_dir: When set, per-dataset logs are written to
            `<log_dir>/<key>.log`. The directory is created if it
            does not exist.

    Raises:
        ValueError: If any requested dataset key is unknown, or if
            both `only_benchmarks` and `exclude_benchmarks` are set.
        ConfigError: When the selected datasets fail pre-flight
            validation (unknown source, or — when explicitly selected
            — disabled or missing-manual entries).
        RuntimeError: If one or more downloads failed.
    """
    if only_benchmarks and exclude_benchmarks:
        raise ValueError(
            "only_benchmarks and exclude_benchmarks are mutually exclusive"
        )

    base_output_dir, datasets = load_config(config_path)
    if output_dir_override:
        base_output_dir = output_dir_override

    if dataset_keys:
        unknown = set(dataset_keys) - set(datasets.keys())
        if unknown:
            raise ValueError(f"Unknown dataset keys: {', '.join(unknown)}")
        selected = {k: v for k, v in datasets.items() if k in dataset_keys}
    else:
        selected = {k: v for k, v in datasets.items() if v.enabled}
        if only_benchmarks:
            selected = {k: v for k, v in selected.items() if v.benchmark}
        elif exclude_benchmarks:
            selected = {k: v for k, v in selected.items() if not v.benchmark}

    _validate_selection(
        selected, base_output_dir, explicit=bool(dataset_keys)
    )

    keys = list(selected.keys())
    workers = resolve_workers(max_workers, len(keys), io_default(len(keys)))

    def kwargs_for(key: str) -> Dict[str, Any]:
        return {
            "config": selected[key],
            "base_output_dir": base_output_dir,
            "force": force,
        }

    _, failures = run_parallel(
        _download_one,
        keys,
        max_workers=workers,
        desc="download",
        pool="thread",
        kwargs_for=kwargs_for,
        log_dir=log_dir,
    )

    if failures:
        lines = [f"Download failed for {len(failures)} dataset(s):"]
        for key, exc in failures:
            if isinstance(exc, DatasetDownloadError):
                lines.append(f"  - {exc}")
            else:
                lines.append(
                    f"  - {key}: {exc.__class__.__name__}: {exc}"
                )
        if log_dir is not None:
            lines.append(f"See {log_dir} for per-dataset logs.")
        raise RuntimeError("\n".join(lines))


def _dir_has_files(path: Path) -> bool:
    """Check if a directory exists and contains entries.

    Args:
        path: Directory path to check.

    Returns:
        bool: True if directory exists and is non-empty.
    """
    if not path.exists():
        return False
    return any(path.iterdir())
