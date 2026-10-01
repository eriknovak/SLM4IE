"""Convert raw tokenizer-evaluation downloads into JSONL records.

Materializes the datasets tagged for tokenizer / morphology evaluation. This
route bypasses the extract -> datatrove -> curate pipeline used by pretraining
corpora and reads the raw download directly, like the `raw`-source path of the
task converters.

Configuration is read from `configs/data/tokenization.yaml`, which declares
`input_dir`, `output_dir`, and the dataset keys to convert. Each key must be
declared in `configs/data/download.yaml` (so its raw subdirectory resolves) and
have a reader registered in `readers/`. Sloleks 3.1 is the seed dataset;
new lexicons are added by registering a reader and listing the key in the
config.

Each output line has this shape:

    {
        "entry_id": "...",
        "lemma":    "hiša",
        "lemma_msd": "Ncfsn",
        "forms": [
            {"form": "hiša", "msd": "Ncfsn"},
            {"form": "hiše", "msd": "Ncfsg"}
        ],
        "dataset": "sloleks",
        "task":    "TOKENIZER"
    }
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, IO, Iterable, List, Optional

from tqdm import tqdm

from slm4ie.data.download.config import DatasetConfig, load_config
from slm4ie.utils.io import open_output
from slm4ie.utils.parallel import (
    configure_script_logging,
    cpu_default,
    resolve_workers,
    run_parallel,
    workers_quiet,
)
from slm4ie.data.tokenization.config import load_tokenization_config
from slm4ie.data.tokenization.readers import READERS
from slm4ie.utils.cli import stamped_log_dir

logger = logging.getLogger(__name__)


@dataclass
class ConversionSummary:
    """Outcome of a tokenization conversion run.

    Attributes:
        records: Records written per converted dataset key.
        skipped: Keys with no registered reader or no raw directory on disk.
        failed: Keys whose conversion raised.
    """

    records: Dict[str, int] = field(default_factory=dict)
    skipped: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)


def write_records(records: Iterable[Dict[str, Any]], out_stream: IO[str]) -> int:
    """Write `records` as JSONL lines and return the count.

    Args:
        records: Records to serialize.
        out_stream: Writable text stream.

    Returns:
        Number of records written.
    """
    count = 0
    for record in records:
        out_stream.write(json.dumps(record, ensure_ascii=False))
        out_stream.write("\n")
        count += 1
    return count


def convert_dataset(
    key: str,
    raw_dir: Path,
    output_dir: Path,
    force: bool = False,
) -> Optional[int]:
    """Convert one tokenizer-eval dataset to `<output_dir>/<key>.jsonl.gz`.

    Args:
        key: Dataset key (must be registered in `READERS`).
        raw_dir: Directory holding the raw download for `key`.
        output_dir: Directory to write the JSONL output into. Created if missing.
        force: When True, overwrite an existing output file.

    Returns:
        Number of records written, or None when no reader is registered for
        `key` or the raw directory is absent. Returns 0 when the output already
        exists and `force` is False.

    Raises:
        ValueError: If the conversion ran but produced zero records, which
            signals a reader/format mismatch rather than success.
    """
    reader = READERS.get(key)
    if reader is None:
        logger.warning("No tokenizer-eval reader registered for dataset %r; skipping.", key)
        return None

    if not raw_dir.exists():
        logger.warning("Raw dir for %r does not exist: %s", key, raw_dir)
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{key}.jsonl.gz"

    if out_path.exists() and not force:
        logger.info("Skipping %r, output already exists: %s (use --force to overwrite)", key, out_path)
        return 0

    logger.info("Converting %s -> %s", raw_dir, out_path)
    progress = tqdm(reader(raw_dir), desc=key, unit="entry", disable=workers_quiet())
    with open_output(out_path) as out_stream:
        try:
            count = write_records(progress, out_stream)
        finally:
            progress.close()

    if count == 0:
        # An empty output almost always means a reader/format mismatch;
        # remove the bogus file and fail loudly instead of exiting 0.
        out_path.unlink(missing_ok=True)
        raise ValueError(
            f"Conversion of {key!r} produced 0 records from {raw_dir}; "
            "likely a reader/format mismatch. No output written."
        )

    logger.info("Wrote %d records to %s", count, out_path)
    return count


def _resolve_dataset_subdir(download_config_path: Path, key: str) -> str:
    """Return the per-dataset raw subdirectory name from `download.yaml`.

    Args:
        download_config_path: Path to download.yaml.
        key: Dataset key.

    Returns:
        The `output_dir` value declared for `key` in download.yaml, or `key`
        itself if no per-dataset override exists.
    """
    _, datasets = load_config(download_config_path)
    cfg: Optional[DatasetConfig] = datasets.get(key)
    return cfg.output_dir if cfg and cfg.output_dir else key


def convert_tokenization_datasets(
    config_path: Path,
    download_config_path: Path,
    dataset_keys: Optional[List[str]] = None,
    force: bool = False,
    max_workers: int = 0,
) -> ConversionSummary:
    """Convert the selected tokenizer-eval datasets to JSONL.

    Args:
        config_path: Path to `configs/data/tokenization.yaml`.
        download_config_path: Path to `configs/data/download.yaml`, used to
            resolve each dataset's raw subdirectory.
        dataset_keys: Keys to convert, or None for every declared dataset.
        force: Overwrite existing outputs.
        max_workers: Worker processes. 0=auto (cpu_count // 2), 1=serial.

    Returns:
        Per-key record counts plus the skipped and failed keys.
    """
    config = load_tokenization_config(config_path)

    if dataset_keys is None:
        keys = list(config.datasets)
        if not keys:
            logger.warning("No datasets declared in %s; nothing to do.", config_path)
            return ConversionSummary()
    else:
        keys = list(dataset_keys)
        unknown = [k for k in keys if k not in config.datasets]
        if unknown:
            logger.warning("Dataset key(s) %s not listed in %s; proceeding anyway.", unknown, config_path)

    # Resolve per-dataset raw dirs in the parent (avoids re-parsing
    # download.yaml inside every worker).
    dataset_dirs = {key: config.input_dir / _resolve_dataset_subdir(download_config_path, key) for key in keys}

    workers = resolve_workers(max_workers, len(keys), cpu_default(len(keys)))
    configure_script_logging(parallel=workers > 1)

    def kwargs_for(key: str) -> Dict[str, Any]:
        """Return per-worker kwargs for `convert_dataset`.

        Args:
            key: Dataset key being processed.

        Returns:
            Keyword arguments for `convert_dataset`.
        """
        return {"raw_dir": dataset_dirs[key], "output_dir": config.output_dir, "force": force}

    results, failures = run_parallel(
        convert_dataset,
        keys,
        max_workers=workers,
        desc="tokenization",
        pool="process",
        kwargs_for=kwargs_for,
        log_dir=stamped_log_dir("tokenization"),
    )

    summary = ConversionSummary(
        records={k: v for k, v in results.items() if v is not None},
        skipped=[k for k, v in results.items() if v is None],
        failed=[k for k, _ in failures],
    )
    logger.info(
        "Done. Converted %d dataset(s), %d records total. Skipped: %s. Failed: %s",
        len(summary.records),
        sum(summary.records.values()),
        summary.skipped or "none",
        summary.failed or "none",
    )
    return summary
