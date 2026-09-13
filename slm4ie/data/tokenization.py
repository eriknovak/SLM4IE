"""Convert raw tokenizer-evaluation downloads into JSONL records.

Materializes the datasets tagged for tokenizer / morphology evaluation. This
route bypasses the extract -> datatrove -> curate pipeline used by pretraining
corpora and reads the raw download directly, like the `raw`-source path of the
task converters.

Configuration is read from `configs/data/tokenization.yaml`, which declares
`input_dir`, `output_dir`, and the dataset keys to convert. Each key must be
declared in `configs/data/download.yaml` (so its raw subdirectory resolves) and
have a reader registered in `READERS` below. Sloleks 3.1 is the seed dataset;
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
from typing import Any, Callable, Dict, IO, Iterable, Iterator, List, Optional

import yaml
from tqdm import tqdm

from slm4ie.data.archives import unpack_archives
from slm4ie.data.catalog import DatasetConfig, load_config
from slm4ie.data.io_utils import open_output, resolve_project_path
from slm4ie.data.parallel import (
    configure_script_logging,
    cpu_default,
    resolve_workers,
    run_parallel,
    workers_quiet,
)
from slm4ie.data.sloleks import iter_sloleks_dir
from slm4ie.data.sloleks_relations import (
    find_word_relations_tsv,
    iter_word_relation_segmentations,
)
from slm4ie.utils.cli import stamped_log_dir

logger = logging.getLogger(__name__)

#: Task tag identifying tokenizer-evaluation datasets.
TOKENIZER_TASK = "TOKENIZER"


def _read_sloleks(raw_dir: Path) -> Iterator[Dict[str, Any]]:
    """Yield Sloleks records, tagging them with dataset/task fields.

    Any archive sitting in `raw_dir` (e.g. the downloaded Sloleks zip) is
    unpacked in place when no XML is found yet, so the download need not be
    unzipped by hand first.

    Args:
        raw_dir: Directory holding the Sloleks XML files (or the zip to unpack).

    Yields:
        Dict[str, Any]: Records as produced by `iter_sloleks_dir`, extended
            with `dataset` and `task` fields.

    Raises:
        FileNotFoundError: If no Sloleks XML files are found under `raw_dir`,
            even after unpacking any archive present.
    """
    if not any(raw_dir.rglob("*.xml")):
        unpack_archives(raw_dir)
    if not any(raw_dir.rglob("*.xml")):
        raise FileNotFoundError(
            f"No XML files found under {raw_dir}. Run `prepare_datasets.py download sloleks` first."
        )
    for record in iter_sloleks_dir(raw_dir):
        record["dataset"] = "sloleks"
        record["task"] = TOKENIZER_TASK
        yield record


def _read_sloleks_relations(raw_dir: Path) -> Iterator[Dict[str, Any]]:
    """Yield Sloleks word-relation derivational segmentations.

    Each record carries one derived lemma's morpheme decomposition (read from
    the underscore column, not heuristically aligned), tagged with the dataset
    and task fields and a `verified` flag for the manually-scored subset. Any
    archive in `raw_dir` (the downloaded zip) is unpacked in place when no TSV
    is found yet, so the download need not be unzipped by hand first.

    Args:
        raw_dir: Directory holding the word-relations TSV (or the zip to unpack).

    Yields:
        Dict[str, Any]: Records as produced by `iter_word_relation_segmentations`,
            extended with `dataset` and `task` fields.

    Raises:
        FileNotFoundError: If no word-relations TSV is found under `raw_dir`,
            even after unpacking any archive present.
    """
    tsv_path = find_word_relations_tsv(raw_dir)
    if tsv_path is None:
        unpack_archives(raw_dir)
        tsv_path = find_word_relations_tsv(raw_dir)
    if tsv_path is None:
        raise FileNotFoundError(
            f"No word-relations TSV found under {raw_dir}. Run `prepare_datasets.py download sloleks_relations` first."
        )
    for record in iter_word_relation_segmentations(tsv_path):
        record["dataset"] = "sloleks_relations"
        record["task"] = TOKENIZER_TASK
        yield record


#: Registry mapping dataset key to a reader callable.
READERS: Dict[str, Callable[[Path], Iterator[Dict[str, Any]]]] = {
    "sloleks": _read_sloleks,
    "sloleks_relations": _read_sloleks_relations,
}


@dataclass
class TokenizationConfig:
    """Resolved contents of `configs/data/tokenization.yaml`.

    Attributes:
        input_dir: Root holding the per-dataset raw download directories.
        output_dir: Directory the `<key>.jsonl.gz` outputs are written to.
        datasets: Dataset keys declared for conversion.
    """

    input_dir: Path
    output_dir: Path
    datasets: List[str] = field(default_factory=list)


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


def load_tokenization_config(config_path: Path) -> TokenizationConfig:
    """Load and validate the tokenization YAML config.

    Args:
        config_path: Path to `configs/data/tokenization.yaml`.

    Returns:
        The parsed config with project-relative directories resolved.

    Raises:
        FileNotFoundError: If `config_path` does not exist.
        ValueError: If required fields are missing or malformed.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Tokenization config not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    input_dir = raw.get("input_dir")
    output_dir = raw.get("output_dir")
    datasets = raw.get("datasets") or []

    missing: List[str] = []
    if not input_dir:
        missing.append("input_dir")
    if not output_dir:
        missing.append("output_dir")
    if missing:
        raise ValueError(f"Tokenization config {config_path} is missing required fields: {', '.join(missing)}")
    if not isinstance(input_dir, str) or not isinstance(output_dir, str):
        raise ValueError(f"Tokenization config {config_path}: `input_dir` and `output_dir` must be strings.")
    if not isinstance(datasets, list) or not all(isinstance(k, str) for k in datasets):
        raise ValueError(f"Tokenization config {config_path}: `datasets` must be a list of strings.")

    return TokenizationConfig(
        input_dir=resolve_project_path(input_dir),
        output_dir=resolve_project_path(output_dir),
        datasets=list(datasets),
    )


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
