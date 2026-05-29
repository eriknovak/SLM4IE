"""Data cleaning, formatting, and splitting utilities."""

import gzip
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from tqdm import tqdm

# Import extractors to trigger registration
import slm4ie.data.extractors.coleslaw  # noqa: F401
import slm4ie.data.extractors.conllu  # noqa: F401
import slm4ie.data.extractors.huggingface  # noqa: F401
import slm4ie.data.extractors.json  # noqa: F401
import slm4ie.data.extractors.jsonl  # noqa: F401
import slm4ie.data.extractors.macocu  # noqa: F401
import slm4ie.data.extractors.tei  # noqa: F401
import slm4ie.data.extractors.text  # noqa: F401
from slm4ie.data.extract import extract_archive
from slm4ie.data.extractors import get_extractor
from slm4ie.data.parallel import (
    cpu_default,
    resolve_workers,
    run_parallel,
    workers_quiet,
)

logger = logging.getLogger(__name__)

#: Minimum number of input files before intra-dataset sharding kicks in.
#: Below this, the per-file overhead of a process pool is not worth it.
_SHARD_MIN_FILES = 8


@dataclass
class ExtractionConfig:
    """Configuration for dataset extraction pipeline.

    Attributes:
        input_dir: Base directory for raw datasets.
        output_dir: Base directory for processed output.
        datasets: Dict mapping dataset key to config dict with 'extractor' and 'domain' keys.
    """

    input_dir: str
    output_dir: str
    datasets: Dict[str, Dict] = field(default_factory=dict)


def load_extraction_config(config_path: Path) -> ExtractionConfig:
    """Load extraction config from YAML file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        ExtractionConfig: Parsed config.

    Raises:
        FileNotFoundError: If config file does not exist.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    return ExtractionConfig(
        input_dir=raw.get("input_dir", "data/raw"),
        output_dir=raw.get("output_dir", "data/processed"),
        datasets=raw.get("datasets", {}),
    )


def _stub_line(doc_id: Optional[str], uid: Optional[str]) -> str:
    """Build an annotations stub line carrying only identifiers.

    Stubs keep the annotations sidecar aligned with the text JSONL
    when an extractor yields a mix of annotated and unannotated
    documents. Downstream readers detect a stub by the absence of
    the parallel-array fields (`forms`, `lemmas`, ...).

    Args:
        doc_id: Document identifier of the unannotated record.
        uid: Globally unique identifier of the unannotated record.

    Returns:
        str: A JSON line with `doc_id` and `uid` only.
    """
    data: Dict[str, Optional[str]] = {}
    if doc_id is not None:
        data["doc_id"] = doc_id
    if uid is not None:
        data["uid"] = uid
    return json.dumps(data, ensure_ascii=False)


def _chunk_files(files: List[Path], n_chunks: int) -> List[List[Path]]:
    """Split files into contiguous, order-preserving slices.

    The first `len(files) % n_chunks` slices get one extra item so the
    sizes differ by at most one. Empty slices are dropped, so the
    result has at most `min(n_chunks, len(files))` chunks.

    Args:
        files (List[Path]): Files to split, in their final order.
        n_chunks (int): Desired number of chunks (clamped to
            `[1, len(files)]`).

    Returns:
        List[List[Path]]: Non-empty chunks whose concatenation equals
            `files`.
    """
    if not files:
        return []
    n = max(1, min(n_chunks, len(files)))
    size, remainder = divmod(len(files), n)
    chunks: List[List[Path]] = []
    start = 0
    for i in range(n):
        end = start + size + (1 if i < remainder else 0)
        chunks.append(files[start:end])
        start = end
    return [c for c in chunks if c]


def _extract_one(
    key: str,
    ds_cfg: Dict[str, Any],
    input_base: Path,
    output_base: Path,
    force: bool,
) -> Optional[int]:
    """Extract one dataset to unified JSONL (and optional annotations).

    Args:
        key: Dataset key (used for log messages and output filenames).
        ds_cfg: Per-dataset config dict with `extractor` and `domain` keys.
        input_base: Base directory under which `<key>/` lives.
        output_base: Directory to write `<key>.jsonl` (+ optional
            `<key>.annotations.jsonl.gz`) into.
        force: When True, overwrite an existing output file.

    Returns:
        Optional[int]: Document count written, or None when the input
            directory is missing (caller treats as a skip, not an error).
            Returns 0 when the output already exists and *force* is False.
    """
    extractor_name = ds_cfg["extractor"]
    domain = ds_cfg["domain"]
    metadata_cfg = ds_cfg.get("metadata")
    input_dir = input_base / key

    if not input_dir.exists():
        logger.warning("Input dir not found for '%s': %s", key, input_dir)
        return None

    text_file = output_base / f"{key}.jsonl"
    ann_file = output_base / f"{key}.annotations.jsonl.gz"

    if text_file.exists() and not force:
        logger.info(
            "Skipping '%s', output already exists: %s "
            "(use --force to re-extract)",
            key,
            text_file,
        )
        return 0

    # Recovery on entry: discard partial files left by a prior
    # crashed run. The final outputs are written atomically below
    # via os.replace, so any `.partial` files imply incomplete work.
    text_partial = output_base / f"{key}.jsonl.partial"
    ann_partial = output_base / f"{key}.annotations.jsonl.gz.partial"
    if text_partial.exists():
        logger.info(
            "Removing orphan partial text file for '%s'", key,
        )
        text_partial.unlink()
    if ann_partial.exists():
        logger.info(
            "Removing orphan partial annotations file for '%s'", key,
        )
        ann_partial.unlink()

    # Decompress any archives before extraction
    for archive in sorted(input_dir.iterdir()):
        if archive.name.endswith(
            (".gz", ".xz", ".zip", ".tgz", ".tar.gz", ".tar.zst", ".tar.zstd")
        ):
            extract_archive(archive, input_dir)

    logger.info("Extracting '%s' with %s extractor", key, extractor_name)

    extractor = get_extractor(extractor_name)

    count = 0
    has_annotations = False

    # Buffer of (doc_id, uid) pairs for text-only docs seen before
    # the first annotated doc. When annotations begin, these are
    # backfilled as stub lines so the two streams stay in lockstep.
    pending_stubs: List[Tuple[Optional[str], Optional[str]]] = []

    with open(text_partial, "w", encoding="utf-8") as tf:
        ann_fh = None
        try:
            for index, doc in enumerate(tqdm(
                extractor.extract(
                    input_dir, key, domain, metadata=metadata_cfg
                ),
                desc=key,
                unit="doc",
                disable=workers_quiet(),
            )):
                if doc.doc_id is None:
                    doc.doc_id = f"idx-{index:014d}"

                tf.write(doc.to_jsonl_line())
                tf.write("\n")

                ann_line = doc.to_annotation_line()
                if ann_line is not None:
                    if ann_fh is None:
                        ann_fh = gzip.open(ann_partial, "wt", encoding="utf-8")
                        has_annotations = True
                        for stub_doc_id, stub_uid in pending_stubs:
                            ann_fh.write(_stub_line(stub_doc_id, stub_uid))
                            ann_fh.write("\n")
                        pending_stubs = []
                    ann_fh.write(ann_line)
                    ann_fh.write("\n")
                elif ann_fh is not None:
                    ann_fh.write(_stub_line(doc.doc_id, doc.uid))
                    ann_fh.write("\n")
                else:
                    pending_stubs.append((doc.doc_id, doc.uid))

                count += 1
        finally:
            if ann_fh is not None:
                ann_fh.close()

    # Promote partials atomically; single-file os.replace is atomic
    # on POSIX. If the function raised mid-stream above, control
    # never reaches here and the orphan recovery on the next run
    # cleans up the `.partial` files.
    os.replace(text_partial, text_file)
    if has_annotations:
        os.replace(ann_partial, ann_file)

    logger.info(
        "Extracted %d documents from '%s' -> %s%s",
        count,
        key,
        text_file,
        f" + {ann_file}" if has_annotations else "",
    )
    return count


def extract_datasets(
    config_path: Path,
    dataset_keys: Optional[List[str]] = None,
    force: bool = False,
    max_workers: int = 0,
    log_dir: Optional[Path] = None,
    input_dir_override: Optional[str] = None,
    output_dir_override: Optional[str] = None,
) -> None:
    """Extract and convert datasets to unified JSONL.

    Args:
        config_path: Path to extraction YAML config.
        dataset_keys: Specific dataset keys to extract. If None, extracts all configured datasets.
        force: When True, re-extract datasets whose output already
            exists. Defaults to False (skip already-extracted datasets).
        max_workers: Number of datasets to extract in parallel.
            `0` (default) picks `min(cpu_count // 2, n_datasets)`;
            `1` runs serially with unwrapped tracebacks; `N > 1`
            spins up that many worker processes.
        log_dir: When set, per-dataset logs are written to
            `<log_dir>/<key>.log`. The directory is created if it
            does not exist. When extractions fail, the failed dataset
            keys are also written to `<log_dir>/failures.txt`, one per
            line, sorted alphabetically.
        input_dir_override: Override the `input_dir` from the YAML
            config. When truthy, this path is used as the base
            directory under which `<key>/` lives.
        output_dir_override: Override the `output_dir` from the YAML
            config. When truthy, processed outputs are written here
            instead of the configured location.

    Raises:
        ValueError: If any requested key is unknown.
        RuntimeError: If one or more dataset extractions failed.
    """
    cfg = load_extraction_config(config_path)

    if dataset_keys:
        unknown = set(dataset_keys) - set(cfg.datasets.keys())
        if unknown:
            raise ValueError(f"Unknown dataset keys: {', '.join(sorted(unknown))}")
        selected = {k: v for k, v in cfg.datasets.items() if k in dataset_keys}
    else:
        selected = cfg.datasets

    input_base = Path(input_dir_override) if input_dir_override else Path(cfg.input_dir)
    output_base = Path(output_dir_override) if output_dir_override else Path(cfg.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    keys = list(selected.keys())
    workers = resolve_workers(max_workers, len(keys), cpu_default(len(keys)))

    def kwargs_for(key: str) -> Dict[str, Any]:
        return {
            "ds_cfg": selected[key],
            "input_base": input_base,
            "output_base": output_base,
            "force": force,
        }

    _, failures = run_parallel(
        _extract_one,
        keys,
        max_workers=workers,
        desc="extract",
        pool="process",
        kwargs_for=kwargs_for,
        log_dir=log_dir,
    )

    if failures:
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            failed_sorted = sorted(k for k, _ in failures)
            (log_dir / "failures.txt").write_text(
                "\n".join(failed_sorted) + "\n",
                encoding="utf-8",
            )
        failed_keys = ", ".join(k for k, _ in failures)
        raise RuntimeError(
            f"Extraction failed for {len(failures)} dataset(s): {failed_keys}"
        )
