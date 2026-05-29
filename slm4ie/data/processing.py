"""Data cleaning, formatting, and splitting utilities."""

import gzip
import json
import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from slm4ie.data.extractors import BaseExtractor, FileBasedExtractor, get_extractor
from slm4ie.data.parallel import (
    resolve_workers,
    run_parallel,
    workers_quiet,
)

logger = logging.getLogger(__name__)

#: Minimum number of input files before intra-dataset sharding kicks in.
#: Below this, the per-file overhead of a process pool is not worth it.
_SHARD_MIN_FILES = 8

#: Chunks per worker for shard work-stealing. Over-provisioning past the
#: worker count keeps all workers busy when shards finish unevenly.
_SHARD_OVERPROVISION = 4


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


@dataclass(frozen=True)
class ShardResult:
    """Outcome of parsing one file shard.

    Attributes:
        index (int): Shard ordinal, used to concatenate in order.
        text_path (Path): Temp file holding this shard's text JSONL.
        ann_path (Path): Temp file holding this shard's gzipped
            annotations (one line per document, real or stub).
        count (int): Documents written by this shard.
        had_real_ann (bool): True if any document carried a real
            annotation line (not just a stub).
    """

    index: int
    text_path: Path
    ann_path: Path
    count: int
    had_real_ann: bool


def _extract_shard(
    index: int,
    files: List[Path],
    key: str,
    extractor_name: str,
    domain: str,
    metadata_cfg: Optional[Dict[str, Any]],
    input_dir: Path,
    tmp_dir: Path,
) -> ShardResult:
    """Parse one file shard into per-shard text + annotation files.

    Writes exactly one annotation line per document (a real line when
    the document carries annotations, otherwise a stub), so the shard
    files stay internally lockstep and concatenate cleanly. Documents
    without a `doc_id` get a shard-namespaced fallback id.

    Args:
        index (int): Shard ordinal.
        files (List[Path]): Files assigned to this shard, in order.
        key (str): Dataset key (used as the `source` field).
        extractor_name (str): Registry name of a `FileBasedExtractor`.
        domain (str): Domain label assigned to every Document.
        metadata_cfg (Optional[Dict[str, Any]]): Optional `metadata:`
            config block forwarded to the extractor.
        input_dir (Path): Dataset root, for sidecar resolution.
        tmp_dir (Path): Directory to write this shard's temp files into.

    Returns:
        ShardResult: Paths, document count, and annotation flag.
    """
    extractor = get_extractor(extractor_name)
    text_path = tmp_dir / f"{index:05d}.jsonl"
    ann_path = tmp_dir / f"{index:05d}.annotations.jsonl.gz"
    count = 0
    had_real_ann = False

    with open(text_path, "w", encoding="utf-8") as tf, gzip.open(
        ann_path, "wt", encoding="utf-8"
    ) as af:
        for local, doc in enumerate(
            extractor.extract_files(
                files, key, domain, input_dir, metadata_cfg
            )
        ):
            if doc.doc_id is None:
                # Shard-namespaced fallback id (vs. the serial writer's
                # global `idx-{index:014d}`). Only differs for extractors
                # that yield null doc_ids; the converted file-based
                # extractors assign real ids except `text`, whose only
                # dataset (cc100) is a single file and never shards.
                doc.doc_id = f"idx-{index:05d}-{local:010d}"
            tf.write(doc.to_jsonl_line())
            tf.write("\n")

            ann_line = doc.to_annotation_line()
            if ann_line is not None:
                had_real_ann = True
                af.write(ann_line)
            else:
                af.write(_stub_line(doc.doc_id, doc.uid))
            af.write("\n")
            count += 1

    return ShardResult(index, text_path, ann_path, count, had_real_ann)


def _extract_serial(
    key: str,
    extractor: BaseExtractor,
    domain: str,
    metadata_cfg: Optional[Dict[str, Any]],
    input_dir: Path,
    text_file: Path,
    ann_file: Path,
) -> int:
    """Stream a dataset to JSONL in a single pass (no sharding).

    This is the original single-process writer: it consumes the
    extractor generator in order, writing the text JSONL and the
    gzipped annotations sidecar in lockstep, buffering stubs until the
    first real annotation appears, then promoting the `.partial` files
    atomically.

    Args:
        key (str): Dataset key (used as `source` and in messages).
        extractor (BaseExtractor): Instantiated extractor.
        domain (str): Domain label assigned to every Document.
        metadata_cfg (Optional[Dict[str, Any]]): Optional `metadata:`
            config block forwarded to the extractor.
        input_dir (Path): Directory containing the raw source data.
        text_file (Path): Final destination for the text JSONL.
        ann_file (Path): Final destination for the gzipped annotations.

    Returns:
        int: Number of documents written.
    """
    text_partial = text_file.parent / f"{text_file.name}.partial"
    ann_partial = ann_file.parent / f"{ann_file.name}.partial"

    count = 0
    has_annotations = False
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
                    # Global fallback id. The sharded path uses a
                    # shard-namespaced scheme; see _extract_shard.
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


def _extract_sharded(
    key: str,
    extractor_name: str,
    domain: str,
    metadata_cfg: Optional[Dict[str, Any]],
    input_dir: Path,
    files: List[Path],
    text_file: Path,
    ann_file: Path,
    shard_workers: int,
) -> int:
    """Parse a dataset's files in parallel shards, then merge in order.

    Splits `files` into ordered chunks, parses each chunk in a worker
    process to its own temp text + annotation shard, then concatenates
    the shards in order into the canonical outputs and promotes them
    atomically. The merged text file is byte-identical to the serial
    writer's; the merged annotations file is a multi-member gzip whose
    decompressed content matches the serial writer's.

    Args:
        key (str): Dataset key.
        extractor_name (str): Registry name of a `FileBasedExtractor`.
        domain (str): Domain label assigned to every Document.
        metadata_cfg (Optional[Dict[str, Any]]): Optional `metadata:`
            config block forwarded to the extractor.
        input_dir (Path): Dataset root, for sidecar resolution.
        files (List[Path]): All input files for this dataset, sorted.
        text_file (Path): Final destination for the text JSONL.
        ann_file (Path): Final destination for the gzipped annotations.
        shard_workers (int): Number of worker processes.

    Returns:
        int: Total number of documents written.
    """
    tmp_dir = text_file.parent / f".{key}.shards"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    n_chunks = min(len(files), max(1, shard_workers) * _SHARD_OVERPROVISION)
    chunks = _chunk_files(files, n_chunks)
    results: List[Optional[ShardResult]] = [None] * len(chunks)

    try:
        # The dataset loop runs serially (run_parallel max_workers=1),
        # so the parent process is single-threaded here and forking the
        # pool is safe. Keep the dataset axis serial if revisiting.
        with ProcessPoolExecutor(max_workers=shard_workers) as executor:
            future_to_index = {
                executor.submit(
                    _extract_shard,
                    i,
                    chunk,
                    key,
                    extractor_name,
                    domain,
                    metadata_cfg,
                    input_dir,
                    tmp_dir,
                ): i
                for i, chunk in enumerate(chunks)
            }
            for future in tqdm(
                as_completed(future_to_index),
                total=len(future_to_index),
                desc=key,
                unit="shard",
                disable=workers_quiet(),
            ):
                idx = future_to_index[future]
                results[idx] = future.result()

        ordered = [r for r in results if r is not None]
        total = sum(r.count for r in ordered)
        # Keep the merged annotations file only if at least one shard
        # produced a real annotation; otherwise every line would be a
        # stub and the serial path would have written no file at all.
        has_annotations = any(r.had_real_ann for r in ordered)

        text_partial = text_file.parent / f"{text_file.name}.partial"
        with open(text_partial, "wb") as out:
            for r in ordered:
                with open(r.text_path, "rb") as src:
                    shutil.copyfileobj(src, out)
        os.replace(text_partial, text_file)

        if has_annotations:
            ann_partial = ann_file.parent / f"{ann_file.name}.partial"
            with open(ann_partial, "wb") as out:
                for r in ordered:
                    with open(r.ann_path, "rb") as src:
                        shutil.copyfileobj(src, out)
            os.replace(ann_partial, ann_file)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info(
        "Extracted %d documents from '%s' (%d shards) -> %s%s",
        total,
        key,
        len(ordered),
        text_file,
        f" + {ann_file}" if has_annotations else "",
    )
    return total


def _extract_one(
    key: str,
    ds_cfg: Dict[str, Any],
    input_base: Path,
    output_base: Path,
    force: bool,
    requested_workers: int = 0,
) -> Optional[int]:
    """Extract one dataset to unified JSONL (and optional annotations).

    Args:
        key: Dataset key (used for log messages and output filenames).
        ds_cfg: Per-dataset config dict with `extractor` and `domain` keys.
        input_base: Base directory under which `<key>/` lives.
        output_base: Directory to write `<key>.jsonl` (+ optional
            `<key>.annotations.jsonl.gz`) into.
        force: When True, overwrite an existing output file.
        requested_workers: Shard-worker count for intra-dataset
            parallelism. `0` means auto (all cores). Sharding only
            engages for `FileBasedExtractor`s with at least
            `_SHARD_MIN_FILES` input files and more than one worker.

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

    # A prior sharded run that was hard-killed (no finally) can leave
    # its shard temp dir behind; remove it so it cannot accumulate.
    stale_shards = output_base / f".{key}.shards"
    if stale_shards.exists():
        logger.info("Removing orphan shard temp dir for '%s'", key)
        shutil.rmtree(stale_shards, ignore_errors=True)

    # Decompress any archives before extraction
    for archive in sorted(input_dir.iterdir()):
        if archive.name.endswith(
            (".gz", ".xz", ".zip", ".tgz", ".tar.gz", ".tar.zst", ".tar.zstd")
        ):
            extract_archive(archive, input_dir)

    logger.info("Extracting '%s' with %s extractor", key, extractor_name)

    extractor = get_extractor(extractor_name)

    cores = os.cpu_count() or 1
    shard_workers = resolve_workers(requested_workers, cores, cores)

    files = (
        extractor.iter_input_files(input_dir)
        if isinstance(extractor, FileBasedExtractor)
        else None
    )

    if (
        files is not None
        and shard_workers > 1
        and len(files) >= _SHARD_MIN_FILES
    ):
        return _extract_sharded(
            key,
            extractor_name,
            domain,
            metadata_cfg,
            input_dir,
            files,
            text_file,
            ann_file,
            shard_workers,
        )

    return _extract_serial(
        key,
        extractor,
        domain,
        metadata_cfg,
        input_dir,
        text_file,
        ann_file,
    )


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
        max_workers: Shard-worker count used *within* each dataset for
            intra-dataset parallelism. Datasets themselves are always
            processed sequentially. `0` (default) means auto (all
            cores); `1` forces the single-pass serial writer; `N > 1`
            parses each dataset's files across `N` worker processes
            (only for file-based extractors with enough input files).
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

    def kwargs_for(key: str) -> Dict[str, Any]:
        return {
            "ds_cfg": selected[key],
            "input_base": input_base,
            "output_base": output_base,
            "force": force,
            "requested_workers": max_workers,
        }

    # Datasets are processed strictly sequentially (max_workers=1 on
    # the dataset axis); parallelism happens *inside* each dataset via
    # shard workers in `_extract_one`. This keeps exactly one process
    # pool alive at a time.
    _, failures = run_parallel(
        _extract_one,
        keys,
        max_workers=1,
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
