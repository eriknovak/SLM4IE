"""Extract raw downloads into the canonical unified JSONL form."""

import gzip
import hashlib
import json
import logging
import os
import shutil
from array import array
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

# Import extractors to trigger registration
import slm4ie.data.extract.extractors.coleslaw  # noqa: F401
import slm4ie.data.extract.extractors.conllu  # noqa: F401
import slm4ie.data.extract.extractors.huggingface  # noqa: F401
import slm4ie.data.extract.extractors.json  # noqa: F401
import slm4ie.data.extract.extractors.jsonl  # noqa: F401
import slm4ie.data.extract.extractors.macocu  # noqa: F401
import slm4ie.data.extract.extractors.tei  # noqa: F401
import slm4ie.data.extract.extractors.text  # noqa: F401
from slm4ie.data.archives import extract_archive
from slm4ie.data.extract.config import load_extract_config
from slm4ie.data.extract.extractors import BaseExtractor, FileBasedExtractor, get_extractor
from slm4ie.utils.io import resolve_project_path
from slm4ie.utils.parallel import (
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


def _hash64(value: str) -> int:
    """Return a 64-bit hash of `value` that is identical across processes.

    Python's own `hash` is salted per interpreter, so shard workers and
    the parent would disagree; blake2b is not.

    Args:
        value: The string to hash.

    Returns:
        int: The hash as an unsigned 64-bit integer.
    """
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "little")


class DuplicateDocumentIdError(ValueError):
    """A dataset repeats a `doc_id` on documents whose text differs.

    `<dataset>:<doc_id>` is the key every downstream consumer joins on
    (see `CONTEXT.md`, "Document id"), so extraction refuses to write a
    dataset that breaks it rather than let the collision surface later
    as under-counted drops or mismatched annotations.
    """

    def __init__(self, key: str, conflicts: int, examples: List[str]) -> None:
        """Format the error for one dataset.

        Args:
            key: Dataset key.
            conflicts: Number of documents whose id repeats an earlier
                document's id with different text.
            examples: Up to a few of the offending ids.
        """
        shown = ", ".join(repr(e) for e in examples)
        super().__init__(
            f"{conflicts} documents in '{key}' repeat an earlier doc_id with different text (e.g. {shown}). "
            "doc_id must be unique within a dataset: give the extractor a positional id or "
            "narrow the input with `include:` in extract.yaml."
        )
        self.key = key
        self.conflicts = conflicts
        self.examples = examples


class _IdLedger:
    """Hashes of every written document's id and text, in write order.

    Sixteen bytes per document, so a dataset of tens of millions of
    documents is checked for repeated ids without holding the ids
    themselves. Shards build one each and the parent concatenates them
    in shard order, which is also the order of the merged output.
    """

    def __init__(self, ids: bytes = b"", texts: bytes = b"") -> None:
        """Start a ledger, optionally from another ledger's serialized arrays.

        Args:
            ids: `array("Q")` bytes of id hashes.
            texts: `array("Q")` bytes of text hashes, parallel to `ids`.
        """
        self.ids = array("Q")
        self.ids.frombytes(ids)
        self.texts = array("Q")
        self.texts.frombytes(texts)

    def __len__(self) -> int:
        """Return the number of documents recorded."""
        return len(self.ids)

    def add(self, doc_id: str, text: str) -> None:
        """Record one written document.

        Args:
            doc_id: The document's id, after any fallback was applied.
            text: The document's text.
        """
        self.ids.append(_hash64(doc_id))
        self.texts.append(_hash64(text))

    def extend(self, other: "_IdLedger") -> None:
        """Append another ledger's documents after this one's.

        Args:
            other: The ledger to append; its order is preserved.
        """
        self.ids.extend(other.ids)
        self.texts.extend(other.texts)

    def serialized(self) -> Tuple[bytes, bytes]:
        """Return the two arrays as bytes, for crossing a process boundary.

        Returns:
            Tuple[bytes, bytes]: Id hashes and text hashes.
        """
        return self.ids.tobytes(), self.texts.tobytes()

    def resolve(self) -> Tuple[bytearray, List[int]]:
        """Decide what to do with every repeated id.

        The first document carrying an id is always kept. A later
        document with the same id is dropped when its text hash equals
        the first one's (the same record stored twice) and reported as
        a conflict when it differs.

        Returns:
            Tuple[bytearray, List[int]]: One keep flag (1/0) per
                document in write order, and the write-order indices
                of the conflicting documents.
        """
        n = len(self.ids)
        keep = bytearray(b"\x01") * n
        conflicts: List[int] = []
        # Stable sort: within one id, write order survives, so the first
        # element of a group is the earliest written document.
        order = sorted(range(n), key=self.ids.__getitem__)
        group_start = 0
        for pos in range(1, n + 1):
            if pos < n and self.ids[order[pos]] == self.ids[order[group_start]]:
                continue
            first = order[group_start]
            for later in order[group_start + 1 : pos]:
                if self.texts[later] == self.texts[first]:
                    keep[later] = 0
                else:
                    conflicts.append(later)
            group_start = pos
        conflicts.sort()
        return keep, conflicts


def _drop_lines(path: Path, keep: Iterable[int], gz: bool) -> None:
    """Rewrite a line-oriented file in place, keeping the flagged lines.

    Args:
        path: The file to filter; plain text or gzip.
        keep: One truthy/falsy flag per line of the file.
        gz: Whether the file is gzip-compressed.

    Raises:
        ValueError: If the file has a different number of lines than
            `keep` has flags.
    """
    opener = gzip.open if gz else open
    filtered = path.with_name(path.name + ".dedup")
    with opener(path, "rb") as fin, opener(filtered, "wb") as fout:
        for flag, line in zip(keep, fin, strict=True):
            if flag:
                fout.write(line)
    os.replace(filtered, path)


def _doc_ids_at(text_files: List[Tuple[Path, int]], indices: List[int]) -> List[str]:
    """Read the `doc_id` of the documents at the given write-order indices.

    Args:
        text_files: The text JSONL files holding the documents in
            order, each with its document count.
        indices: Sorted write-order indices to look up.

    Returns:
        List[str]: The ids found, in index order.
    """
    wanted = list(indices)
    found: List[str] = []
    offset = 0
    for path, count in text_files:
        local = [i - offset for i in wanted if offset <= i < offset + count]
        if local:
            with open(path, encoding="utf-8") as fh:
                for line_idx, line in enumerate(fh):
                    if line_idx in local:
                        found.append(str(json.loads(line).get("doc_id")))
        offset += count
    return found


def _select_files(files: List[Path], input_dir: Path, include: Optional[List[str]]) -> List[Path]:
    """Keep the input files matching the dataset's `include:` globs.

    Args:
        files: Every file the extractor discovered, sorted.
        input_dir: Dataset root the globs are relative to.
        include: Shell-style patterns matched against each file's path
            under `input_dir` (`*` also crosses `/`). None or empty
            keeps every file.

    Returns:
        List[Path]: The matching files, in their original order.

    Raises:
        ValueError: If the patterns match no file at all.
    """
    if not include:
        return files
    kept = [f for f in files if any(fnmatch(f.relative_to(input_dir).as_posix(), pattern) for pattern in include)]
    if not kept:
        raise ValueError(f"include patterns {include!r} match no input file under {input_dir}")
    return kept


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
        id_hashes (bytes): The shard's `_IdLedger` id array, serialized.
        text_hashes (bytes): The shard's `_IdLedger` text array, serialized.
    """

    index: int
    text_path: Path
    ann_path: Path
    count: int
    had_real_ann: bool
    id_hashes: bytes
    text_hashes: bytes


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
        input_dir (Path): Dataset root, for metadata table resolution.
        tmp_dir (Path): Directory to write this shard's temp files into.

    Returns:
        ShardResult: Paths, document count, and annotation flag.
    """
    extractor = get_extractor(extractor_name)
    text_path = tmp_dir / f"{index:05d}.jsonl"
    ann_path = tmp_dir / f"{index:05d}.annotations.jsonl.gz"
    count = 0
    had_real_ann = False
    ledger = _IdLedger()

    with open(text_path, "w", encoding="utf-8") as tf, gzip.open(ann_path, "wt", encoding="utf-8") as af:
        for local, doc in enumerate(extractor.extract_files(files, key, domain, input_dir, metadata_cfg)):
            if doc.doc_id is None:
                # Shard-namespaced fallback id (vs. the serial writer's
                # global `idx-{index:014d}`), so the two schemes disagree
                # for any extractor that yields null doc_ids. No shipped
                # extractor does: this only guards a future one.
                doc.doc_id = f"idx-{index:05d}-{local:010d}"
            tf.write(doc.to_jsonl_line())
            tf.write("\n")
            ledger.add(doc.doc_id, doc.text)

            ann_line = doc.to_annotation_line()
            if ann_line is not None:
                had_real_ann = True
                af.write(ann_line)
            else:
                af.write(_stub_line(doc.doc_id, doc.uid))
            af.write("\n")
            count += 1

    id_hashes, text_hashes = ledger.serialized()
    return ShardResult(index, text_path, ann_path, count, had_real_ann, id_hashes, text_hashes)


def _extract_serial(
    key: str,
    extractor: BaseExtractor,
    domain: str,
    metadata_cfg: Optional[Dict[str, Any]],
    input_dir: Path,
    text_file: Path,
    ann_file: Path,
    files: Optional[List[Path]] = None,
) -> int:
    """Stream a dataset to JSONL in a single pass (no sharding).

    This is the original single-process writer: it consumes the
    extractor generator in order, writing the text JSONL and the
    gzipped annotations sidecar in lockstep, buffering stubs until the
    first real annotation appears, then promoting the `.partial` files
    atomically. Before promotion the id ledger is resolved: repeats of
    an id with identical text are dropped, repeats with different text
    abort the dataset (see `DuplicateDocumentIdError`).

    Args:
        key (str): Dataset key (used as `source` and in messages).
        extractor (BaseExtractor): Instantiated extractor.
        domain (str): Domain label assigned to every Document.
        metadata_cfg (Optional[Dict[str, Any]]): Optional `metadata:`
            config block forwarded to the extractor.
        input_dir (Path): Directory containing the raw source data.
        text_file (Path): Final destination for the text JSONL.
        ann_file (Path): Final destination for the gzipped annotations.
        files (Optional[List[Path]]): The selected input files of a
            `FileBasedExtractor`; None lets the extractor discover them.

    Returns:
        int: Number of documents written.

    Raises:
        DuplicateDocumentIdError: If a `doc_id` repeats with different text.
    """
    text_partial = text_file.parent / f"{text_file.name}.partial"
    ann_partial = ann_file.parent / f"{ann_file.name}.partial"

    count = 0
    has_annotations = False
    pending_stubs: List[Tuple[Optional[str], Optional[str]]] = []
    ledger = _IdLedger()

    if files is not None and isinstance(extractor, FileBasedExtractor):
        documents = extractor.extract_files(files, key, domain, input_dir, metadata_cfg)
    else:
        documents = extractor.extract(input_dir, key, domain, metadata=metadata_cfg)

    with open(text_partial, "w", encoding="utf-8") as tf:
        ann_fh = None
        try:
            for index, doc in enumerate(tqdm(documents, desc=key, unit="doc", disable=workers_quiet())):
                if doc.doc_id is None:
                    # Global fallback id. The sharded path uses a
                    # shard-namespaced scheme; see _extract_shard. Every
                    # shipped extractor assigns its own doc_id, so this
                    # is unreachable today and kept only as a guard.
                    doc.doc_id = f"idx-{index:014d}"

                tf.write(doc.to_jsonl_line())
                tf.write("\n")
                ledger.add(doc.doc_id, doc.text)

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

    keep, conflicts = ledger.resolve()
    if conflicts:
        examples = _doc_ids_at([(text_partial, count)], conflicts[:3])
        text_partial.unlink()
        if has_annotations:
            ann_partial.unlink()
        raise DuplicateDocumentIdError(key, len(conflicts), examples)
    dropped = count - sum(keep)
    if dropped:
        _drop_lines(text_partial, keep, gz=False)
        if has_annotations:
            _drop_lines(ann_partial, keep, gz=True)
        count -= dropped
        logger.info("Dropped %d repeated documents (same doc_id, same text) from '%s'", dropped, key)

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
        input_dir (Path): Dataset root, for metadata table resolution.
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
        # Keep the merged annotations file only if at least one shard
        # produced a real annotation; otherwise every line would be a
        # stub and the serial path would have written no file at all.
        has_annotations = any(r.had_real_ann for r in ordered)

        ledger = _IdLedger()
        for r in ordered:
            ledger.extend(_IdLedger(r.id_hashes, r.text_hashes))
        keep, conflicts = ledger.resolve()
        if conflicts:
            examples = _doc_ids_at([(r.text_path, r.count) for r in ordered], conflicts[:3])
            raise DuplicateDocumentIdError(key, len(conflicts), examples)
        dropped = len(keep) - sum(keep)
        if dropped:
            offset = 0
            for r in ordered:
                shard_keep = keep[offset : offset + r.count]
                offset += r.count
                if 0 in shard_keep:
                    _drop_lines(r.text_path, shard_keep, gz=False)
                    _drop_lines(r.ann_path, shard_keep, gz=True)
            logger.info("Dropped %d repeated documents (same doc_id, same text) from '%s'", dropped, key)
        total = len(keep) - dropped

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
            "Skipping '%s', output already exists: %s (use --force to re-extract)",
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
            "Removing orphan partial text file for '%s'",
            key,
        )
        text_partial.unlink()
    if ann_partial.exists():
        logger.info(
            "Removing orphan partial annotations file for '%s'",
            key,
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
        if archive.name.endswith((".gz", ".xz", ".zip", ".tgz", ".tar.gz", ".tar.zst", ".tar.zstd")):
            extract_archive(archive, input_dir)

    logger.info("Extracting '%s' with %s extractor", key, extractor_name)

    extractor = get_extractor(extractor_name)

    cores = os.cpu_count() or 1
    shard_workers = resolve_workers(requested_workers, cores, cores)

    include = ds_cfg.get("include")
    if isinstance(extractor, FileBasedExtractor):
        files = _select_files(extractor.iter_input_files(input_dir), input_dir, include)
    elif include:
        raise ValueError(f"'{key}': `include:` needs a file-based extractor, and '{extractor_name}' is not one")
    else:
        files = None

    if files is not None and shard_workers > 1 and len(files) >= _SHARD_MIN_FILES:
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
        files=files,
    )


def extract_datasets(
    config_path: Path,
    dataset_keys: Optional[List[str]] = None,
    force: bool = False,
    max_workers: int = 0,
    log_dir: Optional[Path] = None,
    input_dir_override: Optional[str] = None,
    output_dir_override: Optional[str] = None,
    mlflow_enabled: Optional[bool] = None,
) -> None:
    """Extract and convert datasets to unified JSONL.

    Args:
        config_path: Path to extraction YAML config.
        dataset_keys: Specific dataset keys to extract. If None, extracts all configured datasets.
        force: When True, re-extract datasets whose output already
            exists. Defaults to False (skip already-extracted datasets).
            Also re-logs the MLflow extraction build for the same digest.
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
        mlflow_enabled: Tri-state override for MLflow extraction tracking.
            None defers to the config's `mlflow.enabled`; True/False force it
            on/off regardless of config.

    Raises:
        ValueError: If any requested key is unknown.
        RuntimeError: If one or more dataset extractions failed.
    """
    cfg = load_extract_config(config_path)

    if dataset_keys:
        unknown = set(dataset_keys) - set(cfg.datasets.keys())
        if unknown:
            raise ValueError(f"Unknown dataset keys: {', '.join(sorted(unknown))}")
        selected = {k: v for k, v in cfg.datasets.items() if k in dataset_keys}
    else:
        selected = cfg.datasets

    input_base = Path(input_dir_override) if input_dir_override else resolve_project_path(cfg.input_dir)
    output_base = Path(output_dir_override) if output_dir_override else resolve_project_path(cfg.output_dir)
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
        raise RuntimeError(f"Extraction failed for {len(failures)} dataset(s): {failed_keys}")

    enabled = bool(cfg.mlflow.get("enabled", False)) if mlflow_enabled is None else mlflow_enabled
    if enabled:
        from slm4ie.data.extract.tracking import DEFAULT_EXPERIMENT, log_extract_run

        log_extract_run(
            output_base,
            cfg.datasets,
            enabled=True,
            experiment=cfg.mlflow.get("experiment", DEFAULT_EXPERIMENT),
            tracking_uri=cfg.mlflow.get("tracking_uri"),
            force=force,
            artifact_dir=log_dir,
        )
