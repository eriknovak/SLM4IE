"""Readers of the extracted tier.

Extraction leaves two files per dataset under `extracted/`: `<key>.jsonl`
with the text and an optional gzipped `<key>.annotations.jsonl.gz` sidecar
with the parallel-array annotations. These helpers locate the pair and join
it on the fly so no consumer materializes a merged file.
"""

import itertools
import json
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

from slm4ie.utils.io import open_text_stream


def find_dataset_files(
    processed_dir: Path,
    key: str,
) -> Optional[Tuple[Path, Optional[Path]]]:
    """Locate the text JSONL and optional annotations sidecar for a dataset.

    Args:
        processed_dir (Path): Directory containing processed files.
        key (str): Dataset key.

    Returns:
        Optional[Tuple[Path, Optional[Path]]]: A tuple of
            `(text_path, annotations_path)` where `annotations_path`
            is None when the dataset has no annotations sidecar.
            Returns None when no `<key>.jsonl` exists.
    """
    text_path = processed_dir / f"{key}.jsonl"
    if not text_path.exists():
        return None
    ann_path = processed_dir / f"{key}.annotations.jsonl.gz"
    return (text_path, ann_path if ann_path.exists() else None)


def iter_joined_records(
    text_path: Path,
    annotations_path: Optional[Path] = None,
) -> Iterator[Dict[str, Any]]:
    """Iterate records from *text_path*, attaching annotations when present.

    Both files are assumed to share document order. When the text and
    annotation records both carry a `doc_id` (or `uid`) and they
    disagree, this is treated as a hard error: a mismatch indicates
    the files were produced by different runs.

    Args:
        text_path (Path): Path to `<key>.jsonl`.
        annotations_path (Optional[Path]): Path to the annotations
            file, or None if the dataset has no annotations.

    Yields:
        Dict[str, Any]: A text record with, when available, an
            `annotations` field carrying the parallel-array payload
            (`forms`, `lemmas`, `upos`, `feats`, `space_after`,
            `sentences`). Sidecars written before the `space_after`
            array existed simply lack that key — consumers must treat
            a missing array as all-True. Stub lines (no parallel
            arrays) yield records without an `annotations` field.

    Raises:
        ValueError: If a `doc_id`/`uid` mismatch is detected or
            the two streams differ in length.
    """
    if annotations_path is None:
        with text_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
        return

    with text_path.open(encoding="utf-8") as text_fh, open_text_stream(annotations_path) as ann_fh:
        for lineno, (text_line, ann_line) in enumerate(itertools.zip_longest(text_fh, ann_fh), start=1):
            if text_line is None or ann_line is None:
                missing = "text" if text_line is None else "annotations"
                raise ValueError(f"Line counts differ at line {lineno}: {missing} stream exhausted first")

            text_record = json.loads(text_line)
            ann_record = json.loads(ann_line)

            text_id = text_record.get("doc_id")
            ann_id = ann_record.get("doc_id")
            if text_id is not None and ann_id is not None and text_id != ann_id:
                raise ValueError(f"doc_id mismatch at line {lineno}: text={text_id!r} annotations={ann_id!r}")

            text_uid = text_record.get("uid")
            ann_uid = ann_record.get("uid")
            if text_uid is not None and ann_uid is not None and text_uid != ann_uid:
                raise ValueError(f"uid mismatch at line {lineno}: text={text_uid!r} annotations={ann_uid!r}")

            ann_record.pop("doc_id", None)
            ann_record.pop("uid", None)
            if ann_record:
                text_record["annotations"] = ann_record
            yield text_record
