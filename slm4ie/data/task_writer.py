"""Shared helpers for task converters driven by `configs/data/tasks.yaml`.

The three task-family converters (`to_spans`, `to_sentiment`,
`to_superglue`) all share the same output convention — one gzipped
JSONL per declared split under
``<roots.tasks>/<task>/<dataset>/<split>.jsonl.gz``. This module
factors out the deterministic split assignment, output-existence
check, and JSONL split writer so each converter only has to provide
the per-record serialization logic.
"""

import gzip
import hashlib
import json
import logging
from contextlib import ExitStack
from pathlib import Path
from typing import IO, Any, Dict, Iterable, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: Standard split names recognized by the task pipeline.
SPLIT_TRAIN: str = "train"
SPLIT_VAL: str = "val"
SPLIT_TEST: str = "test"


def hash_split(
    key: str,
    splits: Iterable[str],
    train_pct: int = 80,
    val_pct: int = 10,
) -> str:
    """Assign *key* to a split using a deterministic hash bucket.

    The bucket is ``int(blake2s(key).hexdigest()[:8], 16) % 100``;
    buckets ``[0, train_pct)`` go to ``train``, ``[train_pct,
    train_pct + val_pct)`` go to ``val``, the remainder to ``test``.
    Only splits actually present in *splits* are returned: when a
    declared split is missing, examples falling into its bucket are
    redirected to ``train`` (or, if ``train`` is also absent, the
    first split in *splits*).

    Args:
        key: Stable identifier of the example (typically ``doc_id``
            or ``uid``).
        splits: Iterable of split names declared by the entry.
        train_pct: Bucket-percent boundary for the ``train`` split.
        val_pct: Bucket-percent width for the ``val`` split. The
            remainder (``100 - train_pct - val_pct``) goes to
            ``test``.

    Returns:
        One of the split names in *splits*.

    Raises:
        ValueError: If *splits* is empty, or if the percent
            boundaries are not in ``[0, 100]``.
    """
    splits_list = list(splits)
    if not splits_list:
        raise ValueError("`splits` must be a non-empty iterable.")
    if not (0 <= train_pct <= 100 and 0 <= val_pct <= 100):
        raise ValueError("Split percentages must be in [0, 100].")
    if train_pct + val_pct > 100:
        raise ValueError("train_pct + val_pct must not exceed 100.")

    digest = hashlib.blake2s(key.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100

    if bucket < train_pct:
        target = SPLIT_TRAIN
    elif bucket < train_pct + val_pct:
        target = SPLIT_VAL
    else:
        target = SPLIT_TEST

    if target in splits_list:
        return target
    if SPLIT_TRAIN in splits_list:
        return SPLIT_TRAIN
    return splits_list[0]


def outputs_for_splits(
    output_dir: Path,
    splits: Dict[str, str],
) -> Dict[str, Path]:
    """Resolve absolute paths for every declared split.

    Args:
        output_dir: Directory under which split files are written.
        splits: Mapping ``{split_name: filename}`` from the entry.

    Returns:
        Mapping ``{split_name: output_dir / filename}``.
    """
    return {split: output_dir / filename for split, filename in splits.items()}


def all_outputs_exist(outputs: Dict[str, Path]) -> bool:
    """Return True when every output path already exists on disk.

    Args:
        outputs: Result of `outputs_for_splits`.

    Returns:
        True when every value in *outputs* points to an existing file.
    """
    return bool(outputs) and all(path.exists() for path in outputs.values())


def _open_split_streams(
    outputs: Dict[str, Path],
    stack: ExitStack,
) -> Dict[str, IO[str]]:
    """Open writable gzip text streams for every split path.

    Args:
        outputs: Mapping from split name to absolute output path.
        stack: Caller-managed ``ExitStack`` that owns the lifetimes.

    Returns:
        Mapping from split name to a writable text stream.
    """
    streams: Dict[str, IO[str]] = {}
    for split, path in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        streams[split] = stack.enter_context(
            gzip.open(path, "wt", encoding="utf-8")
        )
    return streams


def write_jsonl_splits(
    examples: Iterable[Tuple[str, Dict[str, Any]]],
    outputs: Dict[str, Path],
) -> Dict[str, int]:
    """Stream `(split, example)` pairs to per-split gzipped JSONL files.

    Args:
        examples: Iterable of ``(split_name, example_dict)`` tuples.
            Pairs whose split is absent from *outputs* are dropped
            with a warning (counted at most once per split).
        outputs: Mapping from split name to absolute output path.

    Returns:
        Mapping ``{split_name: written_count}`` covering every split
        in *outputs*; values default to ``0`` when no example landed
        there.
    """
    counts: Dict[str, int] = {split: 0 for split in outputs}
    warned: set = set()
    with ExitStack() as stack:
        streams = _open_split_streams(outputs, stack)
        for split, example in examples:
            stream = streams.get(split)
            if stream is None:
                if split not in warned:
                    logger.warning(
                        "Dropping example for undeclared split %r "
                        "(known: %s).",
                        split, sorted(outputs.keys()),
                    )
                    warned.add(split)
                continue
            stream.write(json.dumps(example, ensure_ascii=False))
            stream.write("\n")
            counts[split] += 1
    return counts


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield parsed JSON objects from a JSONL file (plain or gzipped).

    Args:
        path: Path to a ``.jsonl`` or ``.jsonl.gz`` file.

    Yields:
        Parsed JSON records.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def find_first_existing(
    candidates: List[Path],
) -> Optional[Path]:
    """Return the first path in *candidates* that exists on disk.

    Args:
        candidates: Candidate paths in priority order.

    Returns:
        The first existing path, or ``None`` if none exist.
    """
    for path in candidates:
        if path.exists():
            return path
    return None
