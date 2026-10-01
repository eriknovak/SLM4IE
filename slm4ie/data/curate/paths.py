"""Filesystem layout of a curation run, and the shard helpers every part shares.

`CuratePaths` names every folder under `<output_dir>/`: the stage folders, the
staging and scratch folders under `_partial/`, and the per-stage datatrove
logs. The helpers here only look at the tree — build a symlink view of some
datasets' shards, test whether a dataset has output, fingerprint a shard
layout, count JSONL rows — and import nothing beyond the standard library.
"""

import gzip
import hashlib
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from slm4ie.data.curate.stages import STAGE_DIRS


@dataclass
class CuratePaths:
    """Filesystem locations for one curation run.

    Attributes:
        input_folder: Folder of `<key>.jsonl` extraction outputs that
            the `convert` stage (stage 0) reads from.
        output_dir: Curation output root. Stage folders
            (`00_convert/`, `01_language/`, ...) live directly under
            this path, alongside `_partial/` and `_logs/`.
    """

    input_folder: Path
    output_dir: Path

    def stage_dir(self, stage: str) -> Path:
        """Return the folder under `output_dir` that holds *stage*'s output.

        Args:
            stage: Stage name (see `slm4ie.data.curate.stages.STAGE_NAMES`).

        Returns:
            Absolute path to the stage's output folder.

        Raises:
            KeyError: If *stage* is not a known stage name.
        """
        return self.output_dir / STAGE_DIRS[stage]

    def staging_dir(self, stage: str) -> Path:
        """Return the folder a stage builds into before its output is promoted.

        Args:
            stage: Stage name (see `slm4ie.data.curate.stages.STAGE_NAMES`).

        Returns:
            `<output_dir>/_partial/<stage folder>`.
        """
        return self.output_dir / "_partial" / STAGE_DIRS[stage]

    def scratch_dir(self, stage: str) -> Path:
        """Return the folder for a stage's intermediate files, kept beside its staging folder.

        The dedup stages pass signatures and duplicate lists between their
        steps here. It survives a crash so a resumed run keeps finished
        steps, and is never promoted with the output.

        Args:
            stage: Stage name (see `slm4ie.data.curate.stages.STAGE_NAMES`).

        Returns:
            `<output_dir>/_partial/<stage folder>.scratch`.
        """
        return self.output_dir / "_partial" / f"{STAGE_DIRS[stage]}.scratch"

    def logs_dir(self, stage: str) -> Path:
        """Return the per-stage logging directory.

        Args:
            stage: Stage name (one of `STAGE_NAMES`).

        Returns:
            `<output_dir>/_logs/<stage>` — datatrove's `logging_dir`
            for that stage's executor chain.
        """
        return self.output_dir / "_logs" / stage


def count_jsonl_rows(path: Path) -> int:
    """Count newline-delimited records in a `.jsonl` or `.jsonl.gz` file.

    Counts newline bytes over the raw (decompressed) stream in megabyte
    chunks, skipping per-line UTF-8 decoding — far faster at corpus scale,
    where a full backfill reads tens of millions of rows. Every JSONL
    record is newline-terminated, so a trailing newline is added when the
    final byte is not one (a file with no trailing newline still counts
    its last record).

    Args:
        path: Path to a plain or gzipped JSONL file.

    Returns:
        The number of rows (documents) in the file.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    count = 0
    last = b"\n"
    with opener(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            count += chunk.count(b"\n")
            last = chunk[-1:]
    if last not in (b"\n", b""):
        count += 1
    return count


def filter_stage_subset(stage_dir: Path, keys: List[str], holder: Optional[Path] = None) -> Path:
    """Materialize a folder of symlinks restricted to *keys* under *stage_dir*.

    Args:
        stage_dir: A scoped stage's output folder (e.g.
            `<output_dir>/01_language/`).
        keys: Dataset keys to expose.
        holder: Folder to build the view in, replacing any previous view
            there; a fresh tempdir when omitted. A fixed path keeps a
            resumed corpus stage reading the same file paths.

    Returns:
        Path to the folder mirroring the requested keys via symlinks, so a
        downstream stage's reader walks only the subset's shards.

    Raises:
        FileNotFoundError: If any requested shard folder is missing or
            empty under *stage_dir*.
    """
    missing: List[str] = []
    for key in keys:
        src = stage_dir / key
        if not src.is_dir() or not any(src.glob("*.jsonl.gz")):
            missing.append(key)
    if missing:
        raise FileNotFoundError(
            f"No converted shard folder(s) under {stage_dir} for dataset(s): " + ", ".join(repr(k) for k in missing)
        )
    if holder is None:
        holder = Path(tempfile.mkdtemp(prefix="slm4ie-pretrain-subset-"))
    else:
        shutil.rmtree(holder, ignore_errors=True)
        holder.mkdir(parents=True)
    try:
        for key in keys:
            src = stage_dir / key
            holder_key = holder / key
            holder_key.mkdir()
            for shard in src.glob("*.jsonl.gz"):
                (holder_key / shard.name).symlink_to(shard.resolve())
    except BaseException:
        shutil.rmtree(holder, ignore_errors=True)
        raise
    return holder


def has_stage_output(stage_dir: Path, key: str) -> bool:
    """Return True if *key* has shard output under *stage_dir*.

    Used to drop datasets that produced nothing upstream — declared in the
    roster but never downloaded, or fully filtered out by an earlier stage
    — before a scoped stage tries to read their (nonexistent) shards.

    Args:
        stage_dir: A stage's output folder (e.g. `<output_dir>/00_convert`).
        key: Dataset key to check.

    Returns:
        True if `<stage_dir>/<key>/` exists and holds `.jsonl.gz` shards.
    """
    folder = stage_dir / key
    return folder.is_dir() and any(folder.glob("*.jsonl.gz"))


def shard_layout(view: Path) -> Tuple[int, str]:
    """Count and hash the layout of the shards a corpus stage will read.

    Args:
        view: Symlink view of the stage's input, one folder per dataset.

    Returns:
        Tuple `(shard_count, digest)`; the digest covers each shard's relative
        path and size, which fix how datatrove assigns shards to tasks.
    """
    shards = sorted(view.glob("*/*.jsonl.gz"))
    digest = hashlib.sha256()
    for shard in shards:
        digest.update(f"{shard.relative_to(view)}\t{shard.stat().st_size}\n".encode("utf-8"))
    return len(shards), digest.hexdigest()


def human_bytes(num: int) -> str:
    """Render a byte count as a human-readable string.

    Args:
        num: A non-negative byte count.

    Returns:
        The count scaled to the largest binary unit below 1024, with
        one decimal place for KiB and above (e.g. 1536 -> `"1.5 KiB"`)
        and no decimal for plain bytes (e.g. 512 -> `"512 B"`).
    """
    size = float(num)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if size < 1024.0:
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} EiB"
