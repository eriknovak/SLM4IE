"""Archive decompression utilities for dataset files."""

import gzip
import logging
import lzma
import shutil
import tarfile
import zipfile
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)


def _extract_gzip(archive_path: Path, output_dir: Path) -> Path:
    """Extract a .gz file to output_dir.

    Strips the .gz extension to determine the output filename.
    Skips extraction if the output file already exists.

    Args:
        archive_path (Path): Path to the .gz file.
        output_dir (Path): Directory to write the extracted file.

    Returns:
        Path: Path to the extracted file.
    """
    output_path = output_dir / archive_path.stem
    if output_path.exists():
        logger.info(
            "Skipping extraction, output already exists: %s",
            output_path,
        )
        return output_path

    logger.info(
        "Extracting %s -> %s", archive_path, output_path
    )
    total = archive_path.stat().st_size
    with open(archive_path, "rb") as raw_in:
        with tqdm.wrapattr(
            raw_in,
            "read",
            total=total,
            unit="B",
            unit_scale=True,
            desc=archive_path.name,
        ) as wrapped_in:
            with gzip.GzipFile(fileobj=wrapped_in, mode="rb") as f_in:  # type: ignore[arg-type]
                with open(output_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

    return output_path


def _extract_xz(archive_path: Path, output_dir: Path) -> Path:
    """Extract a .xz file to output_dir.

    Strips the .xz extension to determine the output filename.
    Skips extraction if the output file already exists.

    Args:
        archive_path (Path): Path to the .xz file.
        output_dir (Path): Directory to write the extracted file.

    Returns:
        Path: Path to the extracted file.
    """
    output_path = output_dir / archive_path.stem
    if output_path.exists():
        logger.info(
            "Skipping extraction, output already exists: %s",
            output_path,
        )
        return output_path

    logger.info(
        "Extracting %s -> %s", archive_path, output_path
    )
    total = archive_path.stat().st_size
    with open(archive_path, "rb") as raw_in:
        with tqdm.wrapattr(
            raw_in,
            "read",
            total=total,
            unit="B",
            unit_scale=True,
            desc=archive_path.name,
        ) as wrapped_in:
            with lzma.open(wrapped_in, "rb") as f_in:  # type: ignore[arg-type]
                with open(output_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

    return output_path


def _extract_zip(archive_path: Path, output_dir: Path) -> Path:
    """Extract a .zip file to output_dir.

    Args:
        archive_path (Path): Path to the .zip file.
        output_dir (Path): Directory to extract contents into.

    Returns:
        Path: The output_dir path.
    """
    logger.info(
        "Extracting %s -> %s", archive_path, output_dir
    )
    with zipfile.ZipFile(archive_path, "r") as zf:
        members = zf.infolist()
        for member in tqdm(
            members,
            desc=archive_path.name,
            unit="file",
        ):
            zf.extract(member, output_dir)
    return output_dir


def _extract_tar(archive_path: Path, output_dir: Path) -> Path:
    """Extract a .tar.gz or .tgz file to output_dir.

    Uses filter="data" for safe extraction.

    Args:
        archive_path (Path): Path to the tar archive.
        output_dir (Path): Directory to extract contents into.

    Returns:
        Path: The output_dir path.
    """
    logger.info(
        "Extracting %s -> %s", archive_path, output_dir
    )
    with tarfile.open(archive_path, "r:gz") as tf:
        members = tf.getmembers()
        for member in tqdm(
            members,
            desc=archive_path.name,
            unit="file",
        ):
            tf.extract(member, output_dir, filter="data")
    return output_dir


def extract_archive(
    archive_path: Path, output_dir: Path
) -> Path:
    """Extract an archive file to the specified output directory.

    Detects format by filename extension. Supported formats:
    - .gz (non-tar): gunzip to output_dir, strip .gz extension.
      Skips if output already exists.
    - .xz: lzma-decompress to output_dir, strip .xz extension.
      Skips if output already exists.
    - .zip: extract all contents to output_dir.
    - .tar.gz / .tgz: extract with filter="data" to output_dir.

    Args:
        archive_path (Path): Path to the archive file.
        output_dir (Path): Directory to extract contents into.

    Returns:
        Path: Path to the extracted file (for .gz / .xz) or
            output_dir (for .zip, .tar.gz, .tgz).

    Raises:
        ValueError: If the archive format is not supported.
    """
    name = archive_path.name

    if name.endswith(".tar.gz") or name.endswith(".tgz"):
        return _extract_tar(archive_path, output_dir)
    elif name.endswith(".gz"):
        return _extract_gzip(archive_path, output_dir)
    elif name.endswith(".xz"):
        return _extract_xz(archive_path, output_dir)
    elif name.endswith(".zip"):
        return _extract_zip(archive_path, output_dir)
    else:
        raise ValueError(
            f"Unsupported archive format: {archive_path.suffix}"
        )
