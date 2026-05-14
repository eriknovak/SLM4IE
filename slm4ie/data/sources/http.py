"""HTTP(S) file downloader with streaming, resume, and retry."""

import logging
import time
from pathlib import Path
from typing import List, Tuple
from urllib.parse import urlparse

import requests
from tqdm import tqdm

from slm4ie.data.catalog import DatasetConfig
from slm4ie.data.download import DownloaderResult, _augment_with_note
from slm4ie.data.parallel import workers_quiet

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
CHUNK_SIZE = 8192
REQUEST_TIMEOUT = 30


def download(
    config: DatasetConfig, output_dir: Path, force: bool
) -> DownloaderResult:
    """Download all URLs for a dataset over HTTP(S).

    Streams each URL to disk with a progress bar, supporting resume via
    HTTP Range headers and retry with exponential backoff on transient
    request failures.

    Args:
        config: Dataset configuration with `urls` list.
        output_dir: Directory to save downloaded files.
        force: When True, re-download each URL even if `dest` already
            exists. The destination is replaced atomically via the
            `.part` rename at the end of the stream, so a crash mid
            force-download leaves the previous file intact.

    Returns:
        DownloaderResult: Outcome record with one entry per URL — either
            in `completed` or in `failed` (alongside the exception that
            ended the attempt). A failure on one URL does not abort the
            others.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    completed: List[str] = []
    failed: List[Tuple[str, BaseException]] = []
    for url in config.urls:
        filename = _extract_filename(url)
        dest = output_dir / filename
        part = output_dir / f"{filename}.part"
        try:
            _download_file(url, dest, part, force)
            completed.append(url)
        except (requests.RequestException, OSError) as exc:
            failed.append((url, _augment_with_note(exc, config.note)))

    return DownloaderResult(completed=completed, failed=failed)


def _extract_filename(url: str) -> str:
    """Extract filename from a URL, stripping query params.

    Args:
        url: The download URL.

    Returns:
        str: The filename portion of the URL path.
    """
    parsed = urlparse(url)
    return Path(parsed.path).name


def _download_file(url: str, dest: Path, part: Path, force: bool) -> None:
    """Download a single file with resume and retry support.

    Args:
        url: URL to download from.
        dest: Final destination path.
        part: Temporary .part file path.
        force: When True, clear any leftover `.part` and bypass the
            skip-if-`dest`-exists short circuit. The final rename is
            still atomic so the prior `dest` survives a mid-download
            crash.

    Raises:
        requests.RequestException: If the download fails after
            `MAX_RETRIES` attempts.
    """
    if force:
        part.unlink(missing_ok=True)
    elif dest.exists():
        logger.info("File exists, skipping: %s", dest.name)
        return

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            _stream_download(url, dest, part)
            return
        except requests.RequestException as e:
            if attempt == MAX_RETRIES:
                logger.error(
                    "Failed to download %s after %d attempts: %s",
                    dest.name,
                    MAX_RETRIES,
                    e,
                )
                raise
            wait = 2 ** (attempt - 1)
            logger.warning(
                "Download attempt %d/%d failed for %s, retrying in %ds: %s",
                attempt,
                MAX_RETRIES,
                dest.name,
                wait,
                e,
            )
            time.sleep(wait)


def _stream_download(url: str, dest: Path, part: Path) -> None:
    """Stream download with optional resume.

    Args:
        url: URL to download from.
        dest: Final destination path.
        part: Temporary .part file path.
    """
    headers = {}
    initial_size = 0
    mode = "wb"

    if part.exists():
        initial_size = part.stat().st_size
        headers["Range"] = f"bytes={initial_size}-"
        mode = "ab"
        logger.info(
            "Resuming download of %s from %d bytes",
            dest.name,
            initial_size,
        )

    with requests.get(
        url, stream=True, headers=headers, timeout=REQUEST_TIMEOUT
    ) as resp:
        resp.raise_for_status()

        if initial_size > 0 and resp.status_code != 206:
            initial_size = 0
            mode = "wb"
            logger.info(
                "Server does not support resume, restarting download of %s",
                dest.name,
            )

        total = resp.headers.get("content-length")
        total_size = int(total) + initial_size if total else None

        with (
            open(part, mode) as f,
            tqdm(
                total=total_size,
                initial=initial_size,
                unit="B",
                unit_scale=True,
                desc=dest.name,
                disable=workers_quiet(),
            ) as pbar,
        ):
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
                pbar.update(len(chunk))

    part.rename(dest)
    logger.info("Downloaded: %s", dest.name)
