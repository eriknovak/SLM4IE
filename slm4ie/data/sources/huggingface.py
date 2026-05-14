"""HuggingFace Hub dataset downloader."""

import logging
import os
import shutil
from pathlib import Path
from typing import List, Tuple

from datasets import load_dataset

from slm4ie.data.catalog import DatasetConfig
from slm4ie.data.download import DownloaderResult, _augment_with_note

logger = logging.getLogger(__name__)


def download(
    config: DatasetConfig, output_dir: Path, force: bool
) -> DownloaderResult:
    """Download HuggingFace dataset configs to an output directory.

    Each declared config (typically a language code) is saved as a
    subdirectory in Arrow format via `save_to_disk()`. To make
    completion atomic across process crashes, the download streams into
    a sibling `<cfg>.partial` directory and is renamed to `<cfg>.ready`
    once the writer finishes. The final swap into `<cfg>` removes any
    prior data and renames `.ready` into place. On entry, an orphaned
    `.ready` from a crashed prior run is finished, and any leftover
    `.partial` is discarded.

    Args:
        config: Dataset configuration with `repo_id` and `configs`.
        output_dir: Directory under which per-config subdirectories are
            written.
        force: When True, re-download each config even if `save_path`
            already exists. The previous data is removed only after the
            new download has finished writing into `.partial` and been
            promoted to `.ready`, so a crash mid force-download leaves
            the prior data (or a `.ready` to finish on the next run).

    Returns:
        DownloaderResult: Outcome record with one entry per declared
            config in either `completed` or `failed`. A failure on one
            config (e.g., gated repo without `HF_TOKEN`) does not abort
            the remaining configs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    completed: List[str] = []
    failed: List[Tuple[str, BaseException]] = []
    configs = config.configs or []
    for cfg_name in configs:
        try:
            save_path = output_dir / cfg_name
            staging = save_path.parent / f"{save_path.name}.partial"
            ready = save_path.parent / f"{save_path.name}.ready"

            # Recovery on entry: complete any half-swapped prior run,
            # then discard orphan staging dirs.
            if ready.exists():
                logger.info(
                    "Recovering '%s': finishing swap from .ready marker",
                    cfg_name,
                )
                if save_path.exists():
                    shutil.rmtree(save_path)
                os.rename(ready, save_path)
            if staging.exists():
                logger.info(
                    "Removing orphan staging dir for '%s'", cfg_name,
                )
                shutil.rmtree(staging)

            if save_path.exists() and not force:
                logger.info(
                    "Config '%s' already exists, skipping",
                    cfg_name,
                )
                completed.append(cfg_name)
                continue

            logger.info(
                "Downloading %s config '%s' from HuggingFace",
                config.name,
                cfg_name,
            )
            staging.parent.mkdir(parents=True, exist_ok=True)
            ds = load_dataset(config.repo_id, cfg_name)
            ds.save_to_disk(str(staging))
            os.rename(staging, ready)
            if save_path.exists():
                shutil.rmtree(save_path)
            os.rename(ready, save_path)
            logger.info(
                "Saved %s/%s to %s",
                config.repo_id,
                cfg_name,
                save_path,
            )
            completed.append(cfg_name)
        except Exception as exc:
            failed.append(
                (cfg_name, _augment_with_note(exc, config.note))
            )

    return DownloaderResult(completed=completed, failed=failed)
