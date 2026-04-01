"""Dataset fetchers for HuggingFace Hub, Clarin.si, and other sources."""

import abc
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
import yaml
from datasets import load_dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for a single dataset.

    Attributes:
        key: Config key identifier (e.g., 'classla_web_sl').
        name: Human-readable dataset name.
        enabled: Whether to include in default download.
        source: Source type ('clarin' or 'huggingface').
        urls: Download URLs (CLARIN datasets).
        output_dir: Subdirectory name under base output dir.
        manual: Whether this requires manual download.
        repo_id: HuggingFace repository ID.
        configs: HuggingFace dataset config names.
        note: Informational note.
    """

    key: str
    name: str
    enabled: bool = True
    source: str = ""
    urls: List[str] = field(default_factory=list)
    output_dir: str = ""
    manual: bool = False
    repo_id: Optional[str] = None
    configs: Optional[List[str]] = None
    note: Optional[str] = None

    @classmethod
    def from_dict(cls, key: str, data: Dict) -> "DatasetConfig":
        """Create a DatasetConfig from a config dictionary.

        Args:
            key: The dataset key identifier.
            data: Dictionary of config values.

        Returns:
            DatasetConfig: Populated config instance.
        """
        return cls(
            key=key,
            name=data.get("name", key),
            enabled=data.get("enabled", True),
            source=data.get("source", ""),
            urls=data.get("urls", []),
            output_dir=data.get("output_dir", ""),
            manual=data.get("manual", False),
            repo_id=data.get("repo_id"),
            configs=data.get("configs"),
            note=data.get("note"),
        )


def load_config(
    config_path: Path,
) -> Tuple[str, Dict[str, DatasetConfig]]:
    """Load dataset download configuration from YAML file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Tuple of (output_dir, dict of dataset key to DatasetConfig).

    Raises:
        FileNotFoundError: If config file does not exist.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    output_dir = raw.get("output_dir", "data/raw")
    datasets: Dict[str, DatasetConfig] = {}

    for key, data in raw.get("datasets", {}).items():
        datasets[key] = DatasetConfig.from_dict(key, data)

    return output_dir, datasets


class BaseDownloader(abc.ABC):
    """Abstract base class for dataset downloaders."""

    @abc.abstractmethod
    def download(self, config: DatasetConfig, output_dir: Path) -> Path:
        """Download dataset files to output directory.

        Args:
            config: Dataset configuration.
            output_dir: Directory to save downloaded files.

        Returns:
            Path: The output directory.
        """


class ClarinDownloader(BaseDownloader):
    """Downloads datasets from CLARIN.SI repository via HTTP.

    Supports streaming downloads with progress bars, resume via
    HTTP Range headers, and retry with exponential backoff.
    """

    MAX_RETRIES = 3
    CHUNK_SIZE = 8192

    def download(self, config: DatasetConfig, output_dir: Path) -> Path:
        """Download all URLs for a CLARIN dataset.

        Args:
            config: Dataset configuration with urls list.
            output_dir: Directory to save files.

        Returns:
            Path: The output directory.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        for url in config.urls:
            filename = self._extract_filename(url)
            dest = output_dir / filename
            part = output_dir / f"{filename}.part"
            self._download_file(url, dest, part)

        return output_dir

    def _extract_filename(self, url: str) -> str:
        """Extract filename from a URL, stripping query params.

        Args:
            url: The download URL.

        Returns:
            str: The filename portion of the URL path.
        """
        parsed = urlparse(url)
        return Path(parsed.path).name

    def _download_file(self, url: str, dest: Path, part: Path) -> None:
        """Download a single file with resume and retry support.

        Args:
            url: URL to download from.
            dest: Final destination path.
            part: Temporary .part file path.
        """
        if dest.exists():
            logger.info("File exists, skipping: %s", dest.name)
            return

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                self._stream_download(url, dest, part)
                return
            except requests.RequestException as e:
                if attempt == self.MAX_RETRIES:
                    logger.error(
                        "Failed to download %s after %d " "attempts: %s",
                        dest.name,
                        self.MAX_RETRIES,
                        e,
                    )
                    raise
                wait = 2 ** (attempt - 1)
                logger.warning(
                    "Download attempt %d/%d failed for " "%s, retrying in %ds: %s",
                    attempt,
                    self.MAX_RETRIES,
                    dest.name,
                    wait,
                    e,
                )
                time.sleep(wait)

    def _stream_download(self, url: str, dest: Path, part: Path) -> None:
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

        with requests.get(url, stream=True, headers=headers, timeout=30) as resp:
            resp.raise_for_status()

            if initial_size > 0 and resp.status_code != 206:
                initial_size = 0
                mode = "wb"
                logger.info(
                    "Server does not support resume, " "restarting download of %s",
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
                ) as pbar,
            ):
                for chunk in resp.iter_content(chunk_size=self.CHUNK_SIZE):
                    f.write(chunk)
                    pbar.update(len(chunk))

        part.rename(dest)
        logger.info("Downloaded: %s", dest.name)


class HuggingFaceDownloader(BaseDownloader):
    """Downloads datasets from Hugging Face Hub.

    Uses the datasets library to download and save in Arrow
    format. For gated datasets, the library reads HF_TOKEN
    from the environment.
    """

    def download(self, config: DatasetConfig, output_dir: Path) -> Path:
        """Download HuggingFace dataset configs to output dir.

        Each config (language code) is saved as a subdirectory
        in Arrow format via save_to_disk(). If download fails
        (e.g., gated dataset without token), logs a warning
        and skips that config.

        Args:
            config: Dataset config with repo_id and configs.
            output_dir: Directory to save dataset(s).

        Returns:
            Path: The output directory.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        configs = config.configs or []
        for cfg_name in configs:
            save_path = output_dir / cfg_name
            if save_path.exists():
                logger.info(
                    "Config '%s' already exists, skipping",
                    cfg_name,
                )
                continue

            logger.info(
                "Downloading %s config '%s' from HuggingFace",
                config.name,
                cfg_name,
            )
            try:
                ds = load_dataset(config.repo_id, cfg_name)
                ds.save_to_disk(str(save_path))
                logger.info(
                    "Saved %s/%s to %s",
                    config.repo_id,
                    cfg_name,
                    save_path,
                )
            except Exception as e:
                note = config.note or ""
                logger.warning(
                    "Failed to download %s config '%s'" ": %s. %s",
                    config.name,
                    cfg_name,
                    e,
                    note,
                )

        return output_dir


def download_datasets(
    config_path: Path,
    dataset_keys: Optional[List[str]] = None,
    force: bool = False,
    output_dir_override: Optional[str] = None,
) -> None:
    """Download datasets according to configuration.

    Args:
        config_path: Path to the YAML config file.
        dataset_keys: Specific dataset keys to download.
            If None, downloads all enabled datasets.
        force: Re-download even if output exists.
        output_dir_override: Override base output directory.

    Raises:
        ValueError: If any requested dataset key is unknown.
    """
    base_output_dir, datasets = load_config(config_path)
    if output_dir_override:
        base_output_dir = output_dir_override

    if dataset_keys:
        unknown = set(dataset_keys) - set(datasets.keys())
        if unknown:
            raise ValueError(f"Unknown dataset keys: " f"{', '.join(unknown)}")
        selected = {k: v for k, v in datasets.items() if k in dataset_keys}
    else:
        selected = {k: v for k, v in datasets.items() if v.enabled}

    clarin_dl = ClarinDownloader()
    hf_dl = HuggingFaceDownloader()

    for key, config in selected.items():
        output_path = Path(base_output_dir) / config.output_dir

        if not config.enabled:
            note = config.note or "No details available."
            logger.warning("Dataset '%s' is disabled: %s", key, note)
            continue

        if config.manual:
            if _dir_has_files(output_path):
                logger.info(
                    "Manual dataset '%s' found at %s",
                    key,
                    output_path,
                )
            else:
                note = config.note or "Manual download required."
                logger.warning(
                    "Dataset '%s' requires manual " "download: %s",
                    key,
                    note,
                )
            continue

        if not force and _dir_has_files(output_path):
            logger.info(
                "Dataset '%s' already exists at %s, " "skipping",
                key,
                output_path,
            )
            continue

        logger.info(
            "Downloading '%s' (%s) from %s",
            config.name,
            key,
            config.source,
        )

        if config.source == "clarin":
            clarin_dl.download(config, output_path)
        elif config.source == "huggingface":
            hf_dl.download(config, output_path)
        else:
            logger.error(
                "Unknown source '%s' for dataset '%s'",
                config.source,
                key,
            )


def _dir_has_files(path: Path) -> bool:
    """Check if a directory exists and contains entries.

    Args:
        path: Directory path to check.

    Returns:
        bool: True if directory exists and is non-empty.
    """
    if not path.exists():
        return False
    return any(path.iterdir())
