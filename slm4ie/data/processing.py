"""Data cleaning, formatting, and splitting utilities."""

import gzip
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# Import extractors to trigger registration
import slm4ie.data.extractors.conllu  # noqa: F401
import slm4ie.data.extractors.huggingface  # noqa: F401
import slm4ie.data.extractors.jsonl  # noqa: F401
import slm4ie.data.extractors.macocu  # noqa: F401
import slm4ie.data.extractors.tei  # noqa: F401
import slm4ie.data.extractors.text  # noqa: F401
from slm4ie.data.extract import extract_archive
from slm4ie.data.extractors import get_extractor

logger = logging.getLogger(__name__)


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


def extract_datasets(config_path: Path, dataset_keys: Optional[List[str]] = None) -> None:
    """Extract and convert datasets to unified JSONL.

    Args:
        config_path: Path to extraction YAML config.
        dataset_keys: Specific dataset keys to extract. If None, extracts all configured datasets.

    Raises:
        ValueError: If any requested key is unknown.
    """
    cfg = load_extraction_config(config_path)

    if dataset_keys:
        unknown = set(dataset_keys) - set(cfg.datasets.keys())
        if unknown:
            raise ValueError(f"Unknown dataset keys: {', '.join(sorted(unknown))}")
        selected = {k: v for k, v in cfg.datasets.items() if k in dataset_keys}
    else:
        selected = cfg.datasets

    output_base = Path(cfg.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    for key, ds_cfg in selected.items():
        extractor_name = ds_cfg["extractor"]
        domain = ds_cfg["domain"]
        input_dir = Path(cfg.input_dir) / key

        if not input_dir.exists():
            logger.warning("Input dir not found for '%s': %s", key, input_dir)
            continue

        # Decompress any archives before extraction
        for archive in sorted(input_dir.iterdir()):
            if archive.name.endswith(
                (".gz", ".xz", ".zip", ".tgz", ".tar.gz")
            ):
                extract_archive(archive, input_dir)

        logger.info("Extracting '%s' with %s extractor", key, extractor_name)

        extractor = get_extractor(extractor_name)
        text_file = output_base / f"{key}.jsonl"
        ann_file = output_base / f"{key}.annotations.jsonl.gz"

        count = 0
        has_annotations = False

        with open(text_file, "w", encoding="utf-8") as tf:
            ann_fh = None
            try:
                for doc in extractor.extract(input_dir, key, domain):
                    tf.write(doc.to_jsonl_line())
                    tf.write("\n")

                    ann_line = doc.to_annotation_line()
                    if ann_line is not None:
                        if ann_fh is None:
                            ann_fh = gzip.open(
                                ann_file, "wt", encoding="utf-8"
                            )
                            has_annotations = True
                        ann_fh.write(ann_line)
                        ann_fh.write("\n")

                    count += 1
            finally:
                if ann_fh is not None:
                    ann_fh.close()

        logger.info(
            "Extracted %d documents from '%s' -> %s%s",
            count,
            key,
            text_file,
            f" + {ann_file}" if has_annotations else "",
        )
