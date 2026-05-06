"""HuggingFace Arrow dataset extractor for the SLM4IE pipeline.

Reads HuggingFace Dataset and DatasetDict objects that were
serialized to disk via dataset.save_to_disk(). The input_dir must
contain one subdirectory per dataset config (e.g. one per language
shard or split group).

Example:
    On-disk layout:

        input_dir/
          sl/                    # config 1 (e.g. C4 Slovene)
            dataset_info.json
            state.json
            data-00000-of-00001.arrow
          hr/                    # config 2 (e.g. C4 Croatian)
            ...

    Each row is a dict, e.g. for AllenAI C4:

        {
          "text": "Dober dan, kako ste?",
          "timestamp": datetime(2019, 4, 25, 12, 34, 56),
          "url": "https://example.com/page"
        }

    Schema mapping:
        text:        row["text"] (rows with empty/missing text
                     are skipped).
        source:      provided by caller.
        domain:      provided by caller.
        doc_id:      not produced.
        metadata:    every other column (non-None), with
                     datetime/date values converted to ISO 8601
                     strings.
        annotations: not produced.
"""

import datetime
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List

from datasets import load_from_disk

from slm4ie.data.extractors import BaseExtractor, register_extractor
from slm4ie.data.schema import Document

logger = logging.getLogger(__name__)


def _to_jsonable(value: Any) -> Any:
    """Convert non-JSON-native values to serializable equivalents.

    Currently handles datetime and date values (e.g. C4's timestamp
    column) by emitting ISO 8601 strings. Other values are returned
    unchanged.

    Args:
        value (Any): Arbitrary value from a HuggingFace dataset row.

    Returns:
        Any: A JSON-serializable representation of the value.
    """
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    return value


class HuggingFaceExtractor(BaseExtractor):
    """Extracts Documents from HuggingFace Arrow datasets saved to disk.

    Expects input_dir to contain config subdirectories, each saved
    via dataset.save_to_disk(). Handles both Dataset and DatasetDict
    objects. No annotations are produced; extra columns are stored
    in metadata.
    """

    def extract(
        self,
        input_dir: Path,
        source: str,
        domain: str,
    ) -> Iterator[Document]:
        """Yield Documents from all Arrow config dirs in input_dir.

        Each immediate subdirectory of input_dir is treated as a
        separate dataset config (e.g. a language shard). If loading
        a config dir fails, a warning is logged and that dir is
        skipped.

        Args:
            input_dir (Path): Directory whose subdirectories are
                Arrow datasets saved with save_to_disk().
            source (str): Dataset key assigned to every Document.
            domain (str): Domain label assigned to every Document.

        Yields:
            Document: One document per non-empty row.
        """
        config_dirs: List[Path] = sorted(
            p for p in input_dir.iterdir() if p.is_dir()
        )

        for config_dir in config_dirs:
            try:
                ds = load_from_disk(str(config_dir))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to load dataset from %s: %s", config_dir, exc
                )
                continue

            yield from self._yield_from_dataset(ds, source, domain)

    def _yield_from_dataset(
        self,
        ds: Any,
        source: str,
        domain: str,
    ) -> Iterator[Document]:
        """Yield Documents from a Dataset or DatasetDict.

        Args:
            ds (Any): A Dataset or DatasetDict instance.
            source (str): Dataset key.
            domain (str): Domain label.

        Yields:
            Document: One document per non-empty row.
        """
        if isinstance(ds.column_names, dict):
            # DatasetDict: column_names is {split: [col, ...]}
            for split in ds.keys():
                yield from self._yield_from_split(
                    ds[split], source, domain
                )
        else:
            yield from self._yield_from_split(ds, source, domain)

    def _yield_from_split(
        self,
        split: Any,
        source: str,
        domain: str,
    ) -> Iterator[Document]:
        """Yield Documents from a single Dataset split.

        Args:
            split (Any): A Dataset instance (single split).
            source (str): Dataset key.
            domain (str): Domain label.

        Yields:
            Document: One document per non-empty row.
        """
        for row in split:
            text = row.get("text") or ""
            if not text:
                continue

            metadata: Dict[str, Any] = {
                k: _to_jsonable(v)
                for k, v in row.items()
                if k != "text" and v is not None
            }

            yield Document(
                text=text,
                source=source,
                domain=domain,
                metadata=metadata,
            )


register_extractor("huggingface", HuggingFaceExtractor)
