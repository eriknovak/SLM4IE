r"""External per-document metadata loaded from a TSV/CSV table.

Some datasets (oss, kas, kzb) ship a flat per-document metadata table
alongside their text files. This module loads the table once and serves
per-document attribute dicts that extractors merge into `Document.metadata`.

Example:
    YAML config entry:

        oss:
          extractor: conllu
          domain: scientific
          metadata:
            path: OSS.CoNLL-U/OSS-metadata.tsv
            key_column: id
            key_from: filename_stem
            key_pattern: '^oss-(\d+)$'    # filenames look like oss-10000.conllu
            fields:
              cerif: cerif                # tsv column -> metadata field
              udc: udc
              type: doctype
            splits:
              cerif: '|'                  # split into list on '|'

    Lookup:

        cfg = ds_cfg["metadata"]
        lookup = MetadataLookup.from_config(input_dir, cfg)
        row = lookup.get_for_path(Path("oss-10000.conllu"))
        # row == {"cerif": ["P000", "T270"], "udc": "502(043)", "doctype": "Diplomsko delo"}
"""


import csv
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional, Pattern

from slm4ie.data.io_utils import open_text_stream

logger = logging.getLogger(__name__)

#: Values that should be filtered out as missing/NA. Includes empty
#: strings and the conventional ``"-"`` marker used by OSS/KAS TSVs.
_NA_VALUES = frozenset({"", "-"})

#: Supported strategies for deriving the lookup key from a source file.
_KEY_FROM_VALID = frozenset({"filename_stem"})


class MetadataLookup:
    """Per-document metadata loaded from an external TSV, indexed by key.

    The class name is intentionally distinct from ``Document.metadata``
    so it does not shadow that attribute in tracebacks or imports.

    Attributes:
        key_column: Column in the TSV that holds the lookup key.
        fields: Mapping of source column name to target metadata field
            name (allows renaming on the way in).
        splits: Optional mapping of source column to separator string;
            matched columns are split into lists.
        key_from: Strategy used to derive a lookup key from a file path.
            Currently only ``"filename_stem"`` is supported.
        key_pattern: Optional compiled regex applied to the derived key
            before lookup; the first capture group becomes the actual
            lookup key. Used by OSS where filename stems look like
            ``oss-10000`` but the TSV ``id`` column holds just ``10000``.
    """

    key_column: str
    fields: Dict[str, str]
    splits: Dict[str, str]
    key_from: str
    key_pattern: Optional[Pattern[str]]

    def __init__(
        self,
        path: Path,
        key_column: str,
        fields: Dict[str, str],
        splits: Optional[Dict[str, str]] = None,
        key_from: str = "filename_stem",
        key_pattern: Optional[str] = None,
    ) -> None:
        """Load and index a metadata TSV.

        Args:
            path: Path to a tab-separated file. ``.gz`` is handled
                transparently via :func:`open_text_stream`.
            key_column: Column to index rows by. Must be present in the
                TSV header.
            fields: Mapping ``{tsv_column: metadata_field}``. Only these
                columns appear in ``get()`` results, renamed as needed.
            splits: Optional ``{tsv_column: separator}`` for list-typed
                columns (e.g. pipe-separated keyword lists).
            key_from: Strategy for deriving the lookup key from a file
                path. Currently only ``"filename_stem"``.
            key_pattern: Optional regex applied to the derived key; the
                first capture group becomes the lookup key.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If ``key_from`` is unsupported, ``key_pattern``
                lacks a capture group, or ``key_column`` is missing
                from the TSV header.
        """
        if not path.exists():
            raise FileNotFoundError(f"Metadata table not found: {path}")
        if key_from not in _KEY_FROM_VALID:
            raise ValueError(
                f"Unsupported key_from={key_from!r}; "
                f"expected one of {sorted(_KEY_FROM_VALID)}"
            )

        self.key_column = key_column
        self.fields = dict(fields)
        self.splits = dict(splits or {})
        self.key_from = key_from
        self.key_pattern = re.compile(key_pattern) if key_pattern else None
        if self.key_pattern is not None and self.key_pattern.groups < 1:
            raise ValueError(
                f"key_pattern {key_pattern!r} must contain at least one "
                f"capture group"
            )

        self._rows: Dict[str, Dict[str, Any]] = {}
        self._load(path)

    @classmethod
    def from_config(
        cls,
        input_dir: Path,
        cfg: Dict[str, Any],
    ) -> "MetadataLookup":
        """Build a lookup from a YAML ``metadata:`` block.

        Args:
            input_dir: Per-dataset input directory (the TSV ``path``
                in ``cfg`` is resolved relative to this).
            cfg: Parsed YAML block with keys ``path``, ``key_column``,
                ``fields``, and optionally ``splits``, ``key_from``,
                ``key_pattern``.

        Returns:
            MetadataLookup: A loaded, ready-to-query instance.

        Raises:
            KeyError: If required keys are missing from ``cfg``.
        """
        return cls(
            path=input_dir / cfg["path"],
            key_column=cfg["key_column"],
            fields=cfg["fields"],
            splits=cfg.get("splits"),
            key_from=cfg.get("key_from", "filename_stem"),
            key_pattern=cfg.get("key_pattern"),
        )

    def _load(self, path: Path) -> None:
        """Read the TSV and populate the row index.

        Args:
            path: Path to the TSV file.

        Raises:
            ValueError: If ``self.key_column`` is missing from the header.
        """
        with open_text_stream(path) as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            if reader.fieldnames is None or self.key_column not in reader.fieldnames:
                raise ValueError(
                    f"Metadata table {path} is missing key column "
                    f"{self.key_column!r}; got headers={reader.fieldnames}"
                )
            for row in reader:
                key = (row.get(self.key_column) or "").strip()
                if not key:
                    continue
                self._rows[key] = row
        logger.info(
            "Loaded %d metadata rows from %s (key=%s)",
            len(self._rows),
            path,
            self.key_column,
        )

    def get(self, key: str) -> Dict[str, Any]:
        """Look up a row by key and return the renamed/split fields.

        Args:
            key: Value to match against the configured ``key_column``.

        Returns:
            Dict[str, Any]: Metadata fields for the row, with empty/NA
                values filtered out. ``{}`` when no row matches.
        """
        row = self._rows.get(key)
        if row is None:
            return {}
        return self._project_row(row)

    def get_for_path(self, filepath: Path) -> Dict[str, Any]:
        """Derive a key from ``filepath`` and look up the row.

        Args:
            filepath: Source file currently being extracted.

        Returns:
            Dict[str, Any]: Metadata fields, or ``{}`` if no row matches.
        """
        derived = self._derive_key(filepath)
        if derived is None:
            return {}
        return self.get(derived)

    def _derive_key(self, filepath: Path) -> Optional[str]:
        """Compute the lookup key for ``filepath`` per configured strategy.

        Args:
            filepath: Source file currently being extracted.

        Returns:
            Optional[str]: The lookup key, or None when ``key_pattern``
                does not match the derived stem.
        """
        # Only "filename_stem" is currently supported; validated in __init__.
        derived = filepath.stem
        if self.key_pattern is None:
            return derived
        match = self.key_pattern.search(derived)
        if match is None:
            return None
        return match.group(1)

    def _project_row(self, row: Dict[str, str]) -> Dict[str, Any]:
        """Project a raw row through ``fields`` and ``splits``.

        Drops NA/empty values, renames columns to their target metadata
        field names, and applies list splits where configured.

        Args:
            row: Raw TSV row as a dict of strings.

        Returns:
            Dict[str, Any]: The projected metadata dict.
        """
        out: Dict[str, Any] = {}
        for src_col, dst_field in self.fields.items():
            raw = row.get(src_col)
            if raw is None:
                continue
            value = raw.strip()
            if value in _NA_VALUES:
                continue

            sep = self.splits.get(src_col)
            if sep:
                parts = [p.strip() for p in value.split(sep)]
                parts = [p for p in parts if p and p not in _NA_VALUES]
                if not parts:
                    continue
                out[dst_field] = parts
            else:
                out[dst_field] = value
        return out
