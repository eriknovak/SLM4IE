"""Verify a finished pretokenized output against its manifest.

The check reads every sample of every format back, token ids as numpy arrays,
and confirms four things: each format holds the token and sample totals the
manifest records, every stored `len` equals its sample's id count, every format
holds the same samples in the same order, and no document has samples in both
splits. It also warns, without failing, when the validation split is smaller
than the configured minimum.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import zip_longest
from pathlib import Path

import numpy as np

from slm4ie.data.pretokenize.formats import StoredRow, get_format
from slm4ie.data.pretokenize.run import MANIFEST_FILE
from slm4ie.data.pretokenize.splits import TRAIN, VALIDATION

# ===============================
# Report
# ===============================


@dataclass
class CheckReport:
    """Outcome of checking one output.

    Attributes:
        output_dir: The output checked.
        totals: Format to split to the `samples` and `tokens` read back.
        problems: What failed, one line each; empty when the output is sound.
        warnings: What a reader should know but does not fail the check, such
            as a validation split below the configured minimum (the hash split
            meets the minimum only on average).
    """

    output_dir: Path
    totals: dict[str, dict[str, dict[str, int]]] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Return True when no problem was found.

        Returns:
            Whether the output passed.
        """
        return not self.problems


# ===============================
# Check
# ===============================


def _document(sample_id: str) -> str:
    """Return the document id of a sample id `<document id>#<chunk index>`.

    Args:
        sample_id: The sample id.

    Returns:
        The document id.
    """
    return sample_id.rsplit("#", 1)[0]


def _same(stored: StoredRow | None, other: StoredRow | None) -> bool:
    """Return True when two stored rows hold the same sample.

    Args:
        stored: A row of one format, or None past its end.
        other: The row of another format at the same position.

    Returns:
        True when both are present with equal ids and equal token ids.
    """
    if stored is None or other is None:
        return stored is other
    return stored.id == other.id and np.array_equal(stored.input_ids, other.input_ids)


def check_pretokenized(output_dir: Path) -> CheckReport:
    """Check a finished output's formats against its manifest.

    Args:
        output_dir: The entry's output folder, holding `manifest.json`.

    Returns:
        The totals read back and every problem found.
    """
    report = CheckReport(output_dir=Path(output_dir))
    manifest_path = report.output_dir / MANIFEST_FILE
    if not manifest_path.exists():
        report.problems.append(f"{manifest_path} missing: the output is not complete")
        return report
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    formats = manifest["settings"]["formats"]

    # read every format split by split, in step; validation first, to test train against it
    validation_documents: set[str] = set()
    for split in (VALIDATION, TRAIN):
        readers = [get_format(name).iter_rows(report.output_dir / name / split) for name in formats]
        totals = {name: {"samples": 0, "tokens": 0} for name in formats}
        reported: set[str] = set()
        for position, row in enumerate(zip_longest(*readers)):
            first = row[0]
            if "differ" not in reported and not all(_same(stored, first) for stored in row[1:]):
                report.problems.append(f"{split}: formats differ at sample {position}")
                reported.add("differ")
            for name, stored in zip(formats, row, strict=True):
                if stored is None:
                    continue
                totals[name]["samples"] += 1
                totals[name]["tokens"] += len(stored.input_ids)
                if stored.len != len(stored.input_ids) and name not in reported:
                    report.problems.append(
                        f"{name}/{split}: sample {position} stores len {stored.len} for {len(stored.input_ids)} ids"
                    )
                    reported.add(name)
            if first is None:
                continue
            document = _document(first.id)
            if split == VALIDATION:
                validation_documents.add(document)
            elif document in validation_documents:
                report.problems.append(f"document {document} has samples in both splits")

        # compare each format's totals with the manifest
        expected = manifest["totals"][split]
        for name in formats:
            report.totals.setdefault(name, {})[split] = totals[name]
            for key in ("samples", "tokens"):
                if totals[name][key] != expected[key]:
                    report.problems.append(
                        f"{name}/{split}: {totals[name][key]} {key} read, manifest records {expected[key]}"
                    )

    # warn when the hash split drew fewer validation documents than the minimum
    minimum = manifest["settings"]["min_validation_docs"]
    drawn = manifest["totals"][VALIDATION]["documents"]
    if drawn < minimum:
        report.warnings.append(f"validation split holds {drawn} documents, fewer than min_validation_docs={minimum}")
    return report
