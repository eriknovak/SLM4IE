"""Tests for the KPI-coverage experiment's per-stage funnel in its analysis.py."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict, Tuple

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "experiments" / "data" / "kpi-coverage-slovenian" / "analysis.py"


def _load() -> ModuleType:
    """Import the experiment's analysis.py, whose folder name is no package name.

    Returns:
        The loaded module.
    """
    spec = importlib.util.spec_from_file_location("kpi_coverage_analysis", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


analysis = _load()

#: Documents each fixture source writes at the five per-source stages, then after both dedups.
_COUNTS: Dict[str, Tuple[int, ...]] = {
    "web_a": (100, 90, 80, 70, 60, 50, 40),
    "web_b": (10, 10, 9, 8, 8, 6, 5),
    "med": (20, 20, 20, 18, 18, 9, 3),
}


def _sentinel(folder: Path, records_in: int, records_out: int) -> None:
    """Write a stage sentinel with its document counts.

    Args:
        folder: The unit folder.
        records_in: Documents the unit read.
        records_out: Documents the unit wrote.
    """
    folder.mkdir(parents=True, exist_ok=True)
    payload = {"records_in": records_in, "records_out": records_out, "document_digest": "sum256:abc"}
    (folder / ".complete").write_text(json.dumps(payload), encoding="utf-8")


def _build(root: Path) -> Path:
    """Write a fixture build: per-source sentinels, dedup sentinels and final statistics.

    Args:
        root: Folder to build in.

    Returns:
        The build folder.
    """
    for source, counts in _COUNTS.items():
        previous = counts[0]
        for stage, written in zip(analysis.SCOPED_STAGES, counts[:5]):
            _sentinel(root / stage / source, previous, written)
            previous = written
        per_dataset = root / "07_statistics" / "per_dataset"
        per_dataset.mkdir(parents=True, exist_ok=True)
        final = {
            "dataset": source,
            "domain": _SOURCES[source]["domain"],
            "doc_count": counts[6],
            "word_count": 10 * counts[6],
        }
        (per_dataset / f"{source}.json").write_text(json.dumps(final), encoding="utf-8")
    exact_in, exact_out, final_out = (sum(counts[index] for counts in _COUNTS.values()) for index in (4, 5, 6))
    _sentinel(root / analysis.COUNTED_STAGE, exact_in, exact_out)
    _sentinel(root / "06_sentence_dedup", exact_out, final_out)
    return root


_EXACT = {source: counts[5] for source, counts in _COUNTS.items()}
_SOURCES = {
    "web_a": {"domain": "web", "access": "open"},
    "web_b": {"domain": "web", "access": "gated"},
    "med": {"domain": "medical", "access": "open"},
}


def test_funnel_follows_each_source_through_every_stage(tmp_path: Path) -> None:
    """Each source's row carries its documents after every stage and its final tokens."""
    rows = analysis.funnel_source_rows("loosened", _build(tmp_path), _EXACT, _SOURCES)
    med = next(row for row in rows if row["source"] == "med")
    stages = [med[column] for column in ("convert", "language", "spam", "quality", "repetition")]
    assert stages + [med["exact_dedup"], med["sentence_dedup"]] == list(_COUNTS["med"])
    assert (med["tokens_estimated"], med["retained"]) == (60, 0.15)


def test_funnel_rejects_a_stage_that_read_other_than_the_stage_before_wrote(tmp_path: Path) -> None:
    """A stage reused from another build shows as a break in the chain."""
    build = _build(tmp_path)
    _sentinel(build / "03_quality" / "web_b", 7, 8)
    with pytest.raises(ValueError, match="web_b: 03_quality read 7"):
        analysis.funnel_source_rows("loosened", build, _EXACT, _SOURCES)


def test_funnel_rejects_exact_dedup_counts_that_miss_the_stage_sentinel(tmp_path: Path) -> None:
    """Stale exact-dedup counts fail against the corpus-wide sentinel."""
    with pytest.raises(ValueError, match="exact_dedup documents sum to 66"):
        analysis.funnel_source_rows("loosened", _build(tmp_path), {**_EXACT, "web_a": 51}, _SOURCES)


def test_domain_rows_sum_the_sources_sharing_a_label(tmp_path: Path) -> None:
    """Two web sources fold into one web row, ranked above the smaller medical row."""
    rows = analysis.funnel_domain_rows(analysis.funnel_source_rows("loosened", _build(tmp_path), _EXACT, _SOURCES))
    assert [(row["source_domain"], row["sources"], row["convert"], row["sentence_dedup"]) for row in rows] == [
        ("web", 2, 110, 45),
        ("medical", 1, 20, 3),
    ]


def test_funnel_keeps_a_source_that_lost_its_last_document_before_the_end(tmp_path: Path) -> None:
    """A source with no statistics file ends at zero instead of vanishing from the funnel."""
    build = _build(tmp_path)
    (build / "07_statistics" / "per_dataset" / "med.json").unlink()
    _sentinel(build / analysis.COUNTED_STAGE, 86, 56)
    _sentinel(build / "06_sentence_dedup", 56, 45)
    rows = analysis.funnel_source_rows("loosened", build, {**_EXACT, "med": 0}, _SOURCES)
    med = next(row for row in rows if row["source"] == "med")
    assert (med["repetition"], med["exact_dedup"], med["sentence_dedup"], med["words_final"]) == (18, 0, 0, 0)


def test_funnel_rejects_statistics_for_a_source_no_stage_converted(tmp_path: Path) -> None:
    """A statistics file with no sentinels behind it names a build mixed from two source sets."""
    build = _build(tmp_path)
    stray = build / "07_statistics" / "per_dataset" / "stray.json"
    stray.write_text(
        json.dumps({"dataset": "stray", "domain": "web", "doc_count": 1, "word_count": 1}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r"no stage converted: \['stray'\]"):
        analysis.funnel_source_rows("loosened", build, _EXACT, _SOURCES)
