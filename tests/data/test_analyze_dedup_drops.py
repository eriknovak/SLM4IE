"""Tests for `scripts/data/analyze_dedup_drops.py`.

These tests build a synthetic `_dedup/` tree with two drop stages
plus an input folder, then verify both `collect_reports` (the
machine-readable shape) and `format_table` (the human-readable view)
behave as documented.
"""

import gzip
import json
from pathlib import Path
from typing import Iterable

import pytest

from scripts.data import analyze_dedup_drops as analyzer


def _write_shard(path: Path, records: Iterable[dict]) -> None:
    """Gzip-write JSONL records under *path* (parents created)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _build_scratch(tmp_path: Path) -> Path:
    """Build a synthetic scratch tree with two drop stages."""
    scratch = tmp_path / "_dedup"
    _write_shard(
        scratch / "exact_dropped" / "alpha" / "00000.jsonl.gz",
        [
            {"text": "one two three four", "id": "a:1"},
            {"text": "five six", "id": "a:2"},
        ],
    )
    _write_shard(
        scratch / "sentence_dropped" / "beta" / "00000.jsonl.gz",
        [{"text": "alpha beta gamma", "id": "b:1"}],
    )
    return scratch


class TestCollectReports:
    """`collect_reports` aggregates per-stage drop volume."""

    def test_returns_one_entry_per_drop_folder(self, tmp_path: Path) -> None:
        """Each `*_dropped/` folder under scratch becomes one stage row."""
        scratch = _build_scratch(tmp_path)
        report = analyzer.collect_reports(scratch, input_dir=None)
        names = [s["name"] for s in report["stages"]]
        assert names == ["exact", "sentence"]

    def test_aggregates_doc_and_word_counts(self, tmp_path: Path) -> None:
        """Word counts come from whitespace-splitting the `text` field."""
        scratch = _build_scratch(tmp_path)
        report = analyzer.collect_reports(scratch, input_dir=None)
        by_name = {s["name"]: s for s in report["stages"]}
        assert by_name["exact"]["n_docs"] == 2
        assert by_name["exact"]["n_words"] == 4 + 2
        assert by_name["sentence"]["n_docs"] == 1
        assert by_name["sentence"]["n_words"] == 3

    def test_canonical_stage_order_is_preserved(self, tmp_path: Path) -> None:
        """Canonical pipeline order beats lexical order in the output."""
        scratch = tmp_path / "_dedup"
        for name in ("sentence_dropped", "lang_dropped", "exact_dropped"):
            _write_shard(scratch / name / "x" / "00000.jsonl.gz", [{"text": "a", "id": "z"}])
        report = analyzer.collect_reports(scratch, input_dir=None)
        names = [s["name"] for s in report["stages"]]
        assert names == ["lang", "exact", "sentence"]

    def test_input_dir_populates_totals(self, tmp_path: Path) -> None:
        """Passing input_dir adds a non-null input summary."""
        scratch = _build_scratch(tmp_path)
        input_dir = tmp_path / "datatrove"
        _write_shard(
            input_dir / "alpha" / "00000.jsonl.gz",
            [
                {"text": "one two three", "id": "i:1"},
                {"text": "four five", "id": "i:2"},
                {"text": "six", "id": "i:3"},
            ],
        )
        report = analyzer.collect_reports(scratch, input_dir=input_dir)
        assert report["input"] is not None
        assert report["input"]["n_docs"] == 3
        assert report["input"]["n_words"] == 3 + 2 + 1

    def test_missing_scratch_raises(self, tmp_path: Path) -> None:
        """Nonexistent scratch folder is a hard error."""
        with pytest.raises(FileNotFoundError):
            analyzer.collect_reports(tmp_path / "nope", input_dir=None)


class TestFormatTable:
    """`format_table` renders a fixed-width human view."""

    def test_table_includes_stage_rows(self, tmp_path: Path) -> None:
        """Every stage shows up as a row in the rendered table."""
        scratch = _build_scratch(tmp_path)
        report = analyzer.collect_reports(scratch, input_dir=None)
        rendered = analyzer.format_table(report)
        assert "exact" in rendered
        assert "sentence" in rendered
        assert "n_docs" in rendered

    def test_input_dir_adds_percent_columns(self, tmp_path: Path) -> None:
        """When input_dir is set, the table grows `% docs` / `% words`."""
        scratch = _build_scratch(tmp_path)
        input_dir = tmp_path / "datatrove"
        _write_shard(
            input_dir / "alpha" / "00000.jsonl.gz",
            [{"text": "x y z w", "id": str(i)} for i in range(10)],
        )
        report = analyzer.collect_reports(scratch, input_dir=input_dir)
        rendered = analyzer.format_table(report)
        assert "% docs" in rendered
        assert "% words" in rendered
        assert "(input)" in rendered


class TestParseArgs:
    """`parse_args` enforces the documented CLI shape."""

    def test_scratch_is_required(self) -> None:
        """Calling with no arguments fails."""
        with pytest.raises(SystemExit):
            analyzer.parse_args([])

    def test_input_dir_flag_is_optional(self) -> None:
        """Bare scratch arg parses with input_dir == None."""
        args = analyzer.parse_args(["/tmp/scratch"])
        assert args.scratch == Path("/tmp/scratch")
        assert args.input_dir is None
        assert args.json_output is False

    def test_json_flag_toggles_output(self) -> None:
        """`--json` sets json_output to True."""
        args = analyzer.parse_args(["/tmp/scratch", "--json"])
        assert args.json_output is True
