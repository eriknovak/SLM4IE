"""Tests for the curation-quality experiment's metrics in its analysis.py."""

import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "experiments" / "data" / "curation-quality-slovenian" / "analysis.py"


def _load() -> ModuleType:
    """Import the experiment's analysis.py, whose folder name is no package name.

    Returns:
        The loaded module.
    """
    spec = importlib.util.spec_from_file_location("curation_quality_analysis", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analysis = _load()


def _verdict(coherence: int = 5, text_type: str = "prose", spam: bool = False) -> Dict[str, Any]:
    """Build a judge verdict or a person's label on the rubric.

    Args:
        coherence: Coherence on the 1-5 scale.
        text_type: The text shape.
        spam: The adult-or-spam flag.

    Returns:
        A verdict dict.
    """
    return {"coherence": coherence, "text_type": text_type, "adult_or_spam": spam}


class TestBars:
    """The D8 bars and the Wilson interval every rate is reported with."""

    @pytest.mark.parametrize(
        ("verdict", "lenient", "strict"),
        [
            (_verdict(5), False, False),
            (_verdict(3), False, True),
            (_verdict(2), True, True),
            (_verdict(5, "boilerplate"), True, True),
            (_verdict(5, spam=True), True, True),
        ],
    )
    def test_is_bad_at_both_bars(self, verdict: Dict[str, Any], lenient: bool, strict: bool) -> None:
        """Coherence 3 is bad only at the strict bar; shape and spam at both."""
        assert analysis.is_bad(verdict) is lenient
        assert analysis.is_bad(verdict, strict=True) is strict

    def test_wilson_stays_inside_the_scale(self) -> None:
        """An empty share still has a positive upper bound, and none runs past 0 or 1."""
        low, high = analysis.wilson(0, 10)
        assert low == 0.0 and 0.0 < high < 0.35
        assert analysis.wilson(0, 0) == (0.0, 0.0)

    def test_empty_cell_is_blank_not_zero(self) -> None:
        """A stage with drops but no kept documents reports no residual rate."""
        grouped = {("demo", "spam", "dropped"): [_verdict(1), _verdict(5)]}
        (row,) = analysis.stage_rows(grouped, by_source=False)
        assert row["drop_precision"] == 0.5
        assert row["residual_bad_rate"] == ""


class TestPairwiseRows:
    """A pair counts for one side only when both orders agree."""

    def test_orders_must_agree(self) -> None:
        """Kept twice wins, kept-then-tie is an order disagreement, tie twice is a tie."""
        pairs: List[Dict[str, Any]] = []
        answers: Dict[str, Dict[str, Any]] = {}
        for pair, (first, second) in enumerate([("kept", "kept"), ("kept", "tie"), ("tie", "tie")]):
            for slot, call in (("a", first), ("b", second)):
                comparison_id = f"demo|spam|{pair:02d}|{slot}"
                pairs.append({"id": comparison_id, "dataset": "demo", "stage": "spam", "pair": pair, "kept": slot})
                better = "tie" if call == "tie" else slot
                answers[comparison_id] = {"id": comparison_id, "better": better}

        pooled, per_source = analysis.pairwise_rows(pairs, answers)

        assert pooled["source"] == "ALL" and per_source["source"] == "demo"
        assert pooled["pairs"] == 3
        assert (pooled["kept"], pooled["tie"], pooled["orders_differ"], pooled["dropped"]) == (
            pytest.approx(1 / 3, abs=1e-4),
            pytest.approx(1 / 3, abs=1e-4),
            pytest.approx(1 / 3, abs=1e-4),
            0.0,
        )


class TestAdjudication:
    """The conflict draw and which side a person takes on it."""

    @staticmethod
    def _row(doc_id: str, dataset: str, stage: str, decision: str) -> Dict[str, Any]:
        """Build a sample row in one cell.

        Args:
            doc_id: Document id.
            dataset: Source name.
            stage: Stage of the cell.
            decision: `kept` or `dropped`.

        Returns:
            A sample row.
        """
        return {"id": doc_id, "dataset": dataset, "cells": [{"stage": stage, "decision": decision}]}

    def test_draw_takes_flat_conflicts_from_unlabelled_sources(self) -> None:
        """Dedup cells, labelled sources and mild disagreements are all left out."""
        sample = [
            self._row("a:drop-clean", "a", "quality", "dropped"),
            self._row("a:drop-mild", "a", "quality", "dropped"),
            self._row("b:kept-bad", "b", "spam", "kept"),
            self._row("b:dedup", "b", "exact_dedup", "dropped"),
            self._row("web:drop-clean", "web", "quality", "dropped"),
        ]
        verdicts = {
            "a:drop-clean": {**_verdict(5), "id": "a:drop-clean"},
            "a:drop-mild": {**_verdict(3), "id": "a:drop-mild"},
            "b:kept-bad": {**_verdict(1), "id": "b:kept-bad"},
            "b:dedup": {**_verdict(5), "id": "b:dedup"},
            "web:drop-clean": {**_verdict(5), "id": "web:drop-clean"},
        }

        drawn = analysis.draw_adjudications(sample, verdicts, labelled_sources=["web"], size=10)

        assert sorted(row["id"] for row in drawn) == ["a:drop-clean", "b:kept-bad"]
        assert all(row["conflicts"] for row in drawn)

    def test_side_depends_on_the_conflict_direction(self) -> None:
        """Keeping a dropped document and dropping a kept one both side with the judge."""
        adjudications = [
            {"id": "x", "conflicts": [{"stage": "quality", "decision": "dropped"}]},
            {"id": "y", "conflicts": [{"stage": "spam", "decision": "kept"}]},
            {"id": "z", "conflicts": [{"stage": "spam", "decision": "kept"}]},
        ]
        labels = [{"id": "x", **_verdict(5)}, {"id": "y", **_verdict(1)}, {"id": "z", **_verdict(5)}]

        rows = {row["conflict"]: row for row in analysis.adjudication_rows(adjudications, labels)}

        assert rows["pipeline dropped, judge clean"]["sides_with_judge"] == 1
        assert rows["pipeline kept, judge bad"]["sides_with_judge"] == 1
        assert (rows["all"]["sides_with_judge"], rows["all"]["labelled"]) == (2, 3)


class TestLossRows:
    """Each source's documents split by the stage that dropped them."""

    def test_shares_sum_to_one_and_inflated_input_is_rebased(self) -> None:
        """A doubled language output is the base, so its language loss is zero."""
        counts = {"language": 90, "spam": 80, "quality": 60, "repetition": 50, "exact_dedup": 40, "final": 30}
        funnel = [
            {"source": "plain", "convert": 100, "duplicated_input": False, **counts},
            {"source": "doubled", "convert": 45, "duplicated_input": True, **counts},
            {"source": "TOTAL", "convert": 145, "duplicated_input": 1, **counts},
        ]

        plain, doubled = analysis.loss_rows(funnel)

        for row in (plain, doubled):
            # shares are stored at four decimals, so the sum is exact only to that
            assert sum(row[stage] for stage in analysis.STAGES) + row["kept"] == pytest.approx(1.0, abs=1e-3)
        assert plain["language"] == 0.1 and plain["kept"] == 0.3
        assert doubled["language"] == 0.0 and doubled["base"] == 90


class TestStepTimings:
    """Machine cost read from an executor's per-task stats files."""

    @staticmethod
    def _stats(path: Path, reader: float, work: float, writer: float, documents: int, end: float) -> None:
        """Write one task's stats file with nested block timings.

        Args:
            path: Stats file to write.
            reader: The reader block's seconds.
            work: The filter block's seconds, which include the writer's.
            writer: The writer block's seconds.
            documents: Documents the reader read.
            end: Modification time to stamp, the task's end.
        """
        blocks = [
            {
                "name": "R - READER: Jsonl",
                "time_stats": {"total": reader},
                "stats": {"documents": {"total": documents}},
            },
            {"name": "F - FILTER: Demo", "time_stats": {"total": work}, "stats": {}},
            {"name": "W - WRITER: Jsonl", "time_stats": {"total": writer}, "stats": {}},
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(blocks), encoding="utf-8")
        os.utime(path, (end, end))

    def test_slowest_block_stale_ranks_and_wall_clock(self, tmp_path: Path) -> None:
        """Nested blocks are not summed, stale ranks are skipped, and wall clock spans the tasks."""
        (tmp_path / "executor.json").write_text(json.dumps({"tasks": 2, "workers": 2}), encoding="utf-8")
        self._stats(tmp_path / "stats" / "00000.json", 1.0, 10.0, 8.0, documents=100, end=1_000.0)
        self._stats(tmp_path / "stats" / "00001.json", 1.0, 20.0, 15.0, documents=200, end=1_005.0)
        self._stats(tmp_path / "stats" / "00002.json", 1.0, 999.0, 1.0, documents=9_999, end=500.0)

        timings = analysis._step_timings(tmp_path)

        assert timings["cpu_seconds"] == 30.0
        assert timings["wall_seconds"] == 20.0
        assert timings["documents"] == 300
        assert timings["blocks"]["FILTER: Demo"] == 30.0

    def test_missing_executor_is_none(self, tmp_path: Path) -> None:
        """A step that never ran reports nothing."""
        assert analysis._step_timings(tmp_path) is None
