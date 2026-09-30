"""Tests for the stratified curation-decision sampler."""

import gzip
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from slm4ie.data.curate.sample import (
    JUDGED_STAGES,
    SurvivorIndex,
    draw_stratified_sample,
    resolve_output_dir,
    roster,
    sample_cell,
)
from slm4ie.data.curate.stages import STAGE_DIRS

_DATASET = "demo"


def _document(index: int, source: Optional[str] = None, text: Optional[str] = None) -> Dict[str, Any]:
    """Build one datatrove-shaped document.

    Args:
        index: Document number, used for a stable id.
        source: Value for `metadata.file_path`, naming the upstream shard.
        text: Document text; defaults to a short unique string.

    Returns:
        A document dict as the pipeline writes it.
    """
    metadata: Dict[str, Any] = {"dataset": _DATASET, "domain": "web"}
    if source is not None:
        metadata["file_path"] = f"/tmp/view/{_DATASET}/{source}"
    return {"id": f"{_DATASET}:{index:04d}", "text": text or f"document {index}", "metadata": metadata}


def _write_shard(output_dir: Path, stage: str, name: str, documents: List[Dict[str, Any]]) -> Path:
    """Write documents as one gzipped JSONL shard of a stage.

    Args:
        output_dir: The curation output root.
        stage: Stage name.
        name: Shard file name, e.g. `00000.jsonl.gz`.
        documents: Documents to write.

    Returns:
        Path to the written shard.
    """
    path = output_dir / STAGE_DIRS[stage] / _DATASET / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for doc in documents:
            fh.write(json.dumps(doc) + "\n")
    return path


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file the way the sampler writes it.

    Args:
        path: The JSONL file.

    Returns:
        One dict per line; splits on newline only, so Unicode line separators
        inside a document's text stay part of that document.
    """
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def _ids(rows: List[Dict[str, Any]], decision: str) -> List[str]:
    """Collect the ids of rows drawn into one decision.

    Args:
        rows: Sample rows.
        decision: `kept` or `dropped`.

    Returns:
        Sorted document ids.
    """
    return sorted(row["id"] for row in rows if row["cells"][0]["decision"] == decision)


class TestSampleCell:
    """A cell is the set of documents one stage kept or dropped for one dataset."""

    def test_dropped_documents_are_the_ones_missing_downstream(self, tmp_path: Path) -> None:
        """Drops are the input documents absent from the stage's output."""
        upstream = [_document(index) for index in range(6)]
        _write_shard(tmp_path, "language", "00000.jsonl.gz", upstream)
        _write_shard(tmp_path, "spam", "00000.jsonl.gz", [upstream[0], upstream[3], upstream[5]])

        rows = sample_cell(tmp_path, _DATASET, "spam", 10, 0, 2000, 7)

        assert _ids(rows, "kept") == [f"{_DATASET}:0000", f"{_DATASET}:0003", f"{_DATASET}:0005"]
        assert _ids(rows, "dropped") == [f"{_DATASET}:0001", f"{_DATASET}:0002", f"{_DATASET}:0004"]

    def test_documents_moved_between_shards_are_still_classified(self, tmp_path: Path) -> None:
        """A stage reshuffles its documents across shards, and survivors are still found."""
        upstream = [_document(index) for index in range(6)]
        _write_shard(tmp_path, "language", "00000.jsonl.gz", upstream[:3])
        _write_shard(tmp_path, "language", "00001.jsonl.gz", upstream[3:])
        # The stage keeps four documents but writes them into the opposite shards.
        _write_shard(tmp_path, "spam", "00000.jsonl.gz", [upstream[4], upstream[5]])
        _write_shard(tmp_path, "spam", "00001.jsonl.gz", [upstream[0], upstream[2]])

        rows = sample_cell(tmp_path, _DATASET, "spam", 10, 0, 2000, 7)

        assert _ids(rows, "dropped") == [f"{_DATASET}:0001", f"{_DATASET}:0003"]

    def test_a_document_repeated_upstream_is_counted_once(self, tmp_path: Path) -> None:
        """Sources hold duplicates before dedup; a repeat of a survivor is not a drop."""
        repeated = _document(0)
        _write_shard(tmp_path, "language", "00000.jsonl.gz", [repeated, repeated, _document(1)])
        _write_shard(tmp_path, "spam", "00000.jsonl.gz", [repeated])

        rows = sample_cell(tmp_path, _DATASET, "spam", 10, 0, 2000, 7)

        assert _ids(rows, "kept") == [f"{_DATASET}:0000"]
        assert _ids(rows, "dropped") == [f"{_DATASET}:0001"]

    def test_unequal_shard_counts_are_fine(self, tmp_path: Path) -> None:
        """A stage may write a different number of shards than it read."""
        _write_shard(tmp_path, "language", "00000.jsonl.gz", [_document(0), _document(1)])
        _write_shard(tmp_path, "spam", "00000.jsonl.gz", [_document(0)])
        _write_shard(tmp_path, "spam", "00001.jsonl.gz", [])

        rows = sample_cell(tmp_path, _DATASET, "spam", 10, 0, 2000, 7)

        assert _ids(rows, "dropped") == [f"{_DATASET}:0001"]

    def test_missing_stage_output_is_skipped(self, tmp_path: Path) -> None:
        """A dataset a stage never wrote yields no rows instead of an error."""
        assert sample_cell(tmp_path, _DATASET, "spam", 10, 0, 2000, 7) == []

    def test_per_cell_caps_each_decision(self, tmp_path: Path) -> None:
        """Kept and dropped are each capped at the per-cell size."""
        upstream = [_document(index) for index in range(20)]
        _write_shard(tmp_path, "language", "00000.jsonl.gz", upstream)
        _write_shard(tmp_path, "spam", "00000.jsonl.gz", upstream[:10])

        rows = sample_cell(tmp_path, _DATASET, "spam", 3, 0, 2000, 7)

        assert len(_ids(rows, "kept")) == 3
        assert len(_ids(rows, "dropped")) == 3

    def test_the_same_seed_redraws_the_same_documents(self, tmp_path: Path) -> None:
        """The sample is reproducible from the seed, and a new seed redraws it."""
        upstream = [_document(index) for index in range(40)]
        _write_shard(tmp_path, "language", "00000.jsonl.gz", upstream)
        _write_shard(tmp_path, "spam", "00000.jsonl.gz", upstream[:20])

        first = sample_cell(tmp_path, _DATASET, "spam", 5, 0, 2000, 7)
        again = sample_cell(tmp_path, _DATASET, "spam", 5, 0, 2000, 7)
        other_seed = sample_cell(tmp_path, _DATASET, "spam", 5, 0, 2000, 8)

        assert _ids(first, "dropped") == _ids(again, "dropped")
        assert _ids(first, "dropped") != _ids(other_seed, "dropped")

    def test_text_is_truncated_and_the_full_length_recorded(self, tmp_path: Path) -> None:
        """Rows carry the truncated text and the document's real length."""
        long_document = _document(0, text="a" * 5000)
        _write_shard(tmp_path, "language", "00000.jsonl.gz", [long_document])
        _write_shard(tmp_path, "spam", "00000.jsonl.gz", [long_document])

        row = sample_cell(tmp_path, _DATASET, "spam", 1, 0, 2000, 7)[0]

        assert len(row["text"]) == 2000
        assert row["chars"] == 5000
        assert row["truncated"] is True

    def test_shards_per_cell_bounds_the_search_for_drops(self, tmp_path: Path) -> None:
        """Drops are looked for in a bounded number of input shards, keeps in all of them."""
        for shard in range(4):
            documents = [_document(shard * 10 + offset) for offset in range(4)]
            _write_shard(tmp_path, "language", f"0000{shard}.jsonl.gz", documents)
            _write_shard(tmp_path, "spam", f"0000{shard}.jsonl.gz", documents[:2])

        rows = sample_cell(tmp_path, _DATASET, "spam", 10, 1, 2000, 7)

        assert len(_ids(rows, "dropped")) == 2
        assert len({row["cells"][0]["shard"] for row in rows if row["cells"][0]["decision"] == "dropped"}) == 1
        assert len(_ids(rows, "kept")) == 8


class TestSurvivorIndex:
    """The index answers which ids a stage kept."""

    def test_recorded_ids_are_found_and_others_are_not(self) -> None:
        """An id put into the index is in it; one that was never added is not."""
        index = SurvivorIndex()
        for number in range(100):
            index.add(f"{_DATASET}:{number:04d}")
        index.freeze()

        assert len(index) == 100
        assert f"{_DATASET}:0042" in index
        assert f"{_DATASET}:9999" not in index

    def test_an_empty_index_holds_nothing(self) -> None:
        """A stage that kept nothing reports every document as dropped."""
        index = SurvivorIndex()
        index.freeze()

        assert len(index) == 0
        assert f"{_DATASET}:0000" not in index


class TestDrawStratifiedSample:
    """The written sample carries one row per document and every cell it fell in."""

    def _corpus(self, tmp_path: Path) -> Path:
        """Write a three-stage corpus where each stage drops one document.

        Args:
            tmp_path: Test-local directory used as the curation output root.

        Returns:
            The curation output root.
        """
        documents = [_document(index, source="00000.jsonl.gz") for index in range(4)]
        _write_shard(tmp_path, "convert", "00000.jsonl.gz", documents)
        _write_shard(tmp_path, "language", "00000.jsonl.gz", documents[:3])
        _write_shard(tmp_path, "spam", "00000.jsonl.gz", documents[:2])
        _write_shard(tmp_path, "sentence_dedup", "00000.jsonl.gz", documents[:2])
        return tmp_path

    def test_cells_of_one_document_are_merged(self, tmp_path: Path) -> None:
        """A document drawn into several cells is written once, carrying them all."""
        destination = tmp_path / "out" / "sample.jsonl"

        counts = draw_stratified_sample(
            self._corpus(tmp_path),
            destination,
            stages=("language", "spam"),
            per_cell=10,
            shards_per_cell=0,
        )

        rows = _read_jsonl(destination)
        by_id = {row["id"]: row for row in rows}
        assert counts == {"documents": 4, "cells": 7, "kept": 5, "dropped": 2}
        assert by_id[f"{_DATASET}:0000"]["cells"] == [
            {"stage": "language", "decision": "kept", "shard": "00000.jsonl.gz"},
            {"stage": "spam", "decision": "kept", "shard": "00000.jsonl.gz"},
        ]
        assert by_id[f"{_DATASET}:0002"]["cells"] == [
            {"stage": "language", "decision": "kept", "shard": "00000.jsonl.gz"},
            {"stage": "spam", "decision": "dropped", "shard": "00000.jsonl.gz"},
        ]
        assert [row["id"] for row in rows] == sorted(by_id)

    def test_only_datasets_in_the_final_corpus_are_sampled(self, tmp_path: Path) -> None:
        """Sources that never reached the final corpus are not part of the roster."""
        corpus = self._corpus(tmp_path)
        benchmark = corpus / STAGE_DIRS["language"] / "benchmark"
        benchmark.mkdir(parents=True)

        assert roster(corpus) == [_DATASET]

    def test_a_finished_cell_is_not_drawn_again(self, tmp_path: Path) -> None:
        """A restart reuses the cells an interrupted run already drew."""
        corpus = self._corpus(tmp_path)
        destination = tmp_path / "out" / "sample.jsonl"
        draw_stratified_sample(corpus, destination, stages=("spam",), per_cell=10, shards_per_cell=0)

        # Output removed: a cell drawn again would now find nothing.
        shutil.rmtree(corpus / STAGE_DIRS["spam"])
        counts = draw_stratified_sample(corpus, destination, stages=("spam",), per_cell=10, shards_per_cell=0)

        assert counts["cells"] == 3
        assert counts["dropped"] == 1

    def test_text_with_unicode_line_separators_survives_the_cache(self, tmp_path: Path) -> None:
        """Web text carries   and friends; they must not be read back as row breaks."""
        # U+2028, U+2029 and U+0085 are line breaks to str.splitlines but not to JSON.
        torn = _document(0, text="prva vrstica druga tretja\x85cetrta")
        _write_shard(tmp_path, "language", "00000.jsonl.gz", [torn, _document(1)])
        _write_shard(tmp_path, "spam", "00000.jsonl.gz", [torn])
        destination = tmp_path / "out" / "sample.jsonl"

        counts = draw_stratified_sample(
            tmp_path, destination, datasets=[_DATASET], stages=("spam",), per_cell=10, shards_per_cell=0
        )

        rows = {row["id"]: row for row in _read_jsonl(destination)}
        assert counts == {"documents": 2, "cells": 2, "kept": 1, "dropped": 1}
        assert rows[f"{_DATASET}:0000"]["text"] == "prva vrstica druga tretja\x85cetrta"

    def test_changed_settings_are_not_served_from_the_cache(self, tmp_path: Path) -> None:
        """A different per-cell size draws a fresh cell rather than reusing the old one."""
        corpus = self._corpus(tmp_path)
        destination = tmp_path / "out" / "sample.jsonl"
        draw_stratified_sample(corpus, destination, stages=("spam",), per_cell=10, shards_per_cell=0)

        counts = draw_stratified_sample(corpus, destination, stages=("spam",), per_cell=1, shards_per_cell=0)

        assert counts["kept"] == 1
        assert counts["dropped"] == 1

    def test_an_unjudged_stage_is_refused(self, tmp_path: Path) -> None:
        """A stage that makes no keep-or-drop decision cannot be sampled."""
        with pytest.raises(ValueError, match="not judged stages: statistics"):
            draw_stratified_sample(tmp_path, tmp_path / "sample.jsonl", stages=("statistics",))

    def test_every_judged_stage_has_an_upstream_stage(self) -> None:
        """Convert has no input stage and statistics writes no corpus, so neither is judged."""
        assert "convert" not in JUDGED_STAGES
        assert "statistics" not in JUDGED_STAGES


class TestResolveOutputDir:
    """The stage root comes from the curation config unless overridden."""

    def test_the_config_output_dir_is_resolved(self, tmp_path: Path) -> None:
        """The config's `output_dir` names the folder holding the stages."""
        config = tmp_path / "curate.yaml"
        config.write_text(f"output_dir: {tmp_path / 'pretrain'}\n", encoding="utf-8")

        assert resolve_output_dir(config) == tmp_path / "pretrain"

    def test_an_override_wins(self, tmp_path: Path) -> None:
        """An explicit folder is used without reading the config."""
        assert resolve_output_dir(tmp_path / "missing.yaml", tmp_path / "elsewhere") == tmp_path / "elsewhere"

    def test_a_config_without_an_output_dir_is_an_error(self, tmp_path: Path) -> None:
        """A config that sets no `output_dir` cannot locate the corpus."""
        config = tmp_path / "curate.yaml"
        config.write_text("input_dir: data/extracted\n", encoding="utf-8")

        with pytest.raises(FileNotFoundError, match="no corpus root"):
            resolve_output_dir(config)
