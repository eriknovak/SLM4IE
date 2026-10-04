"""Tests for slm4ie.data.curate.stages.convert (the curate stage-0 module).

The convert stage is stage 0 of the curation pipeline. These tests cover
`convert_record`, `lift_dataset`, and `run_convert_stage`.
"""

import gzip
import json
import os
from pathlib import Path
from typing import Dict, List

import pytest

from slm4ie.data.curate.stages.convert import (
    convert_input_files,
    input_files_digest,
    lift_dataset,
    convert_record,
    run_convert_stage,
)


def _write_jsonl(path: Path, records: List[Dict]) -> None:
    """Write *records* as one JSON object per line to *path*.

    Args:
        path: Destination file path. Parents are created as needed.
        records: Records to serialize.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False))
            fh.write("\n")


def _write_gz_jsonl(path: Path, records: List[Dict]) -> None:
    """Write *records* as one JSON object per line, gzipped.

    Args:
        path: Destination ``.jsonl.gz`` path. Parents are created as
            needed.
        records: Records to serialize.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False))
            fh.write("\n")


def _read_all_shard_records(folder: Path) -> List[Dict]:
    """Return every JSON record across the gzipped shards in *folder*.

    Args:
        folder: Per-dataset shard folder.

    Returns:
        All decoded records, in shard-name order.
    """
    out: List[Dict] = []
    for shard in sorted(folder.glob("*.jsonl.gz")):
        with gzip.open(shard, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    return out


class TestConvertRecord:
    """Unit tests for `convert_record`."""

    def test_text_present_record(self) -> None:
        """A well-formed record produces datatrove's id/text shape."""
        record = {
            "text": "Lepa beseda.",
            "source": "kzb",
            "domain": "scientific",
            "doc_id": "s1",
            "uid": "kzb:s1",
            "metadata": {"sent_id": "s1"},
        }
        out = convert_record(record)

        assert out["text"] == "Lepa beseda."
        assert out["id"] == "kzb:s1"
        # `source` is renamed to `dataset` for the datatrove world.
        assert out["dataset"] == "kzb"
        assert "source" not in out
        assert out["domain"] == "scientific"
        assert out["doc_id"] == "s1"
        # Free-form metadata is flattened onto the top level.
        assert out["sent_id"] == "s1"
        # `metadata` itself does not appear in the output.
        assert "metadata" not in out

    def test_missing_text_raises(self) -> None:
        """A record without the configured text field fails with KeyError."""
        record = {
            "source": "kzb",
            "domain": "web",
            "doc_id": "1",
            "uid": "kzb:1",
        }
        with pytest.raises(KeyError, match="text field"):
            convert_record(record)

    def test_missing_uid_raises(self) -> None:
        """A record without `uid` fails with a KeyError naming the extract step."""
        record = {
            "text": "no uid here",
            "source": "ds",
            "domain": "web",
        }
        with pytest.raises(KeyError, match="prepare_datasets.py extract"):
            convert_record(record)

    def test_empty_uid_raises(self) -> None:
        """An empty-string uid is treated the same as a missing one."""
        record = {
            "text": "x",
            "source": "ds",
            "domain": "web",
            "uid": "",
        }
        with pytest.raises(KeyError, match="prepare_datasets.py extract"):
            convert_record(record)

    def test_metadata_flatten_collision_renamed(self) -> None:
        """Metadata keys clashing with reserved fields get a meta_ prefix."""
        record = {
            "text": "x",
            "source": "ds",
            "domain": "web",
            "doc_id": "1",
            "uid": "ds:1",
            "metadata": {
                "domain": "wrong",
                "annotations": "wrong",
                "url": "ok",
            },
        }
        out = convert_record(record)

        # Reserved fields keep their original (top-level) value...
        assert out["domain"] == "web"
        # ...and the clashing metadata copy is renamed.
        assert out["meta_domain"] == "wrong"
        assert out["meta_annotations"] == "wrong"
        # Non-clashing metadata flows through unchanged.
        assert out["url"] == "ok"

    def test_id_field_preserved_when_not_in_metadata_fields(self) -> None:
        """`id_field` survives even when it's not listed in `metadata_fields`."""
        record = {
            "text": "x",
            "source": "ds",
            "doc_id": "preserved",
            "uid": "ds:preserved",
        }
        out = convert_record(
            record,
            id_field="doc_id",
            metadata_fields=[],  # doc_id missing on purpose
        )
        # The id_field value is still kept verbatim alongside the
        # globally-unique `id` (which comes from uid).
        assert out["doc_id"] == "preserved"
        assert out["id"] == "ds:preserved"

    def test_source_renamed_even_when_listed_in_metadata_fields(self) -> None:
        """`source` is always emitted as `dataset`, never kept verbatim."""
        record = {
            "text": "x",
            "source": "ds",
            "domain": "web",
            "doc_id": "1",
            "uid": "ds:1",
        }
        # Even if a stale config still lists `source`, the output routes
        # it to `dataset` (every downstream stage writer keys on it).
        out = convert_record(record, metadata_fields=["source", "domain"])
        assert out["dataset"] == "ds"
        assert "source" not in out

    def test_annotations_only_emitted_when_present(self) -> None:
        """An `annotations` key is emitted iff the record carries one."""
        with_ann = convert_record(
            {
                "text": "x",
                "source": "ds",
                "doc_id": "1",
                "uid": "ds:1",
                "annotations": {"forms": ["x"]},
            }
        )
        assert with_ann["annotations"] == {"forms": ["x"]}

        without_ann = convert_record({"text": "x", "source": "ds", "doc_id": "1", "uid": "ds:1"})
        assert "annotations" not in without_ann


class TestConvertDataset:
    """Tests for end-to-end per-dataset conversion."""

    def test_writes_shard_when_input_present(self, tmp_path: Path) -> None:
        """`lift_dataset` produces a sharded folder of gzipped JSONL."""
        input_dir = tmp_path / "extracted"
        output_dir = tmp_path / "out"
        _write_jsonl(
            input_dir / "kzb.jsonl",
            [
                {
                    "text": "Lepa beseda.",
                    "source": "kzb",
                    "domain": "sci",
                    "doc_id": "s1",
                    "uid": "kzb:s1",
                }
            ],
        )

        n = lift_dataset("kzb", input_dir=input_dir, output_dir=output_dir)
        assert n == 1

        shard_folder = output_dir / "kzb"
        assert (shard_folder / "00000.jsonl.gz").exists()
        rows = _read_all_shard_records(shard_folder)
        assert len(rows) == 1
        assert rows[0]["id"] == "kzb:s1"
        # The on-disk shard carries `dataset` — every downstream stage
        # writer routes shards by `${dataset}`.
        assert rows[0]["dataset"] == "kzb"

    def test_includes_annotations_when_opted_in(self, tmp_path: Path) -> None:
        """With include_annotations=True, the sidecar is joined into output."""
        input_dir = tmp_path / "extracted"
        output_dir = tmp_path / "out"
        _write_jsonl(
            input_dir / "kzb.jsonl",
            [
                {
                    "text": "Lepa beseda.",
                    "source": "kzb",
                    "domain": "sci",
                    "doc_id": "s1",
                    "uid": "kzb:s1",
                }
            ],
        )
        _write_gz_jsonl(
            input_dir / "kzb.annotations.jsonl.gz",
            [
                {
                    "doc_id": "s1",
                    "uid": "kzb:s1",
                    "forms": ["Lepa", "beseda", "."],
                    "sentences": [[0, 2]],
                }
            ],
        )

        n = lift_dataset(
            "kzb",
            input_dir=input_dir,
            output_dir=output_dir,
            include_annotations=True,
        )
        assert n == 1
        rows = _read_all_shard_records(output_dir / "kzb")
        assert rows[0]["annotations"]["forms"] == ["Lepa", "beseda", "."]

    def test_missing_input_returns_none(self, tmp_path: Path) -> None:
        """No `<key>.jsonl` -> lift_dataset returns None (skip)."""
        input_dir = tmp_path / "extracted"
        input_dir.mkdir()
        result = lift_dataset("ghost", input_dir=input_dir, output_dir=tmp_path / "out")
        assert result is None


class TestRunConvertStage:
    """Tests for the parallel stage runner."""

    def test_runs_all_dataset_keys(self, tmp_path: Path) -> None:
        """The stage runner walks every supplied dataset key."""
        input_dir = tmp_path / "extracted"
        output_dir = tmp_path / "00_convert"
        for key in ("alpha", "beta"):
            _write_jsonl(
                input_dir / f"{key}.jsonl",
                [
                    {
                        "text": f"text from {key}",
                        "source": key,
                        "domain": "web",
                        "doc_id": "1",
                        "uid": f"{key}:1",
                    }
                ],
            )

        results = run_convert_stage(
            input_dir=input_dir,
            output_dir=output_dir,
            dataset_keys=["alpha", "beta"],
            workers=1,
        )
        assert results == {"alpha": 1, "beta": 1}
        assert (output_dir / "alpha" / "00000.jsonl.gz").exists()
        assert (output_dir / "beta" / "00000.jsonl.gz").exists()

    def test_missing_input_returns_none_in_results(self, tmp_path: Path) -> None:
        """A dataset key without input is reported as None in the results."""
        input_dir = tmp_path / "extracted"
        input_dir.mkdir()
        output_dir = tmp_path / "00_convert"

        results = run_convert_stage(
            input_dir=input_dir,
            output_dir=output_dir,
            dataset_keys=["ghost"],
            workers=1,
        )
        assert results == {"ghost": None}


def _write_extracted(input_dir: Path, key: str, text: str = "x") -> Path:
    """Write a minimal extracted `<key>.jsonl` and return its path."""
    input_dir.mkdir(parents=True, exist_ok=True)
    path = input_dir / f"{key}.jsonl"
    path.write_text(text, encoding="utf-8")
    return path


def _digest(input_dir: Path, key: str, annotations: bool = False, previous: object = None) -> str:
    """Return the convert input digest of *key* under *input_dir*."""
    return input_files_digest(convert_input_files(input_dir, key, annotations, previous))  # type: ignore[arg-type]


def test_convert_input_digest_changes_with_content(tmp_path: Path) -> None:
    """Rewriting the source with other bytes of the same size changes the digest."""
    _write_extracted(tmp_path, "news", "abcde")
    before = _digest(tmp_path, "news")
    _write_extracted(tmp_path, "news", "vwxyz")
    assert _digest(tmp_path, "news") != before


def test_convert_input_digest_ignores_mtime(tmp_path: Path) -> None:
    """Touching the source, or rewriting identical bytes, keeps the digest."""
    path = _write_extracted(tmp_path, "news", "abcde")
    before = _digest(tmp_path, "news")
    os.utime(path, ns=(2_000_000_000_000_000_000, 2_000_000_000_000_000_000))
    assert _digest(tmp_path, "news") == before
    _write_extracted(tmp_path, "news", "abcde")
    assert _digest(tmp_path, "news") == before


def test_convert_input_files_reuse_hash_when_size_and_mtime_match(tmp_path: Path) -> None:
    """An untouched file keeps its recorded hash instead of being reread."""
    _write_extracted(tmp_path, "news", "abcde")
    files = convert_input_files(tmp_path, "news", False)
    recorded = {"news.jsonl": {**files["news.jsonl"], "sha256": "recorded"}}  # type: ignore[dict-item]
    assert convert_input_files(tmp_path, "news", False, recorded)["news.jsonl"]["sha256"] == "recorded"  # type: ignore[index]


def test_convert_input_files_mark_absent(tmp_path: Path) -> None:
    """A missing source file is recorded as absent, not raised."""
    assert convert_input_files(tmp_path, "missing", False) == {"missing.jsonl": None}


def test_convert_input_digest_folds_annotations(tmp_path: Path) -> None:
    """With annotations on, the sidecar participates in the digest."""
    _write_extracted(tmp_path, "news", "abc")
    without = _digest(tmp_path, "news")
    (tmp_path / "news.annotations.jsonl.gz").write_bytes(b"gz")
    assert _digest(tmp_path, "news", annotations=True) != without
