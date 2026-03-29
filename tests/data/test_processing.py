"""Tests for slm4ie.data.processing module."""

import json
from pathlib import Path
from typing import Iterator

import pytest
import yaml

from slm4ie.data.extractors import register_extractor, BaseExtractor
from slm4ie.data.schema import Annotations, Document, Token
from slm4ie.data.processing import (
    extract_datasets,
    load_extraction_config,
)

_STUB_DOC = Document(
    text="stub text",
    source="stub",
    domain="test",
    doc_id="stub-1",
)


class _StubExtractor(BaseExtractor):
    """Yields one Document per call, regardless of input."""

    def extract(
        self,
        input_dir: Path,
        source: str,
        domain: str,
    ) -> Iterator[Document]:
        """Yields a single stub document.

        Args:
            input_dir (Path): Not used.
            source (str): Not used.
            domain (str): Not used.

        Yields:
            Document: A single stub document.
        """
        yield Document(
            text="stub text",
            source=source,
            domain=domain,
            doc_id="stub-1",
        )


register_extractor("stub", _StubExtractor)


class _AnnotatedStubExtractor(BaseExtractor):
    """Yields one annotated Document per call."""

    def extract(
        self,
        input_dir: Path,
        source: str,
        domain: str,
    ) -> Iterator[Document]:
        """Yields a single annotated document.

        Args:
            input_dir (Path): Not used.
            source (str): Dataset key.
            domain (str): Domain label.

        Yields:
            Document: A single annotated document.
        """
        yield Document(
            text="Zdravo svet",
            source=source,
            domain=domain,
            doc_id="ann-1",
            annotations=Annotations(
                tokens=[
                    Token(form="Zdravo", lemma="zdrav", upos="INTJ"),
                    Token(
                        form="svet",
                        lemma="svet",
                        upos="NOUN",
                        feats="Case=Nom",
                    ),
                ],
                sentences=[[0, 1]],
            ),
        )


register_extractor("annotated_stub", _AnnotatedStubExtractor)


def _write_config(tmp_path: Path, datasets: dict) -> Path:
    """Write a minimal extraction config YAML to tmp_path.

    Args:
        tmp_path (Path): Temporary directory.
        datasets (dict): Dataset entries for the config.

    Returns:
        Path: Path to the written config file.
    """
    config_data = {
        "input_dir": str(tmp_path / "raw"),
        "output_dir": str(tmp_path / "processed"),
        "datasets": datasets,
    }
    config_file = tmp_path / "extract.yaml"
    config_file.write_text(yaml.dump(config_data))
    return config_file


class TestLoadExtractionConfig:
    """Tests for load_extraction_config."""

    def test_loads_config(self, tmp_path: Path):
        """Verify all fields are parsed correctly."""
        config_file = _write_config(
            tmp_path,
            {
                "ds1": {"extractor": "stub", "domain": "web"},
            },
        )
        cfg = load_extraction_config(config_file)
        assert cfg.input_dir == str(tmp_path / "raw")
        assert cfg.output_dir == str(tmp_path / "processed")
        assert "ds1" in cfg.datasets
        assert cfg.datasets["ds1"]["extractor"] == "stub"
        assert cfg.datasets["ds1"]["domain"] == "web"

    def test_missing_file_raises(self):
        """FileNotFoundError raised for non-existent config."""
        with pytest.raises(FileNotFoundError):
            load_extraction_config(
                Path("/nonexistent/extract.yaml")
            )


class TestExtractDatasets:
    """Tests for extract_datasets orchestrator."""

    def test_extracts_single_dataset(self, tmp_path: Path):
        """Creates raw dir, extracts, verifies JSONL output."""
        config_file = _write_config(
            tmp_path,
            {
                "ds1": {"extractor": "stub", "domain": "web"},
            },
        )
        raw_dir = tmp_path / "raw" / "ds1"
        raw_dir.mkdir(parents=True)
        (raw_dir / "dummy.txt").write_text("placeholder")

        extract_datasets(config_file)

        output_file = tmp_path / "processed" / "ds1.jsonl"
        assert output_file.exists()
        lines = output_file.read_text().strip().splitlines()
        assert len(lines) == 1
        doc = json.loads(lines[0])
        assert doc["source"] == "ds1"
        assert doc["domain"] == "web"
        assert doc["text"] == "stub text"

    def test_extracts_selected_datasets(self, tmp_path: Path):
        """Only the requested dataset key is extracted."""
        config_file = _write_config(
            tmp_path,
            {
                "ds1": {"extractor": "stub", "domain": "web"},
                "ds2": {"extractor": "stub", "domain": "news"},
            },
        )
        (tmp_path / "raw" / "ds1").mkdir(parents=True)
        (tmp_path / "raw" / "ds2").mkdir(parents=True)

        extract_datasets(config_file, dataset_keys=["ds1"])

        processed = tmp_path / "processed"
        assert (processed / "ds1.jsonl").exists()
        assert not (processed / "ds2.jsonl").exists()

    def test_text_jsonl_excludes_annotations(self, tmp_path: Path):
        """Text JSONL file must not contain annotations."""
        config_file = _write_config(
            tmp_path,
            {
                "ds1": {"extractor": "annotated_stub", "domain": "web"},
            },
        )
        raw_dir = tmp_path / "raw" / "ds1"
        raw_dir.mkdir(parents=True)

        extract_datasets(config_file)

        output_file = tmp_path / "processed" / "ds1.jsonl"
        lines = output_file.read_text().strip().splitlines()
        data = json.loads(lines[0])
        assert "annotations" not in data
        assert data["text"] == "Zdravo svet"
        assert data["doc_id"] == "ann-1"

    def test_writes_annotation_file_for_annotated_datasets(
        self, tmp_path: Path
    ):
        """Annotated datasets produce a gzipped annotation file."""
        import gzip

        config_file = _write_config(
            tmp_path,
            {
                "ds1": {"extractor": "annotated_stub", "domain": "web"},
            },
        )
        raw_dir = tmp_path / "raw" / "ds1"
        raw_dir.mkdir(parents=True)

        extract_datasets(config_file)

        ann_file = tmp_path / "processed" / "ds1.annotations.jsonl.gz"
        assert ann_file.exists()

        with gzip.open(ann_file, "rt", encoding="utf-8") as f:
            lines = f.read().strip().splitlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["doc_id"] == "ann-1"
        assert data["forms"] == ["Zdravo", "svet"]
        assert data["lemmas"] == ["zdrav", "svet"]
        assert data["upos"] == ["INTJ", "NOUN"]
        assert data["feats"] == [None, "Case=Nom"]
        assert data["sentences"] == [[0, 1]]

    def test_no_annotation_file_for_text_only_datasets(
        self, tmp_path: Path
    ):
        """Text-only datasets do not produce an annotation file."""
        config_file = _write_config(
            tmp_path,
            {
                "ds1": {"extractor": "stub", "domain": "web"},
            },
        )
        raw_dir = tmp_path / "raw" / "ds1"
        raw_dir.mkdir(parents=True)

        extract_datasets(config_file)

        ann_file = tmp_path / "processed" / "ds1.annotations.jsonl.gz"
        assert not ann_file.exists()

    def test_unknown_key_raises(self, tmp_path: Path):
        """ValueError raised for unknown dataset key."""
        config_file = _write_config(
            tmp_path,
            {
                "ds1": {"extractor": "stub", "domain": "web"},
            },
        )
        with pytest.raises(ValueError, match="unknown_ds"):
            extract_datasets(
                config_file, dataset_keys=["unknown_ds"]
            )
