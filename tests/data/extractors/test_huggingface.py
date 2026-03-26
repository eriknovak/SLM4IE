"""Tests for HuggingFaceExtractor."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from slm4ie.data.extractors.huggingface import HuggingFaceExtractor
from slm4ie.data.schema import Document


class TestHuggingFaceExtractor:
    """Tests for HuggingFaceExtractor."""

    def test_extracts_text_column(self, tmp_path: Path) -> None:
        """Text column values are extracted as Document text."""
        (tmp_path / "sl").mkdir()
        with patch(
            "slm4ie.data.extractors.huggingface.load_from_disk"
        ) as mock_load:
            mock_ds = MagicMock()
            mock_ds.__iter__ = MagicMock(
                return_value=iter([
                    {"text": "Dober dan."},
                    {"text": "Kako si?"},
                ])
            )
            mock_ds.column_names = ["text"]
            mock_load.return_value = mock_ds

            extractor = HuggingFaceExtractor()
            docs = list(
                extractor.extract(tmp_path, source="fineweb2", domain="web")
            )

        assert len(docs) == 2
        assert docs[0].text == "Dober dan."
        assert docs[1].text == "Kako si?"
        assert all(isinstance(d, Document) for d in docs)

    def test_preserves_metadata_columns(self, tmp_path: Path) -> None:
        """Non-text columns with non-None values appear in metadata."""
        (tmp_path / "sl").mkdir()
        with patch(
            "slm4ie.data.extractors.huggingface.load_from_disk"
        ) as mock_load:
            mock_ds = MagicMock()
            mock_ds.__iter__ = MagicMock(
                return_value=iter([
                    {
                        "text": "Primer.",
                        "url": "http://a.com",
                        "language_score": 0.95,
                    }
                ])
            )
            mock_ds.column_names = ["text", "url", "language_score"]
            mock_load.return_value = mock_ds

            extractor = HuggingFaceExtractor()
            docs = list(
                extractor.extract(tmp_path, source="fineweb2", domain="web")
            )

        assert len(docs) == 1
        meta = docs[0].metadata
        assert meta["url"] == "http://a.com"
        assert meta["language_score"] == 0.95
        assert "text" not in meta

    def test_skips_empty_text(self, tmp_path: Path) -> None:
        """Rows with empty or missing text are skipped."""
        (tmp_path / "sl").mkdir()
        with patch(
            "slm4ie.data.extractors.huggingface.load_from_disk"
        ) as mock_load:
            mock_ds = MagicMock()
            mock_ds.__iter__ = MagicMock(
                return_value=iter([
                    {"text": ""},
                    {"text": "Veljavno besedilo."},
                    {"text": None},
                ])
            )
            mock_ds.column_names = ["text"]
            mock_load.return_value = mock_ds

            extractor = HuggingFaceExtractor()
            docs = list(
                extractor.extract(tmp_path, source="fineweb2", domain="web")
            )

        assert len(docs) == 1
        assert docs[0].text == "Veljavno besedilo."

    def test_handles_dataset_dict(self, tmp_path: Path) -> None:
        """DatasetDict with splits is iterated across all splits."""
        (tmp_path / "sl").mkdir()
        with patch(
            "slm4ie.data.extractors.huggingface.load_from_disk"
        ) as mock_load:
            mock_split = MagicMock()
            mock_split.__iter__ = MagicMock(
                return_value=iter([{"text": "Train row."}])
            )
            mock_split.column_names = ["text"]

            mock_dd = MagicMock()
            mock_dd.column_names = {"train": ["text"]}
            mock_dd.keys = MagicMock(return_value=["train"])
            mock_dd.__getitem__ = MagicMock(return_value=mock_split)
            mock_load.return_value = mock_dd

            extractor = HuggingFaceExtractor()
            docs = list(
                extractor.extract(tmp_path, source="fineweb2", domain="web")
            )

        assert len(docs) == 1
        assert docs[0].text == "Train row."

    def test_registered_as_huggingface(self) -> None:
        """HuggingFaceExtractor is registered under 'huggingface'."""
        from slm4ie.data.extractors import get_extractor

        extractor = get_extractor("huggingface")
        assert isinstance(extractor, HuggingFaceExtractor)

    def test_skips_failed_config_dir(self, tmp_path: Path) -> None:
        """Config dirs that fail to load are skipped with a warning."""
        (tmp_path / "sl").mkdir()
        (tmp_path / "hr").mkdir()
        with patch(
            "slm4ie.data.extractors.huggingface.load_from_disk"
        ) as mock_load:
            mock_ds = MagicMock()
            mock_ds.__iter__ = MagicMock(
                return_value=iter([{"text": "Good row."}])
            )
            mock_ds.column_names = ["text"]
            mock_load.side_effect = [
                Exception("corrupt dataset"),
                mock_ds,
            ]

            extractor = HuggingFaceExtractor()
            docs = list(
                extractor.extract(tmp_path, source="fineweb2", domain="web")
            )

        assert len(docs) == 1
        assert docs[0].text == "Good row."
