"""Tests for HuggingFaceExtractor."""

from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from datasets import Dataset, DatasetDict

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


def _save_config(
    root: Path, config: str, rows: List[Dict[str, Any]]
) -> None:
    """Save rows as a single-split Arrow dataset under a config dir.

    Args:
        root (Path): Directory that stands in for `raw/<key>/`.
        config (str): Config subdirectory name to create.
        rows (List[Dict[str, Any]]): Rows to serialize.
    """
    Dataset.from_list(rows).save_to_disk(str(root / config))


class TestHuggingFaceDocIds:
    """Tests for natural-key and positional doc_id assignment."""

    def test_prefers_id_column(self, tmp_path: Path) -> None:
        """An `id` column supplies the doc_id verbatim."""
        _save_config(
            tmp_path,
            "sl",
            [{"text": "Prvi.", "id": "abc"}, {"text": "Drugi.", "id": "def"}],
        )

        docs = list(HuggingFaceExtractor().extract(tmp_path, "c4", "web"))

        assert [d.doc_id for d in docs] == ["abc", "def"]
        assert docs[0].uid == "c4:abc"

    def test_non_string_id_is_coerced(self, tmp_path: Path) -> None:
        """An integer key column becomes a string doc_id."""
        _save_config(
            tmp_path, "sl", [{"text": "Prvi.", "id": 7}, {"text": "Drugi.", "id": 8}]
        )

        docs = list(HuggingFaceExtractor().extract(tmp_path, "c4", "web"))

        assert [d.doc_id for d in docs] == ["7", "8"]

    def test_falls_back_to_url_column(self, tmp_path: Path) -> None:
        """`url` is used when no higher-priority key column is present."""
        _save_config(
            tmp_path,
            "sl",
            [
                {"text": "Prvi.", "url": "https://a.example/1"},
                {"text": "Drugi.", "url": "https://a.example/2"},
            ],
        )

        docs = list(HuggingFaceExtractor().extract(tmp_path, "c4", "web"))

        assert [d.doc_id for d in docs] == [
            "https://a.example/1",
            "https://a.example/2",
        ]

    def test_id_outranks_url(self, tmp_path: Path) -> None:
        """With both columns present, `id` wins on priority order."""
        _save_config(
            tmp_path,
            "sl",
            [{"text": "Prvi.", "id": "x1", "url": "https://a.example/1"}],
        )

        docs = list(HuggingFaceExtractor().extract(tmp_path, "c4", "web"))

        assert docs[0].doc_id == "x1"

    def test_natural_key_stays_in_metadata(self, tmp_path: Path) -> None:
        """The column that supplied the doc_id is not stripped."""
        _save_config(
            tmp_path,
            "sl",
            [{"text": "Prvi.", "id": "x1", "url": "https://a.example/1"}],
        )

        docs = list(HuggingFaceExtractor().extract(tmp_path, "c4", "web"))

        assert docs[0].metadata["id"] == "x1"
        assert docs[0].metadata["url"] == "https://a.example/1"

    def test_empty_natural_key_falls_through(self, tmp_path: Path) -> None:
        """A blank `id` is ignored in favour of the next candidate."""
        _save_config(
            tmp_path,
            "sl",
            [{"text": "Prvi.", "id": "", "url": "https://a.example/1"}],
        )

        docs = list(HuggingFaceExtractor().extract(tmp_path, "c4", "web"))

        assert docs[0].doc_id == "https://a.example/1"

    def test_positional_fallback_for_single_dataset(
        self, tmp_path: Path
    ) -> None:
        """A bare Dataset with no key column gets `<config>:<row>` ids."""
        _save_config(
            tmp_path, "sl", [{"text": "Prvi."}, {"text": "Drugi."}]
        )

        docs = list(HuggingFaceExtractor().extract(tmp_path, "c4", "web"))

        assert [d.doc_id for d in docs] == ["sl:00000000", "sl:00000001"]

    def test_positional_fallback_includes_split_for_dataset_dict(
        self, tmp_path: Path
    ) -> None:
        """A DatasetDict adds the split name between config and row index."""
        DatasetDict(
            {
                "train": Dataset.from_list([{"text": "Prvi."}]),
                "test": Dataset.from_list([{"text": "Drugi."}]),
            }
        ).save_to_disk(str(tmp_path / "sl"))

        docs = list(HuggingFaceExtractor().extract(tmp_path, "c4", "web"))

        assert sorted(d.doc_id for d in docs) == [
            "sl:test:00000000",
            "sl:train:00000000",
        ]

    def test_positional_ids_are_unique_across_configs(
        self, tmp_path: Path
    ) -> None:
        """Two config dirs cannot collide, since ids carry the config."""
        _save_config(tmp_path, "sl", [{"text": "Prvi."}])
        _save_config(tmp_path, "hr", [{"text": "Drugi."}])

        docs = list(HuggingFaceExtractor().extract(tmp_path, "c4", "web"))
        ids = [d.doc_id for d in docs]

        assert sorted(ids) == ["hr:00000000", "sl:00000000"]
        assert len(set(ids)) == len(ids)

    def test_row_index_counts_skipped_rows(self, tmp_path: Path) -> None:
        """Positional ids track the raw row index, not the emitted count.

        Anchoring on the true row offset keeps an id pointing at the same
        row even if text-emptiness filtering changes upstream.
        """
        _save_config(
            tmp_path,
            "sl",
            [{"text": ""}, {"text": "Drugi."}],
        )

        docs = list(HuggingFaceExtractor().extract(tmp_path, "c4", "web"))

        assert [d.doc_id for d in docs] == ["sl:00000001"]

    def test_ids_are_stable_across_runs(self, tmp_path: Path) -> None:
        """Re-extracting the same input reproduces identical ids."""
        _save_config(
            tmp_path, "sl", [{"text": "Prvi."}, {"text": "Drugi."}]
        )

        extractor = HuggingFaceExtractor()
        first = [d.doc_id for d in extractor.extract(tmp_path, "c4", "web")]
        second = [d.doc_id for d in extractor.extract(tmp_path, "c4", "web")]

        assert first == second
