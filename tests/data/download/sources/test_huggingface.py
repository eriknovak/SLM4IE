"""Tests for the HuggingFace download source (slm4ie/data/download/sources/huggingface.py)."""

from pathlib import Path


from unittest.mock import MagicMock, patch

from slm4ie.data.download.config import DatasetConfig
from slm4ie.data.download.run import (
    DownloaderResult,
)
from slm4ie.data.download.sources import huggingface as hf_source


class TestHuggingFaceSource:
    """Tests for the huggingface source downloader."""

    @patch("slm4ie.data.download.sources.huggingface.load_dataset")
    def test_download_single_config(self, mock_load: MagicMock, tmp_path: Path):
        """A single HF config streams via .partial and ends at the final dir."""

        def _save(path: str) -> None:
            p = Path(path)
            p.mkdir(parents=True, exist_ok=True)
            (p / "shard.arrow").write_bytes(b"data")

        mock_ds = MagicMock()
        mock_ds.save_to_disk.side_effect = _save
        mock_load.return_value = mock_ds

        config = DatasetConfig.from_dict(
            "finepdf",
            {
                "enabled": True,
                "source": "huggingface",
                "name": "FinePDF",
                "repo_id": "HuggingFaceFW/finepdfs",
                "configs": ["slv_Latn"],
                "output_dir": "finepdf",
            },
        )
        output_dir = tmp_path / "finepdf"

        result = hf_source.download(config, output_dir, force=False)

        mock_load.assert_called_once_with("HuggingFaceFW/finepdfs", "slv_Latn")
        mock_ds.save_to_disk.assert_called_once_with(str(output_dir / "slv_Latn.partial"))
        assert (output_dir / "slv_Latn" / "shard.arrow").exists()
        assert isinstance(result, DownloaderResult)

    @patch("slm4ie.data.download.sources.huggingface.load_dataset")
    def test_download_multiple_configs(self, mock_load: MagicMock, tmp_path: Path):
        """Each declared HF config triggers its own load_dataset call."""
        mock_ds = MagicMock()
        mock_load.return_value = mock_ds

        config = DatasetConfig.from_dict(
            "finepdf",
            {
                "enabled": True,
                "source": "huggingface",
                "name": "FinePDF",
                "repo_id": "HuggingFaceFW/finepdfs",
                "configs": ["slv_Latn", "deu_Latn"],
                "output_dir": "finepdf",
            },
        )
        output_dir = tmp_path / "finepdf"

        hf_source.download(config, output_dir, force=False)

        assert mock_load.call_count == 2

    @patch("slm4ie.data.download.sources.huggingface.load_dataset")
    def test_gated_failure_now_propagates_to_failed(self, mock_load: MagicMock, tmp_path: Path):
        """Gated/auth failures surface in `failed`, augmented with the note."""
        mock_load.side_effect = Exception("Unauthorized: gated dataset")

        config = DatasetConfig.from_dict(
            "culturax",
            {
                "enabled": True,
                "source": "huggingface",
                "name": "CulturaX",
                "repo_id": "uonlp/CulturaX",
                "configs": ["sl"],
                "output_dir": "culturax",
                "note": "Requires HF_TOKEN.",
            },
        )
        output_dir = tmp_path / "culturax"

        result = hf_source.download(config, output_dir, force=False)

        assert result.completed == []
        assert len(result.failed) == 1
        unit, exc = result.failed[0]
        assert unit == "sl"
        assert "Unauthorized" in str(exc)
        assert "Requires HF_TOKEN." in str(exc)

    @patch("slm4ie.data.download.sources.huggingface.load_dataset")
    def test_per_config_failure_collected(self, mock_load: MagicMock, tmp_path: Path):
        """One failing config does not abort the remaining configs."""

        def _save(path: str) -> None:
            p = Path(path)
            p.mkdir(parents=True, exist_ok=True)
            (p / "shard.arrow").write_bytes(b"data")

        def _load(repo_id: str, cfg_name: str):
            if cfg_name == "bad":
                raise RuntimeError("boom")
            mock_ds = MagicMock()
            mock_ds.save_to_disk.side_effect = _save
            return mock_ds

        mock_load.side_effect = _load

        config = DatasetConfig.from_dict(
            "test_hf",
            {
                "enabled": True,
                "source": "huggingface",
                "name": "Test HF",
                "repo_id": "foo/bar",
                "configs": ["good", "bad"],
                "output_dir": "test_hf",
            },
        )
        output_dir = tmp_path / "test_hf"

        result = hf_source.download(config, output_dir, force=False)

        assert result.completed == ["good"]
        assert len(result.failed) == 1
        assert result.failed[0][0] == "bad"

    @patch("slm4ie.data.download.sources.huggingface.load_dataset")
    def test_atomic_swap_via_ready_marker(self, mock_load: MagicMock, tmp_path: Path):
        """Successful downloads pass through .partial then land at save_path."""

        def _save(path: str) -> None:
            p = Path(path)
            p.mkdir(parents=True, exist_ok=True)
            (p / "shard.arrow").write_bytes(b"data")

        mock_ds = MagicMock()
        mock_ds.save_to_disk.side_effect = _save
        mock_load.return_value = mock_ds

        config = DatasetConfig.from_dict(
            "finepdf",
            {
                "enabled": True,
                "source": "huggingface",
                "name": "FinePDF",
                "repo_id": "HuggingFaceFW/finepdfs",
                "configs": ["slv_Latn"],
                "output_dir": "finepdf",
            },
        )
        output_dir = tmp_path / "finepdf"

        hf_source.download(config, output_dir, force=False)

        save_path = output_dir / "slv_Latn"
        assert save_path.exists()
        assert (save_path / "shard.arrow").read_bytes() == b"data"
        assert not (output_dir / "slv_Latn.partial").exists()
        assert not (output_dir / "slv_Latn.ready").exists()

    @patch("slm4ie.data.download.sources.huggingface.load_dataset")
    def test_recovery_finishes_swap_from_ready_marker(self, mock_load: MagicMock, tmp_path: Path):
        """A pre-existing .ready directory is renamed without re-downloading."""
        output_dir = tmp_path / "finepdf"
        output_dir.mkdir(parents=True)
        ready = output_dir / "slv_Latn.ready"
        ready.mkdir()
        (ready / "shard.arrow").write_bytes(b"recovered")

        config = DatasetConfig.from_dict(
            "finepdf",
            {
                "enabled": True,
                "source": "huggingface",
                "name": "FinePDF",
                "repo_id": "HuggingFaceFW/finepdfs",
                "configs": ["slv_Latn"],
                "output_dir": "finepdf",
            },
        )

        hf_source.download(config, output_dir, force=False)

        mock_load.assert_not_called()
        save_path = output_dir / "slv_Latn"
        assert save_path.exists()
        assert (save_path / "shard.arrow").read_bytes() == b"recovered"
        assert not ready.exists()

    @patch("slm4ie.data.download.sources.huggingface.load_dataset")
    def test_recovery_cleans_orphan_partial(self, mock_load: MagicMock, tmp_path: Path):
        """An orphan .partial directory is removed before a fresh download."""

        def _save(path: str) -> None:
            p = Path(path)
            p.mkdir(parents=True, exist_ok=True)
            (p / "shard.arrow").write_bytes(b"fresh")

        mock_ds = MagicMock()
        mock_ds.save_to_disk.side_effect = _save
        mock_load.return_value = mock_ds

        output_dir = tmp_path / "finepdf"
        output_dir.mkdir(parents=True)
        partial = output_dir / "slv_Latn.partial"
        partial.mkdir()
        (partial / "stale.arrow").write_bytes(b"stale")

        config = DatasetConfig.from_dict(
            "finepdf",
            {
                "enabled": True,
                "source": "huggingface",
                "name": "FinePDF",
                "repo_id": "HuggingFaceFW/finepdfs",
                "configs": ["slv_Latn"],
                "output_dir": "finepdf",
            },
        )

        hf_source.download(config, output_dir, force=False)

        mock_load.assert_called_once()
        save_path = output_dir / "slv_Latn"
        assert (save_path / "shard.arrow").read_bytes() == b"fresh"
        assert not (save_path / "stale.arrow").exists()
        assert not partial.exists()

    @patch("slm4ie.data.download.sources.huggingface.load_dataset")
    def test_force_replaces_existing_via_swap(self, mock_load: MagicMock, tmp_path: Path):
        """`force=True` rewrites save_path through the .partial -> .ready swap."""

        def _save(path: str) -> None:
            p = Path(path)
            p.mkdir(parents=True, exist_ok=True)
            (p / "shard.arrow").write_bytes(b"new")

        mock_ds = MagicMock()
        mock_ds.save_to_disk.side_effect = _save
        mock_load.return_value = mock_ds

        output_dir = tmp_path / "finepdf"
        save_path = output_dir / "slv_Latn"
        save_path.mkdir(parents=True)
        (save_path / "shard.arrow").write_bytes(b"stale")

        config = DatasetConfig.from_dict(
            "finepdf",
            {
                "enabled": True,
                "source": "huggingface",
                "name": "FinePDF",
                "repo_id": "HuggingFaceFW/finepdfs",
                "configs": ["slv_Latn"],
                "output_dir": "finepdf",
            },
        )

        hf_source.download(config, output_dir, force=True)

        mock_load.assert_called_once()
        assert (save_path / "shard.arrow").read_bytes() == b"new"
        assert not (output_dir / "slv_Latn.partial").exists()
        assert not (output_dir / "slv_Latn.ready").exists()

    @patch("slm4ie.data.download.sources.huggingface.load_dataset")
    def test_force_keeps_old_data_until_swap(self, mock_load: MagicMock, tmp_path: Path):
        """Failure after writing to staging leaves the prior save_path intact."""

        def _save_then_fail(path: str) -> None:
            p = Path(path)
            p.mkdir(parents=True, exist_ok=True)
            (p / "partial.arrow").write_bytes(b"halfway")
            raise RuntimeError("writer crashed after staging")

        mock_ds = MagicMock()
        mock_ds.save_to_disk.side_effect = _save_then_fail
        mock_load.return_value = mock_ds

        output_dir = tmp_path / "finepdf"
        save_path = output_dir / "slv_Latn"
        save_path.mkdir(parents=True)
        (save_path / "shard.arrow").write_bytes(b"original")

        config = DatasetConfig.from_dict(
            "finepdf",
            {
                "enabled": True,
                "source": "huggingface",
                "name": "FinePDF",
                "repo_id": "HuggingFaceFW/finepdfs",
                "configs": ["slv_Latn"],
                "output_dir": "finepdf",
            },
        )

        # In step 6, exceptions are still warn-and-continue.
        hf_source.download(config, output_dir, force=True)

        assert (save_path / "shard.arrow").read_bytes() == b"original"

    @patch("slm4ie.data.download.sources.huggingface.load_dataset")
    def test_existing_save_path_skips_when_not_force(self, mock_load: MagicMock, tmp_path: Path):
        """A populated save_path short-circuits without calling load_dataset."""
        output_dir = tmp_path / "finepdf"
        save_path = output_dir / "slv_Latn"
        save_path.mkdir(parents=True)
        (save_path / "shard.arrow").write_bytes(b"existing")

        config = DatasetConfig.from_dict(
            "finepdf",
            {
                "enabled": True,
                "source": "huggingface",
                "name": "FinePDF",
                "repo_id": "HuggingFaceFW/finepdfs",
                "configs": ["slv_Latn"],
                "output_dir": "finepdf",
            },
        )

        hf_source.download(config, output_dir, force=False)
        mock_load.assert_not_called()
        assert (save_path / "shard.arrow").read_bytes() == b"existing"
