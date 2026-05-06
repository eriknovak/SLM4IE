"""Tests for slm4ie.data.download module."""

import logging
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from unittest.mock import MagicMock, patch

from slm4ie.data.download import (
    BaseDownloader,
    HttpDownloader,
    DatasetConfig,
    HuggingFaceDownloader,
    download_datasets,
    load_config,
)


class TestDatasetConfig:
    """Tests for DatasetConfig dataclass."""

    def test_clarin_dataset_from_dict(self):
        data = {
            "enabled": True,
            "source": "clarin",
            "name": "Test Dataset",
            "urls": ["https://example.com/file.gz"],
            "output_dir": "test_dataset",
        }
        config = DatasetConfig.from_dict("test", data)
        assert config.key == "test"
        assert config.name == "Test Dataset"
        assert config.enabled is True
        assert config.source == "clarin"
        assert config.urls == ["https://example.com/file.gz"]
        assert config.output_dir == "test_dataset"
        assert config.manual is False
        assert config.repo_id is None
        assert config.configs is None
        assert config.note is None

    def test_huggingface_dataset_from_dict(self):
        data = {
            "enabled": True,
            "source": "huggingface",
            "name": "FinePDF",
            "repo_id": "HuggingFaceFW/finepdfs",
            "configs": ["slv_Latn"],
            "output_dir": "finepdf",
        }
        config = DatasetConfig.from_dict("finepdf", data)
        assert config.source == "huggingface"
        assert config.repo_id == "HuggingFaceFW/finepdfs"
        assert config.configs == ["slv_Latn"]
        assert config.urls == []

    def test_manual_dataset_from_dict(self):
        data = {
            "enabled": True,
            "source": "clarin",
            "name": "KAS 2.0",
            "manual": True,
            "urls": ["https://example.com/handle"],
            "output_dir": "kas",
            "note": "Download manually.",
        }
        config = DatasetConfig.from_dict("kas", data)
        assert config.manual is True
        assert config.note == "Download manually."

    def test_disabled_dataset_from_dict(self):
        data = {
            "enabled": False,
            "name": "Gigafida 2.0",
            "note": "Not available.",
        }
        config = DatasetConfig.from_dict("gigafida", data)
        assert config.enabled is False
        assert config.source == ""
        assert config.output_dir == ""

    def test_pretraining_dataset_defaults_benchmark_false(self):
        data = {
            "enabled": True,
            "source": "clarin",
            "name": "DS",
            "urls": ["https://example.com/x.gz"],
            "output_dir": "ds",
        }
        config = DatasetConfig.from_dict("ds", data)
        assert config.benchmark is False
        assert config.tasks == []

    def test_benchmark_dataset_from_dict(self):
        data = {
            "enabled": True,
            "benchmark": True,
            "source": "clarin",
            "name": "SUK",
            "urls": ["https://example.com/suk.zip"],
            "output_dir": "suk",
            "tasks": ["POS", "NER", "DEP"],
        }
        config = DatasetConfig.from_dict("suk", data)
        assert config.benchmark is True
        assert config.tasks == ["POS", "NER", "DEP"]


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_valid_config(self, tmp_path: Path):
        config_data = {
            "output_dir": "data/raw",
            "datasets": {
                "test_ds": {
                    "enabled": True,
                    "source": "clarin",
                    "name": "Test",
                    "urls": ["https://example.com/f.gz"],
                    "output_dir": "test_ds",
                },
            },
        }
        config_file = tmp_path / "download.yaml"
        config_file.write_text(yaml.dump(config_data))
        output_dir, datasets = load_config(config_file)
        assert output_dir == "data/raw"
        assert len(datasets) == 1
        assert "test_ds" in datasets
        assert datasets["test_ds"].name == "Test"

    def test_load_config_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_config(Path("/nonexistent/config.yaml"))

    def test_load_config_multiple_datasets(self, tmp_path: Path):
        config_data = {
            "output_dir": "data/raw",
            "datasets": {
                "ds1": {
                    "enabled": True,
                    "source": "clarin",
                    "name": "DS1",
                    "urls": ["https://example.com/1.gz"],
                    "output_dir": "ds1",
                },
                "ds2": {
                    "enabled": False,
                    "name": "DS2",
                    "note": "Disabled.",
                },
            },
        }
        config_file = tmp_path / "download.yaml"
        config_file.write_text(yaml.dump(config_data))
        _output_dir, datasets = load_config(config_file)
        assert len(datasets) == 2
        assert datasets["ds1"].enabled is True
        assert datasets["ds2"].enabled is False


class TestHttpDownloader:
    """Tests for HttpDownloader."""

    def test_is_base_downloader(self):
        downloader = HttpDownloader()
        assert isinstance(downloader, BaseDownloader)

    def test_extract_filename_from_url(self):
        downloader = HttpDownloader()
        url = (
            "https://www.clarin.si/repository/xmlui/bitstream/"
            "handle/11356/1427/classlawiki-sl.conllu.gz"
            "?sequence=6&isAllowed=y"
        )
        assert downloader._extract_filename(url) == (
            "classlawiki-sl.conllu.gz"
        )

    def test_extract_filename_no_query(self):
        downloader = HttpDownloader()
        url = "https://example.com/path/to/file.tar.gz"
        assert downloader._extract_filename(url) == "file.tar.gz"

    @patch("slm4ie.data.download.requests.get")
    def test_download_single_file(
        self, mock_get: MagicMock, tmp_path: Path
    ):
        mock_response = MagicMock()
        mock_response.headers = {"content-length": "100"}
        mock_response.iter_content = MagicMock(
            return_value=[b"x" * 100]
        )
        mock_response.raise_for_status = MagicMock()
        mock_response.__enter__ = MagicMock(
            return_value=mock_response
        )
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_get.return_value = mock_response

        config = DatasetConfig.from_dict(
            "test",
            {
                "enabled": True,
                "source": "clarin",
                "name": "Test",
                "urls": ["https://example.com/test.gz"],
                "output_dir": "test",
            },
        )
        output_dir = tmp_path / "test"

        downloader = HttpDownloader()
        result = downloader.download(config, output_dir)

        assert result == output_dir
        assert (output_dir / "test.gz").exists()
        assert (output_dir / "test.gz").read_bytes() == b"x" * 100
        assert not (output_dir / "test.gz.part").exists()

    @patch("slm4ie.data.download.requests.get")
    def test_download_creates_output_dir(
        self, mock_get: MagicMock, tmp_path: Path
    ):
        mock_response = MagicMock()
        mock_response.headers = {"content-length": "10"}
        mock_response.iter_content = MagicMock(
            return_value=[b"x" * 10]
        )
        mock_response.raise_for_status = MagicMock()
        mock_response.__enter__ = MagicMock(
            return_value=mock_response
        )
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_get.return_value = mock_response

        config = DatasetConfig.from_dict(
            "test",
            {
                "enabled": True,
                "source": "clarin",
                "name": "Test",
                "urls": ["https://example.com/data.gz"],
                "output_dir": "test",
            },
        )
        output_dir = tmp_path / "new_dir"
        assert not output_dir.exists()

        downloader = HttpDownloader()
        downloader.download(config, output_dir)

        assert output_dir.exists()


class TestHuggingFaceDownloader:
    """Tests for HuggingFaceDownloader."""

    def test_is_base_downloader(self):
        downloader = HuggingFaceDownloader()
        assert isinstance(downloader, BaseDownloader)

    @patch("slm4ie.data.download.load_dataset")
    def test_download_single_config(
        self, mock_load: MagicMock, tmp_path: Path
    ):
        mock_ds = MagicMock()
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

        downloader = HuggingFaceDownloader()
        result = downloader.download(config, output_dir)

        mock_load.assert_called_once_with(
            "HuggingFaceFW/finepdfs", "slv_Latn"
        )
        mock_ds.save_to_disk.assert_called_once_with(
            str(output_dir / "slv_Latn")
        )
        assert result == output_dir

    @patch("slm4ie.data.download.load_dataset")
    def test_download_multiple_configs(
        self, mock_load: MagicMock, tmp_path: Path
    ):
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

        downloader = HuggingFaceDownloader()
        downloader.download(config, output_dir)

        assert mock_load.call_count == 2

    @patch("slm4ie.data.download.load_dataset")
    def test_gated_dataset_missing_token_skips(
        self, mock_load: MagicMock, tmp_path: Path
    ):
        mock_load.side_effect = Exception(
            "Unauthorized: gated dataset"
        )

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

        downloader = HuggingFaceDownloader()
        result = downloader.download(config, output_dir)
        assert result == output_dir


class TestDownloadDatasets:
    """Tests for download_datasets orchestrator."""

    def _make_config_file(
        self, tmp_path: Path, datasets: dict
    ) -> Path:
        config_data = {
            "output_dir": str(tmp_path / "raw"),
            "datasets": datasets,
        }
        config_file = tmp_path / "download.yaml"
        config_file.write_text(yaml.dump(config_data))
        return config_file

    @patch.object(HttpDownloader, "download")
    def test_downloads_enabled_datasets(
        self, mock_dl: MagicMock, tmp_path: Path
    ):
        config_file = self._make_config_file(
            tmp_path,
            {
                "ds1": {
                    "enabled": True,
                    "source": "clarin",
                    "name": "DS1",
                    "urls": ["https://example.com/1.gz"],
                    "output_dir": "ds1",
                },
                "ds2": {
                    "enabled": False,
                    "name": "DS2",
                },
            },
        )
        download_datasets(config_file)
        mock_dl.assert_called_once()
        call_config = mock_dl.call_args[0][0]
        assert call_config.key == "ds1"

    @patch.object(HttpDownloader, "download")
    def test_skips_existing_directory(
        self, mock_dl: MagicMock, tmp_path: Path
    ):
        config_file = self._make_config_file(
            tmp_path,
            {
                "ds1": {
                    "enabled": True,
                    "source": "clarin",
                    "name": "DS1",
                    "urls": ["https://example.com/1.gz"],
                    "output_dir": "ds1",
                },
            },
        )
        ds_dir = tmp_path / "raw" / "ds1"
        ds_dir.mkdir(parents=True)
        (ds_dir / "existing.gz").write_bytes(b"data")
        download_datasets(config_file)
        mock_dl.assert_not_called()

    @patch.object(HttpDownloader, "download")
    def test_force_redownloads(
        self, mock_dl: MagicMock, tmp_path: Path
    ):
        config_file = self._make_config_file(
            tmp_path,
            {
                "ds1": {
                    "enabled": True,
                    "source": "clarin",
                    "name": "DS1",
                    "urls": ["https://example.com/1.gz"],
                    "output_dir": "ds1",
                },
            },
        )
        ds_dir = tmp_path / "raw" / "ds1"
        ds_dir.mkdir(parents=True)
        (ds_dir / "existing.gz").write_bytes(b"data")
        download_datasets(config_file, force=True)
        mock_dl.assert_called_once()

    @patch.object(HttpDownloader, "download")
    def test_select_specific_datasets(
        self, mock_dl: MagicMock, tmp_path: Path
    ):
        config_file = self._make_config_file(
            tmp_path,
            {
                "ds1": {
                    "enabled": True,
                    "source": "clarin",
                    "name": "DS1",
                    "urls": ["https://example.com/1.gz"],
                    "output_dir": "ds1",
                },
                "ds2": {
                    "enabled": True,
                    "source": "clarin",
                    "name": "DS2",
                    "urls": ["https://example.com/2.gz"],
                    "output_dir": "ds2",
                },
            },
        )
        download_datasets(
            config_file, dataset_keys=["ds2"]
        )
        mock_dl.assert_called_once()
        call_config = mock_dl.call_args[0][0]
        assert call_config.key == "ds2"

    def test_unknown_dataset_key_raises(
        self, tmp_path: Path
    ):
        config_file = self._make_config_file(
            tmp_path,
            {
                "ds1": {
                    "enabled": True,
                    "source": "clarin",
                    "name": "DS1",
                    "urls": ["https://example.com/1.gz"],
                    "output_dir": "ds1",
                },
            },
        )
        with pytest.raises(ValueError, match="unknown_ds"):
            download_datasets(
                config_file, dataset_keys=["unknown_ds"]
            )

    @patch.object(HttpDownloader, "download")
    def test_only_benchmarks_filters_default_selection(
        self, mock_dl: MagicMock, tmp_path: Path
    ):
        config_file = self._make_config_file(
            tmp_path,
            {
                "pretrain_ds": {
                    "enabled": True,
                    "source": "clarin",
                    "name": "Pretrain",
                    "urls": ["https://example.com/p.gz"],
                    "output_dir": "pretrain_ds",
                },
                "bench_ds": {
                    "enabled": True,
                    "benchmark": True,
                    "source": "clarin",
                    "name": "Bench",
                    "urls": ["https://example.com/b.gz"],
                    "output_dir": "bench_ds",
                    "tasks": ["NER"],
                },
            },
        )
        download_datasets(config_file, only_benchmarks=True)
        mock_dl.assert_called_once()
        assert mock_dl.call_args[0][0].key == "bench_ds"

    @patch.object(HttpDownloader, "download")
    def test_exclude_benchmarks_filters_default_selection(
        self, mock_dl: MagicMock, tmp_path: Path
    ):
        config_file = self._make_config_file(
            tmp_path,
            {
                "pretrain_ds": {
                    "enabled": True,
                    "source": "clarin",
                    "name": "Pretrain",
                    "urls": ["https://example.com/p.gz"],
                    "output_dir": "pretrain_ds",
                },
                "bench_ds": {
                    "enabled": True,
                    "benchmark": True,
                    "source": "clarin",
                    "name": "Bench",
                    "urls": ["https://example.com/b.gz"],
                    "output_dir": "bench_ds",
                },
            },
        )
        download_datasets(config_file, exclude_benchmarks=True)
        mock_dl.assert_called_once()
        assert mock_dl.call_args[0][0].key == "pretrain_ds"

    def test_only_and_exclude_benchmarks_mutually_exclusive(
        self, tmp_path: Path
    ):
        config_file = self._make_config_file(
            tmp_path,
            {
                "ds1": {
                    "enabled": True,
                    "source": "clarin",
                    "name": "DS1",
                    "urls": ["https://example.com/1.gz"],
                    "output_dir": "ds1",
                },
            },
        )
        with pytest.raises(ValueError, match="mutually exclusive"):
            download_datasets(
                config_file,
                only_benchmarks=True,
                exclude_benchmarks=True,
            )

    def test_manual_dataset_logs_note(
        self, tmp_path: Path, caplog
    ):
        config_file = self._make_config_file(
            tmp_path,
            {
                "kas": {
                    "enabled": True,
                    "source": "clarin",
                    "name": "KAS",
                    "manual": True,
                    "urls": ["https://example.com/handle"],
                    "output_dir": "kas",
                    "note": "Download manually.",
                },
            },
        )
        with caplog.at_level(logging.WARNING):
            download_datasets(config_file)
        assert "Download manually." in caplog.text


PROJECT_ROOT = str(
    Path(__file__).resolve().parent.parent.parent
)


class TestCLI:
    """Tests for the CLI entrypoint."""

    def test_cli_help(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.data.download",
                "--help",
            ],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0
        assert "--datasets" in result.stdout
        assert "--config" in result.stdout
        assert "--force" in result.stdout

    def test_cli_unknown_dataset(self, tmp_path: Path):
        config_data = {
            "output_dir": str(tmp_path / "raw"),
            "datasets": {
                "ds1": {
                    "enabled": True,
                    "source": "clarin",
                    "name": "DS1",
                    "urls": ["https://example.com/1.gz"],
                    "output_dir": "ds1",
                },
            },
        }
        config_file = tmp_path / "download.yaml"
        config_file.write_text(yaml.dump(config_data))

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.data.download",
                "--config",
                str(config_file),
                "--datasets",
                "nonexistent",
            ],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        assert result.returncode == 1
