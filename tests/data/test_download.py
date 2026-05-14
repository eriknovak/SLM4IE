"""Tests for slm4ie.data.download module."""

import logging
import subprocess
import sys
from pathlib import Path

import pytest
import requests
import yaml

from unittest.mock import MagicMock, patch

from slm4ie.data.catalog import ConfigError, DatasetConfig
from slm4ie.data.download import (
    DatasetDownloadError,
    DownloaderResult,
    download_datasets,
)
from slm4ie.data.sources import http as http_source
from slm4ie.data.sources import huggingface as hf_source


class TestHttpSource:
    """Tests for the http source downloader."""

    def test_extract_filename_from_url(self):
        """Filename is extracted from a URL with query parameters."""
        url = (
            "https://www.clarin.si/repository/xmlui/bitstream/"
            "handle/11356/1427/classlawiki-sl.conllu.gz"
            "?sequence=6&isAllowed=y"
        )
        assert http_source._extract_filename(url) == (
            "classlawiki-sl.conllu.gz"
        )

    def test_extract_filename_no_query(self):
        """Filename extraction handles URLs without query strings."""
        url = "https://example.com/path/to/file.tar.gz"
        assert http_source._extract_filename(url) == "file.tar.gz"

    @patch("slm4ie.data.sources.http.requests.get")
    def test_download_single_file(
        self, mock_get: MagicMock, tmp_path: Path
    ):
        """A single URL streams to disk and the .part file is removed on success."""
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
                "source": "http",
                "name": "Test",
                "urls": ["https://example.com/test.gz"],
                "output_dir": "test",
            },
        )
        output_dir = tmp_path / "test"

        http_source.download(config, output_dir, force=False)

        assert (output_dir / "test.gz").exists()
        assert (output_dir / "test.gz").read_bytes() == b"x" * 100
        assert not (output_dir / "test.gz.part").exists()

    @patch("slm4ie.data.sources.http.requests.get")
    def test_force_redownloads_existing_dest(
        self, mock_get: MagicMock, tmp_path: Path
    ):
        """`force=True` overwrites an existing destination file."""
        mock_response = MagicMock()
        mock_response.headers = {"content-length": "5"}
        mock_response.iter_content = MagicMock(return_value=[b"fresh"])
        mock_response.raise_for_status = MagicMock()
        mock_response.status_code = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_get.return_value = mock_response

        config = DatasetConfig.from_dict(
            "test",
            {
                "enabled": True,
                "source": "http",
                "name": "Test",
                "urls": ["https://example.com/test.gz"],
                "output_dir": "test",
            },
        )
        output_dir = tmp_path / "test"
        output_dir.mkdir(parents=True)
        dest = output_dir / "test.gz"
        dest.write_bytes(b"stale")

        http_source.download(config, output_dir, force=True)

        assert dest.read_bytes() == b"fresh"

    @patch("slm4ie.data.sources.http.requests.get")
    def test_force_clears_stale_part(
        self, mock_get: MagicMock, tmp_path: Path
    ):
        """`force=True` removes any leftover `.part` so resume does not engage."""
        mock_response = MagicMock()
        mock_response.headers = {"content-length": "5"}
        mock_response.iter_content = MagicMock(return_value=[b"fresh"])
        mock_response.raise_for_status = MagicMock()
        mock_response.status_code = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_get.return_value = mock_response

        config = DatasetConfig.from_dict(
            "test",
            {
                "enabled": True,
                "source": "http",
                "name": "Test",
                "urls": ["https://example.com/test.gz"],
                "output_dir": "test",
            },
        )
        output_dir = tmp_path / "test"
        output_dir.mkdir(parents=True)
        (output_dir / "test.gz").write_bytes(b"old")
        (output_dir / "test.gz.part").write_bytes(b"partial")

        http_source.download(config, output_dir, force=True)

        # No Range header should be present on a force redownload.
        _args, kwargs = mock_get.call_args
        headers = kwargs.get("headers", {}) or {}
        assert "Range" not in headers
        assert (output_dir / "test.gz").read_bytes() == b"fresh"

    @patch("slm4ie.data.sources.http.requests.get")
    def test_per_url_failure_collected(
        self, mock_get: MagicMock, tmp_path: Path
    ):
        """A failing URL is recorded in `failed` while others still complete."""
        good_response = MagicMock()
        good_response.headers = {"content-length": "4"}
        good_response.iter_content = MagicMock(return_value=[b"good"])
        good_response.raise_for_status = MagicMock()
        good_response.status_code = 200
        good_response.__enter__ = MagicMock(return_value=good_response)
        good_response.__exit__ = MagicMock(return_value=False)

        calls = {"n": 0}

        def _side_effect(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return good_response
            raise requests.ConnectionError("network unreachable")

        mock_get.side_effect = _side_effect

        config = DatasetConfig.from_dict(
            "test",
            {
                "enabled": True,
                "source": "http",
                "name": "Test",
                "urls": [
                    "https://example.com/a.gz",
                    "https://example.com/b.gz",
                ],
                "output_dir": "test",
            },
        )
        output_dir = tmp_path / "test"

        result = http_source.download(config, output_dir, force=False)

        assert result.completed == ["https://example.com/a.gz"]
        assert len(result.failed) == 1
        assert result.failed[0][0] == "https://example.com/b.gz"
        assert (output_dir / "a.gz").read_bytes() == b"good"

    @patch("slm4ie.data.sources.http.requests.get")
    def test_download_creates_output_dir(
        self, mock_get: MagicMock, tmp_path: Path
    ):
        """Download creates the output directory if it does not exist."""
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
                "source": "http",
                "name": "Test",
                "urls": ["https://example.com/data.gz"],
                "output_dir": "test",
            },
        )
        output_dir = tmp_path / "new_dir"
        assert not output_dir.exists()

        http_source.download(config, output_dir, force=False)

        assert output_dir.exists()


class TestHuggingFaceSource:
    """Tests for the huggingface source downloader."""

    @patch("slm4ie.data.sources.huggingface.load_dataset")
    def test_download_single_config(
        self, mock_load: MagicMock, tmp_path: Path
    ):
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

        mock_load.assert_called_once_with(
            "HuggingFaceFW/finepdfs", "slv_Latn"
        )
        mock_ds.save_to_disk.assert_called_once_with(
            str(output_dir / "slv_Latn.partial")
        )
        assert (output_dir / "slv_Latn" / "shard.arrow").exists()
        assert isinstance(result, DownloaderResult)

    @patch("slm4ie.data.sources.huggingface.load_dataset")
    def test_download_multiple_configs(
        self, mock_load: MagicMock, tmp_path: Path
    ):
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

    @patch("slm4ie.data.sources.huggingface.load_dataset")
    def test_gated_failure_now_propagates_to_failed(
        self, mock_load: MagicMock, tmp_path: Path
    ):
        """Gated/auth failures surface in `failed`, augmented with the note."""
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

        result = hf_source.download(config, output_dir, force=False)

        assert result.completed == []
        assert len(result.failed) == 1
        unit, exc = result.failed[0]
        assert unit == "sl"
        assert "Unauthorized" in str(exc)
        assert "Requires HF_TOKEN." in str(exc)

    @patch("slm4ie.data.sources.huggingface.load_dataset")
    def test_per_config_failure_collected(
        self, mock_load: MagicMock, tmp_path: Path
    ):
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

    @patch("slm4ie.data.sources.huggingface.load_dataset")
    def test_atomic_swap_via_ready_marker(
        self, mock_load: MagicMock, tmp_path: Path
    ):
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

    @patch("slm4ie.data.sources.huggingface.load_dataset")
    def test_recovery_finishes_swap_from_ready_marker(
        self, mock_load: MagicMock, tmp_path: Path
    ):
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

    @patch("slm4ie.data.sources.huggingface.load_dataset")
    def test_recovery_cleans_orphan_partial(
        self, mock_load: MagicMock, tmp_path: Path
    ):
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

    @patch("slm4ie.data.sources.huggingface.load_dataset")
    def test_force_replaces_existing_via_swap(
        self, mock_load: MagicMock, tmp_path: Path
    ):
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

    @patch("slm4ie.data.sources.huggingface.load_dataset")
    def test_force_keeps_old_data_until_swap(
        self, mock_load: MagicMock, tmp_path: Path
    ):
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

    @patch("slm4ie.data.sources.huggingface.load_dataset")
    def test_existing_save_path_skips_when_not_force(
        self, mock_load: MagicMock, tmp_path: Path
    ):
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

    @patch("slm4ie.data.sources.http.download")
    def test_downloads_enabled_datasets(
        self, mock_dl: MagicMock, tmp_path: Path
    ):
        """Only enabled datasets are passed to the downloader."""
        mock_dl.return_value = DownloaderResult(completed=[], failed=[])
        config_file = self._make_config_file(
            tmp_path,
            {
                "ds1": {
                    "enabled": True,
                    "source": "http",
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

    @patch("slm4ie.data.sources.http.download")
    def test_dispatch_runs_when_output_exists(
        self, mock_dl: MagicMock, tmp_path: Path
    ):
        """The source is still invoked when output exists; per-unit skip is its job."""
        mock_dl.return_value = DownloaderResult(completed=[], failed=[])
        config_file = self._make_config_file(
            tmp_path,
            {
                "ds1": {
                    "enabled": True,
                    "source": "http",
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
        mock_dl.assert_called_once()
        # force defaults to False for the source call.
        _config, _output, force_arg = mock_dl.call_args[0]
        assert force_arg is False

    @patch("slm4ie.data.sources.http.download")
    def test_force_redownloads(
        self, mock_dl: MagicMock, tmp_path: Path
    ):
        """`force=True` flows through to the source even when output exists."""
        mock_dl.return_value = DownloaderResult(completed=[], failed=[])
        config_file = self._make_config_file(
            tmp_path,
            {
                "ds1": {
                    "enabled": True,
                    "source": "http",
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
        _config, _output, force_arg = mock_dl.call_args[0]
        assert force_arg is True

    @patch("slm4ie.data.sources.http.download")
    def test_select_specific_datasets(
        self, mock_dl: MagicMock, tmp_path: Path
    ):
        """`dataset_keys` restricts the run to the named datasets."""
        mock_dl.return_value = DownloaderResult(completed=[], failed=[])
        config_file = self._make_config_file(
            tmp_path,
            {
                "ds1": {
                    "enabled": True,
                    "source": "http",
                    "name": "DS1",
                    "urls": ["https://example.com/1.gz"],
                    "output_dir": "ds1",
                },
                "ds2": {
                    "enabled": True,
                    "source": "http",
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
        """An unknown dataset key in `dataset_keys` raises ValueError."""
        config_file = self._make_config_file(
            tmp_path,
            {
                "ds1": {
                    "enabled": True,
                    "source": "http",
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

    @patch("slm4ie.data.sources.http.download")
    def test_only_benchmarks_filters_default_selection(
        self, mock_dl: MagicMock, tmp_path: Path
    ):
        """`only_benchmarks=True` keeps benchmark datasets only."""
        mock_dl.return_value = DownloaderResult(completed=[], failed=[])
        config_file = self._make_config_file(
            tmp_path,
            {
                "pretrain_ds": {
                    "enabled": True,
                    "source": "http",
                    "name": "Pretrain",
                    "urls": ["https://example.com/p.gz"],
                    "output_dir": "pretrain_ds",
                },
                "bench_ds": {
                    "enabled": True,
                    "benchmark": True,
                    "source": "http",
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

    @patch("slm4ie.data.sources.http.download")
    def test_exclude_benchmarks_filters_default_selection(
        self, mock_dl: MagicMock, tmp_path: Path
    ):
        """`exclude_benchmarks=True` drops benchmark datasets from the run."""
        mock_dl.return_value = DownloaderResult(completed=[], failed=[])
        config_file = self._make_config_file(
            tmp_path,
            {
                "pretrain_ds": {
                    "enabled": True,
                    "source": "http",
                    "name": "Pretrain",
                    "urls": ["https://example.com/p.gz"],
                    "output_dir": "pretrain_ds",
                },
                "bench_ds": {
                    "enabled": True,
                    "benchmark": True,
                    "source": "http",
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
        """Passing both `only_benchmarks` and `exclude_benchmarks` raises."""
        config_file = self._make_config_file(
            tmp_path,
            {
                "ds1": {
                    "enabled": True,
                    "source": "http",
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
        """Manual datasets emit their note as a warning instead of downloading."""
        config_file = self._make_config_file(
            tmp_path,
            {
                "kas": {
                    "enabled": True,
                    "source": "http",
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


class TestFailFastValidation:
    """Tests for `_validate_selection` fail-fast behaviour."""

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

    def test_explicit_disabled_raises_config_error(self, tmp_path: Path):
        """Naming a disabled dataset escalates to ConfigError with the note."""
        config_file = self._make_config_file(
            tmp_path,
            {
                "ds1": {
                    "enabled": True,
                    "source": "http",
                    "name": "DS1",
                    "urls": ["https://example.com/1.gz"],
                    "output_dir": "ds1",
                },
                "ds2": {
                    "enabled": False,
                    "name": "DS2",
                    "note": "License not granted.",
                },
            },
        )
        with pytest.raises(ConfigError) as excinfo:
            download_datasets(
                config_file, dataset_keys=["ds1", "ds2"]
            )
        msg = str(excinfo.value)
        assert "ds2" in msg
        assert "disabled" in msg
        assert "License not granted." in msg

    def test_explicit_manual_missing_raises(self, tmp_path: Path):
        """Naming a manual dataset whose dir is empty raises ConfigError."""
        config_file = self._make_config_file(
            tmp_path,
            {
                "kas": {
                    "enabled": True,
                    "source": "http",
                    "name": "KAS",
                    "manual": True,
                    "urls": ["https://example.com/handle"],
                    "output_dir": "kas",
                    "note": "Download manually.",
                },
            },
        )
        with pytest.raises(ConfigError) as excinfo:
            download_datasets(
                config_file, dataset_keys=["kas"]
            )
        msg = str(excinfo.value)
        assert "kas" in msg
        assert "manual" in msg
        assert "Download manually." in msg

    @patch("slm4ie.data.sources.http.download")
    def test_explicit_manual_present_succeeds(
        self, mock_dl: MagicMock, tmp_path: Path
    ):
        """A manual dataset with files on disk passes validation."""
        config_file = self._make_config_file(
            tmp_path,
            {
                "kas": {
                    "enabled": True,
                    "source": "http",
                    "name": "KAS",
                    "manual": True,
                    "urls": ["https://example.com/handle"],
                    "output_dir": "kas",
                    "note": "Download manually.",
                },
            },
        )
        ds_dir = tmp_path / "raw" / "kas"
        ds_dir.mkdir(parents=True)
        (ds_dir / "data.tar.gz").write_bytes(b"data")
        download_datasets(config_file, dataset_keys=["kas"])
        # The manual branch in _download_one early-returns; the source
        # downloader must not be called.
        mock_dl.assert_not_called()

    def test_unknown_source_raises(self, tmp_path: Path):
        """An unknown `source` triggers ConfigError regardless of explicit mode."""
        config_file = self._make_config_file(
            tmp_path,
            {
                "ds1": {
                    "enabled": True,
                    "source": "bogus",
                    "name": "DS1",
                    "urls": ["https://example.com/1.gz"],
                    "output_dir": "ds1",
                    "note": "Wrong source.",
                },
            },
        )
        with pytest.raises(ConfigError) as excinfo:
            download_datasets(config_file)
        msg = str(excinfo.value)
        assert "ds1" in msg
        assert "bogus" in msg
        assert "Wrong source." in msg

    def test_validation_aggregates_all_problems(self, tmp_path: Path):
        """Multiple problems surface together in a single ConfigError."""
        config_file = self._make_config_file(
            tmp_path,
            {
                "ds1": {
                    "enabled": True,
                    "source": "bogus",
                    "name": "DS1",
                    "urls": ["https://example.com/1.gz"],
                    "output_dir": "ds1",
                },
                "ds2": {
                    "enabled": False,
                    "name": "DS2",
                },
                "ds3": {
                    "enabled": True,
                    "manual": True,
                    "source": "http",
                    "name": "DS3",
                    "urls": ["https://example.com/3.gz"],
                    "output_dir": "ds3",
                },
            },
        )
        with pytest.raises(ConfigError) as excinfo:
            download_datasets(
                config_file, dataset_keys=["ds1", "ds2", "ds3"]
            )
        problems = excinfo.value.problems
        assert len(problems) == 3
        keys_hit = {p.split(":", 1)[0] for p in problems}
        assert keys_hit == {"ds1", "ds2", "ds3"}

    @patch("slm4ie.data.sources.http.download")
    def test_default_mode_skips_disabled_quietly(
        self, mock_dl: MagicMock, tmp_path: Path
    ):
        """Default-mode runs do not raise on disabled entries."""
        mock_dl.return_value = DownloaderResult(completed=[], failed=[])
        config_file = self._make_config_file(
            tmp_path,
            {
                "ds1": {
                    "enabled": True,
                    "source": "http",
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
        )
        download_datasets(config_file)
        mock_dl.assert_called_once()
        assert mock_dl.call_args[0][0].key == "ds1"


class TestDatasetDownloadError:
    """Direct construction tests for `DatasetDownloadError`."""

    def test_message_format_includes_units_and_counts(self):
        """`str(...)` lists dataset key, count, and per-unit summaries."""
        failed = [
            ("sl", RuntimeError("auth required")),
            ("hr", ConnectionError("dns failed")),
        ]
        err = DatasetDownloadError(
            "culturax", failed, n_completed=0, n_total=2
        )
        msg = str(err)
        assert "culturax" in msg
        assert "2/2 sub-units failed" in msg
        assert "sl" in msg
        assert "RuntimeError" in msg
        assert "hr" in msg
        assert "ConnectionError" in msg


class TestIntraDatasetFailureSurfacing:
    """Tests that per-sub-unit failures bubble through `download_datasets`."""

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

    @patch("slm4ie.data.sources.http.download")
    def test_intra_dataset_failures_surface_in_top_level_runtime_error(
        self, mock_dl: MagicMock, tmp_path: Path
    ):
        """A `DatasetDownloadError` from `_download_one` becomes the run error."""
        mock_dl.return_value = DownloaderResult(
            completed=["https://example.com/ok.gz"],
            failed=[
                (
                    "https://example.com/bad.gz",
                    RuntimeError("HTTP 500"),
                ),
            ],
        )
        log_dir = tmp_path / "logs"
        config_file = self._make_config_file(
            tmp_path,
            {
                "ds1": {
                    "enabled": True,
                    "source": "http",
                    "name": "DS1",
                    "urls": [
                        "https://example.com/ok.gz",
                        "https://example.com/bad.gz",
                    ],
                    "output_dir": "ds1",
                },
            },
        )

        with pytest.raises(RuntimeError) as excinfo:
            download_datasets(config_file, log_dir=log_dir)

        msg = str(excinfo.value)
        assert "ds1" in msg
        assert "https://example.com/bad.gz" in msg
        assert str(log_dir) in msg


PROJECT_ROOT = str(
    Path(__file__).resolve().parent.parent.parent
)


class TestCLI:
    """Tests for the CLI entrypoint."""

    def test_cli_help(self):
        """`--help` exits cleanly and lists the main flags."""
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
        assert "datasets" in result.stdout
        assert "--all" in result.stdout
        assert "--config" in result.stdout
        assert "--force" in result.stdout

    def test_cli_requires_selection(self):
        """Bare invocation errors out: must pass datasets or --all."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.data.download",
            ],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        assert result.returncode != 0
        assert "--all" in result.stderr

    def test_cli_unknown_dataset(self, tmp_path: Path):
        """The CLI exits non-zero when a positional names an unknown key."""
        config_data = {
            "output_dir": str(tmp_path / "raw"),
            "datasets": {
                "ds1": {
                    "enabled": True,
                    "source": "http",
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
                "nonexistent",
            ],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        assert result.returncode == 1
