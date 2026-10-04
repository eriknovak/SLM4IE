"""Tests for slm4ie.data.download.run module."""

import logging
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from unittest.mock import MagicMock, patch

from slm4ie.data.download.config import ConfigError
from slm4ie.data.download.run import (
    DatasetDownloadError,
    DownloaderResult,
    download_datasets,
)


class TestDownloadDatasets:
    """Tests for download_datasets orchestrator."""

    def _make_config_file(self, tmp_path: Path, datasets: dict) -> Path:
        config_data = {
            "output_dir": str(tmp_path / "raw"),
            "datasets": datasets,
        }
        config_file = tmp_path / "download.yaml"
        config_file.write_text(yaml.dump(config_data))
        return config_file

    @patch("slm4ie.data.download.sources.http.download")
    def test_downloads_enabled_datasets(self, mock_dl: MagicMock, tmp_path: Path):
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

    @patch("slm4ie.data.download.sources.http.download")
    def test_dispatch_runs_when_output_exists(self, mock_dl: MagicMock, tmp_path: Path):
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

    @patch("slm4ie.data.download.sources.http.download")
    def test_force_redownloads(self, mock_dl: MagicMock, tmp_path: Path):
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

    @patch("slm4ie.data.download.sources.http.download")
    def test_select_specific_datasets(self, mock_dl: MagicMock, tmp_path: Path):
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
        download_datasets(config_file, dataset_keys=["ds2"])
        mock_dl.assert_called_once()
        call_config = mock_dl.call_args[0][0]
        assert call_config.key == "ds2"

    def test_unknown_dataset_key_raises(self, tmp_path: Path):
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
            download_datasets(config_file, dataset_keys=["unknown_ds"])

    @patch("slm4ie.data.download.sources.http.download")
    def test_only_benchmarks_filters_default_selection(self, mock_dl: MagicMock, tmp_path: Path):
        """`only_benchmarks=True` keeps every non-pretrain dataset."""
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
                    "role": "benchmark",
                    "source": "http",
                    "name": "Bench",
                    "urls": ["https://example.com/b.gz"],
                    "output_dir": "bench_ds",
                    "tasks": ["NER"],
                },
                "lexicon_ds": {
                    "enabled": True,
                    "role": "lexicon",
                    "source": "http",
                    "name": "Lexicon",
                    "urls": ["https://example.com/l.gz"],
                    "output_dir": "lexicon_ds",
                    "tasks": ["TOKENIZER"],
                },
            },
        )
        download_datasets(config_file, only_benchmarks=True)
        assert mock_dl.call_count == 2
        keys = {call[0][0].key for call in mock_dl.call_args_list}
        assert keys == {"bench_ds", "lexicon_ds"}

    @patch("slm4ie.data.download.sources.http.download")
    def test_exclude_benchmarks_filters_default_selection(self, mock_dl: MagicMock, tmp_path: Path):
        """`exclude_benchmarks=True` keeps only `role: pretrain` datasets."""
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
                    "role": "benchmark",
                    "source": "http",
                    "name": "Bench",
                    "urls": ["https://example.com/b.gz"],
                    "output_dir": "bench_ds",
                },
                "lexicon_ds": {
                    "enabled": True,
                    "role": "lexicon",
                    "source": "http",
                    "name": "Lexicon",
                    "urls": ["https://example.com/l.gz"],
                    "output_dir": "lexicon_ds",
                },
            },
        )
        download_datasets(config_file, exclude_benchmarks=True)
        mock_dl.assert_called_once()
        assert mock_dl.call_args[0][0].key == "pretrain_ds"

    def test_only_and_exclude_benchmarks_mutually_exclusive(self, tmp_path: Path):
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

    def test_manual_dataset_logs_note(self, tmp_path: Path, caplog):
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

    def _make_config_file(self, tmp_path: Path, datasets: dict) -> Path:
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
            download_datasets(config_file, dataset_keys=["ds1", "ds2"])
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
            download_datasets(config_file, dataset_keys=["kas"])
        msg = str(excinfo.value)
        assert "kas" in msg
        assert "manual" in msg
        assert "Download manually." in msg

    @patch("slm4ie.data.download.sources.http.download")
    def test_explicit_manual_present_succeeds(self, mock_dl: MagicMock, tmp_path: Path):
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
            download_datasets(config_file, dataset_keys=["ds1", "ds2", "ds3"])
        problems = excinfo.value.problems
        assert len(problems) == 3
        keys_hit = {p.split(":", 1)[0] for p in problems}
        assert keys_hit == {"ds1", "ds2", "ds3"}

    @patch("slm4ie.data.download.sources.http.download")
    def test_default_mode_skips_disabled_quietly(self, mock_dl: MagicMock, tmp_path: Path):
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
        err = DatasetDownloadError("culturax", failed, n_completed=0, n_total=2)
        msg = str(err)
        assert "culturax" in msg
        assert "2/2 sub-units failed" in msg
        assert "sl" in msg
        assert "RuntimeError" in msg
        assert "hr" in msg
        assert "ConnectionError" in msg


class TestIntraDatasetFailureSurfacing:
    """Tests that per-sub-unit failures bubble through `download_datasets`."""

    def _make_config_file(self, tmp_path: Path, datasets: dict) -> Path:
        config_data = {
            "output_dir": str(tmp_path / "raw"),
            "datasets": datasets,
        }
        config_file = tmp_path / "download.yaml"
        config_file.write_text(yaml.dump(config_data))
        return config_file

    @patch("slm4ie.data.download.sources.http.download")
    def test_intra_dataset_failures_surface_in_top_level_runtime_error(self, mock_dl: MagicMock, tmp_path: Path):
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


PROJECT_ROOT = str(Path(__file__).resolve().parents[3])


class TestCLI:
    """Tests for the CLI entrypoint."""

    def test_cli_help(self):
        """`--help` exits cleanly and lists the main flags."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.prepare_datasets",
                "download",
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
                "scripts.prepare_datasets",
                "download",
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
                "scripts.prepare_datasets",
                "download",
                "--config",
                str(config_file),
                "nonexistent",
            ],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        assert result.returncode == 1
