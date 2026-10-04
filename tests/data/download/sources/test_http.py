"""Tests for the HTTP download source (slm4ie/data/download/sources/http.py)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

from slm4ie.data.download.config import DatasetConfig
from slm4ie.data.download.sources import http as http_source


class TestHttpSource:
    """Tests for the http source downloader."""

    def test_extract_filename_from_url(self):
        """Filename is extracted from a URL with query parameters."""
        url = (
            "https://www.clarin.si/repository/xmlui/bitstream/"
            "handle/11356/1427/classlawiki-sl.conllu.gz"
            "?sequence=6&isAllowed=y"
        )
        assert http_source._extract_filename(url) == ("classlawiki-sl.conllu.gz")

    def test_extract_filename_no_query(self):
        """Filename extraction handles URLs without query strings."""
        url = "https://example.com/path/to/file.tar.gz"
        assert http_source._extract_filename(url) == "file.tar.gz"

    @patch("slm4ie.data.download.sources.http.requests.get")
    def test_download_single_file(self, mock_get: MagicMock, tmp_path: Path):
        """A single URL streams to disk and the .part file is removed on success."""
        mock_response = MagicMock()
        mock_response.headers = {"content-length": "100"}
        mock_response.iter_content = MagicMock(return_value=[b"x" * 100])
        mock_response.raise_for_status = MagicMock()
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

        http_source.download(config, output_dir, force=False)

        assert (output_dir / "test.gz").exists()
        assert (output_dir / "test.gz").read_bytes() == b"x" * 100
        assert not (output_dir / "test.gz.part").exists()

    @patch("slm4ie.data.download.sources.http.requests.get")
    def test_force_redownloads_existing_dest(self, mock_get: MagicMock, tmp_path: Path):
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

    @patch("slm4ie.data.download.sources.http.requests.get")
    def test_force_clears_stale_part(self, mock_get: MagicMock, tmp_path: Path):
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

    @patch("slm4ie.data.download.sources.http.requests.get")
    def test_per_url_failure_collected(self, mock_get: MagicMock, tmp_path: Path):
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

    @patch("slm4ie.data.download.sources.http.requests.get")
    def test_download_creates_output_dir(self, mock_get: MagicMock, tmp_path: Path):
        """Download creates the output directory if it does not exist."""
        mock_response = MagicMock()
        mock_response.headers = {"content-length": "10"}
        mock_response.iter_content = MagicMock(return_value=[b"x" * 10])
        mock_response.raise_for_status = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
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
