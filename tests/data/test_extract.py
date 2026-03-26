"""Tests for slm4ie.data.extract module."""

import gzip
import tarfile
import zipfile
from pathlib import Path

import pytest

from slm4ie.data.extract import extract_archive


class TestExtractArchive:
    """Tests for extract_archive function."""

    def test_extract_gzip(self, tmp_path: Path):
        """Create .jsonl.gz, extract, verify contents."""
        content = b'{"text": "hello world"}\n{"text": "foo"}\n'
        gz_path = tmp_path / "data.jsonl.gz"
        with gzip.open(gz_path, "wb") as f:
            f.write(content)

        output_dir = tmp_path / "out"
        output_dir.mkdir()

        result = extract_archive(gz_path, output_dir)

        assert result == output_dir / "data.jsonl"
        assert result.exists()
        assert result.read_bytes() == content

    def test_extract_zip(self, tmp_path: Path):
        """Create .zip with inner dir/file, extract, verify."""
        zip_path = tmp_path / "archive.zip"
        inner_content = b"some data\n"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("inner_dir/file.txt", inner_content)

        output_dir = tmp_path / "out"
        output_dir.mkdir()

        result = extract_archive(zip_path, output_dir)

        assert result == output_dir
        extracted = output_dir / "inner_dir" / "file.txt"
        assert extracted.exists()
        assert extracted.read_bytes() == inner_content

    def test_extract_tar_gz(self, tmp_path: Path):
        """Create .tgz, extract, verify contents."""
        inner_file = tmp_path / "source.txt"
        inner_file.write_bytes(b"tgz content\n")
        tgz_path = tmp_path / "archive.tgz"
        with tarfile.open(tgz_path, "w:gz") as tf:
            tf.add(inner_file, arcname="source.txt")

        output_dir = tmp_path / "out"
        output_dir.mkdir()

        result = extract_archive(tgz_path, output_dir)

        assert result == output_dir
        assert (output_dir / "source.txt").exists()
        assert (output_dir / "source.txt").read_bytes() == (
            b"tgz content\n"
        )

    def test_extract_tar_gz_long_ext(self, tmp_path: Path):
        """Test that .tar.gz extension also works."""
        inner_file = tmp_path / "data.txt"
        inner_file.write_bytes(b"tar gz content\n")
        tar_gz_path = tmp_path / "archive.tar.gz"
        with tarfile.open(tar_gz_path, "w:gz") as tf:
            tf.add(inner_file, arcname="data.txt")

        output_dir = tmp_path / "out"
        output_dir.mkdir()

        result = extract_archive(tar_gz_path, output_dir)

        assert result == output_dir
        assert (output_dir / "data.txt").exists()

    def test_extract_skips_if_already_extracted(
        self, tmp_path: Path
    ):
        """Gz skip when output exists — don't overwrite."""
        original_content = b"original data\n"
        new_content = b"new data\n"

        output_dir = tmp_path / "out"
        output_dir.mkdir()

        # Pre-create the output file with original content
        expected_output = output_dir / "data.jsonl"
        expected_output.write_bytes(original_content)

        # Create a gz archive with different content
        gz_path = tmp_path / "data.jsonl.gz"
        with gzip.open(gz_path, "wb") as f:
            f.write(new_content)

        result = extract_archive(gz_path, output_dir)

        assert result == expected_output
        # Content should remain unchanged
        assert expected_output.read_bytes() == original_content

    def test_extract_unknown_format_raises(
        self, tmp_path: Path
    ):
        """Unknown format .xyz raises ValueError."""
        unknown_path = tmp_path / "archive.xyz"
        unknown_path.write_bytes(b"not an archive")

        output_dir = tmp_path / "out"
        output_dir.mkdir()

        with pytest.raises(ValueError, match="Unsupported archive format"):
            extract_archive(unknown_path, output_dir)
