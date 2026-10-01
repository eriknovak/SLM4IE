"""Tests for slm4ie.utils.io helpers."""

import gzip
import sys
from pathlib import Path


from slm4ie.utils.io import (
    find_project_root,
    open_output,
    resolve_project_path,
)


class TestResolveProjectPath:
    """Tests for resolve_project_path."""

    def test_absolute_passes_through(self, tmp_path: Path) -> None:
        """An absolute value is returned unchanged, ignoring root."""
        abs_path = tmp_path / "vault" / "raw"
        assert resolve_project_path(abs_path, root=Path("/repo")) == abs_path

    def test_relative_anchored_to_root(self) -> None:
        """A relative value is joined onto the provided root."""
        result = resolve_project_path("./data/raw", root=Path("/repo"))
        assert result == Path("/repo/data/raw")
        assert result.is_absolute()

    def test_relative_string_accepts_plain_form(self) -> None:
        """A leading `./` is optional and collapses the same way."""
        assert resolve_project_path("data/raw", root=Path("/repo")) == Path("/repo/data/raw")

    def test_default_root_is_project_root(self) -> None:
        """Without an explicit root, values anchor to the project root."""
        result = resolve_project_path("data/raw")
        assert result == find_project_root() / "data" / "raw"
        assert result.is_absolute()


class TestOpenOutput:
    """Tests for open_output."""

    def test_gzip_suffix_writes_gzipped(self, tmp_path: Path) -> None:
        """An output path ending in .gz produces a gzip file."""
        out_path = tmp_path / "out.jsonl.gz"
        with open_output(out_path) as fh:
            fh.write("hello\n")
        with gzip.open(out_path, "rt", encoding="utf-8") as fh:
            assert fh.read() == "hello\n"

    def test_plain_path_writes_plain(self, tmp_path: Path) -> None:
        """Without .gz the output is a plain text file."""
        out_path = tmp_path / "out.jsonl"
        with open_output(out_path) as fh:
            fh.write("hi\n")
        assert out_path.read_text(encoding="utf-8") == "hi\n"

    def test_none_means_stdout(self) -> None:
        """Passing None yields sys.stdout (not closed afterwards)."""
        with open_output(None) as fh:
            assert fh is sys.stdout
