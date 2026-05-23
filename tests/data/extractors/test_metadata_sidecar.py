"""Tests for the external-metadata helper used by extractors."""

import gzip
import textwrap
from pathlib import Path

import pytest

from slm4ie.data.metadata_sidecar import MetadataSidecar


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SAMPLE_TSV = textwrap.dedent("""\
    id\ttitle\tcerif\tudc\tkeywords\tnote
    kas-10000\tFirst\tP000|T270\t005\tfoo|bar\t-
    kas-10001\tSecond\tS212\t-\t\tnonempty
""")


def _write_tsv(tmp_path: Path, content: str = _SAMPLE_TSV, name: str = "meta.tsv") -> Path:
    """Write *content* to *tmp_path / name* and return the path.

    Args:
        tmp_path: pytest tmp_path fixture value.
        content: TSV body to write.
        name: File name.

    Returns:
        Path to the written file.
    """
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMetadataSidecarBasics:
    """Loading, key lookup, and column renaming."""

    def test_lookup_hit_renames_fields(self, tmp_path: Path) -> None:
        """Configured fields are renamed via the mapping."""
        path = _write_tsv(tmp_path)
        sidecar = MetadataSidecar(
            path=path,
            key_column="id",
            fields={"title": "title", "udc": "udc_code"},
        )
        assert sidecar.get("kas-10000") == {"title": "First", "udc_code": "005"}

    def test_lookup_miss_returns_empty(self, tmp_path: Path) -> None:
        """An unknown key yields an empty dict, not None."""
        path = _write_tsv(tmp_path)
        sidecar = MetadataSidecar(
            path=path,
            key_column="id",
            fields={"title": "title"},
        )
        assert sidecar.get("does-not-exist") == {}

    def test_unconfigured_columns_dropped(self, tmp_path: Path) -> None:
        """Columns not listed in ``fields`` never appear in the output."""
        path = _write_tsv(tmp_path)
        sidecar = MetadataSidecar(
            path=path,
            key_column="id",
            fields={"title": "title"},
        )
        assert "cerif" not in sidecar.get("kas-10000")


class TestMetadataSidecarNa:
    """NA / empty value filtering."""

    def test_dash_treated_as_na(self, tmp_path: Path) -> None:
        """A literal ``-`` value is filtered out as NA."""
        path = _write_tsv(tmp_path)
        sidecar = MetadataSidecar(
            path=path,
            key_column="id",
            fields={"note": "note"},
        )
        assert sidecar.get("kas-10000") == {}

    def test_empty_string_treated_as_na(self, tmp_path: Path) -> None:
        """An empty cell is filtered out as NA."""
        path = _write_tsv(tmp_path)
        sidecar = MetadataSidecar(
            path=path,
            key_column="id",
            fields={"keywords": "keywords"},
        )
        # kas-10001 has an empty 'keywords' cell.
        assert sidecar.get("kas-10001") == {}


class TestMetadataSidecarSplits:
    """List splits for pipe/comma-separated columns."""

    def test_pipe_split_produces_list(self, tmp_path: Path) -> None:
        """A configured separator turns the value into a list."""
        path = _write_tsv(tmp_path)
        sidecar = MetadataSidecar(
            path=path,
            key_column="id",
            fields={"cerif": "cerif"},
            splits={"cerif": "|"},
        )
        assert sidecar.get("kas-10000") == {"cerif": ["P000", "T270"]}

    def test_split_drops_na_parts(self, tmp_path: Path) -> None:
        """``-`` parts inside a split list are filtered out."""
        content = textwrap.dedent("""\
            id\tfoo
            x\ta|-|b
        """)
        path = _write_tsv(tmp_path, content=content, name="t.tsv")
        sidecar = MetadataSidecar(
            path=path,
            key_column="id",
            fields={"foo": "foo"},
            splits={"foo": "|"},
        )
        assert sidecar.get("x") == {"foo": ["a", "b"]}


class TestMetadataSidecarKeyDerivation:
    """Mapping file paths to sidecar keys."""

    def test_filename_stem_default(self, tmp_path: Path) -> None:
        """With no pattern, the file stem is used verbatim."""
        path = _write_tsv(tmp_path)
        sidecar = MetadataSidecar(
            path=path,
            key_column="id",
            fields={"title": "title"},
        )
        assert sidecar.get_for_path(Path("kas-10000.xml")) == {"title": "First"}

    def test_regex_extracts_group_one(self, tmp_path: Path) -> None:
        """``key_pattern`` group 1 becomes the actual sidecar key."""
        content = textwrap.dedent("""\
            id\tfield
            10000\tphysics
        """)
        path = _write_tsv(tmp_path, content=content, name="oss.tsv")
        sidecar = MetadataSidecar(
            path=path,
            key_column="id",
            fields={"field": "field"},
            key_pattern=r"^oss-(\d+)$",
        )
        assert sidecar.get_for_path(Path("oss-10000.conllu")) == {"field": "physics"}

    def test_regex_no_match_returns_empty(self, tmp_path: Path) -> None:
        """When the regex doesn't match, sidecar returns ``{}``."""
        path = _write_tsv(tmp_path)
        sidecar = MetadataSidecar(
            path=path,
            key_column="id",
            fields={"title": "title"},
            key_pattern=r"^oss-(\d+)$",
        )
        assert sidecar.get_for_path(Path("kas-10000.xml")) == {}


class TestMetadataSidecarGzip:
    """Gzip-aware reads."""

    def test_loads_gzipped_tsv(self, tmp_path: Path) -> None:
        """``.gz`` files are transparently handled via open_text_stream."""
        path = tmp_path / "meta.tsv.gz"
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write(_SAMPLE_TSV)
        sidecar = MetadataSidecar(
            path=path,
            key_column="id",
            fields={"title": "title"},
        )
        assert sidecar.get("kas-10001") == {"title": "Second"}


class TestMetadataSidecarErrors:
    """Construction-time validation."""

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """A path that doesn't exist raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            MetadataSidecar(
                path=tmp_path / "nope.tsv",
                key_column="id",
                fields={"x": "x"},
            )

    def test_missing_key_column_raises(self, tmp_path: Path) -> None:
        """A key_column not in the header raises ValueError."""
        path = _write_tsv(tmp_path)
        with pytest.raises(ValueError, match="missing key column"):
            MetadataSidecar(
                path=path,
                key_column="not_there",
                fields={"title": "title"},
            )

    def test_unsupported_key_from_raises(self, tmp_path: Path) -> None:
        """Unknown key_from strategy fails fast."""
        path = _write_tsv(tmp_path)
        with pytest.raises(ValueError, match="Unsupported key_from"):
            MetadataSidecar(
                path=path,
                key_column="id",
                fields={"title": "title"},
                key_from="newdoc_id",
            )

    def test_regex_without_group_raises(self, tmp_path: Path) -> None:
        """key_pattern without a capture group is rejected."""
        path = _write_tsv(tmp_path)
        with pytest.raises(ValueError, match="capture group"):
            MetadataSidecar(
                path=path,
                key_column="id",
                fields={"title": "title"},
                key_pattern=r"^oss-\d+$",
            )


class TestMetadataSidecarFromConfig:
    """The ``from_config`` factory mirrors the YAML schema."""

    def test_from_config_resolves_relative_path(self, tmp_path: Path) -> None:
        """The TSV path in the config is resolved against ``input_dir``."""
        nested = tmp_path / "sub"
        nested.mkdir()
        _write_tsv(nested, name="meta.tsv")
        sidecar = MetadataSidecar.from_config(
            input_dir=tmp_path,
            cfg={
                "path": "sub/meta.tsv",
                "key_column": "id",
                "fields": {"title": "title"},
            },
        )
        assert sidecar.get("kas-10000") == {"title": "First"}
