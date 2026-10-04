"""Tests for slm4ie/data/tokenization/config.py."""

from pathlib import Path
from textwrap import dedent

from slm4ie.data.tokenization import config as tokenization


class TestLoadTokenizationConfig:
    """Tests for the tokenization config loader."""

    def test_parses_minimum_required_fields(self, tmp_path: Path):
        """A well-formed tokenization config parses into the expected dict."""
        config = tmp_path / "tokenization.yaml"
        config.write_text(
            dedent(
                """\
                input_dir: /tmp/raw
                output_dir: /tmp/out
                datasets:
                  - sloleks
                """
            ),
            encoding="utf-8",
        )
        cfg = tokenization.load_tokenization_config(config)
        assert cfg.input_dir == Path("/tmp/raw")
        assert cfg.output_dir == Path("/tmp/out")
        assert cfg.datasets == ["sloleks"]
