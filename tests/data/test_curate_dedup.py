"""Tests for the dedup helpers in `slm4ie.data.curate.dedup`.

Structural assertions about the full executor ladder live in
`test_curate_pipeline.py`; this file only covers the small content
getters and config factories.
"""

import importlib.metadata  # noqa: F401  (datatrove workaround)
import importlib.util  # noqa: F401  (datatrove workaround)

import pytest

pytest.importorskip("datatrove")

from datatrove.data import Document  # noqa: E402

from slm4ie.data.curate.dedup import default_exact_config, doc_text, make_exact_config  # noqa: E402


class TestExactDedupHelpers:
    """Helpers used by the exact-dedup signature stage."""

    def test_doc_text_returns_text_payload(self) -> None:
        """`doc_text` extracts the text body for hashing."""
        d = Document(text="hello", id="1", metadata={})
        assert doc_text(d) == "hello"

    def test_default_exact_config_uses_doc_text(self) -> None:
        """The default ExactDedupConfig hashes `doc.text`, not metadata."""
        cfg = default_exact_config()
        assert cfg.content_getter is doc_text

    def test_make_exact_config_defaults_match_default_helper(self) -> None:
        """make_exact_config with no args returns the same shape as default_exact_config."""
        cfg = make_exact_config()
        default = default_exact_config()
        assert cfg.content_getter is doc_text
        assert cfg.hash_config.precision == default.hash_config.precision
        assert cfg.hash_config.hash_fc == default.hash_config.hash_fc
        assert cfg.only_dedup_in_index == default.only_dedup_in_index

    def test_make_exact_config_overrides_precision(self) -> None:
        """make_exact_config threads the precision arg into HashConfig."""
        cfg = make_exact_config(precision=32)
        assert cfg.hash_config.precision == 32

    def test_make_exact_config_overrides_hash_fc(self) -> None:
        """make_exact_config threads hash_fc into HashConfig."""
        cfg = make_exact_config(hash_fc="sha1")
        assert cfg.hash_config.hash_fc == "sha1"

    def test_make_exact_config_overrides_only_dedup_in_index(self) -> None:
        """make_exact_config threads the only_dedup_in_index flag."""
        cfg = make_exact_config(only_dedup_in_index=False)
        assert cfg.only_dedup_in_index is False
