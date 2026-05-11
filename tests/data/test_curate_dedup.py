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

from slm4ie.data.curate.dedup import default_exact_config, doc_text  # noqa: E402


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
