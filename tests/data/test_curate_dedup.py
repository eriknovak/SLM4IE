"""Tests for the dedup helpers in `slm4ie.data.curate.dedup`.

Structural assertions about the full executor ladder live in
`test_curate_pipeline.py`; this file only covers the small content
getters and config factories.
"""

import importlib.metadata  # noqa: F401  (datatrove workaround)
import importlib.util  # noqa: F401  (datatrove workaround)
from pathlib import Path
from typing import Dict

import pytest

pytest.importorskip("datatrove")

from datatrove.data import Document  # noqa: E402

from datatrove.pipeline.dedup import SentDedupConfig, SentenceDedupSignature  # noqa: E402
from datatrove.utils.typeshelper import Languages  # noqa: E402

import slm4ie.data.curate.dedup as dedup_module  # noqa: E402
from slm4ie.data.curate.dedup import (  # noqa: E402
    CompactSentenceDedupSignature,
    default_exact_config,
    doc_text,
    make_exact_config,
)


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

    def test_make_exact_config_defaults_are_pinned(self) -> None:
        """Defaults are 64-bit xxhash, only_dedup_in_index=True, doc_text getter."""
        cfg = make_exact_config()
        assert cfg.content_getter is doc_text
        assert cfg.hash_config.precision == 64
        assert cfg.hash_config.hash_fc == "xxhash"
        assert cfg.only_dedup_in_index is True

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


def _signature_files(folder: Path) -> Dict[str, bytes]:
    """Map each signature file under *folder* (relative path) to its bytes."""
    return {str(f.relative_to(folder)): f.read_bytes() for f in sorted(folder.rglob("*")) if f.is_file()}


class TestCompactSentenceDedupSignature:
    """The compact signature step is a byte-for-byte drop-in for datatrove's."""

    def test_writes_identical_signature_files(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Both steps write the same files for the same documents, across flush boundaries."""
        monkeypatch.setattr(dedup_module, "SIGNATURE_FLUSH_EVERY", 3)
        docs = [
            Document(
                text=" ".join(f"Stavek {i} v dokumentu {d} govori o temi {i % 3}." for i in range(6)),
                id=str(d),
                metadata={},
            )
            for d in range(5)
        ]
        cfg = SentDedupConfig(n_sentences=2, split_sentences=True)
        kwargs = {"config": cfg, "finder_workers": 2, "language": Languages.slovenian}
        SentenceDedupSignature(output_folder=str(tmp_path / "upstream"), **kwargs).run(iter(docs), rank=1)
        CompactSentenceDedupSignature(output_folder=str(tmp_path / "compact"), **kwargs).run(iter(docs), rank=1)

        upstream = _signature_files(tmp_path / "upstream")
        assert upstream
        assert _signature_files(tmp_path / "compact") == upstream
