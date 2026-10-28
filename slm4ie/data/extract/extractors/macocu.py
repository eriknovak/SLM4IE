r"""MaCoCu XML extractor for the SLM4IE pipeline.

Implements the MaCoCu monolingual DTD format (e.g. MaCoCu-sl-2.0): a
<corpus> root containing <doc> elements, each holding one or more <p>
paragraphs. One Document is emitted per <doc> element, with text built
by joining paragraph contents.

Example:
    Raw input:

        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE corpus SYSTEM "MaCoCu-monolingual.dtd">
        <corpus id="MaCoCu-sl-2.0">
          <doc id="macocu.sl.1" title="Page One"
               url="https://example.com/1"
               crawl_date="2022-07-01" lm_score="0.95">
            <p id="macocu.sl.1.1" lang="sl">Dober dan.</p>
            <p id="macocu.sl.1.2" lang="sl">Kako ste?</p>
          </doc>
        </corpus>

    Schema mapping:
        text:        paragraphs (<p> text content) joined with "\n".
        source:      provided by caller.
        domain:      provided by caller.
        doc_id:      id attribute on the <doc> element, else the
                     positional id `<file unit>:<doc ordinal>`.
        native_id:   the id attribute, None when absent.
        metadata:    selected <doc> attributes: title, crawl_date,
                     lang_distr, url, domain, file_type, lm_score
                     (only those present and non-empty).
        annotations: not produced.
"""

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from lxml import etree

from slm4ie.data.extract.extractors import FileBasedExtractor, register_extractor
from slm4ie.data.extract.extractors.assembly import positional_doc_id, relative_unit
from slm4ie.data.schema import Document

# ===============================
# Logging
# ===============================

logger = logging.getLogger(__name__)

# ===============================
# Defaults
# ===============================

_DOC_METADATA_KEYS = (
    "title",
    "crawl_date",
    "lang_distr",
    "url",
    "domain",
    "file_type",
    "lm_score",
)

# ===============================
# Helper functions
# ===============================


def _doc_text(doc_elem: "etree._Element") -> str:
    """Build a document's text by joining its <p> contents.

    Paragraphs are joined with a single newline. Leading and trailing
    whitespace on each paragraph is stripped.

    Args:
        doc_elem: A <doc> element.

    Returns:
        Concatenated paragraph text. Empty if no paragraph yielded any text.
    """
    parts: list[str] = []
    for p_elem in doc_elem.iter("p"):
        text = "".join(p_elem.itertext()).strip()
        if text:
            parts.append(text)
    return "\n".join(parts)


def _doc_metadata(doc_elem: "etree._Element") -> dict[str, Any]:
    """Collect metadata from a <doc> element's attributes.

    Args:
        doc_elem: A <doc> element.

    Returns:
        Selected attributes (title, url, etc.) that are present and non-empty.
    """
    metadata: dict[str, Any] = {}
    for key in _DOC_METADATA_KEYS:
        value = doc_elem.get(key)
        if value:
            metadata[key] = value
    return metadata


# ===============================
# Extractor class
# ===============================


class MacocuExtractor(FileBasedExtractor):
    """Extracts Documents from MaCoCu monolingual XML files.

    Recursively discovers .xml files under the given input directory.
    One Document is emitted per <doc> element with non-empty paragraph
    text. The element's id attribute is used as doc_id.

    Uses lxml's tag-filtered iterparse to stream large corpora without
    loading the entire tree into memory.
    """

    def iter_input_files(self, input_dir: Path) -> list[Path]:
        """Return sorted .xml files under input_dir.

        Args:
            input_dir: Directory searched recursively.

        Returns:
            Sorted .xml file paths.
        """
        return sorted(p for p in input_dir.rglob("*.xml") if p.is_file())

    def extract_files(
        self,
        files: list[Path],
        source: str,
        domain: str,
        input_dir: Path,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[Document]:
        """Yield Documents from the given MaCoCu XML files.

        Parse errors are logged as warnings and the offending file is
        skipped. <doc> elements with no paragraph text are skipped
        silently.

        Args:
            files: .xml files to parse, in order.
            source: Dataset key assigned to every Document.
            domain: Domain label assigned to every Document.
            input_dir: Dataset root; names the unit of a
                positional id when a <doc> carries no id attribute.
            metadata: Ignored.

        Yields:
            One document per non-empty <doc> element.
        """
        del metadata
        for filepath in files:
            try:
                yield from self._parse_file(filepath, source, domain, relative_unit(filepath, input_dir))
            except etree.XMLSyntaxError as exc:
                logger.warning("Skipping %s — parse error: %s", filepath, exc)

    # =================================
    # Private methods
    # =================================

    def _parse_file(
        self,
        filepath: Path,
        source: str,
        domain: str,
        unit: str,
    ) -> Iterator[Document]:
        """Stream-parse one MaCoCu XML file.

        Args:
            filepath: Path to the MaCoCu XML file.
            source: Dataset key.
            domain: Domain label.
            unit: The file's unit name for positional ids.

        Yields:
            One document per non-empty <doc> element, with
                the <doc> id attribute as `doc_id` and `native_id`, or
                the positional id `<unit>:<doc ordinal>` when absent.
        """
        for ordinal, (_, elem) in enumerate(etree.iterparse(str(filepath), events=("end",), tag="doc")):
            text = _doc_text(elem)
            if text:
                native_id = elem.get("id")
                yield Document(
                    text=text,
                    source=source,
                    domain=domain,
                    doc_id=native_id if native_id is not None else positional_doc_id(unit, ordinal),
                    native_id=native_id,
                    metadata=_doc_metadata(elem),
                )

            elem.clear()


# ===============================
# Registration
# ===============================

register_extractor("macocu", MacocuExtractor)
