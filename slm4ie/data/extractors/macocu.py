"""MaCoCu XML extractor for the SLM4IE pipeline.

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
        text:        paragraphs (<p> text content) joined with "\\n".
        source:      provided by caller.
        domain:      provided by caller.
        doc_id:      id attribute on the <doc> element.
        metadata:    selected <doc> attributes: title, crawl_date,
                     lang_distr, url, domain, file_type, lm_score
                     (only those present and non-empty).
        annotations: not produced.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List
from xml.etree import ElementTree

from slm4ie.data.extractors import BaseExtractor, register_extractor
from slm4ie.data.schema import Document

logger = logging.getLogger(__name__)

_DOC_METADATA_KEYS = (
    "title",
    "crawl_date",
    "lang_distr",
    "url",
    "domain",
    "file_type",
    "lm_score",
)


def _doc_text(doc_elem: "ElementTree.Element") -> str:
    """Build a document's text by joining its <p> contents.

    Paragraphs are joined with a single newline. Leading and trailing
    whitespace on each paragraph is stripped.

    Args:
        doc_elem (ElementTree.Element): A <doc> element.

    Returns:
        str: Concatenated paragraph text. Empty if no paragraph
            yielded any text.
    """
    parts: List[str] = []
    for p_elem in doc_elem.iter("p"):
        text = "".join(p_elem.itertext()).strip()
        if text:
            parts.append(text)
    return "\n".join(parts)


def _doc_metadata(doc_elem: "ElementTree.Element") -> Dict[str, Any]:
    """Collect metadata from a <doc> element's attributes.

    Args:
        doc_elem (ElementTree.Element): A <doc> element.

    Returns:
        Dict[str, Any]: Selected attributes (title, url, etc.) that
            are present and non-empty.
    """
    metadata: Dict[str, Any] = {}
    for key in _DOC_METADATA_KEYS:
        value = doc_elem.get(key)
        if value:
            metadata[key] = value
    return metadata


class MacocuExtractor(BaseExtractor):
    """Extracts Documents from MaCoCu monolingual XML files.

    Recursively discovers .xml files under the given input directory.
    One Document is emitted per <doc> element with non-empty paragraph
    text. The element's id attribute is used as doc_id.

    Uses ElementTree.iterparse to stream large corpora without loading
    the entire tree into memory.
    """

    def extract(
        self,
        input_dir: Path,
        source: str,
        domain: str,
    ) -> Iterator[Document]:
        """Yield Documents from all MaCoCu XML files under input_dir.

        Parse errors are logged as warnings and the offending file is
        skipped. <doc> elements with no paragraph text are skipped
        silently.

        Args:
            input_dir (Path): Directory containing .xml files
                (searched recursively).
            source (str): Dataset key assigned to every Document.
            domain (str): Domain label assigned to every Document.

        Yields:
            Document: One document per non-empty <doc> element.
        """
        files = sorted(p for p in input_dir.rglob("*.xml") if p.is_file())

        for filepath in files:
            try:
                yield from self._parse_file(filepath, source, domain)
            except ElementTree.ParseError as exc:
                logger.warning(
                    "Skipping %s — parse error: %s", filepath, exc
                )

    def _parse_file(
        self,
        filepath: Path,
        source: str,
        domain: str,
    ) -> Iterator[Document]:
        """Stream-parse one MaCoCu XML file.

        Args:
            filepath (Path): Path to the MaCoCu XML file.
            source (str): Dataset key.
            domain (str): Domain label.

        Yields:
            Document: One document per non-empty <doc> element.
        """
        for _, elem in ElementTree.iterparse(filepath, events=("end",)):
            if elem.tag != "doc":
                continue

            text = _doc_text(elem)
            if text:
                yield Document(
                    text=text,
                    source=source,
                    domain=domain,
                    doc_id=elem.get("id"),
                    metadata=_doc_metadata(elem),
                )

            elem.clear()


register_extractor("macocu", MacocuExtractor)
