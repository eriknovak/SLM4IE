"""MaCoCu XML extractor for the SLM4IE pipeline.

Supports the custom MaCoCu monolingual DTD format with ``<tu>``
translation units containing ``<tuv>``/``<p>`` paragraph text and
quality scores.
"""

import logging
from pathlib import Path
from typing import Iterator
from xml.etree import ElementTree

from slm4ie.data.extractors import BaseExtractor, register_extractor
from slm4ie.data.schema import Document

logger = logging.getLogger(__name__)


class MacocuExtractor(BaseExtractor):
    """Extracts Documents from MaCoCu XML files.

    Processes files matching ``*.xml`` in the top-level input directory
    (non-recursive). One Document is produced per ``<tu>`` element with
    non-empty text content. Text is the space-joined text of all ``<p>``
    children within the ``<tuv>`` element. The quality score from the
    ``score`` attribute is stored in ``metadata["score"]``.
    """

    def extract(
        self,
        input_dir: Path,
        source: str,
        domain: str,
    ) -> Iterator[Document]:
        """Yield Documents from all MaCoCu XML files in *input_dir*.

        Scans only the top-level directory (not recursive). Parse
        errors are logged as warnings and the file is skipped. ``<tu>``
        elements with empty text are skipped silently.

        Args:
            input_dir (Path): Directory containing ``*.xml`` files.
            source (str): Dataset key assigned to every Document.
            domain (str): Domain label assigned to every Document.

        Yields:
            Document: One document per non-empty ``<tu>`` element.
        """
        for filepath in sorted(
            p for p in input_dir.iterdir() if p.suffix == ".xml"
        ):
            try:
                tree = ElementTree.parse(filepath)
            except ElementTree.ParseError as exc:
                logger.warning(
                    "Skipping %s — parse error: %s", filepath, exc
                )
                continue

            root = tree.getroot()
            for tu_elem in root.iter("tu"):
                parts = []
                for p_elem in tu_elem.iter("p"):
                    text = (p_elem.text or "").strip()
                    if text:
                        parts.append(text)

                if not parts:
                    continue

                yield Document(
                    text=" ".join(parts),
                    source=source,
                    domain=domain,
                    doc_id=tu_elem.get("id"),
                    metadata={"score": tu_elem.get("score")},
                )


register_extractor("macocu", MacocuExtractor)
