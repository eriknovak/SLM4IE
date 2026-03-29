"""TEI XML extractor for the SLM4IE pipeline.

Supports annotated TEI (with ``<w>`` elements) and plain TEI (``<p>``
text only). Handles ParlaMint-SI, siParl, KAS, Janes-Forum,
Janes-Blog, and Janes-News formats.
"""

import logging
from pathlib import Path
from typing import Iterator, List, Optional, Tuple
from xml.etree import ElementTree

from slm4ie.data.extractors import BaseExtractor, register_extractor
from slm4ie.data.schema import Annotations, Document, Token

_TEI_NS = "http://www.tei-c.org/ns/1.0"
_XML_NS = "http://www.w3.org/XML/1998/namespace"

_W_TAG = f"{{{_TEI_NS}}}w"
_PC_TAG = f"{{{_TEI_NS}}}pc"
_S_TAG = f"{{{_TEI_NS}}}s"
_P_TAG = f"{{{_TEI_NS}}}p"
_NAME_TAG = f"{{{_TEI_NS}}}name"
_XML_ID = f"{{{_XML_NS}}}id"

logger = logging.getLogger(__name__)


def _parse_msd(
    msd: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """Parse a morphosyntactic description string into UPOS and feats.

    The ``UPosTag=X`` part is extracted as UPOS; all remaining
    ``Key=Value`` parts are rejoined with ``|`` as the feats string.

    Args:
        msd (Optional[str]): MSD string such as
            ``"UPosTag=NOUN|Case=Nom|Gender=Masc"``, or None.

    Returns:
        Tuple[Optional[str], Optional[str]]: ``(upos, feats)`` where
            either may be None if not present.
    """
    if not msd:
        return None, None

    parts = msd.split("|")
    upos: Optional[str] = None
    remaining: List[str] = []

    for part in parts:
        if part.startswith("UPosTag="):
            upos = part[len("UPosTag="):]
        else:
            remaining.append(part)

    feats = "|".join(remaining) if remaining else None
    return upos, feats


def _extract_tokens_from_sentence(
    s_elem: "ElementTree.Element",
) -> List[Token]:
    """Extract Token objects from a ``<s>`` element.

    Iterates direct and ``<name>``-wrapped children to find all
    ``<w>`` and ``<pc>`` token elements.

    Args:
        s_elem (ElementTree.Element): The ``<s>`` XML element.

    Returns:
        List[Token]: Extracted tokens in document order.
    """
    tokens: List[Token] = []

    for child in s_elem:
        if child.tag in (_W_TAG, _PC_TAG):
            upos, feats = _parse_msd(child.get("msd"))
            tokens.append(
                Token(
                    form=child.text or "",
                    lemma=child.get("lemma"),
                    upos=upos,
                    feats=feats,
                )
            )
        elif child.tag == _NAME_TAG:
            for wc in child:
                if wc.tag in (_W_TAG, _PC_TAG):
                    upos, feats = _parse_msd(wc.get("msd"))
                    tokens.append(
                        Token(
                            form=wc.text or "",
                            lemma=wc.get("lemma"),
                            upos=upos,
                            feats=feats,
                        )
                    )

    return tokens


def _parse_annotated(
    root: "ElementTree.Element",
    source: str,
    domain: str,
) -> Iterator[Document]:
    """Yield Documents from an annotated TEI tree (has ``<w>``).

    One Document is produced per ``<s>`` element. Text is the
    space-joined token forms. ``doc_id`` comes from ``xml:id`` on
    ``<s>``.

    Args:
        root (ElementTree.Element): Parsed XML root element.
        source (str): Dataset key.
        domain (str): Domain label.

    Yields:
        Document: One document per sentence.
    """
    for s_elem in root.iter(_S_TAG):
        tokens = _extract_tokens_from_sentence(s_elem)
        if not tokens:
            continue

        text = " ".join(t.form for t in tokens)
        doc_id = s_elem.get(_XML_ID)
        annotations = Annotations(
            tokens=tokens,
            sentences=[[0, len(tokens) - 1]],
        )
        yield Document(
            text=text,
            source=source,
            domain=domain,
            doc_id=doc_id,
            annotations=annotations,
        )


def _parse_plain(
    root: "ElementTree.Element",
    source: str,
    domain: str,
) -> Iterator[Document]:
    """Yield Documents from a plain TEI tree (no ``<w>`` elements).

    One Document is produced per ``<p>`` element with non-empty text.
    ``doc_id`` comes from ``xml:id`` on ``<p>``.

    Args:
        root (ElementTree.Element): Parsed XML root element.
        source (str): Dataset key.
        domain (str): Domain label.

    Yields:
        Document: One document per non-empty paragraph.
    """
    for p_elem in root.iter(_P_TAG):
        text = "".join(p_elem.itertext()).strip()
        if not text:
            continue
        doc_id = p_elem.get(_XML_ID)
        yield Document(
            text=text,
            source=source,
            domain=domain,
            doc_id=doc_id,
        )


class TeiExtractor(BaseExtractor):
    """Extracts Documents from TEI XML files.

    Handles both annotated TEI (containing ``<w>`` word elements) and
    plain TEI (paragraph text only). Recursively discovers all
    ``*.xml`` files under the given input directory.

    Supported corpora: ParlaMint-SI, siParl, KAS, Janes-Forum,
    Janes-Blog, Janes-News.
    """

    def extract(
        self,
        input_dir: Path,
        source: str,
        domain: str,
    ) -> Iterator[Document]:
        """Yield Documents from all TEI XML files in *input_dir*.

        Args:
            input_dir (Path): Directory to search recursively for
                ``*.xml`` files.
            source (str): Dataset key assigned to every Document.
            domain (str): Domain label assigned to every Document.

        Yields:
            Document: Extracted documents in unified schema format.
        """
        for filepath in sorted(input_dir.rglob("*.xml")):
            try:
                tree = ElementTree.parse(filepath)
            except ElementTree.ParseError as exc:
                logger.warning(
                    "Skipping %s — parse error: %s", filepath, exc
                )
                continue

            root = tree.getroot()
            is_annotated = next(root.iter(_W_TAG), None) is not None

            if is_annotated:
                yield from _parse_annotated(root, source, domain)
            else:
                yield from _parse_plain(root, source, domain)


register_extractor("tei", TeiExtractor)
