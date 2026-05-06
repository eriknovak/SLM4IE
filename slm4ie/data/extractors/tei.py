"""TEI XML extractor for the SLM4IE pipeline.

Supports annotated TEI (with <w> elements) and plain TEI (<p> text
only). Handles ParlaMint-SI, siParl, KAS, Janes-Forum, Janes-Blog,
and Janes-News formats.

Example:
    Annotated TEI (one Document per <s>):

        <TEI xmlns="http://www.tei-c.org/ns/1.0">
          <text>
            <body>
              <s xml:id="ParlaMint-SI.s1">
                <w lemma="dober"
                   msd="UPosTag=ADJ|Case=Nom|Gender=Masc">Dober</w>
                <w lemma="dan" msd="UPosTag=NOUN|Case=Nom">dan</w>
                <pc msd="UPosTag=PUNCT">.</pc>
              </s>
            </body>
          </text>
        </TEI>

    KAS-style ana attribute instead of msd:

        <w lemma="dober" ana="mte:Agpmsny">Dober</w>

    Plain TEI (one Document per <p>):

        <TEI xmlns="http://www.tei-c.org/ns/1.0">
          <text>
            <body>
              <p xml:id="janes.p1">Dober dan, kako ste?</p>
            </body>
          </text>
        </TEI>

    Schema mapping:
        text:        annotated -> token forms joined with a single
                     space. Plain -> the <p> element's text content
                     (stripped).
        source:      provided by caller.
        domain:      provided by caller.
        doc_id:      xml:id of the originating <s> (annotated) or
                     <p> (plain).
        metadata:    not produced.
        annotations: annotated only.
            tokens.form:  text content of <w> / <pc>.
            tokens.lemma: lemma attribute.
            tokens.upos:  derived from msd (UPosTag=...) or from
                          ana (mte:<MULTEXT-East-v6 code>).
            tokens.feats: remaining Key=Value parts of msd, or
                          MTE=<code> when only ana is present.
            sentences:    single span [0, len(tokens)-1].
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

# MULTEXT-East v6 category code (1st char of compact MSD) → UPOS.
# Refined by the 2nd char in `_mte_to_upos` for N/V/C distinctions.
_MTE_CATEGORY_UPOS = {
    "N": "NOUN",
    "V": "VERB",
    "A": "ADJ",
    "P": "PRON",
    "R": "ADV",
    "S": "ADP",
    "C": "CCONJ",
    "M": "NUM",
    "Q": "PART",
    "I": "INTJ",
    "Y": "X",
    "X": "X",
    "Z": "PUNCT",
}

logger = logging.getLogger(__name__)


def _parse_msd(
    msd: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """Parse a morphosyntactic description string into UPOS and feats.

    The UPosTag=X part is extracted as UPOS; all remaining Key=Value
    parts are rejoined with "|" as the feats string.

    Args:
        msd (Optional[str]): MSD string such as
            "UPosTag=NOUN|Case=Nom|Gender=Masc", or None.

    Returns:
        Tuple[Optional[str], Optional[str]]: (upos, feats) where
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


def _mte_to_upos(mte: str) -> Optional[str]:
    """Map a MULTEXT-East v6 compact MSD to a UPOS tag.

    Uses the first character (category) for the base mapping and the
    second character (type) to refine NOUN vs PROPN, VERB vs AUX, and
    CCONJ vs SCONJ.

    Args:
        mte (str): Compact MSD such as "Ncnsn" or "Z".

    Returns:
        Optional[str]: UPOS tag, or None if the category is unknown.
    """
    if not mte:
        return None

    cat = mte[0]
    upos = _MTE_CATEGORY_UPOS.get(cat)
    if upos is None:
        return None

    if len(mte) >= 2:
        sub = mte[1]
        if cat == "N" and sub == "p":
            return "PROPN"
        if cat == "V" and sub == "a":
            return "AUX"
        if cat == "C" and sub == "s":
            return "SCONJ"

    return upos


def _parse_ana(
    ana: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """Parse a TEI ana attribute into UPOS and feats.

    Looks for a mte:<descriptor> value (MULTEXT-East v6 compact MSD,
    used by KAS) and maps it to UPOS. The raw descriptor is preserved
    in feats as MTE=<descriptor>.

    Args:
        ana (Optional[str]): The ana attribute value, or None.

    Returns:
        Tuple[Optional[str], Optional[str]]: (upos, feats) where
            either may be None if no MTE descriptor is found.
    """
    if not ana:
        return None, None

    for part in ana.split():
        if part.startswith("mte:"):
            mte = part[len("mte:"):]
            if not mte:
                return None, None
            return _mte_to_upos(mte), f"MTE={mte}"

    return None, None


def _parse_morph(
    elem: "ElementTree.Element",
) -> Tuple[Optional[str], Optional[str]]:
    """Extract UPOS and feats from a token element.

    Prefers msd (ParlaMint, siParl) over ana (KAS).

    Args:
        elem (ElementTree.Element): A <w> or <pc> element.

    Returns:
        Tuple[Optional[str], Optional[str]]: (upos, feats).
    """
    msd = elem.get("msd")
    if msd:
        return _parse_msd(msd)
    return _parse_ana(elem.get("ana"))


def _extract_tokens_from_sentence(
    s_elem: "ElementTree.Element",
) -> List[Token]:
    """Extract Token objects from a <s> element.

    Iterates direct and <name>-wrapped children to find all <w> and
    <pc> token elements.

    Args:
        s_elem (ElementTree.Element): The <s> XML element.

    Returns:
        List[Token]: Extracted tokens in document order.
    """
    tokens: List[Token] = []

    for child in s_elem:
        if child.tag in (_W_TAG, _PC_TAG):
            upos, feats = _parse_morph(child)
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
                    upos, feats = _parse_morph(wc)
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
    """Yield Documents from an annotated TEI tree (has <w>).

    One Document is produced per <s> element. Text is the
    space-joined token forms. The doc_id comes from the xml:id
    attribute on <s>.

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
    """Yield Documents from a plain TEI tree (no <w> elements).

    One Document is produced per <p> element with non-empty text.
    The doc_id comes from the xml:id attribute on <p>.

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

    Handles both annotated TEI (containing <w> word elements) and
    plain TEI (paragraph text only). Recursively discovers all .xml
    files under the given input directory.

    Supported corpora: ParlaMint-SI, siParl, KAS, Janes-Forum,
    Janes-Blog, Janes-News.
    """

    def extract(
        self,
        input_dir: Path,
        source: str,
        domain: str,
    ) -> Iterator[Document]:
        """Yield Documents from all TEI XML files in input_dir.

        Args:
            input_dir (Path): Directory to search recursively for
                .xml files.
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
