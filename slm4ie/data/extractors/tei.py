"""TEI XML extractor for the SLM4IE pipeline.

Yields one `Document` per real document, aggregating sentence-level
`<s>` units the way each corpus encodes them structurally:

- **Annotated TEI with `<u>` utterances** (ParlaMint-SI, siParl):
  one Document per `<u>`. The utterance's `who` and `ana`
  attributes flow into `Document.metadata`.
- **Annotated TEI without `<u>`** (KAS, similar academic corpora):
  one Document per file (`doc_id` = filename stem).
- **Plain TEI** (no `<w>` elements; e.g. Janes-Forum/Blog/News):
  unchanged — one Document per `<p>`.

Sentence boundaries inside an aggregated document are preserved in
`Annotations.sentences` as inclusive `[start, end]` token-index
spans over the flat `Annotations.tokens` list.

Example:
    Annotated TEI with two utterances:

        <TEI xmlns="http://www.tei-c.org/ns/1.0">
          <text>
            <body>
              <u xml:id="u1" who="#chair">
                <s xml:id="u1.s1">
                  <w lemma="dober" msd="UPosTag=ADJ">Dober</w>
                  <w lemma="dan" msd="UPosTag=NOUN">dan</w>
                  <pc msd="UPosTag=PUNCT">.</pc>
                </s>
                <s xml:id="u1.s2">
                  <w lemma="hvala" msd="UPosTag=NOUN">Hvala</w>
                  <pc msd="UPosTag=PUNCT">.</pc>
                </s>
              </u>
              <u xml:id="u2" who="#regular">
                <s xml:id="u2.s1">
                  <w lemma="ja" msd="UPosTag=INTJ">Ja</w>
                  <pc msd="UPosTag=PUNCT">.</pc>
                </s>
              </u>
            </body>
          </text>
        </TEI>

    Yields two Documents — the first with `doc_id="u1"`,
    `metadata={"who": "#chair"}`, `annotations.sentences=[[0, 2],
    [3, 4]]`, and `text` formed by joining its two sentence strings
    with a newline; the second with `doc_id="u2"`,
    `metadata={"who": "#regular"}`, `annotations.sentences=[[0,
    1]]`, and a single-sentence `text`.

    Schema mapping (annotated):
        text:        per-sentence token forms joined with a single
                     space, then sentence strings joined with newlines
                     so datatrove's sentence splitter retains the cue.
        source:      provided by caller.
        domain:      provided by caller.
        doc_id:      `<u>`'s `xml:id` (utterance path) or the
                     source filename's stem (per-file fallback).
        metadata:    `who` / `ana` from the `<u>` element (utterance
                     path) layered on top of any per-file fields
                     from a `metadata:` config block (see
                     `MetadataSidecar`); empty when neither applies.
        annotations:
            tokens:    flat concatenation across every contained `<s>`.
            sentences: one inclusive `[start, end]` per sentence,
                       in reading order.

    Plain TEI: one Document per `<p>` with `doc_id` from the
    paragraph's `xml:id`, no annotations, and per-file `metadata:`
    fields when configured.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
from lxml import etree

from slm4ie.data.extractors import FileBasedExtractor, register_extractor
from slm4ie.data.metadata_sidecar import MetadataSidecar
from slm4ie.data.schema import Annotations, Document, Token

_TEI_NS = "http://www.tei-c.org/ns/1.0"
_XML_NS = "http://www.w3.org/XML/1998/namespace"

_W_TAG = f"{{{_TEI_NS}}}w"
_PC_TAG = f"{{{_TEI_NS}}}pc"
_S_TAG = f"{{{_TEI_NS}}}s"
_P_TAG = f"{{{_TEI_NS}}}p"
_U_TAG = f"{{{_TEI_NS}}}u"
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
    elem: "etree._Element",
) -> Tuple[Optional[str], Optional[str]]:
    """Extract UPOS and feats from a token element.

    Prefers msd (ParlaMint, siParl) over ana (KAS).

    Args:
        elem (etree._Element): A <w> or <pc> element.

    Returns:
        Tuple[Optional[str], Optional[str]]: (upos, feats).
    """
    msd = elem.get("msd")
    if msd:
        return _parse_msd(msd)
    return _parse_ana(elem.get("ana"))


def _extract_tokens_from_sentence(
    s_elem: "etree._Element",
) -> List[Token]:
    """Extract Token objects from a <s> element.

    Iterates direct and <name>-wrapped children to find all <w> and
    <pc> token elements.

    Args:
        s_elem (etree._Element): The <s> XML element.

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


def _build_document(
    s_elems: List["etree._Element"],
    doc_id: str,
    source: str,
    domain: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Document]:
    """Combine `<s>` siblings into one document-level Document.

    Sentence texts are formed by space-joining each sentence's token
    forms; the sentence strings are then joined with newlines.
    Annotations carry the flat token sequence and inclusive
    `[start, end]` token-index spans for every contained sentence.

    Args:
        s_elems (List[etree._Element]): The `<s>` elements
            that constitute this document, in reading order.
        doc_id (str): Identifier for the resulting Document.
        source (str): Dataset key.
        domain (str): Domain label.
        metadata (Optional[Dict[str, Any]]): Optional metadata to
            attach to the Document. For utterance-level docs this
            typically merges per-file `MetadataSidecar` fields with
            the utterance's `who` / `ana` attributes.

    Returns:
        Optional[Document]: One Document, or `None` when every
            `<s>` element parsed to zero tokens.
    """
    sentence_texts: List[str] = []
    flat_tokens: List[Token] = []
    spans: List[List[int]] = []
    cursor = 0

    for s_elem in s_elems:
        tokens = _extract_tokens_from_sentence(s_elem)
        if not tokens:
            continue
        sentence_texts.append(" ".join(t.form for t in tokens))
        start = cursor
        flat_tokens.extend(tokens)
        cursor += len(tokens)
        spans.append([start, cursor - 1])

    if not flat_tokens:
        return None

    annotations = Annotations(tokens=flat_tokens, sentences=spans)
    return Document(
        text="\n".join(sentence_texts),
        source=source,
        domain=domain,
        doc_id=doc_id,
        metadata=dict(metadata) if metadata else {},
        annotations=annotations,
    )


def _utterance_metadata(u_elem: "etree._Element") -> Dict[str, Any]:
    """Return `Document.metadata` derived from a `<u>` element.

    Args:
        u_elem (etree._Element): The `<u>` element.

    Returns:
        Dict[str, Any]: `who` and `ana` attribute values when
            present; the key is omitted when the attribute is absent
            so that callers can use `"key" in metadata` cleanly.
    """
    md: Dict[str, Any] = {}
    who = u_elem.get("who")
    if who is not None:
        md["who"] = who
    ana = u_elem.get("ana")
    if ana is not None:
        md["ana"] = ana
    return md


def _parse_annotated_with_utterances(
    root: "etree._Element",
    source: str,
    domain: str,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Iterator[Document]:
    """Yield one Document per `<u>` element in an annotated TEI tree.

    Each `<u>`'s sentence descendants are aggregated into a single
    Document; the utterance's `xml:id` becomes the `doc_id` and
    its `who` / `ana` attributes flow into `metadata`, layered on
    top of any per-file fields supplied via *extra_metadata*.

    Args:
        root (etree._Element): Parsed XML root element.
        source (str): Dataset key.
        domain (str): Domain label.
        extra_metadata (Optional[Dict[str, Any]]): Per-file fields
            from `MetadataSidecar`. Utterance attributes (`who`,
            `ana`) take precedence on key collision since they are
            more specific.

    Yields:
        Document: One document per non-empty `<u>`.
    """
    base: Dict[str, Any] = dict(extra_metadata) if extra_metadata else {}
    for u_elem in root.iter(_U_TAG):
        s_elems = list(u_elem.iter(_S_TAG))
        doc_id = u_elem.get(_XML_ID) or ""
        merged = {**base, **_utterance_metadata(u_elem)}
        doc = _build_document(
            s_elems=s_elems,
            doc_id=doc_id,
            source=source,
            domain=domain,
            metadata=merged,
        )
        if doc is not None:
            yield doc


def _parse_annotated_per_file(
    root: "etree._Element",
    source: str,
    domain: str,
    doc_id: str,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Iterator[Document]:
    """Yield exactly one Document by aggregating every `<s>` in *root*.

    Used when an annotated TEI file has no `<u>` elements (e.g.
    KAS). `doc_id` is supplied by the caller (typically the
    source filename's stem).

    Args:
        root (etree._Element): Parsed XML root element.
        source (str): Dataset key.
        domain (str): Domain label.
        doc_id (str): Identifier for the single produced Document.
        extra_metadata (Optional[Dict[str, Any]]): Per-file fields
            from `MetadataSidecar`, copied onto the produced Document.

    Yields:
        Document: One document per file, or nothing if the file had
            no token-bearing sentences.
    """
    s_elems = list(root.iter(_S_TAG))
    doc = _build_document(
        s_elems=s_elems,
        doc_id=doc_id,
        source=source,
        domain=domain,
        metadata=extra_metadata,
    )
    if doc is not None:
        yield doc


def _parse_plain(
    root: "etree._Element",
    source: str,
    domain: str,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Iterator[Document]:
    """Yield Documents from a plain TEI tree (no <w> elements).

    One Document is produced per <p> element with non-empty text.
    The doc_id comes from the xml:id attribute on <p>.

    Args:
        root (etree._Element): Parsed XML root element.
        source (str): Dataset key.
        domain (str): Domain label.
        extra_metadata (Optional[Dict[str, Any]]): Per-document
            fields copied into every yielded Document.metadata.

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
            metadata=dict(extra_metadata) if extra_metadata else {},
        )


class TeiExtractor(FileBasedExtractor):
    """Extracts Documents from TEI XML files.

    For annotated TEI (containing `<w>` word elements), the
    grouping strategy depends on whether the file uses `<u>`
    utterance wrappers:

    - With `<u>`: one Document per utterance (parliamentary
      corpora — ParlaMint-SI, siParl). Speaker (`who`) and topic
      (`ana`) attributes flow into `Document.metadata`.
    - Without `<u>`: one Document per file (academic corpora
      like KAS). `doc_id` falls back to the filename's stem.

    Plain TEI (no `<w>`) keeps the per-`<p>` Document behavior
    used by Janes-Forum/Blog/News (currently disabled in the
    download catalogue but supported).
    """

    def iter_input_files(self, input_dir: Path) -> List[Path]:
        """Return sorted .xml files under input_dir.

        Args:
            input_dir (Path): Directory searched recursively.

        Returns:
            List[Path]: Sorted .xml file paths.
        """
        return sorted(input_dir.rglob("*.xml"))

    def extract_files(
        self,
        files: List[Path],
        source: str,
        domain: str,
        input_dir: Path,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Document]:
        """Yield Documents from the given TEI XML files.

        Args:
            files (List[Path]): .xml files to parse, in order.
            source (str): Dataset key assigned to every Document.
            domain (str): Domain label assigned to every Document.
            input_dir (Path): Dataset root, used to locate an optional
                `MetadataSidecar` TSV.
            metadata (Optional[Dict[str, Any]]): Optional `metadata:`
                config block describing an external per-document TSV.

        Yields:
            Document: Extracted documents in unified schema format.
        """
        sidecar: Optional[MetadataSidecar] = (
            MetadataSidecar.from_config(input_dir, metadata)
            if metadata
            else None
        )

        for filepath in files:
            try:
                tree = etree.parse(str(filepath))
            except etree.XMLSyntaxError as exc:
                logger.warning(
                    "Skipping %s — parse error: %s", filepath, exc
                )
                continue

            extra = sidecar.get_for_path(filepath) if sidecar else {}
            root = tree.getroot()
            is_annotated = root.find(f".//{_W_TAG}") is not None

            if not is_annotated:
                yield from _parse_plain(root, source, domain, extra)
                continue

            # Annotated: prefer <u>-based grouping when the file uses
            # it; otherwise fall back to one Document per file.
            if root.find(f".//{_U_TAG}") is not None:
                yield from _parse_annotated_with_utterances(
                    root, source, domain, extra
                )
            else:
                yield from _parse_annotated_per_file(
                    root, source, domain, doc_id=filepath.stem, extra_metadata=extra
                )


register_extractor("tei", TeiExtractor)
