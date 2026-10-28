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
        text:        per-sentence token forms joined with single
                     spaces, honouring the TEI `join` attribute
                     (see `_extract_tokens_from_sentence`), then
                     sentence strings joined with newlines so
                     datatrove's sentence splitter retains the cue.
        source:      provided by caller.
        domain:      provided by caller.
        doc_id:      `<u>`'s `xml:id` (utterance path) or the
                     source filename's stem (per-file fallback).
        metadata:    `who` / `ana` from the `<u>` element (utterance
                     path) layered on top of any per-file fields
                     from a `metadata:` config block (see
                     `MetadataTable`); empty when neither applies.
        annotations:
            tokens:    flat concatenation across every contained `<s>`.
            sentences: one inclusive `[start, end]` per sentence,
                       in reading order.

    Plain TEI: one Document per `<p>` with `doc_id` from the
    paragraph's `xml:id`, no annotations, and per-file `metadata:`
    fields when configured.
"""

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from lxml import etree

from slm4ie.data.extract.extractors import FileBasedExtractor, register_extractor
from slm4ie.data.extract.metadata_table import MetadataTable
from slm4ie.data.schema import Annotations, Document, Token, render_sentence

# ===============================
# Defaults
# ===============================

_TEI_NS = "http://www.tei-c.org/ns/1.0"
_XML_NS = "http://www.w3.org/XML/1998/namespace"

_W_TAG = f"{{{_TEI_NS}}}w"
_PC_TAG = f"{{{_TEI_NS}}}pc"
_S_TAG = f"{{{_TEI_NS}}}s"
_P_TAG = f"{{{_TEI_NS}}}p"
_BODY_TAG = f"{{{_TEI_NS}}}body"
_U_TAG = f"{{{_TEI_NS}}}u"

#: Files larger than this are parsed with a memory-bounded streaming
#: reader instead of a full in-memory DOM. GigaFida contains whole-
#: document segments up to hundreds of MB; an `etree.parse` DOM of such
#: a file is ~10-15x its size, which OOMs the extractor under parallel
#: sharding. Smaller files keep the faster full-DOM path.
_LARGE_FILE_BYTES = 64 * 1024 * 1024
_NAME_TAG = f"{{{_TEI_NS}}}name"
_XML_ID = f"{{{_XML_NS}}}id"

# MULTEXT-East v6 category (1st char of compact MSD) to UPOS; `_mte_to_upos` refines it
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


# ===============================
# Helper functions
# ===============================


def _parse_msd(
    msd: str | None,
) -> tuple[str | None, str | None]:
    """Parse a morphosyntactic description string into UPOS and feats.

    The UPosTag=X part is extracted as UPOS; all remaining Key=Value
    parts are rejoined with "|" as the feats string.

    Args:
        msd: MSD string such as
            "UPosTag=NOUN|Case=Nom|Gender=Masc", or None.

    Returns:
        `(upos, feats)`; either may be None if not present.
    """
    if not msd:
        return None, None

    # split the MSD into UPOS and feats
    parts = msd.split("|")
    upos: str | None = None
    remaining: list[str] = []

    for part in parts:
        if part.startswith("UPosTag="):
            upos = part[len("UPosTag=") :]
        else:
            remaining.append(part)

    feats = "|".join(remaining) if remaining else None
    return upos, feats


def _mte_to_upos(mte: str) -> str | None:
    """Map a MULTEXT-East v6 compact MSD to a UPOS tag.

    Uses the first character (category) for the base mapping and the
    second character (type) to refine NOUN vs PROPN, VERB vs AUX, and
    CCONJ vs SCONJ.

    Args:
        mte: Compact MSD such as "Ncnsn" or "Z".

    Returns:
        UPOS tag, or None if the category is unknown.
    """
    if not mte:
        return None

    # map the category
    cat = mte[0]
    upos = _MTE_CATEGORY_UPOS.get(cat)
    if upos is None:
        return None

    # refine by the type character
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
    ana: str | None,
) -> tuple[str | None, str | None]:
    """Parse a TEI ana attribute into UPOS and feats.

    Looks for a mte:<descriptor> value (MULTEXT-East v6 compact MSD,
    used by KAS) and maps it to UPOS. The raw descriptor is preserved
    in feats as MTE=<descriptor>.

    Args:
        ana: The ana attribute value, or None.

    Returns:
        `(upos, feats)`; either may be None if no MTE descriptor is found.
    """
    if not ana:
        return None, None

    for part in ana.split():
        if part.startswith("mte:"):
            mte = part[len("mte:") :]
            if not mte:
                return None, None
            return _mte_to_upos(mte), f"MTE={mte}"

    return None, None


def _parse_morph(
    elem: "etree._Element",
) -> tuple[str | None, str | None]:
    """Extract UPOS and feats from a token element.

    Prefers msd (ParlaMint, siParl) over ana (KAS).

    Args:
        elem: A <w> or <pc> element.

    Returns:
        (upos, feats).
    """
    msd = elem.get("msd")
    if msd:
        return _parse_msd(msd)
    return _parse_ana(elem.get("ana"))


def _extract_tokens_from_sentence(
    s_elem: "etree._Element",
) -> list[Token]:
    """Extract Token objects from a <s> element.

    Iterates direct and <name>-wrapped children to find all <w> and
    <pc> token elements. Each token's `space_after` is derived from
    the TEI/CLARIN.SI `join` attribute, which marks the *absence* of
    whitespace: `right` means no space to the token's right (KAS and
    ParlaMint place it on the token preceding punctuation), `left`
    means no space to its left, and `both` means neither side. For
    consecutive tokens A, B the space between them is dropped when A
    is right-joined or B is left-joined.

    Args:
        s_elem: The <s> XML element.

    Returns:
        Extracted tokens in document order, with
            `space_after` set from the `join` attributes.
    """
    # collect the tokens and their join attributes
    tokens: list[Token] = []
    joins: list[str | None] = []

    def _append(elem: "etree._Element") -> None:
        upos, feats = _parse_morph(elem)
        tokens.append(
            Token(
                form=elem.text or "",
                lemma=elem.get("lemma"),
                upos=upos,
                feats=feats,
            )
        )
        joins.append(elem.get("join"))

    for child in s_elem:
        if child.tag in (_W_TAG, _PC_TAG):
            _append(child)
        elif child.tag == _NAME_TAG:
            for wc in child:
                if wc.tag in (_W_TAG, _PC_TAG):
                    _append(wc)

    # drop the space where a join attribute says so
    for i, join in enumerate(joins):
        next_join = joins[i + 1] if i + 1 < len(joins) else None
        if join in ("right", "both") or next_join in ("left", "both"):
            tokens[i].space_after = False

    return tokens


def _build_document(
    s_elems: list["etree._Element"],
    doc_id: str,
    source: str,
    domain: str,
    metadata: dict[str, Any] | None = None,
    native_id: str | None = None,
) -> Document | None:
    """Combine `<s>` siblings into one document-level Document.

    Sentence texts are reconstructed from each sentence's tokens,
    honouring `Token.space_after` (derived from the TEI `join`
    attribute); the sentence strings are then joined with newlines.
    Annotations carry the flat token sequence and inclusive
    `[start, end]` token-index spans for every contained sentence.

    Args:
        s_elems: The `<s>` elements
            that constitute this document, in reading order.
        doc_id: Identifier for the resulting Document.
        source: Dataset key.
        domain: Domain label.
        metadata: Optional metadata to
            attach to the Document. For utterance-level docs this
            typically merges per-file `MetadataTable` fields with
            the utterance's `who` / `ana` attributes.
        native_id: The element's own `xml:id` when
            the document is one `<u>`; None when it is a whole file
            and `doc_id` is the filename stem.

    Returns:
        One Document, or `None` when every
            `<s>` element parsed to zero tokens.
    """
    # aggregate the sentences
    sentence_texts: list[str] = []
    flat_tokens: list[Token] = []
    spans: list[list[int]] = []
    cursor = 0

    for s_elem in s_elems:
        tokens = _extract_tokens_from_sentence(s_elem)
        if not tokens:
            continue
        sentence_texts.append(render_sentence(tokens))
        start = cursor
        flat_tokens.extend(tokens)
        cursor += len(tokens)
        spans.append([start, cursor - 1])

    if not flat_tokens:
        return None

    # build the document
    annotations = Annotations(tokens=flat_tokens, sentences=spans)
    return Document(
        text="\n".join(sentence_texts),
        source=source,
        domain=domain,
        doc_id=doc_id,
        native_id=native_id,
        metadata=dict(metadata) if metadata else {},
        annotations=annotations,
    )


def _stream_per_file_document(
    filepath: Path,
    doc_id: str,
    source: str,
    domain: str,
    metadata: dict[str, Any] | None = None,
) -> tuple[Document | None, bool]:
    """Stream-aggregate a large per-file annotated TEI into one Document.

    Reads `<s>` elements via `etree.iterparse` (with `huge_tree`) and
    clears each one immediately after extracting its tokens, so peak
    memory tracks the accumulated token list rather than a full DOM.
    The result is identical to `_parse_annotated_per_file` for the
    per-file case: iterparse emits `<s>` end events in document order,
    the same order `root.iter` yields them.

    The function only handles the per-file annotated structure (no
    `<u>`). It detects an utterance- or plain-structured file and signals
    the caller to fall back to the full-DOM path: if the first `<s>` is
    wrapped in a `<u>`, or if the file has no `<s>` at all.

    Args:
        filepath: Path to the large TEI XML file.
        doc_id: Identifier for the produced Document.
        source: Dataset key.
        domain: Domain label.
        metadata: Per-file metadata table fields.

    Returns:
        `(document, handled)`. When
            `handled` is True the streaming path applies: `document` is
            the aggregated Document, or `None` if every `<s>` parsed to
            zero tokens. When `handled` is False the file is not
            per-file annotated and the caller must use the full-DOM path.
    """
    # stream the sentences
    sentence_texts: list[str] = []
    flat_tokens: list[Token] = []
    spans: list[list[int]] = []
    cursor = 0
    saw_sentence = False

    context = etree.iterparse(str(filepath), events=("end",), tag=_S_TAG, huge_tree=True)
    for _, s_elem in context:
        if not saw_sentence:
            saw_sentence = True
            parent = s_elem.getparent()
            if parent is not None and parent.tag == _U_TAG:
                # utterance-structured: the caller falls back to the full-DOM path
                return None, False

        tokens = _extract_tokens_from_sentence(s_elem)
        if tokens:
            sentence_texts.append(render_sentence(tokens))
            start = cursor
            flat_tokens.extend(tokens)
            cursor += len(tokens)
            spans.append([start, cursor - 1])

        # free the parsed sentence so the tree never outgrows one sentence
        s_elem.clear()
        while s_elem.getprevious() is not None:
            del s_elem.getparent()[0]

    if not saw_sentence:
        # no <s> at all: plain or empty, the caller decides
        return None, False

    if not flat_tokens:
        return None, True

    # build the document
    annotations = Annotations(tokens=flat_tokens, sentences=spans)
    document = Document(
        text="\n".join(sentence_texts),
        source=source,
        domain=domain,
        doc_id=doc_id,
        metadata=dict(metadata) if metadata else {},
        annotations=annotations,
    )
    return document, True


def _utterance_metadata(u_elem: "etree._Element") -> dict[str, Any]:
    """Return `Document.metadata` derived from a `<u>` element.

    Args:
        u_elem: The `<u>` element.

    Returns:
        `who` and `ana` attribute values when
            present; the key is omitted when the attribute is absent
            so that callers can use `"key" in metadata` cleanly.
    """
    md: dict[str, Any] = {}
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
    extra_metadata: dict[str, Any] | None = None,
) -> Iterator[Document]:
    """Yield one Document per `<u>` element in an annotated TEI tree.

    Each `<u>`'s sentence descendants are aggregated into a single
    Document; the utterance's `xml:id` becomes the `doc_id` and
    its `who` / `ana` attributes flow into `metadata`, layered on
    top of any per-file fields supplied via *extra_metadata*.

    Args:
        root: Parsed XML root element.
        source: Dataset key.
        domain: Domain label.
        extra_metadata: Per-file fields
            from `MetadataTable`. Utterance attributes (`who`,
            `ana`) take precedence on key collision since they are
            more specific.

    Yields:
        One document per non-empty `<u>`.
    """
    # build one document per utterance
    base: dict[str, Any] = dict(extra_metadata) if extra_metadata else {}
    for u_elem in root.iter(_U_TAG):
        s_elems = list(u_elem.iter(_S_TAG))
        native_id = u_elem.get(_XML_ID)
        merged = {**base, **_utterance_metadata(u_elem)}
        doc = _build_document(
            s_elems=s_elems,
            doc_id=native_id or "",
            source=source,
            domain=domain,
            metadata=merged,
            native_id=native_id,
        )
        if doc is not None:
            yield doc


def _parse_annotated_per_file(
    root: "etree._Element",
    source: str,
    domain: str,
    doc_id: str,
    extra_metadata: dict[str, Any] | None = None,
) -> Iterator[Document]:
    """Yield exactly one Document by aggregating every `<s>` in *root*.

    Used when an annotated TEI file has no `<u>` elements (e.g.
    KAS). `doc_id` is supplied by the caller (typically the
    source filename's stem).

    Args:
        root: Parsed XML root element.
        source: Dataset key.
        domain: Domain label.
        doc_id: Identifier for the single produced Document.
        extra_metadata: Per-file fields
            from `MetadataTable`, copied onto the produced Document.

    Yields:
        One document per file, or nothing if the file had
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
    extra_metadata: dict[str, Any] | None = None,
) -> Iterator[Document]:
    """Yield Documents from a plain TEI tree (no <w> elements).

    One Document is produced per `<p>` element with non-empty text,
    scoped to the `<body>` so that `<p>` elements in the `<teiHeader>`
    (copyright notices, COBISS/COMARC catalog codes) are never emitted
    as documents. A file whose body has no paragraphs (e.g. a dedup
    segment emptied to a single `<gap>`) yields nothing. The doc_id
    comes from the xml:id attribute on `<p>`.

    Args:
        root: Parsed XML root element.
        source: Dataset key.
        domain: Domain label.
        extra_metadata: Per-document
            fields copied into every yielded Document.metadata.

    Yields:
        One document per non-empty body paragraph.
    """
    body = root.find(f".//{_BODY_TAG}")
    if body is None:
        return

    # build one document per paragraph
    for p_elem in body.iter(_P_TAG):
        text = "".join(p_elem.itertext()).strip()
        if not text:
            continue
        doc_id = p_elem.get(_XML_ID)
        yield Document(
            text=text,
            source=source,
            domain=domain,
            doc_id=doc_id,
            native_id=doc_id,
            metadata=dict(extra_metadata) if extra_metadata else {},
        )


# ===============================
# Extractor class
# ===============================


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

    def iter_input_files(self, input_dir: Path) -> list[Path]:
        """Return sorted .xml files under input_dir.

        Args:
            input_dir: Directory searched recursively.

        Returns:
            Sorted .xml file paths.
        """
        return sorted(input_dir.rglob("*.xml"))

    def extract_files(
        self,
        files: list[Path],
        source: str,
        domain: str,
        input_dir: Path,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[Document]:
        """Yield Documents from the given TEI XML files.

        Args:
            files: .xml files to parse, in order.
            source: Dataset key assigned to every Document.
            domain: Domain label assigned to every Document.
            input_dir: Dataset root, used to locate an optional
                `MetadataTable` TSV.
            metadata: Optional `metadata:`
                config block describing an external per-document TSV.

        Yields:
            Extracted documents in unified schema format.
        """
        table: MetadataTable | None = MetadataTable.from_config(input_dir, metadata) if metadata else None

        for filepath in files:
            extra = table.get_for_path(filepath) if table else {}

            # stream large files: a multi-GB DOM would OOM the extractor under sharding
            # (utterance and plain files are never large; they fall through to the DOM path)
            try:
                is_large = filepath.stat().st_size > _LARGE_FILE_BYTES
            except OSError:
                is_large = False
            if is_large:
                doc, handled = _stream_per_file_document(filepath, filepath.stem, source, domain, extra)
                if handled:
                    if doc is not None:
                        yield doc
                    continue

            # parse the full DOM
            try:
                tree = etree.parse(str(filepath))
            except etree.XMLSyntaxError as exc:
                logger.warning("Skipping %s — parse error: %s", filepath, exc)
                continue

            root = tree.getroot()
            is_annotated = root.find(f".//{_W_TAG}") is not None

            if not is_annotated:
                yield from _parse_plain(root, source, domain, extra)
                continue

            # annotated: group by <u> when present, else one document per file
            if root.find(f".//{_U_TAG}") is not None:
                yield from _parse_annotated_with_utterances(root, source, domain, extra)
            else:
                yield from _parse_annotated_per_file(root, source, domain, doc_id=filepath.stem, extra_metadata=extra)


register_extractor("tei", TeiExtractor)
