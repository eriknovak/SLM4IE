"""Parser for the Sloleks 3.x Slovenian inflectional lexicon XML.

Sloleks 3.0/3.1 is distributed on CLARIN.SI as a custom `<lexicon>` XML
(not TEI); each zip contains ~100 split XML files plus XSDs and a
mezzanine file. Every `<entry>` carries its headword lemma at
`head/headword/lemma` and its inflected forms under
`body/wordFormList/wordForm`, where each word form has a JOS `<msd>` and
one or more orthographic surface forms at
`formRepresentations/orthographyList/orthography/form`.

This module walks those files entry-by-entry with `iterparse` (bounded
memory) and yields per-lemma records with `entry_id`, `lemma`,
`lemma_msd`, and `forms` keys, where `forms` is a list of
`{"form", "msd"}` pairs covering both the lemma form and every inflected
form. Accentuation and pronunciation `<form>` siblings inside a word form
are deliberately ignored — only orthographic forms are emitted.

The parser is namespace-agnostic (it matches on local element names) so
it tolerates either a namespaced or bare `<lexicon>` root.

Used by `scripts/data/to_tokenization.py` to materialize a
tokenizer/morphology evaluation JSONL. Sloleks is intentionally absent
from `configs/data/extract.yaml`, so it never enters the
extract/datatrove/curate pipelines.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


def _local_name(tag: str) -> str:
    """Return the local part of an XML element tag.

    Args:
        tag: The element tag, optionally Clark-notation namespaced
            (e.g. `"{http://www.tei-c.org/ns/1.0}entry"`).

    Returns:
        str: The local name with any namespace prefix stripped.
    """
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def _text_or_none(elem: Optional[ET.Element]) -> Optional[str]:
    """Return stripped text content of *elem*, or None when empty.

    Args:
        elem: An XML element, or None.

    Returns:
        Optional[str]: The trimmed text content, or None when the
            element is missing or its text is blank.
    """
    if elem is None or elem.text is None:
        return None
    text = elem.text.strip()
    return text or None


def _iter_children(elem: ET.Element, name: str) -> Iterator[ET.Element]:
    """Yield direct children of *elem* whose local name is *name*.

    Args:
        elem: The parent element.
        name: The local (namespace-stripped) tag name to match.

    Yields:
        ET.Element: Matching direct child elements, in document order.
    """
    for child in elem:
        if _local_name(child.tag) == name:
            yield child


def _first_descendant(elem: ET.Element, path: List[str]) -> Optional[ET.Element]:
    """Return the first element reached by following *path* of local names.

    Args:
        elem: The element to search from.
        path: Sequence of local names to descend through, each matched
            against direct children of the previous level.

    Returns:
        Optional[ET.Element]: The element at the end of the first matching
            chain, or None when any level is absent.
    """
    current: Optional[ET.Element] = elem
    for name in path:
        if current is None:
            return None
        current = next(_iter_children(current, name), None)
    return current


def _entry_id(entry: ET.Element) -> Optional[str]:
    """Return a stable identifier for a Sloleks `<entry>`.

    Prefers the `sloleksId` of the `head/lexicalUnit` element, then its
    `sloleksKey`.

    Args:
        entry: A Sloleks `<entry>` element.

    Returns:
        Optional[str]: The lexical-unit identifier, or None when absent.
    """
    head = next(_iter_children(entry, "head"), None)
    if head is None:
        return None
    unit = next(_iter_children(head, "lexicalUnit"), None)
    if unit is None:
        return None
    return unit.get("sloleksId") or unit.get("sloleksKey")


def _entry_lemma(entry: ET.Element) -> Optional[str]:
    """Return the headword lemma of a Sloleks `<entry>`.

    Args:
        entry: A Sloleks `<entry>` element.

    Returns:
        Optional[str]: The lemma text from `head/headword/lemma`, or None
            when absent.
    """
    lemma = _first_descendant(entry, ["head", "headword", "lemma"])
    return _text_or_none(lemma)


def _wordform_msd(wordform: ET.Element) -> Optional[str]:
    """Return the JOS MSD of a `<wordForm>` element.

    Args:
        wordform: A `<wordForm>` element.

    Returns:
        Optional[str]: The text of the direct `<msd>` child, or None when
            absent.
    """
    msd = next(_iter_children(wordform, "msd"), None)
    return _text_or_none(msd)


def _wordform_orth_forms(wordform: ET.Element) -> List[str]:
    """Return the orthographic surface forms of a `<wordForm>`.

    Reads only `formRepresentations/orthographyList/orthography/form`,
    deliberately ignoring the accentuation and pronunciation `<form>`
    siblings that share the same parent.

    Args:
        wordform: A `<wordForm>` element.

    Returns:
        List[str]: Orthographic form strings, in document order (one per
            `<orthography>` variant).
    """
    forms: List[str] = []
    for reps in _iter_children(wordform, "formRepresentations"):
        for orth_list in _iter_children(reps, "orthographyList"):
            for orth in _iter_children(orth_list, "orthography"):
                form = next(_iter_children(orth, "form"), None)
                text = _text_or_none(form)
                if text is not None:
                    forms.append(text)
    return forms


def _entry_to_record(entry: ET.Element) -> Optional[Dict[str, Any]]:
    """Convert a single Sloleks `<entry>` element into a JSONL-ready record.

    Args:
        entry: An `<entry>` element from a Sloleks lexicon file.

    Returns:
        Optional[Dict[str, Any]]: A record with `entry_id`, `lemma`,
            `lemma_msd`, and `forms` keys; or None when the entry has no
            lemma or no orthographic forms.
    """
    lemma = _entry_lemma(entry)
    if lemma is None:
        return None

    forms: List[Dict[str, Optional[str]]] = []
    lemma_msd: Optional[str] = None

    body = next(_iter_children(entry, "body"), None)
    if body is not None:
        for word_list in _iter_children(body, "wordFormList"):
            for wordform in _iter_children(word_list, "wordForm"):
                msd = _wordform_msd(wordform)
                for orth in _wordform_orth_forms(wordform):
                    forms.append({"form": orth, "msd": msd})
                    if orth == lemma and lemma_msd is None:
                        lemma_msd = msd

    if not forms:
        return None

    # Fall back to the first form's MSD when no form orthographically
    # matches the headword (e.g. suppletive or multi-word lemmas).
    if lemma_msd is None:
        lemma_msd = forms[0]["msd"]

    return {
        "entry_id": _entry_id(entry),
        "lemma": lemma,
        "lemma_msd": lemma_msd,
        "forms": forms,
    }


def iter_sloleks_entries(xml_path: Path) -> Iterator[Dict[str, Any]]:
    """Stream entries from a single Sloleks lexicon XML file.

    Uses `xml.etree.ElementTree.iterparse` so memory use stays bounded
    even on the multi-GB merged distribution.

    Args:
        xml_path: Path to a Sloleks lexicon XML file.

    Yields:
        Dict[str, Any]: One record per `<entry>` element with a lemma and
            at least one orthographic form, with `entry_id`, `lemma`,
            `lemma_msd`, and `forms` keys.
    """
    context = ET.iterparse(str(xml_path), events=("end",))
    for _, elem in context:
        if _local_name(elem.tag) != "entry":
            continue
        record = _entry_to_record(elem)
        if record is not None:
            yield record
        elem.clear()


def iter_sloleks_dir(root: Path) -> Iterator[Dict[str, Any]]:
    """Stream entries from every Sloleks XML file under *root*.

    Walks recursively, sorts files by name for determinism, and skips
    schema (.xsd) files and the mezzanine sidecar.

    Args:
        root: Directory that contains the unzipped Sloleks distribution
            (typically `/vault/data/SLM4IE/raw/sloleks/`).

    Yields:
        Dict[str, Any]: Records as produced by `iter_sloleks_entries`.
    """
    files = sorted(root.rglob("*.xml"))
    for path in files:
        if path.name.endswith("_mezzanine.xml"):
            continue
        yield from iter_sloleks_entries(path)
