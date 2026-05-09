"""Parser for the Sloleks 3.x Slovenian inflectional lexicon (TEI XML).

Sloleks 3.0/3.1 is distributed exclusively as TEI XML on CLARIN.SI;
each zip contains ~100 split XML files plus XSDs and a mezzanine file.
This module walks those files entry-by-entry and yields per-lemma
records with `entry_id`, `lemma`, `lemma_msd`, and `forms` keys, where
`forms` is a list of `{"form", "msd"}` pairs covering both the lemma
form and every inflected form.

The parser is intentionally namespace-agnostic and tolerates minor
schema variations (lemma MSD on `<gramGrp>` outside of any `<form>`,
MSD encoded via `<gram type="msd">`, `<msd>`, or a `feats` attribute,
etc.).

Used by `scripts/data/to_tokenizer_eval.py` to materialize a
tokenizer/morphology evaluation JSONL. Sloleks is intentionally absent
from `configs/data/extract.yaml`, so it never enters the
extract/datatrove/curate pipelines.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

#: TEI default namespace used by Sloleks 3.x XML files.
TEI_NS = "http://www.tei-c.org/ns/1.0"


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


def _find_msd(elem: ET.Element) -> Optional[str]:
    """Locate an MSD value attached to *elem*.

    Tries, in order: a child `<msd>` element, a child `<gram type="msd">`
    element, and finally a `feats` attribute. Search is restricted to
    direct descendants so we do not accidentally pick up the MSD of a
    sibling form.

    Args:
        elem: The element to inspect (typically `<form>` or
            `<entry>`/`<gramGrp>`).

    Returns:
        Optional[str]: The MSD string, or None when none is found.
    """
    for child in elem.iter():
        local = _local_name(child.tag)
        if local == "msd":
            text = _text_or_none(child)
            if text is not None:
                return text
        elif local == "gram":
            attr_type = child.get("type") or ""
            if attr_type.lower() == "msd":
                text = _text_or_none(child)
                if text is not None:
                    return text
    feats = elem.get("feats")
    if feats:
        return feats.strip() or None
    return None


def _form_orth(form_elem: ET.Element) -> Optional[str]:
    """Return the orthographic form text from a `<form>` element.

    Args:
        form_elem: A TEI `<form>` element.

    Returns:
        Optional[str]: The text inside the first descendant `<orth>`
            element, or None when none is present.
    """
    for child in form_elem.iter():
        if _local_name(child.tag) == "orth":
            text = _text_or_none(child)
            if text is not None:
                return text
    return None


def _is_lemma_form(form_elem: ET.Element) -> bool:
    """Return True if *form_elem* describes a lemma (headword) form.

    Args:
        form_elem: A TEI `<form>` element.

    Returns:
        bool: True when the element's `type` attribute is `"lemma"`
            (case-insensitive), False otherwise.
    """
    return (form_elem.get("type") or "").lower() == "lemma"


def _entry_to_record(entry: ET.Element) -> Optional[Dict[str, Any]]:
    """Convert a single TEI `<entry>` element into a JSONL-ready record.

    Args:
        entry: A `<entry>` element from a Sloleks TEI file.

    Returns:
        Optional[Dict[str, Any]]: A record with `entry_id`, `lemma`,
            `lemma_msd`, and `forms` keys; or None when the entry has
            no extractable lemma orthography.
    """
    entry_id = entry.get("{http://www.w3.org/XML/1998/namespace}id") or entry.get("id")

    lemma: Optional[str] = None
    lemma_msd: Optional[str] = None
    forms: List[Dict[str, Optional[str]]] = []
    lemma_form_idx: Optional[int] = None

    for child in list(entry):
        local = _local_name(child.tag)
        if local == "form":
            orth = _form_orth(child)
            msd = _find_msd(child)
            if _is_lemma_form(child):
                if lemma is None and orth is not None:
                    lemma = orth
                if lemma_msd is None and msd is not None:
                    lemma_msd = msd
                if orth is not None:
                    lemma_form_idx = len(forms)
                    forms.append({"form": orth, "msd": msd})
            elif orth is not None:
                forms.append({"form": orth, "msd": msd})
        elif local == "gramGrp" and lemma_msd is None:
            lemma_msd = _find_msd(child)

    if lemma is None:
        return None

    # Backfill the lemma form's msd if it was declared on a sibling
    # <gramGrp> rather than inside the <form type="lemma"> itself.
    if lemma_form_idx is not None and forms[lemma_form_idx]["msd"] is None:
        forms[lemma_form_idx]["msd"] = lemma_msd

    return {
        "entry_id": entry_id,
        "lemma": lemma,
        "lemma_msd": lemma_msd,
        "forms": forms,
    }


def iter_sloleks_entries(xml_path: Path) -> Iterator[Dict[str, Any]]:
    """Stream entries from a single Sloleks TEI XML file.

    Uses `xml.etree.ElementTree.iterparse` so memory use stays bounded
    even on the multi-GB merged distribution.

    Args:
        xml_path: Path to a Sloleks TEI XML file.

    Yields:
        Dict[str, Any]: One record per `<entry>` element, with
            `entry_id`, `lemma`, `lemma_msd`, and `forms` keys.
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
