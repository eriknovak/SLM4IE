"""The reference topic page on the new builder (ADR-0005).

`topic_page` turns one `reference/<topic>/README.md` into the values
`templates/topic.html` draws: the kicker and title; the meta line (the open
or closed badge, the entry count, how many are checked against code, the
open ideas); the Question and the Introduction; the sticky section bar; then
the map, the compare switch, the coupling grid, the open ideas, the records
that use the topic and the Entries table. Only the Question and the
Introduction come from the README. Every other part is drawn from what the
entries, `reference/ideas.yaml`, the programmes and the records already
store (tables.md, Topic tables). Every entry the page names shows its title
with its status square; a part with nothing to show says so in words.

Rows the page reads beyond the spec's: an entry's Facts row `<short> —
<note>` groups on the short value and shows the note under its group, its
Difference row `Change` is the Changes line on its map node, its Sources
Code and Paper rows fill the Entries table, and a programme's
`planned[].ticket` names the experiment an idea became.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

from markupsafe import Markup, escape

# record imports builder, which imports this module: read record's names at call time
from . import markup, record as records
from .builder import (
    ENTRY_STATUS,
    ENVIRONMENT,
    Model,
    PageView,
    Site,
    entry_page_id,
    entry_ref,
    entry_title,
    view,
)
from .checks import topic as topic_checks
from .document import read_document
from .entry_page import (
    IDENT,
    NOTE,
    REFERENCE,
    EntryText,
    couples,
    entry_keys,
    link_idents,
    ordered_rows,
    written,
    written_block,
)

# ===============================
# Defaults
# ===============================

TEMPLATE = "topic.html"
PROGRAMMES = "programmes"
# how the coupling lede names the topic itself as a source of links
HERE = "this topic"
# a topic's status → the badge it draws
STATUS_BADGE = {"open": "running", "closed": "draft"}
# an idea's Touches relation → the label of its line on the idea card; "" draws no line
RELATION_LINES = {"changes": "changes", "already has": "already in", "does not apply": ""}
NOT_ASSESSED = "not assessed"
# the entry chips an idea card line shows before it counts the rest
CHIPS_SHOWN = 3
# a section's anchor → its heading, the colour of its chapter label, its name in the section bar
SECTIONS = {
    "map": ("Map", "accent", "Map"),
    "compare": ("Compare one fact at a time", "accent", "Compare"),
    "used": ("Where the parts are used", "accent", "Where used"),
    "ideas": ("Open ideas", "accent", "Open ideas"),
    "used-by": ("Used by", "muted", "Used by"),
    "entries": ("Entries", "muted", "Entries"),
}
NO_QUESTION = "The topic has no question yet."
NO_INTRODUCTION = "Not written yet. The topic’s introduction is written once its entries have descriptions."
NO_ENTRIES = "No entry yet. Each entry is a file beside this README."
NO_MAP = "No entry names its predecessor yet, so there is no map."
NO_COMPARE = "The topic names no fact to compare yet: its compare list is empty."
NO_FACT = "No entry has a fact yet, so the compare switch is hidden."
NO_COUPLING = (
    "No entry here couples to an entry of another topic yet. "
    "The grid fills from the couples_to of entries on either side."
)
NO_IDEAS_CACHE = (
    "No ideas loaded: there is no reference/ideas.yaml yet. "
    "A build without --offline reads the ideas from the tracker."
)
NO_IDEAS = "No open ideas. Ideas live as GitHub issues; none touches these entries yet."
NO_USE = "No experiment varies or uses an entry of this topic yet."
NO_CODE = "no code cited"
NO_PAPER = "no paper"
NO_IDEA = "no open idea"
MAP_NOTE = (
    "Each variant sits after the entry it is a variant of and states what it changes. "
    "Squares are status: filled checked against code, outlined from paper, dashed draft."
)
COMPARE_LEDE = "Pick a fact; entries group by their value. A group of one is the difference worth reading."
IDEAS_LEDE = "Ideas live as GitHub issues. This page shows only which entries they touch and the experiment each became."
# a cited code path in backticks, with an optional `:<line>` or `:<from>-<to>`
CODE_PATH = re.compile(r"`([\w./-]+\.[A-Za-z]\w*)(?::\d+(?:-\d+)?)?`")
# a paper cited by its address alone: a markdown link's target, or a bare one
PAPER_URL = re.compile(r"\]\((https?://[^)\s]+)\)|(?<![(\w])(https?://[^\s)>,]+)")
TICKET = re.compile(r"#?(\d+)")
NUMBER_WORDS = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
INLINE = ENVIRONMENT.get_template("components/inline.html").module

# ===============================
# Main function
# ===============================


@view("topic")
def topic_page(model: Model, site: Site) -> PageView:
    """Builds the topic page's values from one topic README and the site around it.

    Args:
        model: The topic.
        site: Every document: the topic's entries, the entries of other
            topics they couple with, and the records that vary or use them.

    Returns:
        The page view `templates/topic.html` renders.
    """
    entries = {m.id: m for m in site.of_kind("entry")}
    mine = sorted(
        (m for m in entries.values() if m.id.startswith(f"{model.id}/")),
        key=lambda m: entry_title(m).casefold(),
    )
    pid = f"topic-{model.id}"
    title = str(model.value("title") or model.id.replace("-", " ").capitalize())
    status = model.value("status") or "open"
    text = EntryText(model, entries)
    ideas = read_ideas(model, site, mine)

    # the lead: the question and the introduction
    question = model.section("Question")
    introduction = model.section("Introduction")
    question_html = text.inline(written_block(question.text)) if question else ""
    introduction_html = text.block(introduction.text) if introduction else ""

    # the sections, in page order
    used_by = read_used_by(site, mine, entries)
    sections = [
        _section("map", read_map(mine, text), no_map(mine)),
        _section("compare", *read_compare(model, mine, entries, text)),
        _section("used", read_coupling(mine, entries, site), NO_COUPLING),
        _section("ideas", ideas["cards"] if ideas else None, NO_IDEAS if ideas else NO_IDEAS_CACHE),
        _section("used-by", used_by, NO_USE),
        _section("entries", read_entries(mine, entries, ideas, pid), NO_ENTRIES),
    ]
    counts = {
        "ideas": len(ideas["cards"]) if ideas else 0,
        "used-by": len(used_by),
        "entries": len(mine),
    }
    bar = [
        {"href": f"#{pid}/{s['key']}", "name": SECTIONS[s["key"]][2], "n": counts.get(s["key"], 0)}
        for s in sections
    ]

    return PageView(
        kind="topic",
        id=pid,
        title=title,
        template=TEMPLATE,
        context={
            "kicker": model.document.path.relative_to(site.corpus.experiments.parent).as_posix(),
            "badge": STATUS_BADGE.get(status, "draft"),
            "status": status,
            "meta": meta_line(mine, ideas),
            "question": question_html,
            "no_question": NO_QUESTION,
            "introduction": introduction_html,
            "no_introduction": NO_INTRODUCTION,
            "bar": bar,
            "sections": sections,
            "map_note": MAP_NOTE,
            "compare_lede": COMPARE_LEDE,
            "ideas_lede": IDEAS_LEDE,
        },
    )


# ===============================
# Meta line
# ===============================


def meta_line(mine: list[Model], ideas: dict | None) -> list[str]:
    """The meta line after the badge: entries, checked ones, open ideas."""
    statuses = [m.value("status") or "draft" for m in mine]
    parts = [_count(len(mine), "entry", "entries")]
    if mine and all(s == "draft" for s in statuses):
        parts.append("all drafts")
    elif mine:
        parts.append(f"{statuses.count('checked')} checked against code")
    if ideas is None:
        parts.append("ideas not loaded")
    elif ideas["cards"]:
        parts.append(_count(len(ideas["cards"]), "open idea", "open ideas"))
    else:
        parts.append("no open ideas")
    return parts


# ===============================
# Map
# ===============================


def read_map(mine: list[Model], text: EntryText) -> dict | None:
    """The variant trees, each predecessor above its variants, and the entries
    in no tree; None while no entry names a predecessor in the topic."""
    inside = {m.id: m for m in mine}
    parent = {
        m.id: m.value("variant_of")
        for m in mine
        if m.value("variant_of") in inside and m.value("variant_of") != m.id
    }
    if not parent:
        return None
    children: dict[str, list[Model]] = {}
    for m in mine:
        if m.id in parent:
            children.setdefault(parent[m.id], []).append(m)
    placed: set[str] = set()

    def node(m: Model) -> dict:
        placed.add(m.id)
        status = m.value("status") or "draft"
        change = m.row("Difference from predecessor", "Change") if m.id in parent else ""
        return {
            "href": f"#{entry_page_id(m.id)}",
            "title": entry_title(m),
            "square": ENTRY_STATUS[status][0],
            "word": ENTRY_STATUS[status][1],
            "kind": m.value("kind") or "",
            "summary": text.inline(m.value("summary") or ""),
            "change": text.inline(change),
            "planned": status == "draft",
            "children": [node(c) for c in children.get(m.id, []) if c.id not in placed],
        }

    # each root with variants is a tree; a cycle has no root, so its first entry roots it
    trees = [node(m) for m in mine if m.id not in parent and m.id in children]
    for m in mine:
        if m.id in parent and m.id not in placed:
            trees.append(node(m))
    loose = [node(m) for m in mine if m.id not in placed]
    return {"trees": trees, "loose": loose}


def no_map(mine: list[Model]) -> str:
    """The map's empty state: why there is no map, and how the entries become its nodes."""
    if not mine:
        return NO_ENTRIES
    word = "draft" if all((m.value("status") or "draft") == "draft" for m in mine) else "entry"
    if len(mine) == 1:
        return f"{NO_MAP} The one {word} appears as a node once it says what it is a variant of."
    many = _count(len(mine), word, "drafts" if word == "draft" else "entries", words=True)
    return f"{NO_MAP} The {many} appear as nodes once each says what it is a variant of."


# ===============================
# Compare
# ===============================


def read_compare(
    model: Model, mine: list[Model], entries: dict[str, Model], text: EntryText
) -> tuple[list[dict] | None, str]:
    """One panel per compare label: its groups by short value, largest first,
    each entry's note under its group, and the entries lacking the row; with
    the empty state when there is no label or no row at all."""
    labels = [str(label) for label in model.value("compare") or []]
    if not labels:
        return None, NO_COMPARE
    pid = f"topic-{model.id}"
    panels, missing_counts = [], []
    for index, label in enumerate(labels, 1):
        groups: dict[str, dict] = {}
        lacking = []
        for m in mine:
            short, note = fact(m, label)
            if not short:
                lacking.append(entry_ref(m.id, entries))
                continue
            key = markup.plain_text(markup.inline_html(short)).casefold().rstrip(".")
            group = groups.setdefault(
                key, {"value": text.inline(markup.capitalize(short)), "entries": [], "notes": []}
            )
            group["entries"].append(entry_ref(m.id, entries))
            if note:
                group["notes"].append({"title": entry_title(m), "html": text.inline(note)})
        ordered = sorted(groups.values(), key=lambda g: -len(g["entries"]))
        panels.append(
            {"id": f"{pid}/fact-{index}", "label": label, "groups": ordered, "lacking": lacking}
        )
        missing_counts.append((label, len(lacking)))
    if not any(p["groups"] for p in panels):
        counts = ", ".join(f"{label} in {n}" for label, n in missing_counts)
        return None, f"{NO_FACT} Facts missing: {counts}." if mine else NO_FACT
    return panels, ""


def fact(model: Model, label: str) -> tuple[str, str]:
    """An entry's Facts row for a compare label, matched in any case: (short value, note)."""
    section = model.section("Facts")
    for name, row in (section.rows if section else {}).items():
        if name.lower() == label.lower():
            short, _, note = written(row.text).partition(NOTE)
            return short.strip(), note.strip()
    return "", ""


# ===============================
# Where the parts are used
# ===============================


def read_coupling(mine: list[Model], entries: dict[str, Model], site: Site) -> dict | None:
    """The coupling grid: this topic's coupled entries as rows, the entries of
    other topics they couple with (either way) as columns; None without one."""
    inside = {m.id for m in mine}
    links: dict[tuple[str, str], str] = {}
    sources: list[str] = []
    roles: list[str] = []
    for other in entries.values():
        for key, role in couples(other):
            tip = f"{entry_title(entries[key])} is the {role} of {entry_title(other)}" if role and key in entries else ""
            if other.id in inside and key in entries and key not in inside:
                links[(other.id, key)] = tip
                source = HERE
            elif key in inside and other.id not in inside:
                links[(key, other.id)] = tip
                source = other.id.split("/", 1)[0]
            else:
                continue
            if source not in sources:
                sources.append(source)
            if role and role not in roles:
                roles.append(role)
    if not links:
        return None

    # rows and columns in title order, each column with its topic's title
    rows = [m for m in mine if any(r == m.id for r, _ in links)]
    columns = sorted({c for _, c in links}, key=lambda c: entry_title(entries[c]).casefold())
    titles = {topic: read_topic_title(topic, site) for topic in {c.split("/", 1)[0] for c in columns}}

    # the lede: where the links come from, the roles they give, the rest
    named = [read_topic_title(s, site) if s != HERE else s for s in sources]
    lede = f"of entries in {_and(named)}"
    lede += f": the {_or(roles)} each names." if roles else "."
    others = len(mine) - len(rows)
    if others:
        lede += f" The other {_count(others, 'entry', 'entries', words=True)} couple with none."
    return {
        "lede": Markup("From the <code>couples_to</code> ") + lede,
        "columns": [
            {"href": f"#{entry_page_id(c)}", "title": entry_title(entries[c]), "topic": titles[c.split("/", 1)[0]]}
            for c in columns
        ],
        "rows": [
            {
                "entry": entry_ref(m.id, entries),
                "cells": [
                    {"on": (m.id, c) in links, "tip": links.get((m.id, c), "")} for c in columns
                ],
            }
            for m in rows
        ],
    }


def read_topic_title(topic: str, site: Site) -> str:
    """A topic's title from its README.md, else its folder name, capitalised."""
    readme = site.corpus.experiments / REFERENCE / topic / "README.md"
    meta = read_document(readme, "topic", site.corpus.experiments).frontmatter if readme.is_file() else {}
    return str(meta.get("title") or topic.replace("-", " ").capitalize())


# ===============================
# Open ideas
# ===============================


def read_ideas(model: Model, site: Site, mine: list[Model]) -> dict | None:
    """The open ideas touching an entry here, one card each: its issue, the
    entries it changes, already has and leaves unassessed, the planned
    experiment its ticket became, the records testing it. None without a cache."""
    reference = site.corpus.experiments / REFERENCE
    if not (reference / topic_checks.IDEAS_FILE).is_file():
        return None
    entries = {m.id: m for m in site.of_kind("entry")}
    inside = {m.id for m in mine}
    planned = read_planned(site)
    pid = f"topic-{model.id}"

    cards = []
    for idea in topic_checks.read_ideas(reference):
        touched = [t for t in idea.touches if t.entry in inside]
        if idea.state != topic_checks.OPEN or not touched:
            continue
        lines = []
        for relation, label in RELATION_LINES.items():
            keys = [t.entry for t in touched if t.relation == relation]
            if label and keys:
                lines.append(_chip_line(label, keys, entries))
        named = {t.entry for t in touched}
        if left := [m.id for m in mine if m.id not in named]:
            lines.append(_chip_line(NOT_ASSESSED, left, entries))
        cards.append(
            {
                "id": f"{pid}/idea-{idea.number}",
                "number": idea.number,
                "title": idea.title,
                "href": idea.url,
                "entries": sorted(named),
                "lines": lines,
                "planned": planned.get(idea.number, []),
                "tested": [
                    records.record_pill(r)
                    for r in site.of_kind("record")
                    if idea.number in _tickets(r.value("tests"))
                ],
            }
        )
    return {"cards": cards}


def read_planned(site: Site) -> dict[int, list[Markup]]:
    """Each planned experiment whose `ticket` names an issue, as a pill to its programme, by issue number."""
    found: dict[int, list[Markup]] = {}
    folder = site.corpus.experiments / PROGRAMMES
    for path in sorted(folder.glob("*.md")) if folder.is_dir() else []:
        meta = read_document(path, "programme", site.corpus.experiments).frontmatter
        slug = meta.get("slug") or path.stem
        for item in meta.get("planned") or []:
            if not isinstance(item, dict):
                continue
            for number in _tickets(item.get("ticket")):
                name = str(item.get("short") or item.get("slug") or item.get("title") or "")
                found.setdefault(number, []).append(
                    INLINE.pill("planned", name, str(item.get("title") or ""), f"#programme-{slug}")
                )
    return found


# ===============================
# Used by and Entries
# ===============================


def read_used_by(site: Site, mine: list[Model], entries: dict[str, Model]) -> list[dict]:
    """One line per record that varies or uses an entry here: its pill, then those entries."""
    inside = {m.id for m in mine}
    lines = []
    for record in site.of_kind("record"):
        varies = [k for k in entry_keys(record.value("varies")) if k in inside]
        uses = [k for k in entry_keys(record.value("uses")) if k in inside]
        if varies or uses:
            lines.append(
                {
                    "record": records.record_pill(record),
                    "varies": [entry_ref(k, entries) for k in varies],
                    "uses": [entry_ref(k, entries) for k in uses],
                }
            )
    return lines


def read_entries(mine: list[Model], entries: dict[str, Model], ideas: dict | None, pid: str) -> dict | None:
    """The Entries table: each entry's square and title, kind, status word,
    code files, first paper with a count, and the open ideas touching it; a
    column empty in every row is left out."""
    if not mine:
        return None
    rows = []
    for m in mine:
        status = m.value("status") or "draft"
        code, papers = sources(m)
        rows.append(
            {
                "entry": entry_ref(m.id, entries),
                "kind": m.value("kind") or "",
                "word": ENTRY_STATUS[status][1],
                "code": code,
                "paper": papers[0] if papers else None,
                "more": len(papers) - 1,
                "more_tip": ", ".join(p["text"] for p in papers[1:]),
                "ideas": [
                    {"number": c["number"], "href": f"#{c['id']}"}
                    for c in (ideas["cards"] if ideas else [])
                    if m.id in c["entries"]
                ],
            }
        )
    return {
        "rows": rows,
        "has_code": any(r["code"] for r in rows),
        "has_paper": any(r["paper"] for r in rows),
        "has_ideas": ideas is not None and any(r["ideas"] for r in rows),
        "no_code": NO_CODE,
        "no_paper": NO_PAPER,
        "no_idea": NO_IDEA,
    }


def sources(model: Model) -> tuple[list[dict], list[dict]]:
    """An entry's code files (name and path, each once) and its paper
    identifiers (arXiv, doi or an address), from its Sources rows."""
    section = model.section("Sources")
    code, papers, seen = [], [], set()
    for label, values in ordered_rows(section.body if section else ()):
        for value in map(written, values):
            if label == "Code":
                for match in CODE_PATH.finditer(value):
                    if match.group(1) not in seen:
                        seen.add(match.group(1))
                        code.append({"name": match.group(1).rsplit("/", 1)[-1], "path": match.group(1)})
            elif label == "Paper":
                ids = [f"{m.group(1)}:{m.group(2)}" for m in IDENT.finditer(value)]
                papers += [{"text": i, "html": Markup(markup.inline_html(link_idents(i)))} for i in ids]
                if not ids:
                    papers += [_url_paper(m.group(1) or m.group(2)) for m in PAPER_URL.finditer(value)]
    return code, papers


# ===============================
# Helper functions
# ===============================


def _url_paper(url: str) -> dict:
    # a paper known by its address: the host as its text, the address on hover
    host = urlparse(url).netloc.removeprefix("www.") or url
    return {"text": url, "html": Markup(f'<a href="{escape(url)}" title="{escape(url)}">{escape(host)}</a>')}


def _chip_line(label: str, keys: list[str], entries: dict[str, Model]) -> dict:
    refs = sorted((entry_ref(k, entries) for k in keys), key=lambda e: e["title"].casefold())
    rest = refs[CHIPS_SHOWN:]
    return {
        "label": label,
        "entries": refs[:CHIPS_SHOWN],
        "more": len(rest),
        "more_tip": ", ".join(e["title"] for e in rest),
    }


def _tickets(value) -> list[int]:
    # issue numbers from a `tests` list or a `ticket`: 7, "7" or "#7"
    items = value if isinstance(value, list) else [value] if value not in (None, "") else []
    found = []
    for item in items:
        match = TICKET.fullmatch(str(item).strip())
        if match:
            found.append(int(match.group(1)))
    return found


def _section(key: str, data, empty: str) -> dict:
    title, color, _ = SECTIONS[key]
    return {"key": key, "title": title, "color": color, "data": data, "empty": empty}


def _count(n: int, one: str, many: str, words: bool = False) -> str:
    # with words, a count under ten is spelled out, as a sentence reads it
    shown = NUMBER_WORDS[n] if words and n in NUMBER_WORDS else str(n)
    return f"{shown} {one if n == 1 else many}"


def _and(names: list[str]) -> str:
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def _or(names: list[str]) -> str:
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} or {names[-1]}"
