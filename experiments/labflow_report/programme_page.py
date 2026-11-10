"""The research programme page on the new builder (ADR-0005).

`programme_page` turns one `programmes/<slug>.md` into the values
`templates/programme.html` draws: the kicker and short title; the meta line
(the open or closed badge, how many experiments, how many planned, the last
result); the goal line (the Question's first sentence and `why:`); the
sticky section bar; then the map with the experiments table under it, the
threads matrix, the beliefs folded to their claims, the open questions and
the Needs table. Only the goal, the beliefs, the thread rows and the open
questions come from the programme. The map cards and edges, the thread
cells, the experiments table and the Needs table are drawn from what the
member records and the `planned` items already store (tables.md, Programme
tables). A part with nothing to show says so in words, so a programme
planned before its first record, or one whose members have not concluded,
draws its empty states.

The map is laid out here, not in the browser: columns by `builds_on` depth,
one line into each card from its main parent (the earliest card it builds
on), drawn in one SVG under the cards with its label in the column gap
before the child. A card's other parents are chips on it, and their links
are drawn only while the card is hovered or tapped. A map wider than its
column gets an overview in its head row, drawn here from the same layout;
the page script shows it, moves its frame and jumps to a card from its
experiments-table number. On a phone the map is an outline: each card under
its main parent.
"""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass, field
from pathlib import Path

from markupsafe import Markup

# record imports builder, which imports this module: read record's names at call time
from . import markup
from . import record as records
from .builder import ENVIRONMENT, OUTCOME, Model, PageView, Site, view
from .checks import programme as programme_checks
from .checks.figures import FIGURE_LINK
from .checks.record import RECORD_CITATION, RECORD_DECISION, unwritten

# ===============================
# Defaults
# ===============================

TEMPLATE = "programme.html"
# a programme's status → the badge it draws
STATUS_BADGE = {"open": "running", "closed": "draft"}
# a section's anchor → its heading and its name in the section bar
SECTIONS = {
    "map": ("Why each experiment exists", "Map"),
    "threads": ("Where each experiment fits", "Threads"),
    "believe": ("What we now believe", "Believe"),
    "open": ("Open questions", "Open"),
    "needs": ("Needs", "Needs"),
}
BELIEVE = "What we now believe"
THREADS = "Threads"
OPEN = "Open"
CONCLUDED = "concluded"
PLANNED = "planned"
# a matrix with more experiment columns than this heads each by its number alone
NAMED_COLUMNS = 6
# a run of this many clauses in one cell, one outcome and consecutive numbers, shows as a range chip
RANGE_CLAUSES = 3
# an edge label is cut to this many words; the whole bullet shows on hover
LABEL_WORDS = 10
CAVEAT = "Caveat"
# a belief figure is drawn this wide, or at its own width when wider, up to the column
BELIEF_FIGURE_PX = 480
SENTENCE_STOPS = (".", "!", "?")
NO_GOAL = "The programme has no question yet."
NO_MAP = "No experiment yet: the programme names no member and no planned experiment."
NO_THREADS = "No thread yet. Each line of inquiry joins as a Threads row once the programme names it."
NO_BELIEF = "Nothing yet. The programme record’s summary is written once its first experiment concludes."
NO_OPEN = (
    "No open question is recorded yet. A question joins this list when a finding raises it, "
    "with the planned experiment that takes it on."
)
NO_OPEN_CLOSED = "Nothing is open: the programme is closed."
NO_NEEDS = (
    "No needs recorded yet. Each planned record’s Design names the people, compute, "
    "data access and external services it needs."
)
NO_CLAUSE = "no clause settled yet"
FINDINGS_ONLY = "findings only, no clause"
NOT_STARTED = "not started"
NO_TICKET = "no ticket"
NOT_CONCLUDED = "not concluded"
MAP_LEDE = (
    "Each card draws one line, from the earliest experiment it builds on, and the line carries the result "
    "that made the card necessary. Its other parents show as chips; hover or tap a card to draw those links. "
    "A dashed amber line into planned work carries the open question it answers."
)
TREE_LEDE = (
    "Each card sits under the earliest experiment it builds on, with the reason that experiment gave. "
    "Chips name its other parents; tap a card to see those links."
)
SKELETON_MAP_LEDE = (
    "No experiment has concluded yet. Every card is planned; a dashed line joins each to the first "
    "experiment it builds on, and its other parents show as chips. When an experiment concludes, its "
    "card and line turn solid and the line names the result that made the next one necessary."
)
THREADS_LEDE = "Threads down, experiments across in date order. A cell holds the clauses an experiment settled for that thread."
THREADS_PLANNED = "A dashed ? marks a thread a planned experiment will settle."
THREADS_NOW = "Now follows one rule: every clause one outcome gives that outcome, a mix gives partial."
SKELETON_THREADS_LEDE = (
    "Threads down, experiments across in date order. A concluded experiment joins as a column "
    "with the clauses it settled for each thread."
)
BELIEVE_LEDE = (
    "The programme’s own summary, folded to its claims. Each opens to its reasoning, with a "
    "chip where a finding is cited: 2·F1 is finding F1 of experiment 2."
)
NEEDS_NOTE = "Amber: hosted, quota-bound, restricted or manual, or any external dependency. The full row is on hover."
# a belief or thread bullet: its bold lead, then the rest
LEAD = re.compile(r"^- \*\*(.+?)\*\*:?\s*(.*)$", re.DOTALL)
# a Builds on bullet: the parent's slug, an optional (uses), then its text
BUILDS_ON = re.compile(r"^`?([a-z0-9][a-z0-9-]*)`?(?:\s*\(uses\))?\s*([:—–])\s*(.*)$", re.DOTALL)
# the parent entry a Builds on bullet cites: `[<slug>:F<n>]`, `[<slug> D<n>]` or "its finding F<n>"
CITED = re.compile(r"\[([a-z0-9][a-z0-9-]*)[: ]([FD]\d+)\]|\bits (?:finding|decision) ([FD]\d+)\b")
OWN_CITATION = re.compile(r"\(\s*its (?:finding|decision) [FD]\d+\s*\)")
# the short form after the slug's dash: `— <label>: <text>`
SHORT_LABEL = re.compile(r"^([^:]+?):\s+(.*)$", re.DOTALL)
# parentheses that hold only citations: dropped once every citation in them is a chip
CITATIONS_ONLY = re.compile(r"\(\s*((?:\[[a-z0-9][a-z0-9-]*:(?:F\d+|[HCQ]\d+)\](?:\s*[,;]\s*)?)+)\s*\)")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
TRAILING = re.compile(r"[\s—–,;:-]+$")
NOTE_AFTER = re.compile(r"^not yet planned:?\s*(.*?)\.?$")
INLINE = ENVIRONMENT.get_template("components/inline.html").module
WORDS = ENVIRONMENT.get_template("components/words.html").module

# the map's geometry in CSS pixels
NODE_W = 190
NODE_H = 96
# a planned card is compact, centred in its row
PLANNED_H = 60
# a card's text, in lines of at most so many characters: its name beside the mark, its
# question, its result and its status line each wrap to at most two lines
CARD_PAD = 24
CARD_GAP = 4
NAME_CHARS = 18
NAME_LINE = 19
TEXT_CHARS = 26
TEXT_LINE = 17
CARD_LINES = 2
# a card's "also builds on" chips: the first row with its label, then each further row
ALSO_H = 42
CHIP_ROW_H = 21
# a chip's width: its characters, its padding and the gap after it
CHIP_CHAR_W = 7
CHIP_PAD_W = 16
CHIP_ROOM = NODE_W - 24
COL_GAP = 180
ROW_GAP = 72
PAD_X = 16
PAD_Y = 16
# a parent's trunk runs this far into the gap after it; trunks that share rows step apart
TRUNK = 14
STAGGER = 8
ARROW = 7
# an edge label: three lines of 16px text, its bottom just above its line, clear of the trunks
LABEL_H = 48
LABEL_LIFT = 6
LABEL_INSET = 12
LABEL_W = COL_GAP - TRUNK - LABEL_INSET - 8
# a hover link: its lanes and climbs step apart, its ends spread along a card side
LINK_STEP = 6
LANE_MARGIN = 10
PORT_INSET = 22
PORT_STEP = 24
# a Builds on label is at most this many words before its colon
SHORT_LABEL_WORDS = 6
# the overview: the whole map in 200×90 px, corners and frame inset in those pixels
OVERVIEW_W = 200
OVERVIEW_H = 90
OVERVIEW_CARD_CORNER = 1.5
OVERVIEW_FRAME_CORNER = 3
OVERVIEW_FRAME_INSET = 1

# ===============================
# Programme classes
# ===============================


@dataclass(frozen=True)
class Member:
    """One member record as the page shows it: its number in conclusion order and its parts."""

    model: Model
    slug: str
    number: int
    word: str
    short: str
    title: str
    concluded: datetime.date | None
    clauses: tuple = ()
    findings: tuple = ()
    readings: dict = field(default_factory=dict)

    def tip(self, code: str) -> str:
        """The hover card of one of the record's findings or clauses, "" for an unknown code."""
        for f in self.findings:
            if f.id == code:
                return _tip(f"{f.id} — {f.title}", f.text("Summary"), self.short)
        for c in self.clauses:
            if c.id == code:
                return _tip(f"{c.id} — {c.claim}", c.conditions, self.short)
        return ""


@dataclass(frozen=True)
class Planned:
    """One `planned` item: an experiment the programme names but has not started."""

    slug: str
    short: str
    title: str
    category: str
    ticket: str
    builds_on: tuple[str, ...]
    threads: tuple[str, ...]
    number: int = 0


# ===============================
# Main function
# ===============================


@view("programme")
def programme_page(model: Model, site: Site) -> PageView:
    """Builds the programme page's values from one programme and the site around it.

    Args:
        model: The programme.
        site: Every document: the member records, the records they build on,
            the other programmes.

    Returns:
        The page view `templates/programme.html` renders.
    """
    slug = str(model.value("slug") or model.document.path.stem)
    pid = f"programme-{slug}"
    title = str(model.value("title") or slug)
    short = str(model.value("short") or title)
    status = model.value("status") or "open"
    members = read_members(model, site)
    planned = read_planned(model, {m.slug for m in members})
    concluded = [m for m in members if m.concluded]

    # the goal: the Question's first sentence, then why the project needs it
    question = model.section("Question")
    first = markup.first_sentence(markup.unguided(question.text)) if question else ""
    why = model.value("why") or ""
    goal = cite_html(" ".join(p for p in (first, why) if p), members) if first else ""

    # the sections, in page order
    opened = read_open(model, site, planned)
    threads = read_threads(model, members, planned)
    beliefs = read_beliefs(model, site, members, pid)
    needs = read_needs(members, site)
    the_map = read_map(model, site, members, planned, opened["answers"])
    sections = [
        _section("map", the_map, NO_MAP, MAP_LEDE if concluded else SKELETON_MAP_LEDE, len(members) + len(planned), TREE_LEDE if concluded else ""),
        _section("threads", threads, NO_THREADS, threads_lede(threads, members), len(threads["rows"]) if threads else 0),
        _section("believe", beliefs, NO_BELIEF, BELIEVE_LEDE if beliefs else "", len(beliefs)),
        _section("open", opened["rows"], NO_OPEN_CLOSED if status == "closed" else NO_OPEN, opened["lede"], len(opened["rows"])),
        _section("needs", needs, NO_NEEDS, "", len(needs["rows"]) if needs else 0),
    ]
    bar = [{"href": f"#{pid}/{s['key']}", "name": SECTIONS[s["key"]][1], "n": s["n"]} for s in sections]

    return PageView(
        kind="programme",
        id=pid,
        title=short,
        template=TEMPLATE,
        context={
            "kicker": model.document.path.relative_to(site.corpus.experiments.parent).as_posix(),
            "full_title": title,
            "badge": STATUS_BADGE.get(status, "draft"),
            "status": status,
            "meta": meta_line(members, planned),
            "goal": goal,
            "no_goal": NO_GOAL,
            "bar": bar,
            "sections": sections,
            "needs_note": NEEDS_NOTE,
        },
    )


# ===============================
# Members and planned experiments
# ===============================


def read_members(model: Model, site: Site) -> list[Member]:
    """The member records, numbered in conclusion order, unconcluded ones last in `members` order."""
    by_path = {m.document.path.resolve(): m for m in site.of_kind("record")}
    found = []
    for slug, document in programme_checks.members(model.document, site.corpus):
        record = by_path.get(document.path.resolve())
        if record is not None and all(m.id != record.id for _, m in found):
            found.append((slug, record))

    # conclusion order: by date, the unconcluded last in the order `members` lists them
    def conclusion_order(item: tuple[int, tuple[str, Model]]) -> tuple:
        index, (_, record) = item
        when = record.value(CONCLUDED)
        return (when is None, when or datetime.date.min, index)

    ordered = sorted(enumerate(found), key=conclusion_order)
    members = []
    for number, (_, (slug, record)) in enumerate(ordered, 1):
        title = str(record.value("title") or slug)
        members.append(
            Member(
                model=record,
                slug=slug,
                number=number,
                word=records.record_word(record),
                short=str(record.value("short") or title),
                title=title,
                concluded=record.value(CONCLUDED),
                clauses=tuple(records.read_clauses(record)),
                findings=tuple(records.read_findings(record)),
                readings=records.read_readings(record),
            )
        )
    return members


def read_planned(model: Model, member_slugs: set[str]) -> list[Planned]:
    """The `planned` items that are maps with a slug, each once and never a member's slug,
    numbered on from the last member in `planned` order."""
    found, seen = [], set(member_slugs)
    for item in model.value(PLANNED) or []:
        if not isinstance(item, dict) or not item.get("slug") or str(item["slug"]) in seen:
            continue
        slug = str(item["slug"])
        seen.add(slug)
        found.append(
            Planned(
                slug=slug,
                short=str(item.get("short") or slug),
                title=str(item.get("title") or ""),
                category=str(item.get("category") or ""),
                ticket=str(item.get("ticket") or "").strip(),
                builds_on=tuple(str(s) for s in _listed(item.get("builds_on"))),
                threads=tuple(str(s) for s in _listed(item.get("threads"))),
                number=len(member_slugs) + len(found) + 1,
            )
        )
    return found


def meta_line(members: list[Member], planned: list[Planned]) -> list[str]:
    """The meta line after the badge: experiments, planned ones, the last result."""
    dates = [m.concluded for m in members if m.concluded]
    return [
        counted(len(members), "experiment", "experiments"),
        counted(len(planned), "planned", "planned"),
        f"last result {month_day(max(dates))}" if dates else "no result yet",
    ]


# ===============================
# Map
# ===============================


def read_map(
    model: Model, site: Site, members: list[Member], planned: list[Planned], answers: dict[str, str]
) -> dict | None:
    """The map's cards and lines laid out, its phone outline, and the experiments table under it; None with neither."""
    if not members and not planned:
        return None
    edges: list[dict] = []
    keys: dict[str, dict] = {}
    concluded: dict[str, datetime.date | None] = {}
    others = other_programmes(model, site)
    records_by_slug = {records.record_slug(r): r for r in site.of_kind("record")}

    # a card per member, per planned experiment and per parent from outside the programme
    for m in members:
        line = f"{month_day(m.concluded)} · {m.word}" if m.concluded else (m.model.stage or "draft")
        asks = record_question(m.model)
        result = _result(m.model, m.findings) if m.concluded else ""
        keys[m.slug] = _node(m.slug, "member", m.word, m.number, m.short, m.title, f"#{m.slug}", line, asks=asks, result=result)
        concluded[m.slug] = m.concluded
    for p in planned:
        ticket = ticket_chip(p.ticket, site.github)
        asks = markup.capitalize(p.title)
        keys[p.slug] = _node(p.slug, PLANNED, PLANNED, p.number, p.short, p.title, "", "", ticket, asks)
        concluded[p.slug] = None

    def outside(record: Model) -> str:
        key = records.record_slug(record)
        if key not in keys:
            when = record.value(CONCLUDED)
            where = others.get(record.document.folder.resolve(), "no programme")
            line = f"in {where}" + (f" · {month_day(when)}" if when else "")
            short = str(record.value("short") or record.value("title") or key)
            number = len(members) + len(planned) + sum(n["kind"] == "other" for n in keys.values()) + 1
            asks = record_question(record)
            result = _result(record, records.read_findings(record)) if when else ""
            keys[key] = _node(key, "other", records.record_word(record), number, short, str(record.value("title") or key), f"#{key}", line, asks=asks, result=result)
            concluded[key] = when
        return key

    # member edges, earliest parent first, carry the reason the child's Builds on bullet gives
    for m in members:
        bullets = builds_on_bullets(m.model)
        named = list(bullets)
        found = []
        for index, parent in enumerate(records.parents(m.model, site)):
            source = records.record_slug(parent)
            source = source if source in keys and keys[source]["kind"] == "member" else outside(parent)
            first = named.index(source) if source in named else len(named)
            found.append((_earliest(concluded[source], first, index), source))
        for _, source in sorted(found):
            label, tip, chip, code = edge_label(bullets.get(source, ("", "")), source)
            also = also_chip(keys[source], code)
            edges.append({"src": source, "dst": m.slug, "planned": False, "label": label, "tip": tip, "chip": chip, "also": also})

    # planned edges, earliest parent first: the line into a planned card carries the open question it answers
    for p in planned:
        answer = answers.get(p.slug, "")
        label = f"answers: {_cut(answer)}" if answer else ""
        found = []
        for index, source in enumerate(p.builds_on):
            if source not in keys and source in records_by_slug:
                source = outside(records_by_slug[source])
            if source in keys:
                found.append((_earliest(concluded[source], index, index), source))
        for _, source in sorted(found):
            also = also_chip(keys[source], "")
            edges.append({"src": source, "dst": p.slug, "planned": True, "label": label, "tip": answer, "chip": "", "also": also})

    laid = place(list(keys.values()), edges)
    return {**laid, "overview": overview(laid["width"], laid["height"]), "table": read_experiments(members, planned, site)}


def other_programmes(model: Model, site: Site) -> dict:
    """Each record folder of another programme's members → that programme's short name."""
    found = {}
    for g in records.read_programmes(site.corpus.experiments):
        if g["slug"] != (model.value("slug") or model.document.path.stem):
            for folder in g["members"]:
                found.setdefault(folder, g["short"])
    return found


def builds_on_bullets(record: Model) -> dict[str, tuple[str, str]]:
    """The record's Builds on bullets by the parent slug each opens with: (short label, text).

    The label is the `— <label>:` short form after the slug, at most six
    words, else "".
    """
    found = {}
    for text in records.items(records.section_row(record, "Hypothesis", "Builds on")):
        if not (match := BUILDS_ON.match(text)):
            continue
        rest, label = match.group(3).strip(), ""
        short = SHORT_LABEL.match(rest) if match.group(2) == "—" else None
        if short and len(short.group(1).split()) <= SHORT_LABEL_WORDS:
            label, rest = short.group(1).strip(), short.group(2).strip()
        found[match.group(1)] = (label, rest)
    return found


def edge_label(bullet: tuple[str, str], parent: str) -> tuple[str, str, Markup, str]:
    """An edge's label, its hover text, the chip of the parent entry it cites, and that entry's code.

    The label is the bullet's short label, else its last sentence cut. The
    cited entry is a finding or decision of the parent the bullet names, as
    `[<parent>:F<n>]`, `[<parent> D<n>]` or "its finding F<n>": the first in
    its last sentence, else its first.
    """
    label, text = bullet
    if not text and not label:
        return "", "", Markup(""), ""
    # the entry cited in the sentence the label shows, else the bullet's first
    codes = [(m.start(), m.group(3) or m.group(2)) for m in CITED.finditer(text) if m.group(3) or m.group(1) == parent]
    tail = text.rindex(SENTENCE_END.split(text.rstrip())[-1]) if text.strip() else 0
    code = next((c for at, c in codes if at >= tail), codes[0][1] if codes else "")
    bare = OWN_CITATION.sub("", RECORD_DECISION.sub("", RECORD_CITATION.sub("", text)))
    plain = markup.plain_text(markup.inline_html(bare))
    plain = re.sub(r"\s+([.,;:])", r"\1", re.sub(r"\(\s*\)", "", plain)).strip()
    last = TRAILING.sub("", SENTENCE_END.split(plain)[-1]).rstrip(".")
    chip = INLINE.ct(code[0].lower(), code, f"#{parent}/{code.lower()}") if code else Markup("")
    return label or _cut(last), plain, chip, code


def also_chip(parent: dict, code: str) -> Markup:
    """A card's chip for one of its other parents: the parent's number and the entry it cites, the parent named on hover."""
    name = str(parent["number"]) if parent["number"] else parent["name"]
    text = f"{name}·{code}" if code else name
    title = f"{parent['number']} {parent['name']}" if parent["number"] else parent["name"]
    kind = code[0].lower() if code else "i"
    return Markup('<span class="ct {}" title="{}">{}</span>').format(kind, title, text)


def place(nodes: list[dict], edges: list[dict]) -> dict:
    """Lays the map out as the ProgrammeMapMain board does: one line into each card, from its main parent.

    A card's column is one past its deepest parent. Its main parent is the
    first of its edges, in the order given, from an earlier column; the main
    lines form a forest, laid out depth first so each card sits in the
    nearest row its line reaches without crossing a card or another line.
    A main line leaves its parent's right side, runs along the parent's
    trunk in the gap after it, and enters the child's left side; its label
    sits just above it in the gap before the child. Every other edge is a
    hover link: drawn only while one of its cards is hot, through the row
    and column gaps, where no card sits.

    Args:
        nodes: The cards, each with its `key` and `kind`, in the order they are numbered.
        edges: Each edge's `src` and `dst` keys, whether it is `planned`, its
            `label`, `tip`, `chip` and `also` chip; a card's edges come
            earliest parent first.

    Returns:
        The canvas `width` and `height`; the `nodes` with `x`, `y`, `w`, `h`,
        their `also` chips and the keys of the cards they link to; the main
        `edges` each with its SVG `path`, `arrow` and label box; the hover
        `links` each with its `path` and `arrow`; each column's left as
        `columns`; and the phone outline as `tree` rows.
    """
    keys = [n["key"] for n in nodes]
    order = {k: i for i, k in enumerate(keys)}
    parents = {k: [e["src"] for e in edges if e["dst"] == k and e["src"] in order] for k in keys}

    # the column: one past the deepest parent; a cycle is cut where it closes
    column: dict[str, int] = {}

    def depth(key: str, seen: tuple = ()) -> int:
        if key not in column:
            column[key] = 1 + max((depth(p, (*seen, key)) for p in parents[key] if p not in seen and p != key), default=-1)
        return column[key]

    for key in keys:
        depth(key)

    # the main line into each card: its first edge from an earlier column; the rest are hover links
    drawn = [e for e in edges if e["src"] in column and e["dst"] in column and column[e["src"]] < column[e["dst"]]]
    main: dict[str, dict] = {}
    for e in drawn:
        main.setdefault(e["dst"], e)
    links = [e for e in drawn if main[e["dst"]] is not e]
    children: dict[str, list[str]] = {k: [] for k in keys}
    for e in main.values():
        children[e["src"]].append(e["dst"])

    # each card's chips, its height and its row; then the trunks, which may widen a column gap
    for n in nodes:
        n["also"] = [e.get("also", "") for e in links if e["dst"] == n["key"]]
        n["h"] = _card_height(n)
    row, trunks = _rows(keys, column, main, children)
    slot, widest = _slots(trunks)
    rows = max(row.values()) + 1
    columns = max(column.values()) + 1
    gap_w = [COL_GAP + STAGGER * widest.get(g, 0) for g in range(columns - 1)]
    col_x = [PAD_X + c * NODE_W + sum(gap_w[:c]) for c in range(columns)]
    row_h = [max(n["h"] for n in nodes if row[n["key"]] == r) for r in range(rows)]
    below = any(row[e["src"]] == row[e["dst"]] == rows - 1 for e in links)
    # the top gap holds the labels of the first row's lines, however short its cards
    top = max(PAD_Y, PAD_Y + LABEL_H + LABEL_LIFT - row_h[0] / 2)
    gap_h = [top] + [ROW_GAP] * (rows - 1) + [ROW_GAP if below else PAD_Y]
    row_y = [sum(gap_h[: r + 1]) + sum(row_h[:r]) for r in range(rows)]

    # the cards' boxes: a member fills its row, a planned card is compact and centred in it
    for n in nodes:
        r = row[n["key"]]
        h = n["h"] if n.get("kind") == PLANNED else row_h[r]
        n.update(x=col_x[column[n["key"]]], y=row_y[r] + (row_h[r] - h) / 2, w=NODE_W, h=h)
    box = {n["key"]: n for n in nodes}

    # the main lines: out of the parent's right side, along its trunk, into the child's left side
    for e in main.values():
        parent, child = box[e["src"]], box[e["dst"]]
        x1, y1 = parent["x"] + NODE_W, parent["y"] + parent["h"] / 2
        x2, y2 = child["x"], child["y"] + child["h"] / 2
        trunk = x1 + TRUNK + STAGGER * slot[e["src"]]
        e["path"] = f"M{_px(x1)} {_px(y1)}H{_px(x2)}" if y1 == y2 else f"M{_px(x1)} {_px(y1)}H{_px(trunk)}V{_px(y2)}H{_px(x2)}"
        e["arrow"] = _arrow(x2, y2, "left")
        gap = column[e["dst"]] - 1
        left = x2 - gap_w[gap] + TRUNK + STAGGER * widest.get(gap, 0) + LABEL_INSET
        e.update(lx=_px(left), ly=_px(y2 - LABEL_LIFT - LABEL_H), lw=LABEL_W)

    # the hover links, and each card's linked cards
    gaps = [(row_y[g - 1] + row_h[g - 1] if g else 0, row_y[g] if g < rows else row_y[-1] + row_h[-1] + gap_h[-1]) for g in range(rows + 1)]
    _route_links(links, box, row, column, gaps, [x - g for x, g in zip(col_x[1:], gap_w)])
    for n in nodes:
        n["links"] = " ".join(dict.fromkeys(e["src"] if e["dst"] == n["key"] else e["dst"] for e in drawn if n["key"] in (e["src"], e["dst"])))

    width = 2 * PAD_X + columns * NODE_W + sum(gap_w)
    height = gaps[-1][1]
    return {
        "width": width,
        "height": height,
        "nodes": nodes,
        "edges": list(main.values()),
        "links": links,
        "columns": col_x,
        "tree": _tree(keys, box, main, children),
    }


def overview(width: float, height: float) -> dict:
    """The overview's corners and frame in map units, for a map drawn into 200×90 px.

    The overview stretches the map to fit, so a pixel is a different number
    of map units across and down; each corner radius and the frame's inset
    are given both ways.

    Args:
        width: The map canvas's width.
        height: The map canvas's height.

    Returns:
        The cards' `rx` and `ry`, the frame's `frame_rx` and `frame_ry`,
        and the frame's top `frame_y` and height `frame_h`.
    """
    across, down = width / OVERVIEW_W, height / OVERVIEW_H
    return {
        "rx": _px(round(OVERVIEW_CARD_CORNER * across, 1)),
        "ry": _px(round(OVERVIEW_CARD_CORNER * down, 1)),
        "frame_rx": _px(round(OVERVIEW_FRAME_CORNER * across, 1)),
        "frame_ry": _px(round(OVERVIEW_FRAME_CORNER * down, 1)),
        "frame_y": _px(round(OVERVIEW_FRAME_INSET * down, 1)),
        "frame_h": _px(round(height - 2 * OVERVIEW_FRAME_INSET * down, 1)),
    }


def _rows(keys: list[str], column: dict[str, int], main: dict[str, dict], children: dict[str, list[str]]) -> tuple[dict, dict]:
    # depth first over the main-line forest, the larger subtree first so long chains run straight;
    # each card takes the row nearest its parent where its line crosses no card, with the fewest
    # line crossings. Returns each card's row and each gap's trunks (gap g sits after column g).
    order = {k: i for i, k in enumerate(keys)}
    size: dict[str, int] = {}

    def count(key: str) -> int:
        size[key] = 1 + sum(count(c) for c in children[key])
        return size[key]

    roots = [k for k in keys if k not in main]
    for key in roots:
        count(key)
    row: dict[str, int] = {}
    taken: set[tuple[int, int]] = set()
    trunks: dict[int, dict[str, tuple[int, int]]] = {}
    passing: dict[int, set[int]] = {}

    def crossings(key: str, r: int) -> int:
        c, n = column[key], 0
        if key in main:
            p = main[key]["src"]
            cp, rp = column[p], row[p]
            n += sum(a <= r <= b for g in range(cp + 1, c) for a, b in trunks.get(g, {}).values())
            lo, hi = trunks.get(cp, {}).get(p, (rp, rp))
            lo, hi = min(lo, r), max(hi, r)
            n += sum(a <= hi and lo <= b for q, (a, b) in trunks.get(cp, {}).items() if q != p)
            n += sum(lo <= s <= hi for s in passing.get(cp, ()))
        if children[key]:
            n += sum(a <= r <= b for a, b in trunks.get(c, {}).values())
        return n

    def settle(key: str) -> None:
        c = column[key]
        p = main[key]["src"] if key in main else None
        start, between = (row[p], range(column[p] + 1, c)) if p else (0, range(0))
        fresh = max((r for _, r in taken), default=-1) + 1
        free = [r for r in range(start, fresh + 1) if (c, r) not in taken and all((x, r) not in taken for x in between)]
        r = min(free, key=lambda r: (crossings(key, r), r))
        row[key] = r
        taken.add((c, r))
        if p:
            cp = column[p]
            taken.update((x, r) for x in between)
            for g in range(cp + 1, c):
                passing.setdefault(g, set()).add(r)
            lo, hi = trunks.setdefault(cp, {}).get(p, (row[p], row[p]))
            trunks[cp][p] = (min(lo, r), max(hi, r))
        if children[key]:
            trunks.setdefault(c, {}).setdefault(key, (r, r))
        for child in sorted(children[key], key=lambda k: (-size[k], order[k])):
            settle(child)

    for key in roots:
        settle(key)
    return row, trunks


def _slots(trunks: dict[int, dict[str, tuple[int, int]]]) -> tuple[dict[str, int], dict[int, int]]:
    # each parent's trunk slot in its gap, trunks sharing a row a slot apart; and each gap's last slot
    slot: dict[str, int] = {}
    widest: dict[int, int] = {}
    for gap, spans in trunks.items():
        ends: list[int] = []
        for key, (lo, hi) in sorted(spans.items(), key=lambda item: item[1]):
            k = next((i for i, end in enumerate(ends) if end < lo), len(ends))
            ends[k:k + 1] = [hi]
            slot[key] = k
        widest[gap] = len(ends) - 1
    return slot, widest


def _route_links(links: list[dict], box: dict[str, dict], row: dict[str, int], column: dict[str, int], gaps: list, gap_left: list) -> None:
    # each hover link leaves its parent's bottom (or top, for a child above), runs along the row gap
    # beside it, climbs in the column gap before the child to the row gap beside the child, and enters
    # the child's top (or bottom). Links into one card nest: the farther one keeps to the outside.
    for e in links:
        rp, rc = row[e["src"]], row[e["dst"]]
        if rc > rp:
            e.update(out="bottom", into="top", gaps=(rp + 1, rc))
        else:
            e.update(out="top" if rc < rp else "bottom", into="bottom", gaps=(rp if rc < rp else rp + 1, rc + 1))
        # nesting rank: the more rows away, then the fewer columns away, the farther out; in one row,
        # the more columns away
        cols = column[e["dst"]] - column[e["src"]]
        e["far"] = (abs(rc - rp), -cols) if rc != rp else (0, cols)

    # climbs: in each column gap, the farther link into a card climbs further right
    climbs: dict[int, list[dict]] = {}
    for e in links:
        if e["gaps"][0] != e["gaps"][1]:
            climbs.setdefault(column[e["dst"]] - 1, []).append(e)
    climb_x: dict[int, float] = {}
    for gap, group in climbs.items():
        group.sort(key=lambda e: (row[e["dst"]], e["into"], e["far"]))
        room = COL_GAP - 2 * LANE_MARGIN
        step = LINK_STEP if len(group) < 2 else min(LINK_STEP, room / (len(group) - 1))
        for k, e in enumerate(group):
            climb_x[id(e)] = gap_left[gap] + LANE_MARGIN + k * step

    # ports: links into a side spread over its left half, inner first; links out of it over its
    # right half, inner first from the right
    def spread(group: list[dict], b: dict, inward: bool) -> dict[int, float]:
        room = b["w"] / 2 - PORT_INSET - 4
        step = PORT_STEP if len(group) < 2 else min(PORT_STEP, room / (len(group) - 1))
        if inward:
            return {id(e): b["x"] + PORT_INSET + k * step for k, e in enumerate(group)}
        return {id(e): b["x"] + b["w"] - PORT_INSET - k * step for k, e in enumerate(group)}

    # lanes: in each row gap, a link's run lies above or below another's so that neither crosses
    # the other's vertical ends, wherever both can show at once (they share a card). The ports follow
    # the lanes, so the stacking runs again: first with the ports by how far each link reaches
    ports: dict[tuple[int, str], float] = {}
    lane_y: dict[tuple[int, int], float] = {}

    def lane(e: dict, end: int, side: str) -> float:
        # a link's lane beside a card side, ranked from the card outward
        y = lane_y[(e["gaps"][end], id(e))]
        return -y if side == "top" else y

    for _ in range(3):
        for key, b in box.items():
            for side in ("top", "bottom"):
                into = [e for e in links if e["dst"] == key and e["into"] == side]
                into.sort(key=(lambda e: lane(e, 1, side)) if lane_y else (lambda e: e["far"]))
                ports.update({(i, "in"): x for i, x in spread(into, b, True).items()})
        for key, b in box.items():
            for side in ("top", "bottom"):
                out = [e for e in links if e["src"] == key and e["out"] == side]
                out.sort(key=(lambda e: lane(e, 0, side)) if lane_y else (lambda e: -climb_x.get(id(e), ports[(id(e), "in")])))
                ports.update({(i, "out"): x for i, x in spread(out, b, False).items()})
        runs = {id(e): _runs(e, ports[(id(e), "out")], climb_x.get(id(e), 0.0), ports[(id(e), "in")]) for e in links}
        lanes: dict[int, list[dict]] = {}
        for e in links:
            for gap in runs[id(e)]:
                lanes.setdefault(gap, []).append(e)
        for gap, group in lanes.items():
            stacked = _stack(group, {id(e): runs[id(e)][gap] for e in group})
            top, bottom = gaps[gap]
            step = LINK_STEP if len(stacked) < 2 else min(LINK_STEP, (bottom - top - 2 * LANE_MARGIN) / (len(stacked) - 1))
            first = (top + bottom) / 2 - step * (len(stacked) - 1) / 2
            for k, e in enumerate(stacked):
                lane_y[(gap, id(e))] = first + k * step

    # the paths
    for e in links:
        parent, child = box[e["src"]], box[e["dst"]]
        x1, x2 = ports[(id(e), "out")], ports[(id(e), "in")]
        y1 = parent["y"] + (parent["h"] if e["out"] == "bottom" else 0)
        y2 = child["y"] + (child["h"] if e["into"] == "bottom" else 0)
        first = lane_y[(e["gaps"][0], id(e))]
        if e["gaps"][0] == e["gaps"][1]:
            e["path"] = f"M{_px(x1)} {_px(y1)}V{_px(first)}H{_px(x2)}V{_px(y2)}"
        else:
            last, climb = lane_y[(e["gaps"][1], id(e))], climb_x[id(e)]
            e["path"] = f"M{_px(x1)} {_px(y1)}V{_px(first)}H{_px(climb)}V{_px(last)}H{_px(x2)}V{_px(y2)}"
        e["arrow"] = _arrow(x2, y2, e["into"])


def _runs(e: dict, out_x: float, climb: float, in_x: float) -> dict[int, tuple]:
    # a hover link's horizontal run in each row gap it uses, as its two vertical ends: (x, the way
    # the end goes, "up" or "down")
    start = "up" if e["out"] == "bottom" else "down"
    end = "down" if e["into"] == "top" else "up"
    first, last = e["gaps"]
    if first == last:
        return {first: ((out_x, start), (in_x, end))}
    down = last > first
    return {first: ((out_x, start), (climb, "down" if down else "up")), last: ((climb, "up" if down else "down"), (in_x, end))}


def _stack(group: list[dict], ends: dict[int, tuple]) -> list[dict]:
    # the runs of one row gap, top to bottom: a run lies above another whose span holds one of its
    # ends going up, and below one whose span holds one of its ends going down; only runs of links
    # sharing a card are ordered, and a cycle is broken at the run first in card order
    above: dict[int, set[int]] = {id(e): set() for e in group}
    for i, a in enumerate(group):
        for b in group[i + 1 :]:
            if not {a["src"], a["dst"]} & {b["src"], b["dst"]}:
                continue
            for one, other in ((a, b), (b, a)):
                lo, hi = sorted(x for x, _ in ends[id(other)])
                for x, way in ends[id(one)]:
                    if lo < x < hi:
                        upper, lower = (one, other) if way == "up" else (other, one)
                        above[id(lower)].add(id(upper))
    stacked: list[dict] = []
    left = list(group)
    while left:
        placed = {id(e) for e in stacked}
        ready = [e for e in left if above[id(e)] <= placed] or left[:1]
        stacked.append(ready[0])
        left.remove(ready[0])
    return stacked


def _tree(keys: list[str], box: dict[str, dict], main: dict[str, dict], children: dict[str, list[str]]) -> list[dict]:
    # the phone outline: depth first in card order, each card under its main parent with that line's
    # label; a guide per level is a rail (v), a branch with more below (t), the last branch (l) or blank
    order = {k: i for i, k in enumerate(keys)}
    found: list[dict] = []

    def walk(key: str, guides: list[bool]) -> None:
        marks = [("t" if more else "l") if level == len(guides) - 1 else ("v" if more else "") for level, more in enumerate(guides)]
        kids = sorted(children[key], key=order.get)
        found.append({"node": box[key], "guides": marks, "edge": main.get(key), "kids": bool(kids)})
        for k, child in enumerate(kids):
            walk(child, [*guides, k < len(kids) - 1])

    for key in keys:
        if key not in main:
            walk(key, [])
    return found


def read_experiments(members: list[Member], planned: list[Planned], site: Site) -> list[dict]:
    """The experiments table: members in number order, then the planned ones, each by its number and short name, each says what it lacks."""
    rows = []
    for m in members:
        rows.append(
            {
                "number": m.number,
                "name": m.short,
                "href": f"#{m.slug}",
                "title": m.title if m.title != m.short else "",
                "category": m.model.value("category") or INLINE.missing("no category"),
                "state": state_badge(m.word),
                "date": month_day(m.concluded) if m.concluded else INLINE.missing(NOT_CONCLUDED),
                "ticket": ticket_chip(m.model.value("ticket"), site.github),
                "planned": False,
            }
        )
    for p in planned:
        rows.append(
            {
                "number": p.number,
                "name": p.short,
                "href": "",
                "title": markup.capitalize(p.title),
                "category": p.category or INLINE.missing(NOT_STARTED),
                "state": state_badge(PLANNED),
                "date": "",
                "ticket": ticket_chip(p.ticket, site.github),
                "planned": True,
            }
        )
    return rows


def ticket_chip(ticket, github: str) -> Markup:
    """A ticket as an issue chip, linked to GitHub when the repo is known; "no ticket" in words."""
    text = str(ticket or "").strip()
    if not text:
        return INLINE.missing(NO_TICKET)
    if re.match(r"https?://", text):
        return INLINE.link("issue", text)
    number = text.lstrip("#")
    if not number.isdigit():
        return Markup.escape(text)
    return INLINE.iss(number, "", records.github_url(github, "issues", number) if github else "")


# ===============================
# Threads
# ===============================


def read_threads(model: Model, members: list[Member], planned: list[Planned]) -> dict | None:
    """The threads matrix: a row per Threads row, a column per member then per planned experiment, then Now."""
    rows = []
    for _, text in programme_checks.bullets(model.section(THREADS)):
        if not (match := LEAD.match(text)) or unwritten(text):
            continue
        name, body = match.group(1).rstrip(":").strip(), match.group(2)
        cited = RECORD_CITATION.findall(body)
        cells = [thread_cell(m, [code for slug, code in cited if slug == m.slug]) for m in members]
        words = [w for cell in cells for w in cell["words"]]
        has_findings = any(cell["findings"] for cell in cells)
        rows.append(
            {
                "name": Markup(markup.inline_html(name)),
                "plain": markup.plain_text(markup.inline_html(name)),
                "asks": Markup(markup.inline_html(TRAILING.sub("", RECORD_CITATION.sub("", body)))),
                "cells": [cell["html"] for cell in cells],
                "planned": [name.casefold() in {t.casefold() for t in p.threads} for p in planned],
                "now": now_mark(words, has_findings),
                "words": [w for cell in cells for w in cell["drawn"]],
                "findings_only": has_findings and not words,
            }
        )
    if not rows:
        return None
    # past NAMED_COLUMNS every head drops its name and keeps its number
    named = len(members) + len(planned) <= NAMED_COLUMNS
    columns = [
        {"head": f"{m.number} {m.short}" if named else str(m.number), "title": m.title, "sub": month_day(m.concluded) if m.concluded else m.word}
        for m in members
    ] + [{"head": f"{p.number} {p.short}" if named else str(p.number), "title": p.title or p.short, "sub": PLANNED} for p in planned]
    drawn = {w for row in rows for w in row["words"] + row["now"]["words"]}
    if any(any(r["planned"]) for r in rows):
        drawn.add(PLANNED)
    return {"columns": columns, "rows": rows, "key": sorted(drawn, key=list(OUTCOME).index)}


def thread_cell(member: Member, codes: list[str]) -> dict:
    """One member's cell in a thread row: its clause chips (a long run of one outcome as a range) and finding chips."""
    clauses = {c.id: c for c in member.clauses}
    findings = {f.id for f in member.findings}
    pieces: list[Markup] = []
    words: list[str] = []
    drawn: list[str] = []
    run: list = []

    def flush() -> None:
        if len(run) >= RANGE_CLAUSES:
            first, last = run[0][0], run[-1][0]
            pieces.append(WORDS.chip(f"{first.id}–{last.id}", run[0][1], False, f"#{member.slug}/{first.id.lower()}"))
        else:
            pieces.extend(records.clause_chip(c, member.readings, word, member.slug) for c, word in run)
        run.clear()

    for code in codes:
        clause = clauses.get(code)
        if code.startswith("F"):
            flush()
            pieces.append(
                INLINE.ct("f", code, f"#{member.slug}/{code.lower()}", member.tip(code))
                if code in findings
                else INLINE.missing(f"{code} not found")
            )
            continue
        if clause is None:
            flush()
            pieces.append(INLINE.missing(f"{code} not found"))
            continue
        word = records.clause_word(clause, list(member.findings), member.readings)
        words.append(word)
        split = [] if clause.id.startswith("Q") else records.reading_split(member.readings.get(clause.id, []))
        drawn.extend([w for w, _ in split] if len(split) > 1 else [word])
        if clause.superseded_by:
            flush()
            chip = records.clause_chip(clause, member.readings, word, member.slug)
            pieces.append(Markup('<span class="sup" title="superseded by {}">{}</span>').format(clause.superseded_by, chip))
            continue
        if run and not _continues(run[-1][0], clause, run[-1][1], word, member):
            flush()
        run.append((clause, word))
    flush()
    return {
        "html": Markup(" ").join(pieces),
        "words": words,
        "drawn": drawn,
        "findings": any(code.startswith("F") for code in codes),
    }


def now_mark(words: list[str], has_findings: bool) -> dict:
    """Where a thread stands: one outcome when its clauses agree, partial for a mix; words for none."""
    read = [w for w in words if w != PLANNED]
    if not words:
        return {"word": "", "text": FINDINGS_ONLY if has_findings else NO_CLAUSE, "words": []}
    if not read:
        return {"word": PLANNED, "text": "", "words": [PLANNED]}
    kinds = {OUTCOME.get(w, w) for w in read}
    if len(kinds) == 1:
        return {"word": read[0], "text": "", "words": [read[0]]}
    return {"word": "partial", "text": "mixed", "words": ["partial"]}


def threads_lede(threads: dict | None, members: list[Member]) -> str:
    """The threads lede: the reading rule, and the threads cited by findings alone."""
    if not threads:
        return ""
    if not any(m.concluded for m in members):
        return SKELETON_THREADS_LEDE
    parts = [THREADS_LEDE]
    if any(any(r["planned"]) for r in threads["rows"]):
        parts.append(THREADS_PLANNED)
    parts.append(THREADS_NOW)
    if only := [r["plain"] for r in threads["rows"] if r["findings_only"]]:
        parts.append(f"{and_join(only)} {'has' if len(only) == 1 else 'have'} findings but no clause yet.")
    return " ".join(parts)


# ===============================
# Beliefs, open questions and needs
# ===============================


def read_beliefs(model: Model, site: Site, members: list[Member], pid: str) -> list[dict]:
    """One fold per belief: its bold claim, the findings it cites, its reasoning, its figure; a caveat marked.

    A Figure row belongs to the belief above it. Its figures are numbered in
    page order, each 480 px wide or its own width when wider, up to the column.
    """
    beliefs = []
    for _, text in programme_checks.bullets(model.section(BELIEVE)):
        if unwritten(text):
            continue
        if programme_checks.FIGURE_LINE.match(text):
            if beliefs and not beliefs[-1]["figure"]:
                # a missing file shows as missing, with no number and no count
                n = sum(1 for b in beliefs if b["figure"] and not b["figure"]["missing"]) + 1
                figure = belief_figure(model, site, members, text, pid, n)
                if figure:
                    beliefs[-1]["figure"] = figure
                if figure and not figure["missing"]:
                    beliefs[-1]["count"] = " · ".join(c for c in (beliefs[-1]["count"], "1 figure") if c)
            continue
        match = LEAD.match(text)
        claim, body = (match.group(1).rstrip(":").strip(), match.group(2)) if match else (text[2:], "")
        cited = dict.fromkeys(RECORD_CITATION.findall(body))
        beliefs.append(
            {
                "claim": cite_html(claim, members),
                "body": cite_html(markup.capitalize(body), members) if body else "",
                "count": counted(len(cited), "finding", "findings") if cited else "",
                "caveat": claim.casefold() == CAVEAT.casefold(),
                "figure": None,
            }
        )
    return beliefs


def belief_figure(model: Model, site: Site, members: list[Member], row: str, pid: str, n: int) -> dict | None:
    """A belief's Figure row as the n-th figure of the page; None when it links no figure.

    A member's figure names its source in the caption, `From 4·F1, Figure 2
    of <record>`, as the record's page numbers it; the file line gives the
    path from the repo root.
    """
    link = FIGURE_LINK.search(row)
    if link is None:
        return None
    caption, src = link.groups()
    folder = model.document.folder
    path = (folder / src).resolve()
    width = max(BELIEF_FIGURE_PX, markup.figure_width(path) or 0)
    caption = markup.capitalize(caption.strip())
    for member in members:
        if member.model.document.folder.resolve() in path.parents:
            caption = f"{caption}{'' if caption.endswith(SENTENCE_STOPS) else '.'} {figure_source(member, site, path)}"
            break
    root = site.corpus.experiments.parent.resolve()
    return records.read_figure_row(
        folder,
        src,
        cite_html(caption, members),
        f"Figure {n}",
        f"{pid}/figure-{n}",
        width=f"{width:.0f}px",
        shown=path.relative_to(root).as_posix() if root in path.parents else "",
    )


def figure_source(member: Member, site: Site, path: Path) -> str:
    """`From [<slug>:F<n>], Figure <m> of [<short>](#<slug>)`: the finding drawing the file and its number there."""
    finding, label = records.figure_place(member.model, site, path)
    record = f"[{member.short}](#{member.slug})"
    if not label:
        return f"From {record}."
    if not finding:
        return f"From {label} of {record}."
    return f"From [{member.slug}:{finding}], {label} of {record}."


def read_open(model: Model, site: Site, planned: list[Planned]) -> dict:
    """The open questions table, its lede, and the open question each planned experiment answers."""
    by_slug = {p.slug: p for p in planned}
    known = {records.record_slug(r): r for r in site.of_kind("record")}
    for g in records.read_programmes(site.corpus.experiments):
        for slug, (short, title) in g["planned"].items():
            by_slug.setdefault(slug, Planned(slug, short, title, "", "", (), ()))
    rows, answers = [], {}
    taking: set[str] = set()
    for _, text in programme_checks.bullets(model.section(OPEN)):
        if unwritten(text):
            continue
        ending = programme_checks.ending_of(text)
        starts = list(programme_checks.ENDING_START.finditer(text))
        question = (text[: starts[-1].start()] if starts else text)[2:].strip()
        slugs = programme_checks.answering_slugs(ending)
        pills, note, planned_note = [], "", False
        for slug in slugs:
            if slug in by_slug:
                pills.append(INLINE.pill(PLANNED, by_slug[slug].short, by_slug[slug].title))
                answers.setdefault(slug, markup.plain_text(markup.inline_html(question)))
            elif slug in known:
                pills.append(records.record_pill(known[slug]))
            else:
                pills.append(Markup.escape(slug))
            taking.add(slug)
        if slugs:
            note = ending.split(",", 1)[1].strip() if "," in ending else ""
        elif match := NOTE_AFTER.match(ending):
            planned_note, note = True, match.group(1)
        rows.append(
            {
                "question": Markup(markup.inline_html(markup.capitalize(question))),
                "pills": pills,
                "not_planned": planned_note,
                "note": Markup(markup.inline_html(note)) if note else "",
                "no_ending": not slugs and not planned_note,
            }
        )
    return {"rows": rows, "answers": answers, "lede": open_lede(rows, taking, by_slug)}


def open_lede(rows: list[dict], taking: set[str], planned: dict[str, Planned]) -> str:
    """'N questions are open; M planned experiments take on K of them, and J are not yet planned.'"""
    n = len(rows)
    if not n:
        return ""
    answered = sum(1 for r in rows if r["pills"])
    head = f"{n} question is open" if n == 1 else f"{n} questions are open"
    kind = "planned experiment" if taking <= set(planned) else "experiment"
    if answered == n:
        one = f"{'an' if kind.startswith('e') else 'a'} {kind}"
        return f"{head}, and {one} takes it on." if n == 1 else f"{head}, and {one} takes on each."
    if not answered:
        return f"{head}, and none is planned yet."
    takers = counted(len(taking), kind, f"{kind}s")
    rest = n - answered
    verb = "takes" if len(taking) == 1 else "take"
    return f"{head}; {takers} {verb} on {answered} of them, and {rest} {'is' if rest == 1 else 'are'} not yet planned."


def read_needs(members: list[Member], site: Site) -> dict | None:
    """Per member: People, Compute, Data and External as short leads, a heavy one amber, the full row on hover."""
    rows = []
    for m in members:
        prose = records.Prose(m.model, m.slug, site, list(m.clauses), list(m.findings))
        needs = records.read_needs(m.model, prose)
        # the full row in the report's hover card: its label, then its text escaped for the card's HTML
        for need, (label, _) in zip(needs, records.NEEDS):
            full = markup.plain_text(markup.inline_html(markup.unguided(m.model.row("Design", label))))
            need["tip"] = f"{label}\n{Markup.escape(full)}" if full else ""
        rows.append({"member": records.record_pill(m.model), "needs": needs})
    if not any(n["text"] for r in rows for n in r["needs"]):
        return None
    return {"names": [label for label, _ in records.NEEDS], "rows": rows}


# ===============================
# Helper functions
# ===============================


def cite_html(text: str, members: list[Member]) -> Markup:
    """Programme prose as HTML, each `[<slug>:F<n>]` a chip `<n>·F<n>` with its hover card.

    A citation naming no member, or an id the member lacks, stays as written,
    and parentheses holding only citations go only when every one is a chip.
    """
    by_slug = {m.slug: m for m in members}

    def chip(match: re.Match) -> str:
        member = by_slug.get(match.group(1))
        code = match.group(2)
        tip = member.tip(code) if member else ""
        if not tip:
            return match.group(0)
        kind = "f" if code.startswith("F") else "h"
        return str(INLINE.ct(kind, f"{member.number}·{code}", f"#{member.slug}/{code.lower()}", tip))

    def group(match: re.Match) -> str:
        inside = RECORD_CITATION.sub(chip, match.group(1))
        return inside if not RECORD_CITATION.search(inside) else f"({inside})"

    html_ = CITATIONS_ONLY.sub(group, markup.inline_html(text))
    return Markup(RECORD_CITATION.sub(chip, html_))


def _arrow(x: float, y: float, side: str) -> str:
    # the arrow head at a child's port, pointing into the card
    if side == "left":
        return f"M{_px(x - ARROW)} {_px(y - 4)}L{_px(x)} {_px(y)}L{_px(x - ARROW)} {_px(y + 4)}"
    back = -ARROW if side == "top" else ARROW
    return f"M{_px(x - 4)} {_px(y + back)}L{_px(x)} {_px(y)}L{_px(x + 4)} {_px(y + back)}"


def record_question(record: Model) -> str:
    """The question a record's experiment answers, as plain text for a card or ledger row."""
    return markup.capitalize(markup.plain_text(markup.inline_html(markup.unguided(record.row("Hypothesis", "Question")))))


def _result(record: Model, findings) -> str:
    # a concluded card's result: the first sentence of the Summary opening its key finding card
    deciding = records.deciding_keys(record, list(findings))
    summary = markup.unguided(deciding[0].text("Summary")) if deciding else ""
    return markup.capitalize(markup.first_sentence(markup.plain_text(markup.inline_html(summary)))) if summary else ""


def _card_height(node: dict) -> int:
    # a card's height: its name, question, result and status line as they wrap, then a row of chips per
    # row its "also builds on" needs; never below the compact planned or the plain member card
    name = f"{node['number']} {node['name']}" if node.get("number") else str(node.get("name", ""))
    status = f"planned · {markup.plain_text(str(node.get('ticket', '')))}" if node.get("kind") == PLANNED else str(node.get("line", ""))
    height = CARD_PAD + _lines(name, NAME_CHARS) * NAME_LINE + CARD_GAP + _lines(status, TEXT_CHARS) * TEXT_LINE
    for text in (node.get("asks"), node.get("result")):
        if text:
            height += CARD_GAP + _lines(text, TEXT_CHARS) * TEXT_LINE
    if node["also"]:
        rows, used = 1, 0
        for chip in node["also"]:
            width = len(markup.plain_text(str(chip))) * CHIP_CHAR_W + CHIP_PAD_W
            if used and used + width > CHIP_ROOM:
                rows, used = rows + 1, 0
            used += width
        height += ALSO_H + (rows - 1) * CHIP_ROW_H
    return max(height, PLANNED_H if node.get("kind") == PLANNED else NODE_H)


def _lines(text: str, width: int) -> int:
    # how many lines a text wraps to at so many characters a line, at most CARD_LINES
    lines, used = 1, 0
    for word in text.split():
        need = len(word) + (1 if used else 0)
        if used and used + need > width:
            lines, used, need = lines + 1, 0, len(word)
        used += need
        while used > width:
            lines, used = lines + 1, used - width
    return min(lines, CARD_LINES)


def _earliest(when: datetime.date | None, named: int, listed: int) -> tuple:
    # a parent's rank for the main line: concluded first, by date; then as its Builds on bullet names it
    return (when is None, when or datetime.date.max, named, listed)


def _node(key, kind, word, number, name, title, href, line, ticket="", asks="", result="") -> dict:
    return {
        "key": key,
        "kind": kind,
        "word": word,
        "number": number,
        "name": name,
        "title": title,
        "href": href,
        "line": line,
        "ticket": ticket,
        "asks": asks,
        "tip": f"Question\n{Markup.escape(asks)}" if asks else "",
        "result": result,
        "result_tip": f"Result\n{Markup.escape(result)}" if result else "",
    }


def _continues(previous, clause, previous_word: str, word: str, member: Member) -> bool:
    # the next clause of a run: same letter, the next number, the same outcome, never split
    return (
        clause.id[0] == previous.id[0]
        and int(clause.id[1:]) == int(previous.id[1:]) + 1
        and word == previous_word
        and len(records.reading_split(member.readings.get(clause.id, []))) <= 1
        and len(records.reading_split(member.readings.get(previous.id, []))) <= 1
    )


def state_badge(word: str) -> Markup:
    """An experiment's badge; a planned one takes the dashed draft badge of the experiments table."""
    if word == PLANNED:
        return Markup('<span class="badge b-draft">planned</span>')
    return WORDS.badge(word)


def _tip(head: str, body: str, label: str) -> str:
    inline = re.sub(r'<span class="math[^"]*">.*?</span>', "", markup.inline_html(body), flags=re.DOTALL)
    return f"{head}\n<strong>{label}:</strong> {inline}"


def _cut(text: str) -> str:
    words = text.split()
    return " ".join(words[:LABEL_WORDS]) + ("…" if len(words) > LABEL_WORDS else "")


def _listed(value) -> list:
    return value if isinstance(value, list) else [value] if value not in (None, "") else []


def month_day(date: datetime.date) -> str:
    """A date as `MM-DD`."""
    return date.strftime("%m-%d")


def _px(value: float) -> str:
    return f"{value:g}"


def _section(key: str, data, empty: str, lede: str, n: int, tree_lede: str = "") -> dict:
    # tree_lede: the lede a phone shows over the map's outline, in place of `lede`
    return {"key": key, "title": SECTIONS[key][0], "data": data, "empty": empty, "lede": lede, "n": n, "tree_lede": tree_lede}


def counted(n: int, one: str, many: str) -> str:
    """`n` with its noun, singular for one."""
    return f"{n} {one if n == 1 else many}"


def and_join(names: list[str]) -> str:
    """Names joined with commas and a final "and"."""
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"
