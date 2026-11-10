"""The Overview page on the new builder (ADR-0005).

`overview_page` turns the book, `experiments/README.md`, and the site
around it into the values `templates/overview.html` draws: the kicker and
title; the book's intro line; one scorecard per programme (a meter with
one segment per experiment, the latest result, what can start now and the
open questions) with the outcome legend under them; the book's Main
results; one ledger per programme (a row per member with its clauses, why
each was settled and its outcome, then the planned experiments), and one
for the records no programme lists; the reference shelf; and the Notes for
maintainers fold. Only the intro, Main results and the notes come from the
book. The scorecards, ledgers and shelf are drawn from what the
programmes, records and reference entries already store (tables.md,
Overview tables): a ledger row reads the member's
`tables/clause-readings.csv` for its clause chips and its
`tables/verdict-by-arm.csv` for the adopted arm. A part with nothing to
show says so in words, so a programme planned before its first record
draws its skeleton.
"""

from __future__ import annotations

import dataclasses
import datetime
import re

from markupsafe import Markup

# record imports builder, which imports this module: read record's names at call time
from . import markup
from . import programme_page as programmes
from . import record as records
from .builder import ENTRY_STATUS, ENVIRONMENT, Model, PageView, Site, view
from .checks import book as book_checks
from .checks import programme as programme_checks
from .checks.record import RECORD_CITATION, unwritten

# ===============================
# Defaults
# ===============================

TEMPLATE = "overview.html"
PAGE_ID = "overview"
TITLE = "Overview"
MAIN_RESULTS = "Main results"
NOTES = "Notes for maintainers"
OPEN = "Open"
PLANNED = "planned"
ADOPTED = "adopted"
YES = ("yes", "true")
# a clause read on its own counts in the held line; these words decide nothing
UNCALLED = ("inconclusive", "undecided", "planned", "running")
HELD = ("confirmed", "better")
# the outcome legend lists the words the meters draw in this order
LEGEND_ORDER = (
    "confirmed", "refuted", "partial", "inconclusive", "adopted", "rejected",
    "answered", "unanswered", "running", "planned",
)  # fmt: skip
# a later record's relation word → the words its tag on a ledger row reads
QUALIFIED = {"qualifies": "qualified by", "overturns": "overturned by", "supports": "supported by"}
CITED_BY = "cited by"
MORE_FROM_THE_BOOK = "More from the book"
NOT_IN_A_PROGRAMME = "Not in a programme"
EXPERIMENTS = "Experiments"
LEGEND_NOTE = "one segment per experiment, oldest first"
SKELETON_LEGEND_NOTE = "every experiment is planned; the meter fills as each concludes"
SKELETON_LEDGER = (
    "No experiment has concluded yet. Every row is planned; a row gains its verdicts and "
    "outcome when its experiment concludes."
)
NO_LATEST = "no experiment has concluded yet"
NO_OPEN = "no open question recorded yet"
NOTHING_PLANNED = "nothing planned"
NO_CLAUSE = "no clause written yet"
NO_QUESTION = "question not written yet"
NO_GOAL = "The programme has no question yet."
NO_MAIN_RESULTS = (
    "No main result yet. labflow:report writes three to six here, each citing the finding "
    "that carries it, once experiments conclude."
)
NO_EXPERIMENTS = (
    "No experiment yet. Each programme draws a scorecard and a ledger here once "
    "labflow:start plans it."
)
SHELF_LEDE = "the methods the experiments build on."
SHELF_DRAFTS = "all drafts: no entry is checked against code or a paper yet."
# a Main results bullet: its bold claim, then the support and its `— bears on:` tail
RESULT = re.compile(r"^- \*\*(.+?)\*\*\s*(.*)$", re.DOTALL)
INLINE = ENVIRONMENT.get_template("components/inline.html").module

# ===============================
# Main function
# ===============================


@view("book")
def overview_page(model: Model | None, site: Site) -> PageView:
    """Builds the Overview's values from the book and the site around it.

    Args:
        model: The book, `experiments/README.md`; None when the repo has none,
            which leaves out the kicker, the intro and the notes fold and
            draws Main results empty.
        site: Every document: the programmes, records and reference entries.

    Returns:
        The page view `templates/overview.html` renders.
    """
    by_slug = {records.record_slug(r): r for r in site.of_kind("record")}
    members = {slug: as_member(r) for slug, r in by_slug.items()}

    # one scorecard and one ledger per programme, then the records none lists
    cards, ledgers, listed = [], [], set()
    for programme in site.of_kind("programme"):
        read = read_programme(programme, site)
        listed |= {m.slug for m in read["members"]}
        cards.append(read_scorecard(read, members))
        ledgers.append(read_ledger(read, site, members))
    loose = sorted(
        (m for slug, m in members.items() if slug not in listed),
        key=lambda m: (m.concluded is None, m.concluded or datetime.date.min, m.slug),
    )
    if loose:
        numbered = [dataclasses.replace(m, number=n) for n, m in enumerate(loose, 1)]
        title = NOT_IN_A_PROGRAMME if ledgers else EXPERIMENTS
        ledgers.append(
            {
                "id": f"{PAGE_ID}/experiments",
                "title": title,
                "goal": "",
                "lede": "",
                "rows": [member_row(m, members) for m in numbered],
            }
        )

    # the outcome legend: every word a meter draws
    drawn = {word for card in cards for word, _ in card["meter"]}
    legend = [w for w in LEGEND_ORDER if w in drawn] + sorted(drawn - set(LEGEND_ORDER))
    skeleton = drawn <= {PLANNED}
    concluded = any(m.concluded for m in members.values())

    return PageView(
        kind="overview",
        id=PAGE_ID,
        title=TITLE,
        template=TEMPLATE,
        context={
            "kicker": model.document.path.relative_to(site.corpus.experiments.parent).as_posix() if model else "",
            "intro": read_intro(model) if model else "",
            "cards": cards,
            "legend": legend,
            "legend_note": SKELETON_LEGEND_NOTE if skeleton else LEGEND_NOTE,
            "results": read_main_results(model, members, concluded) if model else {"rows": [], "empty": NO_MAIN_RESULTS},
            "ledgers": ledgers,
            "no_experiments": NO_EXPERIMENTS,
            "shelf": read_shelf(site),
            "notes": read_notes(model) if model else None,
        },
    )


def render_without_book(site: Site) -> str:
    """The Overview's HTML for a site whose experiments folder holds no README.md.

    Args:
        site: Every document: the programmes, records and reference entries.

    Returns:
        The page's `<section class="page">` HTML, as `render_page` gives it for the book.
    """
    page = overview_page(None, site)
    return ENVIRONMENT.get_template(page.template).render(page=page, **page.context)


# ===============================
# Book parts
# ===============================


def read_intro(model: Model) -> Markup:
    """The book's intro line: the first paragraph under its title."""
    paragraphs = book_checks.intro(model.document)
    first = paragraphs[0][1] if paragraphs else ""
    return Markup(markup.inline_html(first)) if first and not unwritten(first) else Markup("")


def read_main_results(model: Model, members: dict, concluded: bool) -> dict | None:
    """Main results: per bullet its claim, bears-on line, finding chips and experiment pills.

    Returns None when the book holds none and no record has concluded, so a
    skeleton overview leaves the section out.
    """
    rows = []
    for _, text in book_checks.written_bullets(model.section(MAIN_RESULTS)):
        match = RESULT.match(text)
        if match is None:
            continue
        claim, support = match.group(1), match.group(2)
        bears = support.rsplit(book_checks.BEARS_ON, 1)[1].strip() if book_checks.BEARS_ON in support else ""
        chips, pills = [], []
        for slug, code in dict.fromkeys(book_checks.main_results_citations(support)):
            member = members.get(slug)
            if member is None:
                continue
            if tip := member.tip(code):
                chips.append(INLINE.ct("f", code, f"#{slug}/{code.lower()}", tip))
            if (pill := records.record_pill(member.model)) not in pills:
                pills.append(pill)
        rows.append(
            {
                "claim": Markup(markup.inline_html(claim)),
                "bears": Markup(markup.inline_html(markup.capitalize(bears))) if bears else "",
                "chips": chips,
                "pills": pills,
            }
        )
    if not rows and not concluded:
        return None
    return {"rows": rows, "empty": NO_MAIN_RESULTS}


def read_notes(model: Model) -> dict | None:
    """The foot fold: Notes for maintainers, its first line the summary.

    A book without that section, whose intro runs past one paragraph, folds
    the further paragraphs under "More from the book".
    """
    section = model.section(NOTES)
    text = markup.unguided(section.text) if section else ""
    if text.strip():
        lines = text.strip().split("\n\n", 1)
        summary = markup.plain_text(markup.inline_html(lines[0]))
        body = lines[1] if len(lines) > 1 else ""
        return {"title": NOTES, "summary": summary, "body": Markup(markup.markdown_html(body))}
    rest = [text for _, text in book_checks.intro(model.document)[1:] if not unwritten(text)]
    if not rest:
        return None
    return {"title": MORE_FROM_THE_BOOK, "summary": "", "body": Markup(markup.markdown_html("\n\n".join(rest)))}


# ===============================
# Programmes and scorecards
# ===============================


def read_programme(programme: Model, site: Site) -> dict:
    """One programme's parts the overview shows: names, goal, members, planned and Open."""
    slug = str(programme.value("slug") or programme.document.path.stem)
    title = str(programme.value("title") or slug)
    found = programmes.read_members(programme, site)
    planned = programmes.read_planned(programme, {m.slug for m in found})
    question = programme.section("Question")
    goal = markup.first_sentence(markup.unguided(question.text)) if question else ""
    opened = []
    for _, text in programme_checks.bullets(programme.section(OPEN)):
        if not unwritten(text):
            opened.append(bool(programme_checks.answering_slugs(programme_checks.ending_of(text))))
    return {
        "slug": slug,
        "pid": f"programme-{slug}",
        "short": str(programme.value("short") or title),
        "title": title,
        "goal": Markup(markup.inline_html(goal)) if goal else "",
        "members": found,
        "planned": planned,
        "open": opened,
    }


def read_scorecard(programme: dict, members: dict) -> dict:
    """A programme's scorecard: meter, counts, latest result, what can start now, open questions."""
    found, planned = programme["members"], programme["planned"]
    concluded = [m for m in found if m.concluded]
    running = [m for m in found if not m.concluded and m.word == "running"]

    # the latest result: the newest concluded member, its deciding key finding's title
    latest = None
    if concluded:
        newest = max(concluded, key=lambda m: (m.concluded, m.number))
        deciding = records.deciding_keys(newest.model, list(newest.findings))
        latest = {
            "word": newest.word,
            "name": newest.short,
            "title": newest.title,
            "href": f"#{newest.slug}",
            "result": Markup(markup.inline_html(deciding[0].title)) if deciding else "",
            "date": programmes.month_day(newest.concluded),
        }

    # can start now: planned experiments whose every parent has concluded
    done = {slug for slug, m in members.items() if m.concluded}
    startable = [p for p in planned if set(p.builds_on) <= done]
    waits = ""
    if planned and not startable:
        first = planned[0]
        names = [members[s].short if s in members else _planned_short(s, planned) for s in first.builds_on if s not in done]
        waits = f"{first.short} waits on {programmes.and_join(names)}"

    # the open questions, and how many a planned experiment takes on
    n, taken = len(programme["open"]), sum(programme["open"])
    counts = [f"{len(concluded)} concluded"]
    counts += [f"{len(running)} running"] if running else []
    counts += [f"{len(planned)} planned"] if planned else []
    return {
        "short": programme["short"],
        "title": programme["title"],
        "href": f"#{programme['pid']}",
        "counts": " · ".join(counts),
        "meter": [(m.word, 1) for m in found] + [(PLANNED, 1) for _ in planned],
        "latest": latest,
        "no_latest": NO_LATEST,
        "startable": [INLINE.pill(PLANNED, p.short, markup.capitalize(p.title)) for p in startable],
        "waits": waits,
        "nothing_planned": NOTHING_PLANNED if not planned else "",
        "open_href": f"#{programme['pid']}/open",
        "open": open_line(n, taken),
        "no_open": NO_OPEN,
    }


def open_line(n: int, taken: int) -> dict | None:
    """'N questions, each taken on by a planned experiment', '…, none planned' or '…, K not yet planned'."""
    if not n:
        return None
    head = f"{n} question" if n == 1 else f"{n} questions"
    if taken == n:
        return {"head": head, "rest": "taken on by a planned experiment" if n == 1 else "each taken on by a planned experiment", "unplanned": ""}
    if not taken:
        return {"head": head, "rest": "none planned" if n > 1 else "not yet planned", "unplanned": ""}
    return {"head": head, "rest": "", "unplanned": f"{n - taken} not yet planned"}


# ===============================
# Ledgers
# ===============================


def read_ledger(programme: dict, site: Site, members: dict) -> dict:
    """One programme's ledger: its goal, a row per member in number order, then the planned rows."""
    rows = [member_row(m, members) for m in programme["members"]]
    rows += [planned_row(p, site, members, programme["planned"]) for p in programme["planned"]]
    skeleton = not any(m.concluded for m in programme["members"])
    return {
        "id": f"{PAGE_ID}/{programme['slug']}",
        "title": programme["short"],
        "full_title": programme["title"],
        "goal": programme["goal"] or INLINE.missing(NO_GOAL),
        "lede": SKELETON_LEDGER if skeleton and rows else "",
        "rows": rows,
    }


def member_row(member: programmes.Member, members: dict) -> dict:
    """A member's ledger row: name, question and meta; its clauses; its outcome and tags."""
    model = member.model
    question = programmes.record_question(model)
    category = model.value("category") or ""
    when = programmes.month_day(member.concluded) if member.concluded else member.word
    return {
        "planned": False,
        "number": member.number if member.concluded else "",
        "word": member.word,
        "name": member.short,
        "title": member.title if member.title != member.short else "",
        "href": f"#{member.slug}",
        "question": question or INLINE.missing(NO_QUESTION),
        "meta": " · ".join(p for p in (category, when) if p),
        "clauses": read_clause_cell(member),
        "badge": programmes.state_badge(member.word),
        "held": held_line(member) if member.concluded else "",
        "arm": adopted_arm(model) if member.word == ADOPTED else "",
        "tags": read_tags(member, members),
    }


def planned_row(planned: programmes.Planned, site: Site, members: dict, others: list[programmes.Planned]) -> dict:
    """A planned experiment's ledger row: its question, what opened it or what it builds on, its ticket."""
    opened = opened_by(planned.slug, members)
    parents = [
        records.record_pill(members[s].model) if s in members else INLINE.pill(PLANNED, _planned_short(s, others))
        for s in planned.builds_on
    ]
    return {
        "planned": True,
        "word": PLANNED,
        "name": planned.short,
        "title": "",
        "question": markup.capitalize(planned.title) if planned.title else INLINE.missing(NO_QUESTION),
        "ticket": programmes.ticket_chip(planned.ticket, site.github),
        "opened": opened,
        "parents": [] if opened else parents,
        "no_clause": NO_CLAUSE,
        "badge": programmes.state_badge(PLANNED),
    }


def read_clause_cell(member: programmes.Member) -> dict:
    """A member's clauses as the ledger shows them, grouped by the key finding that decided them.

    A clause one key finding decides alone is a line: its chip, its claim and
    that finding. Clauses one key finding decides together share one chip
    row with the finding under it. In a concluded record the clauses no key
    finding decides share one row too, with the key findings that decide no
    clause under it. Groups follow their first clause. An unconcluded
    record's clauses are lines with their claims. A record with no clause
    lists its key findings; one with neither says so.
    """
    findings = list(member.findings)
    keys = [f for f in findings if f.weight == "key"]
    live = [c for c in member.clauses if not c.superseded_by]
    if not live:
        if keys:
            return {"kind": "findings", "findings": [_finding(member, f) for f in keys]}
        return {"kind": "missing", "text": NO_CLAUSE}

    # group the clauses by their first deciding key finding; None holds the undecided
    groups: dict = {}
    for clause in live:
        word = records.clause_word(clause, findings, member.readings)
        deciding = [f for f in records.deciding_findings(clause, findings) if f.weight == "key"]
        finding = deciding[0] if deciding and member.concluded else None
        line = {
            "chip": records.clause_chip(clause, member.readings, word, member.slug),
            "claim": Markup(markup.inline_html(clause.claim)),
            "finding": _finding(member, finding) if finding else None,
        }
        key = finding.id if finding else (None if member.concluded else clause.id)
        groups.setdefault(key, []).append(line)

    # one clause is a line; several share a chip row with what decided them under it
    deciding_ids = {key for key in groups if key and key.startswith("F")}
    items = []
    for key, lines in groups.items():
        if len(lines) == 1:
            items.append({"kind": "line", **lines[0]})
            continue
        under = [_finding(member, f) for f in keys if f.id == key] if key else [
            _finding(member, f) for f in keys if f.id not in deciding_ids
        ]
        items.append({"kind": "row", "chips": [line["chip"] for line in lines], "findings": under})
    return {"kind": "groups", "parts": items}


def held_line(member: programmes.Member) -> str:
    """'N of M held', or '0 of M called' when no clause was decided; then each question's word."""
    findings = list(member.findings)
    tested, questions = [], []
    for clause in member.clauses:
        if clause.superseded_by:
            continue
        word = records.clause_word(clause, findings, member.readings)
        if clause.id.startswith("Q"):
            questions.append(f"{clause.id} {word}")
        elif clause.standard != "measured" and clause.role != "diagnostic":
            tested.append(word)
    parts = []
    if tested:
        called = [w for w in tested if w not in UNCALLED]
        held = sum(w in HELD for w in tested)
        parts.append(f"{held} of {len(tested)} held" if called else f"0 of {len(tested)} called")
    return ", ".join(parts + questions)


def adopted_arm(model: Model) -> str:
    """The arm `tables/verdict-by-arm.csv` marks adopted; "" without the table or a `yes` row."""
    path = records.artifact_path(model, "verdict-by-arm")
    rows = [r for r in markup.read_table(path) if (r.get("adopted") or "").strip().lower() in YES]
    return rows[0].get("arm", "") if len(rows) == 1 else ""


def read_tags(member: programmes.Member, members: dict) -> list[dict]:
    """One tag per later record citing a finding of this one: its relation, name and finding."""
    tags, seen = [], set()
    for other in members.values():
        if other.slug == member.slug or other.slug in seen:
            continue
        for finding in other.findings:
            text = "\n".join(records.row_source(r) for r in finding.entry.rows.values())
            match = next(
                (m for m in RECORD_CITATION.finditer(text) if m.group(1) == member.slug and m.group(2).startswith("F")),
                None,
            )
            if match is None:
                continue
            relation = records.RELATION_BEFORE.search(text[: match.start()])
            tags.append(
                {
                    "word": other.word,
                    "text": f"{QUALIFIED.get(relation.group(1), CITED_BY) if relation else CITED_BY} {other.short} {finding.id}",
                    "href": f"#{other.slug}/{finding.id.lower()}",
                }
            )
            seen.add(other.slug)
            break
    return tags


def opened_by(slug: str, members: dict) -> dict | None:
    """The record whose Verdict Next item ends `— planned as <slug>`: its pill and the clauses it cites.

    A finding the item cites stands for the clauses its Reading decides; one
    deciding none shows as its own chip.
    """
    for member in members.values():
        for item in records.items(records.section_row(member.model, "Verdict", "Next")):
            planned = records.PLANNED_AS.search(item)
            if planned is None or planned.group(1) != slug:
                continue
            codes: list[str] = []
            clauses = list(member.clauses)
            for code in records.CITATION.findall(item):
                if code[0] in "HCQ":
                    codes.append(code)
                elif code.startswith("F"):
                    finding = next((f for f in member.findings if f.id == code), None)
                    decided = records.finding_clauses(finding, clauses) if finding else []
                    codes += [c.id for c in decided] or [code]
            codes = list(dict.fromkeys(codes))
            chips = programmes.thread_cell(member, codes)["html"] if codes else ""
            return {"chips": chips, "pill": records.record_pill(member.model)}
    return None


# ===============================
# Reference shelf
# ===============================


def read_shelf(site: Site) -> dict | None:
    """The reference shelf: one card per topic with its entry count, and the lede."""
    topics = site.of_kind("topic")
    if not topics:
        return None
    entries = site.of_kind("entry")
    cards, total, drafts = [], 0, 0
    for topic in sorted(topics, key=lambda t: str(t.value("title") or t.id).casefold()):
        own = [e for e in entries if e.id.split("/", 1)[0] == topic.id]
        statuses = [e.value("status") or "draft" for e in own]
        n_drafts = statuses.count("draft")
        total, drafts = total + len(own), drafts + n_drafts
        cards.append(
            {
                "id": f"{PAGE_ID}/topic-{topic.id}",
                "title": str(topic.value("title") or topic.id),
                "href": f"#topic-{topic.id}",
                "count": f"{len(own)} drafts" if own and n_drafts == len(own) else str(len(own)),
                "tip": ", ".join(
                    f"{statuses.count(s)} {word}" for s, (_, word) in ENTRY_STATUS.items() if statuses.count(s)
                ),
            }
        )
    head = f"{programmes.counted(len(topics), 'topic', 'topics')}, {programmes.counted(total, 'entry', 'entries')}"
    lede = f"{head}, {SHELF_DRAFTS}" if total and drafts == total else f"{head}: {SHELF_LEDE}"
    return {"cards": cards, "lede": lede}


# ===============================
# Helper functions
# ===============================


def as_member(record: Model) -> programmes.Member:
    """A record as the programme page's Member, unnumbered: its word, names, clauses and findings."""
    slug = records.record_slug(record)
    title = str(record.value("title") or slug)
    return programmes.Member(
        model=record,
        slug=slug,
        number=0,
        word=records.record_word(record),
        short=str(record.value("short") or title),
        title=title,
        concluded=record.value("concluded"),
        clauses=tuple(records.read_clauses(record)),
        findings=tuple(records.read_findings(record)),
        readings=records.read_readings(record),
    )


def _finding(member: programmes.Member, finding) -> dict:
    # a key finding as the ledger names it: its title and its chip with the hover card
    return {
        "title": Markup(markup.inline_html(finding.title)),
        "chip": INLINE.ct("f", finding.id, f"#{member.slug}/{finding.id.lower()}", member.tip(finding.id)),
    }


def _planned_short(slug: str, planned: list[programmes.Planned]) -> str:
    return next((p.short for p in planned if p.slug == slug), slug)
