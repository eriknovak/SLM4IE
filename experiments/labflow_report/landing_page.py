"""The Research and Reference landing pages on the new builder (ADR-0005).

A click on the rail's Research or Reference tab opens one of these pages
(`specs/landing.yaml`; boards ResearchLanding and ReferenceLanding). Neither
page has a document of its own: both are drawn from what the programmes,
records, topics, entries and `reference/ideas.yaml` already store.

`research_page` lists every experiment, one section per programme in the
rail's order: its members in `members:` order, then its planned experiments
dashed, then the records no programme lists. Its lede counts the concluded,
running and planned experiments, tallies the outcomes, names what runs now
and the latest conclusion, and counts the Open questions no experiment
answers yet; a show filter and an order switch sit above the sections.
`reference_page` lists every topic as a card with its entries as chips, the
open ideas touching them and a link to its Compared table.

Effects stay at the edge: the caller passes in the running slugs, as for the
rail, so the views read only the site.
"""

from __future__ import annotations

import datetime

from markupsafe import Markup

# record imports builder, which imports the page views: read record's names at call time
from . import markup, rail
from . import programme_page as programmes
from . import record as records
from .builder import (
    ENTRY_STATUS,
    ENVIRONMENT,
    OUTCOME,
    Model,
    PageView,
    Site,
    entry_page_id,
    entry_title,
)
from .checks import programme as programme_checks
from .checks import topic as topic_checks
from .checks.record import unwritten
from .entry_page import written_block

# ===============================
# Defaults
# ===============================

RESEARCH = "research"
REFERENCE = "reference"
RESEARCH_TITLE = "Research"
REFERENCE_TITLE = "Reference"
RESEARCH_TEMPLATE = "research.html"
REFERENCE_TEMPLATE = "reference.html"
RUNNING = "running"
PLANNED = "planned"
OPEN = "Open"
# the outcome words in the order the tally, the filter and a programme head list them
OUTCOME_ORDER = (
    "confirmed", "adopted", "partial", "inconclusive", "refuted", "rejected",
    "answered", "unanswered", "better", "equivalent", "worse", "undecided",
)  # fmt: skip
RESEARCH_INTRO = "Every experiment, one programme at a time: what each tested and how it came out."
RESEARCH_INTRO_LOOSE = "Every experiment: what each tested and how it came out."
NOTHING_RUNNING = "Nothing running."
NOTHING_RUNNING_YET = "Nothing running yet: no experiment has a worktree or an exp/ branch."
RUNNING_LABEL = "Running"
NOT_IN_A_PROGRAMME = "Not in a programme"
EXPERIMENTS = "Experiments"
LOOSE_KEY = "loose"
NO_RECORDS = "no records yet"
NO_QUESTION = "question not written yet"
NO_CATEGORY = "no category"
NO_PROGRAMME_EXPERIMENT = "No experiment yet: the programme names no member and no planned experiment."
FILTER_NONE = "No {} experiments in this programme."
NO_EXPERIMENTS = "No experiment yet. Each programme lists its experiments here once labflow:start plans them."
ORDERS = (("members", "programme order"), ("new", "newest first"), ("old", "oldest first"))
REFERENCE_LEDE = "the methods the experiments build on, one topic per question."
REFERENCE_LEDE_EMPTY = "no entries yet: each topic fills as entries are written for it."
TOPIC_STATUS_BADGE = {"open": "running", "closed": "draft"}
NO_TOPIC_QUESTION = "The topic has no question yet."
NO_ENTRIES_YET = "no entries yet"
NO_ENTRIES = "No entries yet. An entry is one Markdown file beside this topic’s README."
NO_TABLE_YET = "The Compared table shows once the topic has entries."
NO_COMPARE = "The topic names no fact to compare yet."
INLINE = ENVIRONMENT.get_template("components/inline.html").module

# ===============================
# Main functions
# ===============================


def research_page(site: Site, running: frozenset[str] = frozenset()) -> PageView:
    """Builds the Research landing page's values from the programmes and records.

    Args:
        site: Every document: the programmes and the records.
        running: The slugs of the experiments that run, from `rail.running_slugs`.

    Returns:
        The page view `templates/research.html` renders.
    """
    record_models = site.of_kind("record")
    by_path = {r.document.path.resolve(): r for r in record_models}
    slugs = {records.record_slug(r) for r in record_models}
    programme_models = site.of_kind("programme")

    # one group per programme: its members in members order, then its planned experiments
    groups, listed = [], set()
    for programme in programme_models:
        members = [by_path[d.path.resolve()] for _, d in programme_checks.members(programme.document, site.corpus)]
        members = [r for n, r in enumerate(members) if r not in members[:n]]
        listed |= {r.document.path.resolve() for r in members}
        planned = programmes.read_planned(programme, slugs)
        groups.append(programme_group(programme, site, members, planned, running))

    # the records no programme lists, in conclusion order, last
    loose = sorted(
        (r for path, r in by_path.items() if path not in listed),
        key=lambda r: (r.value("concluded") is None, r.value("concluded") or datetime.date.min, records.record_slug(r)),
    )
    if loose:
        title = NOT_IN_A_PROGRAMME if programme_models else EXPERIMENTS
        rows = [record_row(r, n, running) for n, r in enumerate(loose)]
        groups.append({"key": LOOSE_KEY, "title": title, "href": "", "path": "", "full_title": "", "summary": "", "rows": rows, "empty": ""})

    # the counts, the tally and the filter, each experiment once though two programmes list it
    rows = list({r["key"]: r for g in groups for r in g["rows"]}.values())
    words = [r["word"] for r in rows]
    outcomes = ordered_outcomes(words)
    concluded = [r for r in rows if not r["planned"] and r["word"] not in (RUNNING, PLANNED)]
    stats = [(len(concluded), "concluded"), (words.count(RUNNING), "running"), (words.count(PLANNED), "planned")]
    filters = [("all", "All", len(rows))]
    filters += [(w, w.capitalize(), words.count(w)) for w in outcomes]
    filters += [(RUNNING, RUNNING_LABEL, words.count(RUNNING)), (PLANNED, PLANNED.capitalize(), words.count(PLANNED))]

    return PageView(
        kind=RESEARCH,
        id=RESEARCH,
        title=RESEARCH_TITLE,
        template=RESEARCH_TEMPLATE,
        context={
            "kicker": f"{site.corpus.experiments.name}/programmes" if programme_models else site.corpus.experiments.name,
            "intro": RESEARCH_INTRO if programme_models else RESEARCH_INTRO_LOOSE,
            "stats": stats,
            "tally": [(w, words.count(w)) for w in outcomes],
            "now": running_now(rows, concluded, record_models, running),
            "open": open_not_planned(programme_models),
            "filters": filters,
            "orders": ORDERS,
            "groups": groups,
            "filter_none": FILTER_NONE,
            "no_experiments": NO_EXPERIMENTS,
        },
    )


def reference_page(site: Site) -> PageView | None:
    """Builds the Reference landing page's values from the topics, entries and ideas.

    Args:
        site: Every document: the topics and their entries.

    Returns:
        The page view `templates/reference.html` renders; None when the
        experiments folder has no `reference/`.
    """
    folder = site.corpus.experiments / REFERENCE
    if not folder.is_dir():
        return None
    entries = site.of_kind("entry")
    topic_models = {m.id: m for m in site.of_kind("topic")}
    topics = sorted(set(topic_models) | {m.id.split("/", 1)[0] for m in entries})
    ideas = [i for i in topic_checks.read_ideas(folder) if i.state == topic_checks.OPEN]

    # the lede: topic and entry counts, entries by status square, the open ideas
    statuses = [e.value("status") or "draft" for e in entries]
    head = programmes.counted(len(topics), "topic", "topics")
    counts = programmes.counted(len(entries), "entry", "entries")
    lede = f"{head}, {counts}: {REFERENCE_LEDE}" if entries else f"{head}, {REFERENCE_LEDE_EMPTY}"
    squares = [(square, statuses.count(status), word) for status, (square, word) in ENTRY_STATUS.items()]
    has_cache = (folder / topic_checks.IDEAS_FILE).is_file()

    return PageView(
        kind=REFERENCE,
        id=REFERENCE,
        title=REFERENCE_TITLE,
        template=REFERENCE_TEMPLATE,
        context={
            "kicker": f"{site.corpus.experiments.name}/{REFERENCE}/",
            "lede": lede,
            "squares": squares,
            "ideas": len(ideas) if has_cache else None,
            "cards": [topic_card(t, topic_models.get(t), entries, ideas) for t in topics],
        },
    )


def render_research(site: Site, running: frozenset[str] = frozenset()) -> str:
    """The Research landing page's HTML.

    Args:
        site: Every document.
        running: The slugs of the experiments that run, from `rail.running_slugs`.

    Returns:
        The page's `<section class="page">` HTML.
    """
    page = research_page(site, running)
    return ENVIRONMENT.get_template(page.template).render(page=page, **page.context)


def render_reference(site: Site) -> str:
    """The Reference landing page's HTML, or "" without `reference/`.

    Args:
        site: Every document.

    Returns:
        The page's `<section class="page">` HTML, or "".
    """
    page = reference_page(site)
    if page is None:
        return ""
    return ENVIRONMENT.get_template(page.template).render(page=page, **page.context)


# ===============================
# Research parts
# ===============================


def programme_group(programme: Model, site: Site, members: list[Model], planned: list, running: frozenset[str]) -> dict:
    """One programme's section: its head, a row per member, then a dashed row per planned experiment."""
    names = rail.programme_names(programme)
    rows = [record_row(r, n, running) for n, r in enumerate(members)]
    rows += [planned_row(p, len(rows) + n, running) for n, p in enumerate(planned)]

    # the head's count: the records, then each word the rows carry
    words = [r["word"] for r in rows]
    counts = [f"{words.count(w)} {w}" for w in (*ordered_outcomes(words), RUNNING, PLANNED) if words.count(w)]
    head = programmes.counted(len(members), "record", "records") if members else NO_RECORDS
    return {
        "key": names["pid"].removeprefix("programme-"),
        "title": names["short"],
        "href": f"#{names['pid']}",
        "path": programme.document.path.relative_to(site.corpus.experiments).as_posix(),
        "full_title": names["title"] if names["title"] != names["short"] else "",
        "summary": f"{head} · {', '.join(counts)}" if counts else head,
        "rows": rows,
        "empty": NO_PROGRAMME_EXPERIMENT,
    }


def record_row(record: Model, position: int, running: frozenset[str]) -> dict:
    """A record's row: its mark, short and title, Question, category, badge and concluded date."""
    short, title = rail.record_names(record)
    word = rail.rail_word(record, running) or str(record.value("status") or "")
    concluded = record.value("concluded")
    question = programmes.record_question(record)
    category = record.value("category") or ""
    return {
        "key": records.record_slug(record),
        "planned": False,
        "word": word,
        "mark": word in OUTCOME,
        "short": short,
        "title": title,
        "href": f"#{records.record_slug(record)}",
        "question": question or INLINE.missing(NO_QUESTION),
        "category": category or INLINE.missing(NO_CATEGORY),
        "has_category": bool(category),
        "badge": programmes.state_badge(word) if word else "",
        "position": position,
        "date": concluded.isoformat() if concluded else "",
        "when": programmes.month_day(concluded) if concluded else "",
    }


def planned_row(planned, position: int, running: frozenset[str]) -> dict:
    """A planned experiment's row: unlinked and undated, its title as the question it will ask."""
    word = RUNNING if planned.slug in running else PLANNED
    question = markup.capitalize(planned.title) if planned.title else ""
    return {
        "key": planned.slug,
        "planned": True,
        "word": word,
        "mark": True,
        "short": planned.short,
        "title": question or planned.short,
        "href": "",
        "question": question or INLINE.missing(NO_QUESTION),
        "category": planned.category or INLINE.missing(NO_CATEGORY),
        "has_category": bool(planned.category),
        "badge": programmes.state_badge(word),
        "position": position,
        "date": "",
        "when": "",
    }


def running_now(rows: list[dict], concluded: list[dict], record_models, running: frozenset[str]) -> dict:
    """The running line: each running experiment as a pill, or that nothing runs; then the latest conclusion."""
    pills = [INLINE.pill(RUNNING, r["short"], r["title"], r["href"]) for r in rows if r["word"] == RUNNING]
    say = RUNNING_LABEL if pills else NOTHING_RUNNING if concluded else NOTHING_RUNNING_YET
    dated = [r for r in record_models if r.value("concluded") and rail.rail_word(r, running) not in (RUNNING, PLANNED)]
    latest = None
    if dated:
        newest = max(dated, key=lambda r: (r.value("concluded"), records.record_slug(r)))
        short, title = rail.record_names(newest)
        word = rail.rail_word(newest, running)
        latest = {
            "pill": INLINE.pill(word or PLANNED, short, title, f"#{records.record_slug(newest)}"),
            "date": programmes.month_day(newest.value("concluded")),
        }
    return {"say": say, "pills": pills, "latest": latest}


def open_not_planned(programme_models) -> dict | None:
    """The Open questions no experiment answers yet: the total and each programme's count."""
    found = []
    for programme in programme_models:
        n = 0
        for _, text in programme_checks.bullets(programme.section(OPEN)):
            if not unwritten(text) and not programme_checks.answering_slugs(programme_checks.ending_of(text)):
                n += 1
        if n:
            names = rail.programme_names(programme)
            found.append({"label": names["short"], "href": f"#{names['pid']}/open", "n": n})
    total = sum(p["n"] for p in found)
    if not total:
        return None
    head = "1 open question is" if total == 1 else f"{total} open questions are"
    return {"text": f"{head} not yet planned:", "programmes": found}


def ordered_outcomes(words: list[str]) -> list[str]:
    """The outcome words present, in OUTCOME_ORDER, any other mark word after them."""
    present = {w for w in words if w in OUTCOME and w not in (RUNNING, PLANNED)}
    return [w for w in OUTCOME_ORDER if w in present] + sorted(present - set(OUTCOME_ORDER))


# ===============================
# Reference parts
# ===============================


def topic_card(topic: str, model: Model | None, entries, ideas: list) -> dict:
    """One topic's card: title, badge, entry counts, Question, entry chips, idea chips and its foot."""
    pid = f"topic-{topic}"
    title = str((model.value("title") if model else None) or topic.replace("-", " ").capitalize())
    status = (model.value("status") if model else None) or "open"
    mine = sorted((e for e in entries if e.id.startswith(f"{topic}/")), key=lambda e: entry_title(e).casefold())

    # the meta line: the entry count, then each status's count
    statuses = [e.value("status") or "draft" for e in mine]
    parts = [f"{statuses.count(s)} {word}" for s, (_, word) in ENTRY_STATUS.items() if statuses.count(s)]
    count = programmes.counted(len(mine), "entry", "entries")
    meta = f"{count} · {', '.join(parts)}" if mine else NO_ENTRIES_YET

    # the question, the entry chips and the open ideas touching them
    question = model.section("Question") if model else None
    question_text = written_block(question.text) if question else ""
    chips = []
    for entry in mine:
        square, word = ENTRY_STATUS.get(entry.value("status") or "draft", ENTRY_STATUS["draft"])
        name = entry_title(entry)
        chips.append({"title": name, "href": f"#{entry_page_id(entry.id)}", "tip": str(entry.value("summary") or name), "square": square, "word": word})
    inside = {e.id for e in mine}
    touching = [i for i in ideas if any(t.entry in inside for t in i.touches)]

    # the foot: the Compared table by its fact count, or why there is none yet
    facts = len(model.value("compare") or []) if model else 0
    compare = f"Compared table · {programmes.counted(facts, 'fact', 'facts')} →" if mine and facts else ""
    return {
        "id": f"{REFERENCE}/{topic}",
        "title": title,
        "href": f"#{pid}" if model else "",
        "badge": TOPIC_STATUS_BADGE.get(status, "draft"),
        "status": status,
        "meta": meta,
        "question": Markup(markup.inline_html(question_text)) if question_text else INLINE.missing(NO_TOPIC_QUESTION),
        "entries": chips,
        "none": NO_ENTRIES,
        "ideas": [{"number": i.number, "title": i.title, "href": i.url} for i in touching],
        "compare": compare,
        "compare_href": f"#{pid}/compare",
        "no_compare": NO_COMPARE if mine else NO_TABLE_YET,
    }
