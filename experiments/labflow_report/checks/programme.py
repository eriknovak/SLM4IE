"""Checks the programme spec (`specs/programme.yaml`) names beyond the built-in ones.

A programme is `programmes/<slug>.md`. Its `members` are record folders; a
member has concluded once its `status` is `concluded`, and from then on What
we now believe, Threads and Open are due (Open not once the programme is
closed, which needs Open and `planned` empty). A citation `[<slug>:F<n>]` or
`[<slug>:H|C|Q<n>]` names a member that holds the finding or clause, and
every clause of every member sits in a Threads row (a member without
clauses, each of its key findings). A member at `exploring` is a stub with
no clause or finding; a `dropped` one ended with no belief, so it makes no
section due, but runs among the concluded by its `concluded:` date when it
has one. A `**Caveat:**` belief sits last. Each
Open bullet ends `— answered by <slug>[ and <slug>][, <note>]` or `— not yet
planned[: <note>]`, the note free to hold dashes. Each `planned` item is a map with `slug`, `short` (two to
five words), `title` (a question ending in `?`) and `category` (the record
folder it will take); its optional `ticket`, `builds_on` and `threads` name
an issue, known slugs and Threads rows. `members` run in conclusion order, the others last.
The programme cites no table or figure, since the page draws every part from
the members' records and `planned`, but in a belief's Figure row: a member's
figure by its path, or one the programme's script wrote into
`programmes/<slug>/figures/`. A Figure row belongs to the belief above it,
so it is no belief of its own.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from ..document import Document
from ..document import Section as DocumentSection
from ..rules import WORD, Corpus, check, split_code
from ..spec import Field, Rule, Section
from .entry import assets
from .figures import figure_rows, resolved
from .record import RECORD_CITATION, clauses, entry_ids, prose_lines, unwritten, weights
from .topic import CLOSING, QUESTION_MARK

# ===============================
# Defaults
# ===============================

RECORD = "record"
PROGRAMME = "programme"
CONCLUDED = "concluded"
# a member ended by a conclusion or a dropped exploration; it runs by its concluded date
ENDED = (CONCLUDED, "dropped")
CLOSED = "closed"
THREADS = "Threads"
BELIEFS = "What we now believe"
FIGURE_ROW = "Figure"
FIGURES_FOLDER = "figures"
FIGURE_LINE = re.compile(r"^- \*\*Figure\*\*:")
OPEN = "Open"
KEY = "key"
PLANNED = "planned"
MEMBERS = "members"
# the keys every planned item holds
PLANNED_KEYS = ("slug", "short", "title")
# a planned short is two to five plain words, as the programme's own
SHORT_WORDS = (2, 5)
# the folders a record may take, as the template's category comment lists them
CATEGORIES = ("data", "methods", "validation")
TICKET = re.compile(r"^#\d+$|://|\.md$")
CAVEAT = re.compile(r"^- \*\*Caveat\b", re.IGNORECASE)
BULLET = "- "
ENDING_DASH = " — "
# the dash that opens an ending; a note may hold dashes of its own
ENDING_START = re.compile(r" — (?=answered by|not yet planned)")
# an answered-by ending starts with a slug, so "answered by TBD" is no ending
ANSWERED_BY = re.compile(r"^answered by\s+(?=`?[a-z0-9])(.*)$")
NOT_PLANNED = re.compile(r"^not yet planned(?::\s*\S.*|\.)?$")
# the first slug after `answered by`; more follow after ` and `, and a `,` opens the note
SLUG = re.compile(r"`?([a-z0-9][a-z0-9-]*)`?")
NEXT_SLUG = re.compile(r"\s+and\s+`?([a-z0-9][a-z0-9-]*)`?")

# ===============================
# Section checks
# ===============================


@check("due-once-concluded")
def due_once_concluded(
    document: Document, part: Section, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """The section is written once a member has concluded; Open not once the programme is closed."""
    section = document.sections.get(part.heading)
    if section is not None and section.text:
        return
    if part.heading == OPEN and document.frontmatter.get("status") == CLOSED:
        return
    concluded = [
        slug for slug, member in members(document, corpus) if is_concluded(member)
    ]
    if concluded:
        yield (section.line if section else 1), f"{', '.join(concluded)} concluded"


@check("member-citations")
def member_citations(
    document: Document, part: None, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each `[<slug>:F<n>]` or `[<slug>:H|C|Q<n>]` names a member holding that finding or clause."""
    held = dict(members(document, corpus))
    for number, text in prose_lines(document):
        for match in RECORD_CITATION.finditer(text):
            slug, cited_id = match.groups()
            member = held.get(slug)
            if member is None:
                yield number, f"[{slug}:{cited_id}] names no member '{slug}'"
            elif cited_id not in cited_ids(member):
                yield number, f"[{slug}:{cited_id}] names no {cited_id} in {slug}"


@check("threads-cover-members")
def threads_cover_members(
    document: Document, part: Section, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Every clause of every member, or a clause-less member's every key finding, sits in a row."""
    section = document.sections.get(part.heading)
    line = section.line if section else 1
    # a row still holding guidance keeps its citations; the guidance rule reports it
    prose = split_code(section.body)[0] if section else ""
    cited = {match.groups() for match in RECORD_CITATION.finditer(prose)}
    for slug, member in members(document, corpus):
        found = clauses(member)
        ids = (
            [clause.id for clause in found]
            if found
            else [fid for fid, weight, _ in weights(member) if weight == KEY]
        )
        what = "clause" if found else "key finding"
        for missing in [i for i in ids if (slug, i) not in cited]:
            yield line, f"{what} {slug} {missing} sits in no thread"


@check("caveat-last")
def caveat_last(
    document: Document, part: Section, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """No belief follows one that opens `**Caveat:**`."""
    caveat = None
    for number, text in bullets(document.sections.get(part.heading)):
        if FIGURE_LINE.match(text):
            continue
        if CAVEAT.match(text):
            caveat = caveat or number
        elif caveat is not None:
            yield caveat, f"a belief follows it on line {number}"
            return


@check("open-ending")
def open_ending(
    document: Document, part: Section, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each written bullet ends `— answered by <slug>…` or `— not yet planned[: <note>]`."""
    for number, text in bullets(document.sections.get(part.heading)):
        if unwritten(text):
            continue
        ending = ending_of(text)
        if not (ANSWERED_BY.match(ending) or NOT_PLANNED.match(ending)):
            yield number, f"it ends '{text[-40:]}'"


@check("answered-by")
def answered_by(
    document: Document, part: Section, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each slug after `answered by` names a record or a planned experiment."""
    known = known_slugs(corpus)
    for number, text in bullets(document.sections.get(part.heading)):
        if unwritten(text):
            continue
        for slug in answering_slugs(ending_of(text)):
            if slug not in known:
                yield number, f"'{slug}' is no record or planned slug"


@check("programme-assets")
def programme_assets(
    document: Document, part: None, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """No `![caption](path)` link outside code, but a belief Figure row's own figure.

    A belief's Figure row cites a figure in a member's `figures/` or in the
    programme's `programmes/<slug>/figures/`.
    """
    rows = figure_rows(document.sections.get(BELIEFS), FIGURE_ROW)
    own = document.folder / programme_slug(document) / FIGURES_FOLDER
    folders = {own.resolve()} | {
        (record.folder / FIGURES_FOLDER).resolve()
        for _, record in members(document, corpus)
    }
    elsewhere = f"no member's figures/ nor programmes/{programme_slug(document)}/figures/"
    for asset in assets(document):
        in_row = asset.section == BELIEFS and any(row.holds(asset.line) for row in rows)
        if not in_row:
            yield asset.line, f"`{asset.path}`"
        elif resolved(document, asset.path).parent not in folders:
            yield asset.line, f"`{asset.path}` sits in {elsewhere}"


# ===============================
# Frontmatter checks
# ===============================


@check("planned-keys")
def planned_keys(
    document: Document, part: Field, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each planned item is a map holding slug, short and title."""
    value = document.frontmatter.get(part.key)
    line = document.key_lines.get(part.key, 1)
    for number, item in enumerate(value if isinstance(value, list) else [], start=1):
        if not isinstance(item, dict):
            yield line, f"item {number} is '{item}', not a map"
            continue
        missing = [key for key in PLANNED_KEYS if not item.get(key)]
        if missing:
            yield line, f"{name(item, number)} has no {', '.join(missing)}"


@check("planned-short")
def planned_short(
    document: Document, part: Field, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each planned short is two to five words."""
    low, high = SHORT_WORDS
    for line, number, item in planned_items(document, part):
        short = item.get("short")
        count = len(WORD.findall(str(short))) if short else low
        if not low <= count <= high:
            yield line, f"{name(item, number)} short is {count} words"


@check("planned-question")
def planned_question(
    document: Document, part: Field, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each planned title is a question ending in `?`."""
    for line, number, item in planned_items(document, part):
        title = str(item.get("title") or "")
        if title and not title.rstrip(CLOSING).endswith(QUESTION_MARK):
            yield line, f"{name(item, number)} title ends '{title.rstrip()[-20:]}'"


@check("planned-category")
def planned_category(
    document: Document, part: Field, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """A planned category, when given, is a record folder."""
    for line, number, item in planned_items(document, part):
        category = item.get("category")
        if category not in (None, "") and category not in CATEGORIES:
            yield line, f"{name(item, number)} category is '{category}'"


@check("planned-category-required")
def planned_category_required(
    document: Document, part: Field, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each planned item states its category, the Research page's chip."""
    for line, number, item in planned_items(document, part):
        if item.get("category") in (None, ""):
            yield line, f"{name(item, number)} has no category"


@check("planned-ticket")
def planned_ticket(
    document: Document, part: Field, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """A planned ticket, when given, is quoted `"#<n>"`, a URL or a markdown path."""
    for line, number, item in planned_items(document, part):
        ticket = item.get("ticket")
        if ticket in (None, ""):
            continue
        if not isinstance(ticket, str) or not TICKET.search(ticket):
            yield line, f"{name(item, number)} ticket is '{ticket}'"


@check("planned-builds-on")
def planned_builds_on(
    document: Document, part: Field, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """A planned builds_on is a list of record or planned slugs."""
    known = known_slugs(corpus)
    for line, number, item in planned_items(document, part):
        parents = item.get("builds_on")
        if parents in (None, ""):
            continue
        if not isinstance(parents, list):
            yield line, f"{name(item, number)} builds_on is '{parents}', not a list"
            continue
        for parent in parents:
            if str(parent) not in known:
                yield (
                    line,
                    f"{name(item, number)} builds on '{parent}', no record or planned slug",
                )


@check("planned-threads")
def planned_threads(
    document: Document, part: Field, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """A planned threads is a list of Threads row names, matched in any case."""
    section = document.sections.get(THREADS)
    rows = {label.lower() for label in (section.rows if section else {})}
    for line, number, item in planned_items(document, part):
        names = item.get("threads")
        if names in (None, ""):
            continue
        if not isinstance(names, list):
            yield line, f"{name(item, number)} threads is '{names}', not a list"
            continue
        for thread in names:
            if str(thread).lower() not in rows:
                yield (
                    line,
                    f"{name(item, number)} names thread '{thread}', no Threads row",
                )


@check("planned-slugs")
def planned_slugs(
    document: Document, part: Field, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """No planned slug repeats or names a member."""
    member_slugs = {slug for slug, _ in members(document, corpus)}
    seen = set()
    for line, _, item in planned_items(document, part):
        slug = str(item.get("slug") or "")
        if slug in member_slugs:
            yield line, f"'{slug}' is a member"
        elif slug and slug in seen:
            yield line, f"'{slug}' is planned twice"
        seen.add(slug)


@check("members-order")
def members_order(
    document: Document, part: Field, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Ended members run in `concluded:` order; the others follow them."""
    previous = None
    for slug, member in members(document, corpus):
        date = str(member.frontmatter.get(CONCLUDED) or "") if is_ended(member) else ""
        if previous is not None and date and (not previous[1] or date < previous[1]):
            when = f"concluded {date}"
            before = f"concluded {previous[1]}" if previous[1] else "not concluded"
            yield (
                document.key_lines[part.key],
                f"{slug} ({when}) follows {previous[0]} ({before})",
            )
            return
        previous = (slug, date)


@check("closed-programme")
def closed_programme(
    document: Document, part: Field, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """A closed programme has no Open bullet and no planned experiment."""
    if document.frontmatter.get(part.key) != CLOSED:
        return
    line = document.key_lines[part.key]
    open_points = len(bullets(document.sections.get(OPEN)))
    planned = document.frontmatter.get(PLANNED)
    if open_points:
        yield line, f"Open holds {open_points} bullet(s)"
    if isinstance(planned, list) and planned:
        yield line, f"planned holds {len(planned)} experiment(s)"


# ===============================
# Helper functions
# ===============================


def members(document: Document, corpus: Corpus) -> list[tuple[str, Document]]:
    """(slug, record) of each member that resolves to a record, in `members` order.

    Args:
        document: The programme.
        corpus: Every document read, for the member records.

    Returns:
        The members; a folder that names no record is left out, since the
        built-in link check reports it.
    """
    records = {d.folder.resolve(): d for d in corpus.documents if d.kind == RECORD}
    listed = document.frontmatter.get(MEMBERS)
    found = []
    for folder in listed if isinstance(listed, list) else []:
        record = records.get((document.folder / str(folder)).resolve())
        if record is not None:
            found.append((record_slug(record), record))
    return found


def known_slugs(corpus: Corpus) -> set[str]:
    """The slug of every record and of every planned experiment of every programme."""
    slugs = {record_slug(d) for d in corpus.documents if d.kind == RECORD}
    for programme in (d for d in corpus.documents if d.kind == PROGRAMME):
        planned = programme.frontmatter.get(PLANNED)
        for item in planned if isinstance(planned, list) else []:
            if isinstance(item, dict) and item.get("slug"):
                slugs.add(str(item["slug"]))
    return slugs


def answering_slugs(ending: str) -> list[str]:
    """The slugs an Open ending names after `answered by`; none for another ending."""
    match = ANSWERED_BY.match(ending)
    if match is None:
        return []
    rest = match.group(1)
    first = SLUG.match(rest)
    if first is None:
        return []
    slugs, position = [first.group(1)], first.end()
    while more := NEXT_SLUG.match(rest, position):
        slugs.append(more.group(1))
        position = more.end()
    return slugs


def programme_slug(document: Document) -> str:
    return str(document.frontmatter.get("slug") or document.path.stem)


def record_slug(record: Document) -> str:
    return str(record.frontmatter.get("slug") or record.folder.name)


def is_concluded(record: Document) -> bool:
    return record.frontmatter.get("status") == CONCLUDED


def is_ended(record: Document) -> bool:
    return record.frontmatter.get("status") in ENDED


def cited_ids(record: Document) -> set[str]:
    return entry_ids(record, "Findings") | {clause.id for clause in clauses(record)}


def bullets(section: DocumentSection | None) -> list[tuple[int, str]]:
    """(line, text) of each top-level `- ` bullet, its indented continuation joined."""
    if section is None:
        return []
    found, in_bullet = [], False
    for number, line in enumerate(section.body, start=section.line + 1):
        if line.startswith(BULLET):
            found.append([number, line.strip()])
            in_bullet = True
        elif in_bullet and line.startswith((" ", "\t")) and line.strip():
            found[-1][1] += " " + line.strip()
        elif line.strip():
            # a top-level line outside a bullet ends it; the form rule reports it
            in_bullet = False
    return [(number, text) for number, text in found]


def ending_of(text: str) -> str:
    """The text after the ` — ` that opens a bullet's ending, else after its last ` — `.

    Backticks and the closing stop are kept.
    """
    starts = list(ENDING_START.finditer(text))
    if starts:
        return text[starts[-1].end() :].strip()
    if ENDING_DASH not in text:
        return ""
    return text.rsplit(ENDING_DASH, 1)[1].strip()


def planned_items(document: Document, part: Field) -> Iterator[tuple[int, int, dict]]:
    """(key line, 1-based number, item) of each planned item that is a map."""
    value = document.frontmatter.get(part.key)
    line = document.key_lines.get(part.key, 1)
    for number, item in enumerate(value if isinstance(value, list) else [], start=1):
        if isinstance(item, dict):
            yield line, number, item


def name(item: dict, number: int) -> str:
    """A planned item as a message names it: its slug, else its place in the list."""
    return f"planned '{item['slug']}'" if item.get("slug") else f"planned item {number}"
