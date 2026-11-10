"""Checks the book spec (`specs/book.yaml`) names beyond the built-in ones.

The book is `experiments/README.md`. It opens with `# Findings` and one
written intro paragraph that links GLOSSARY.md; its `##` headings are
`Main results`, one chapter per record category (data, methods,
validation) and `Notes for maintainers`, in that order. Main results is
written once a record concludes: three to six bullets (fewer only while
the concluded records hold fewer key findings), each ending `— bears on:
<one clause>` and citing findings of concluded records as `[<slug>:F<n>]`,
never through a link.
A chapter holds one line per key finding,
`- **[F<n>](<category>/<slug>/#<anchor>) — <title>.** <Summary>`, linking a
key finding of a record in that category; every key finding of a concluded
or abandoned record has one. The book cites no table or figure. The intro
and the chapters sit outside the declared sections, so the checks here
read the file and the chapter headings themselves.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Iterator

from ..document import HEADING, Document, normalize_link
from ..document import Section as DocumentSection
from ..rules import INLINE_CODE, Corpus, check
from ..spec import Rule, Section
from .entry import IMAGE_LINK
from .programme import (
    BULLET,
    CATEGORIES,
    CONCLUDED,
    KEY,
    RECORD,
    bullets,
    is_concluded,
    record_slug,
)
from .record import RECORD_CITATION, entry_ids, prose_lines, unwritten, weights

# ===============================
# Defaults
# ===============================

TITLE = "# Findings"
MAIN_RESULTS = "Main results"
NOTES = "Notes for maintainers"
HEADINGS = (MAIN_RESULTS, *CATEGORIES, NOTES)
ABANDONED = "abandoned"
FINDINGS = "Findings"
# Main results holds three to six bullets once three key findings exist
RESULTS = (3, 6)
BEARS_ON = " — bears on:"
FENCE = "```"
# a link's target, without its title; `#` opens the anchor
LINK = re.compile(r"\]\(\s*([^)\s]+)[^)]*\)")
# `- **[F<n>](<link>) — <title>.**`: the bold holds the finding link and its title
CHAPTER_LINE = re.compile(r"^- \*\*\[(F\d+)\]\(\s*([^)\s]+)[^)]*\) — \S.*?\*\*")
FINDING_PREFIX = "F"
# a finding cited by a link, `[F<n>](<link>)`, where Main results wants `[<slug>:F<n>]`
LINK_CITATION = re.compile(r"\[(F\d+)\]\(\s*([^)\s]+)[^)]*\)")
CITATION_FORM = "[<slug>:F<n>]"
NESTED = (
    "has a chapter line nested under a record bullet; move it up to a top-level line"
)

# ===============================
# Title and intro checks
# ===============================


@check("book-title")
def book_title(
    document: Document, part: None, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """The first line is `# Findings`."""
    lines = file_lines(document)
    first = lines[0] if lines else ""
    if first.rstrip() != TITLE:
        yield 1, f"it reads '{first[:40]}'"


@check("book-intro")
def book_intro(
    document: Document, part: None, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """A written paragraph sits between the title and the first `##`."""
    paragraphs = intro(document)
    if not paragraphs:
        yield 1, "no paragraph under the title"
    for number, text in paragraphs:
        if unwritten(text):
            yield number, "template guidance left in it"


@check("intro-paragraph")
def intro_paragraph(
    document: Document, part: None, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """The intro is one paragraph."""
    paragraphs = intro(document)
    if len(paragraphs) > 1:
        yield paragraphs[1][0], f"{len(paragraphs)} paragraphs"


@check("intro-glossary")
def intro_glossary(
    document: Document, part: None, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """The intro links GLOSSARY.md."""
    paragraphs = intro(document)
    if not paragraphs:
        return
    targets = {
        normalize_link(match.group(1).split("#")[0])
        for _, text in paragraphs
        for match in LINK.finditer(text)
    }
    if "GLOSSARY" not in targets:
        yield paragraphs[0][0], "no GLOSSARY.md link"


@check("book-headings")
def book_headings(
    document: Document, part: None, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each `##` heading is Main results, a category chapter or Notes for maintainers, once and in that order."""
    seen, last = set(), None
    for number, heading in headings(document):
        if heading not in HEADINGS:
            yield number, f"'## {heading}' is no book heading"
            continue
        if heading in seen:
            yield number, f"'## {heading}' appears twice"
        elif last is not None and HEADINGS.index(heading) < HEADINGS.index(last):
            yield number, f"'## {heading}' follows '## {last}'"
        seen.add(heading)
        last = heading


# ===============================
# Main results checks
# ===============================


@check("results-due")
def results_due(
    document: Document, part: Section, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Main results holds a written bullet once a record has concluded."""
    section = document.sections.get(part.heading)
    if written_bullets(section):
        return
    concluded = [record for record in records(corpus) if is_concluded(record)]
    if concluded:
        line = section.line if section else 1
        yield (
            line,
            f"{len(concluded)} record(s) concluded, among them {record_slug(concluded[0])}",
        )


@check("results-count")
def results_count(
    document: Document, part: Section, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Three to six written bullets; fewer only while the concluded records hold fewer key findings."""
    section = document.sections.get(part.heading)
    count = len(written_bullets(section))
    if count == 0:
        return
    keys = sum(
        len(key_findings(record)) for record in records(corpus) if is_concluded(record)
    )
    low, high = RESULTS
    fewest = max(1, min(low, keys))
    if count > high:
        yield section.line, f"{count} results"
    elif count < fewest:
        yield (
            section.line,
            f"{count} result(s), and the concluded records hold {keys} key findings",
        )


@check("bears-on")
def bears_on(
    document: Document, part: Section, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each written bullet ends `— bears on: <one clause>`."""
    for number, text in written_bullets(document.sections.get(part.heading)):
        if BEARS_ON not in text or not text.rsplit(BEARS_ON, 1)[1].strip():
            yield number, f"it ends '{text[-40:]}'"


@check("results-cite")
def results_cite(
    document: Document, part: Section, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each written bullet cites a finding, and only as `[<slug>:F<n>]`."""
    for number, text in written_bullets(document.sections.get(part.heading)):
        prose = INLINE_CODE.sub(" ", text)
        links = [match.group(1) for match in LINK_CITATION.finditer(prose)]
        if links:
            yield (
                number,
                f"it cites [{links[0]}](…) by link; cite it as {CITATION_FORM}",
            )
        elif not main_results_citations(prose):
            yield number, f"it cites no finding; cite one as {CITATION_FORM}"


@check("results-findings")
def results_findings(
    document: Document, part: Section, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each finding a result cites is held by a concluded record."""
    by_slug = {record_slug(record): record for record in records(corpus)}
    for number, text in prose_lines(document, part.heading):
        for slug, cited_id in main_results_citations(text):
            cited = f"[{slug}:{cited_id}]"
            record = by_slug.get(slug)
            if not cited_id.startswith(FINDING_PREFIX):
                yield number, f"{cited} cites a clause, not a finding"
            elif record is None:
                yield number, f"{cited} names no record '{slug}'"
            elif not is_concluded(record):
                status = record.frontmatter.get("status")
                yield number, f"{cited}: {slug} is {status}, not {CONCLUDED}"
            elif cited_id not in entry_ids(record, FINDINGS):
                yield number, f"{cited} names no {cited_id} in {slug}"


# ===============================
# Chapter checks
# ===============================


@check("chapter-lines")
def chapter_lines(
    document: Document, part: None, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each block of a chapter is a chapter line `- **[F<n>](<link>) — <title>.** …`."""
    for heading in CATEGORIES:
        for number, text in blocks(document.sections.get(heading)):
            if unwritten(text):
                yield number, f"template guidance left under '## {heading}'"
            elif not CHAPTER_LINE.match(text):
                yield number, f"'{text[:40]}' under '## {heading}'"


@check("chapter-findings")
def chapter_findings(
    document: Document, part: None, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each chapter line links a finding a record of that chapter's category holds."""
    by_id = {record.id: record for record in records(corpus)}
    for heading, number, finding, target in finding_links(document, CATEGORIES):
        record = by_id.get(target)
        category = target.split("/")[0]
        if record is None:
            yield number, f"[{finding}] links '{target}', no record"
        elif category != heading:
            yield number, f"{target} is a {category} record, under '## {heading}'"
        elif finding not in entry_ids(record, FINDINGS):
            yield number, f"[{finding}] names no {finding} in {target}"


@check("chapter-key")
def chapter_key(
    document: Document, part: None, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each chapter line links a key finding: supporting and minor ones stay out."""
    by_id = {record.id: record for record in records(corpus)}
    for _, number, finding, target in finding_links(document, CATEGORIES):
        record = by_id.get(target)
        weight = finding_weights(record).get(finding) if record else None
        if weight is not None and weight != KEY:
            yield number, f"{target} {finding} is {weight}"


@check("chapter-coverage")
def chapter_coverage(
    document: Document, part: None, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Every key finding of a concluded or abandoned record has one top-level chapter line.

    A line nested under another bullet is no chapter line; the message says so.
    """
    outside = [h for h in document.sections if h not in (MAIN_RESULTS, NOTES)]
    linked = Counter(
        (target, finding) for _, _, finding, target in finding_links(document, outside)
    )
    nested = set(nested_finding_links(document, outside))
    for record in records(corpus):
        if not (is_concluded(record) or record.frontmatter.get("status") == ABANDONED):
            continue
        chapter = document.sections.get(record.id.split("/")[0])
        line = chapter.line if chapter else 1
        for finding in key_findings(record):
            count = linked[(record.id, finding)]
            if count == 0 and (record.id, finding) in nested:
                yield line, f"{record.id} {finding} {NESTED}"
            elif count == 0:
                yield line, f"{record.id} {finding} has no chapter line"
            elif count > 1:
                yield line, f"{record.id} {finding} has {count} chapter lines"


# ===============================
# Table and figure checks
# ===============================


@check("book-assets")
def book_assets(
    document: Document, part: None, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """The book cites no table or figure: no `![caption](path)` link outside code.

    The intro sits under no `##`, so the whole file is read, not its sections.
    """
    in_fence = False
    for number, line in enumerate(file_lines(document), start=1):
        if line.lstrip().startswith(FENCE):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for match in IMAGE_LINK.finditer(INLINE_CODE.sub("", line)):
            if "<" not in match.group(1):
                yield number, f"`{match.group(1)}`"


# ===============================
# Helper functions
# ===============================


def main_results_citations(text: str) -> list[tuple[str, str]]:
    """(slug, id) of each citation a Main results line holds, as `[<slug>:F<n>]` names it."""
    return [match.groups() for match in RECORD_CITATION.finditer(text)]


def file_lines(document: Document) -> list[str]:
    return document.path.read_text(encoding="utf-8").splitlines()


def intro(document: Document) -> list[tuple[int, str]]:
    """(line, text) of each paragraph between the title and the first `##`, its lines joined."""
    lines = file_lines(document)
    paragraphs: list[list] = []
    previous_blank = True
    for number, line in enumerate(lines[1:], start=2):
        if HEADING.match(line):
            break
        if not line.strip():
            previous_blank = True
            continue
        if previous_blank:
            paragraphs.append([number, line.strip()])
        else:
            paragraphs[-1][1] += " " + line.strip()
        previous_blank = False
    return [(number, text) for number, text in paragraphs]


def headings(document: Document) -> list[tuple[int, str]]:
    """(line, heading) of each `##` heading outside fenced code, repeats kept."""
    found, in_fence = [], False
    for number, line in enumerate(file_lines(document), start=1):
        if line.lstrip().startswith(FENCE):
            in_fence = not in_fence
        elif not in_fence and (match := HEADING.match(line)):
            found.append((number, match.group(1)))
    return found


def written_bullets(section: DocumentSection | None) -> list[tuple[int, str]]:
    return [(number, text) for number, text in bullets(section) if not unwritten(text)]


def blocks(section: DocumentSection | None) -> list[tuple[int, str]]:
    """(line, text) of each block of a section: a bullet, a heading or a paragraph.

    A bullet's indented or wrapped lines join it; a blank line ends a paragraph.
    """
    if section is None:
        return []
    found: list[list] = []
    previous_blank = True
    for number, line in enumerate(section.body, start=section.line + 1):
        if not line.strip():
            previous_blank = True
            continue
        indented = line.startswith((" ", "\t"))
        opens = line.startswith((BULLET, "#")) or (previous_blank and not indented)
        if opens or not found:
            found.append([number, line.strip()])
        else:
            found[-1][1] += " " + line.strip()
        previous_blank = False
    return [(number, text) for number, text in found]


def finding_links(
    document: Document, sections: Iterable[str]
) -> Iterator[tuple[str, int, str, str]]:
    """(heading, line, F<n>, record id) of each chapter line in the named sections."""
    for heading in sections:
        for number, text in blocks(document.sections.get(heading)):
            match = CHAPTER_LINE.match(text)
            if match and not unwritten(text):
                finding, link = match.groups()
                yield heading, number, finding, normalize_link(link.split("#")[0])


def nested_finding_links(
    document: Document, sections: Iterable[str]
) -> Iterator[tuple[str, str]]:
    """(record id, F<n>) of each indented chapter line, nested under another bullet."""
    for heading in sections:
        section = document.sections.get(heading)
        for line in section.body if section else ():
            match = CHAPTER_LINE.match(line.lstrip())
            if line.startswith((" ", "\t")) and match:
                finding, link = match.groups()
                yield normalize_link(link.split("#")[0]), finding


def records(corpus: Corpus) -> list[Document]:
    return [d for d in corpus.documents if d.kind == RECORD]


def finding_weights(record: Document) -> dict[str, str]:
    return {finding: weight for finding, weight, _ in weights(record)}


def key_findings(record: Document) -> list[str]:
    return [finding for finding, weight, _ in weights(record) if weight == KEY]
