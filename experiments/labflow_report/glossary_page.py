"""The Glossary page on the new builder: every metric and term the report uses (ADR-0005).

`glossary_page` turns the glossary's `Model` into the values
`templates/glossary.html` draws: the kicker and title, the lede with the
term count, the kind switch beside the filter box, the A–Z index, then every
term in name order under its letter head, each with its section label, its
aliases and a definition the page clamps to two lines. The kind switch is a
filter link per section with its term count: All, Metrics, each term group
in file order, then Other. A glossary with no term line that parses draws
the empty page. The top bar's breadcrumb (Glossary) comes from the rail, not
from this view.

The view reads terms only through `checks.glossary.terms`, the reader the
glossary checks use, so the page and the checker never disagree on a line.
Terms are not linked from page text, so the page shows no page counts.
"""

from __future__ import annotations

import re
import string
import unicodedata

from markupsafe import Markup

from . import markup
from .builder import Model, PageView, Site, view
from .checks.glossary import METRICS, OTHER, Term, terms

# ===============================
# Defaults
# ===============================

TEMPLATE = "glossary.html"
PAGE_ID = "glossary"
KICKER = "experiments/GLOSSARY.md"
EMPTY_LEDE = "No terms yet."
EMPTY_TEXT = (
    "The glossary is empty. A term joins it when a record's Design names a metric "
    "or a page defines a word."
)
# the letter head of a name that does not start with A–Z (`κ`, `5-fold`), first in the index
OTHER_LETTER = "#"
LETTERS = (OTHER_LETTER, *string.ascii_uppercase)
# a group's anchor carries this prefix, so a group never shares one with a term
GROUP_PREFIX = "group-"

# ===============================
# Main function
# ===============================


@view("glossary")
def glossary_page(model: Model, site: Site) -> PageView:
    """Builds the Glossary page's values from the glossary.

    Args:
        model: The glossary.
        site: Every document; the page reads none but the glossary.

    Returns:
        The page view `templates/glossary.html` renders.
    """
    found = terms(model.document)

    # the kind switch: All, Metrics, each group in file order, then Other
    kinds = [
        {"id": "", "href": f"#{PAGE_ID}", "kind": "", "name": "All", "n": len(found)}
    ]
    kinds += [
        {
            "id": section_id(name),
            "href": f"#{section_id(name)}",
            "kind": name,
            "name": name,
            "n": n,
        }
        for name, n in section_counts(found).items()
    ]

    # every term in name order, under its letter head
    groups: dict[str, list[dict]] = {}
    for term in sorted(
        found, key=lambda t: (LETTERS.index(letter(t.name)), t.name.casefold(), t.line)
    ):
        groups.setdefault(letter(term.name), []).append(term_view(term))
    index = [
        {"letter": x, "href": f"#{letter_id(x)}" if x in groups else ""}
        for x in LETTERS
        if x != OTHER_LETTER or x in groups
    ]

    return PageView(
        kind="glossary",
        id=PAGE_ID,
        title="Glossary",
        template=TEMPLATE,
        context={
            "kicker": KICKER,
            "lede": f"{len(found)} {'term' if len(found) == 1 else 'terms'} the report uses.",
            "kinds": kinds,
            "index": index,
            "groups": [
                {"id": letter_id(x), "letter": x, "terms": items}
                for x, items in groups.items()
            ],
            "empty": {"lede": EMPTY_LEDE, "text": EMPTY_TEXT},
        },
    )


# ===============================
# Helper functions
# ===============================


def anchor(name: str) -> str:
    """A glossary name as its anchor: lowercase words joined by dashes."""
    return re.sub(r"[^\w]+", "-", name.lower()).strip("-")


def section_counts(found: list[Term]) -> dict[str, int]:
    """Each section's term count: Metrics first, the groups in file order, Other last."""
    counts: dict[str, int] = {}
    for term in found:
        counts[term.section] = counts.get(term.section, 0) + 1
    rank = {METRICS: 0, OTHER: 2}
    return dict(sorted(counts.items(), key=lambda item: rank.get(item[0], 1)))


def section_id(name: str) -> str:
    """A section's anchor: `glossary/metrics`, `glossary/other`, else `glossary/group-<slug>`."""
    slug = anchor(name)
    return f"{PAGE_ID}/{slug if name in (METRICS, OTHER) else GROUP_PREFIX + slug}"


def letter(name: str) -> str:
    """The letter head a name files under: its first letter without accents, else OTHER_LETTER."""
    first = unicodedata.normalize("NFKD", name.lstrip())[:1].upper()
    return first if first and first in string.ascii_uppercase else OTHER_LETTER


def letter_id(head: str) -> str:
    """The anchor of a letter head."""
    return f"{PAGE_ID}/letter-{'other' if head == OTHER_LETTER else head.lower()}"


def term_view(term: Term) -> dict:
    """What one term row shows: its anchor, kind, name, aliases, definition and filter text."""
    return {
        "id": f"{PAGE_ID}/{anchor(term.name)}",
        "kind": term.section,
        "name": term.name,
        "aliases": ", ".join(term.aliases),
        "definition": Markup(markup.inline_html(markup.capitalize(term.definition))),
        "q": " ".join(
            [
                term.name,
                *term.aliases,
                markup.plain_text(markup.inline_html(term.definition)),
            ]
        ).lower(),
    }
