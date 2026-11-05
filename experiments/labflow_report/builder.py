"""The spec-led builder: validated model → page view → HTML (ADR-0005).

The pipeline has three stages:

1. `read_site` reads every document the specs declare, runs their rules and
   wraps each document in a `Model`. A model hands out only the parts its
   spec declares, typed as the spec says, so a page cannot read a part the
   spec does not name.
2. A view, registered with `@view("<kind>")`, turns one model into a
   `PageView`: the page's id, title, template and the values it shows.
3. `render_page` renders that view with Jinja2. Page templates extend
   `templates/page.html`; each item on the Components board is one macro of
   the same name, in `templates/components/<board section>.html` (`words`,
   `inline`, `evidence`, `answer`, `findings`, `decisions`, `inputs`,
   `programme`, `overview`, `reference`, `rail`), plus the shared `layout`.

The old builder takes over a page kind from here one kind at a time
(strangler): `build_report.build()` asks `render_page(model, site)` for each
page and keeps its own `render_*` function wherever the answer is None.
"""

from __future__ import annotations

import datetime
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import jinja2

from .document import Document, Section
from .rules import DATE, PYTHON_TYPES, Breach, Corpus, read_corpus, run_rules
from .spec import DocumentSpec

# ===============================
# Defaults
# ===============================

TEMPLATES = Path(__file__).resolve().parent / "templates"

# the mark kind each outcome word draws; a mark kind maps to itself
OUTCOME = {
    "confirmed": "ok",
    "answered": "ok",
    "refuted": "bad",
    "unanswered": "bad",
    "partial": "part",
    "inconclusive": "inc",
    "planned": "plan",
    "running": "run",
    "adopted": "adopt",
    "rejected": "rej",
    "better": "bet",
    "worse": "wor",
    "equivalent": "eq",
    "undecided": "und",
}

# the 16×16 glyph path inside each mark; an empty path draws a bare mark
GLYPHS = {
    "ok": "M4 8.4l2.6 2.6L12 5.6",
    "bad": "M5 5l6 6M11 5l-6 6",
    "part": "M2.8 9.8C3.6 7.4 5.4 6.2 8 8s4.4.6 5.2-1.8",
    "inc": "M5.3 5.9a2.7 2.7 0 1 1 3.9 2.4c-.8.4-1.2 1-1.2 1.8v.4M8 12.7v.5",
    "bet": "M8 12.2V4.6M4.9 7.6 8 4.4l3.1 3.2",
    "wor": "M8 3.8v7.6M4.9 8.4 8 11.6l3.1-3.2",
    "eq": "M4.6 6.4h6.8M4.6 9.6h6.8",
    "und": "M3.4 8h9.2M5.4 5.9 3.3 8l2.1 2.1M10.6 5.9 12.7 8l-2.1 2.1",
    "adopt": "M8 4.4v7.2M4.4 8h7.2",
    "rej": "M4.4 8h7.2",
    "plan": "",
    "run": "",
}

ENVIRONMENT = jinja2.Environment(
    loader=jinja2.FileSystemLoader(TEMPLATES),
    autoescape=True,
    undefined=jinja2.StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)
ENVIRONMENT.globals.update(OUTCOME=OUTCOME, GLYPHS=GLYPHS)

# ===============================
# Model classes
# ===============================


@dataclass(frozen=True)
class Model:
    """One document, checked against its spec, read only through the parts it declares."""

    document: Document
    spec: DocumentSpec
    breaches: tuple[Breach, ...] = ()

    @property
    def kind(self) -> str:
        return self.document.kind

    @property
    def id(self) -> str:
        return self.document.id

    @property
    def stage(self) -> str | None:
        return self.value(self.spec.stage_key) if self.spec.stage_key else None

    def value(self, key: str):
        """Returns a frontmatter value as its spec types it.

        A date becomes a `datetime.date`. A value that breaks its declared
        type or allowed values is None, and so is an empty one; the breach
        names it. A list keeps only its allowed items.

        Raises:
            KeyError: If the spec declares no such key.
        """
        part = self.spec.part(f"frontmatter.{key}")
        value = self.document.frontmatter.get(key)
        if value is None or value == "" or value == [] or value == {}:
            return None

        # type the value
        if part.type == "date":
            if isinstance(value, datetime.date):
                return value
            if not (isinstance(value, str) and DATE.match(value)):
                return None
            # a quoted date the calendar lacks, such as 2026-02-30
            try:
                return datetime.date.fromisoformat(value)
            except ValueError:
                return None
        if part.type in PYTHON_TYPES:
            expected = PYTHON_TYPES[part.type]
            # a bool is an int to Python, never to the spec
            if not isinstance(value, expected) or (
                part.type == "int" and isinstance(value, bool)
            ):
                return None

        # keep the allowed values
        if not part.values:
            return value
        if isinstance(value, list):
            return [item for item in value if item in part.values]
        return value if value in part.values else None

    def section(self, heading: str) -> Section | None:
        """Returns a `## heading` section, or None when the document lacks it.

        Raises:
            KeyError: If the spec declares no such section.
        """
        self.spec.part(f"section.{heading}")
        return self.document.sections.get(heading)

    def row(self, heading: str, label: str) -> str:
        """Returns the text of a `- **label**:` row, or "" when the document lacks it.

        Raises:
            KeyError: If the spec declares no such row.
        """
        self.spec.part(f"section.{heading}.{label}")
        section = self.document.sections.get(heading)
        row = section.rows.get(label) if section else None
        return row.text if row else ""


@dataclass(frozen=True)
class Site:
    """Every model in the experiments folder, and the ids that links resolve against."""

    models: tuple[Model, ...]
    corpus: Corpus

    def of_kind(self, kind: str) -> tuple[Model, ...]:
        return tuple(m for m in self.models if m.kind == kind)

    def get(self, kind: str, id: str) -> Model | None:
        return next((m for m in self.models if m.kind == kind and m.id == id), None)


# ===============================
# View classes
# ===============================


@dataclass(frozen=True)
class PageView:
    """What one page shows: its anchor, title, template and the values the template reads."""

    kind: str
    id: str
    title: str
    template: str
    context: Mapping = field(default_factory=dict)


View = Callable[[Model, Site], PageView]
VIEWS: dict[str, View] = {}


def view(kind: str) -> Callable[[View], View]:
    """Registers the view that renders every document of one kind on the new builder."""

    def register(function: View) -> View:
        if kind in VIEWS:
            raise ValueError(f"view for page kind '{kind}' is registered twice")
        VIEWS[kind] = function
        return function

    return register


# ===============================
# Main functions
# ===============================


def read_site(experiments: Path, specs: dict[str, DocumentSpec]) -> Site:
    """Reads every document the specs declare into a model with its breaches.

    Args:
        experiments: The experiments folder.
        specs: The specs by document kind.

    Returns:
        The site: every model, sorted by path, and the corpus of ids.
    """
    documents, corpus = read_corpus(experiments, specs)
    models = tuple(
        Model(
            document,
            specs[document.kind],
            tuple(run_rules(document, specs[document.kind], corpus)),
        )
        for document in documents
    )
    return Site(models, corpus)


def page_styles() -> str:
    """The CSS of every ported page kind, each file scoped to its own pages.

    Returns:
        The rules of every `templates/*.css`, in file name order.
    """
    return "".join(
        path.read_text(encoding="utf-8") for path in sorted(TEMPLATES.glob("*.css"))
    )


def render_page(model: Model, site: Site) -> str | None:
    """Renders one document's page on the new builder.

    Args:
        model: The document to render.
        site: Every model, for pages that show or link to others.

    Returns:
        The page's `<section class="page">` HTML, or None when its kind has no
        view yet and the old builder still renders it.
    """
    build = VIEWS.get(model.kind)
    if build is None:
        return None
    page = build(model, site)
    template = ENVIRONMENT.get_template(page.template)
    return template.render(page=page, **page.context)


# ===============================
# Page views
# ===============================

# each module registers its page kind's view with `@view` as it is imported
from . import record  # noqa: E402, F401
