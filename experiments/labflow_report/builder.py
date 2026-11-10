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

`build_report.build()` asks `render_page(model, site)` for every page and
joins them into one file with the rail.
"""

from __future__ import annotations

import datetime
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import jinja2

from .document import Document, Section
from .markup import EMPTY_CELL
from .rules import DATE, PYTHON_TYPES, Breach, Corpus, read_corpus, run_rules
from .spec import DocumentSpec

# ===============================
# Defaults
# ===============================

TEMPLATES = Path(__file__).resolve().parent / "templates"
# the rules of the shared layout macros, scoped to the pages that draw them
LAYOUT_STYLES = "layout.css"

# a GitHub remote in its ssh, ssh:// and https forms; the groups are owner and name
GITHUB_REMOTE = re.compile(
    r"^(?:git@github\.com:|ssh://git@github\.com/|https?://(?:[^@/]+@)?github\.com/)"
    r"([^/]+)/([^/]+?)(?:\.git)?/?$"
)

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
    "done": "done",
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
    "done": "M4 8.4l2.6 2.6L12 5.6",
    "plan": "",
    "run": "",
}

# a reference entry's status → (the class of its square, its word)
ENTRY_STATUS = {
    "checked": ("chk", "checked against code"),
    "from-paper": ("pap", "from paper"),
    "draft": ("dft", "draft"),
}
ENTRY_PAGE_PREFIX = "ref-"

ENVIRONMENT = jinja2.Environment(
    loader=jinja2.FileSystemLoader(TEMPLATES),
    autoescape=True,
    undefined=jinja2.StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)
ENVIRONMENT.globals.update(OUTCOME=OUTCOME, GLYPHS=GLYPHS, EMPTY_CELL=EMPTY_CELL)

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
    """Every model in the experiments folder, the ids that links resolve against,
    and the GitHub repo (`owner/name`, or "") that issue and PR numbers link to."""

    models: tuple[Model, ...]
    corpus: Corpus
    github: str = ""

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


def read_site(
    experiments: Path, specs: dict[str, DocumentSpec], github: str = ""
) -> Site:
    """Reads every document the specs declare into a model with its breaches.

    Args:
        experiments: The experiments folder.
        specs: The specs by document kind.
        github: The GitHub repo as `owner/name`, from `github_repo`; "" leaves
            issue and PR numbers as text.

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
    return Site(models, corpus, github)


def github_repo(remote: str | None) -> str:
    """The `owner/name` of a GitHub remote URL.

    Args:
        remote: The `origin` remote, as `git remote get-url` gives it.

    Returns:
        `owner/name`, or "" when the remote is absent or not on GitHub.
    """
    match = GITHUB_REMOTE.match((remote or "").strip())
    return f"{match.group(1)}/{match.group(2)}" if match else ""


def page_styles() -> str:
    """The CSS of every ported page kind, each file scoped to its own pages,
    and `layout.css`, scoped to the pages that draw its section bar.

    Returns:
        The rules of every `templates/*.css`, in file name order.
    """
    return "".join(
        path.read_text(encoding="utf-8") for path in sorted(TEMPLATES.glob("*.css"))
    )


def render_page(model: Model, site: Site) -> str:
    """Renders one document's page.

    Args:
        model: The document to render.
        site: Every model, for pages that show or link to others.

    Returns:
        The page's `<section class="page">` HTML.

    Raises:
        KeyError: If no view is registered for the model's kind.
    """
    if model.kind not in VIEWS:
        raise KeyError(f"no page view for document kind '{model.kind}'")
    page = VIEWS[model.kind](model, site)
    template = ENVIRONMENT.get_template(page.template)
    return template.render(page=page, **page.context)


def render_legend() -> str:
    """Renders the legend: every mark the pages draw, each drawn by the macro
    the pages use, with what it means.

    Returns:
        The legend's `<dialog>` HTML, for the page shell to hold once.
    """
    return ENVIRONMENT.get_template("legend.html").render().strip()


# ===============================
# Entry names
# ===============================


def entry_page_id(entry_id: str) -> str:
    """A reference entry's page anchor: `ref-<topic>-<entry>`."""
    return ENTRY_PAGE_PREFIX + entry_id.replace("/", "-")


def entry_title(model: Model) -> str:
    """A reference entry's title, else its file name."""
    return str(model.value("title") or model.document.path.stem)


def entry_ref(key: str, entries: Mapping[str, Model], role: str = "") -> dict:
    """How a page names a reference entry, for the `entry_link` macro.

    Args:
        key: The entry's id, `<topic>/<entry>`.
        entries: Every entry model by id.
        role: The role a `couples_to` item gives it, or "".

    Returns:
        Its title, page link, hover text (title and summary) and status
        square and word; an unknown key keeps only its text.
    """
    other = entries.get(key)
    if other is None:
        return {"title": key, "href": "", "tip": "", "square": "", "word": "", "role": role}
    status = other.value("status") or "draft"
    title = entry_title(other)
    summary = other.value("summary") or other.value("kind") or ""
    return {
        "title": title,
        "href": f"#{entry_page_id(key)}",
        "tip": f"{title}\n{summary}" if summary else title,
        "square": ENTRY_STATUS[status][0],
        "word": ENTRY_STATUS[status][1],
        "role": role,
    }


# ===============================
# Page views
# ===============================

# each module registers its page kind's view with `@view` as it is imported
from . import entry_page, glossary_page, overview_page, programme_page, record, topic_page  # noqa: E402, F401
