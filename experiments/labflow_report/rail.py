"""The rail on the new builder: the sidebar and top bar every page shares (ADR-0005).

`rail_view` turns the site into the values `templates/components/rail.html`
draws (`specs/rail.yaml`): an icon strip of tabs and one list per tab.
Overview lists the programmes and what changed lately; Research each
programme's members in `members:` order, then its planned experiments, then
the records no programme lists; Reference each topic with its entries, shut
until the open page sits in it; Glossary its sections. Each list opens with
a head linking the tab's landing page, and a click on a tab opens that page.
`render_rail` and `render_top` render the rail and the top bar beside it.

Effects stay at the edge: `running_slugs` asks git which experiments run and
the caller passes the answer in, so the view itself reads only the site.
"""

from __future__ import annotations

import datetime
import subprocess
from pathlib import Path

from . import programme_page as programmes
from . import record as records
from .builder import (
    ENTRY_STATUS,
    ENVIRONMENT,
    OUTCOME,
    Model,
    Site,
    entry_page_id,
    entry_title,
)
from .checks import programme as programme_checks
from .checks.glossary import terms
from .checks.rail import brand_line
from .glossary_page import section_counts, section_id

# ===============================
# Defaults
# ===============================

TEMPLATE = "components/rail.html"
# concluded this many days before the newest conclusion: a new result
RAIL_NEW_DAYS = 14
# tab → its title, what its filter filters, its landing page and its icon path
TABS = {
    "overview": ("Overview", "pages", "overview", "M3 9.5 10 4l7 5.5V16H3z"),
    "research": ("Research", "experiments", "research", "M8 3h4M9 3v5l-5 8h12l-5-8V3"),
    "reference": (
        "Reference",
        "entries",
        "reference",
        "M4 4h5a2 2 0 0 1 2 2v10a2 2 0 0 0-2-2H4zM16 4h-3a2 2 0 0 0-2 2v10a2 2 0 0 1 2-2h3z",
    ),
    "terms": ("Glossary", "terms", "glossary", "M3 15 6.5 5 10 15M4.3 11.5h4.4M12 9h3.5a2 2 0 0 1 0 4H12V5h3a2 2 0 0 1 0 4"),
}
REFERENCE = "reference"
PROGRAMMES = "Programmes"
NOT_IN_A_PROGRAMME = "Not in a programme"
EXPERIMENTS = "Experiments"
GLOSSARY_GROUP = "t-glossary"
RUNNING = "running"
PLANNED = "planned"
# a record that ended with no outcome word shows an empty slot, never planned
ENDED = ("concluded", "abandoned")
# a running experiment's worktree sits here under the main checkout
WORKTREES = Path(".claude") / "worktrees"
# an experiment's branch, local or on any remote
EXPERIMENT_BRANCHES = ("refs/heads/exp/*", "refs/remotes/*/exp/*")
EXPERIMENT_PREFIX = "exp/"

# ===============================
# Main functions
# ===============================


def rail_view(site: Site, running: frozenset[str] = frozenset()) -> list[dict]:
    """Builds the rail's values from the site: one list per tab, Overview first.

    Args:
        site: Every document: the programmes, records, topics, entries and
            the glossary.
        running: The slugs of the experiments that run, from `running_slugs`.

    Returns:
        One panel per tab, each with its tab id, title, filter word, landing
        page id and links; Reference only when `experiments/reference/` exists.
    """
    record_models = site.of_kind("record")
    since, changed = changed_since(record_models)
    new = {records.record_slug(r) for r in changed}
    lists = {"overview": overview_items(site, since, changed, running)}
    lists["research"] = research_items(site, running, new)
    if (site.corpus.experiments / REFERENCE).is_dir():
        lists["reference"] = reference_items(site)
    lists["terms"] = glossary_items(site)
    return [
        {"tab": tab, "title": TABS[tab][0], "filter": TABS[tab][1], "landing": TABS[tab][2], "icon": TABS[tab][3], "links": links}
        for tab, links in lists.items()
    ]


def render_rail(site: Site, running: frozenset[str] = frozenset()) -> str:
    """Renders the rail: the tab strip, one list per tab and the key hints.

    Args:
        site: Every document.
        running: The slugs of the experiments that run, from `running_slugs`.

    Returns:
        The rail's HTML, for the page shell's `<nav class="side">`.
    """
    module = ENVIRONMENT.get_template(TEMPLATE).module
    return str(module.rail(rail_view(site, running))).strip()


def render_top(brand: str, repo_url: str | None = None) -> str:
    """Renders the top bar: the rail toggle, the brand, the crumb, the legend and theme buttons.

    Args:
        brand: The short repo name, from `brand_name`.
        repo_url: The repo's https URL; the brand links it when given.

    Returns:
        The top bar's HTML.
    """
    module = ENVIRONMENT.get_template(TEMPLATE).module
    return str(module.top_bar(brand, repo_url or "")).strip()


def brand_name(site: Site, repo_url: str | None = None) -> str:
    """The top bar's short name: `brand:` under `## labflow` in CLAUDE.md, read as
    the rail check reads it, else the repo name of origin, else the repo folder.

    Args:
        site: Every document; its `claude` model is the repo's CLAUDE.md.
        repo_url: The repo's https URL, or None outside a checkout with a remote.

    Returns:
        The brand.
    """
    claude = next(iter(site.of_kind("claude")), None)
    if claude is not None and (found := brand_line(claude.document, claude.spec.part("section.labflow"))):
        return found[1]
    # a worktree folder carries a suffix the repo name has not
    if repo_url:
        return repo_url.rstrip("/").rsplit("/", 1)[-1]
    return site.corpus.experiments.resolve().parent.name


def running_slugs(experiments: Path) -> frozenset[str]:
    """The experiments that run: an open worktree under the main checkout's
    `.claude/worktrees`, or an `exp/<slug>` branch, local or remote, that HEAD
    has not merged.

    Args:
        experiments: The experiments folder.

    Returns:
        Their slugs; none outside a git checkout.
    """
    # the main checkout's root, also from inside a linked worktree
    common = _git(experiments, "rev-parse", "--path-format=absolute", "--git-common-dir")
    root = Path(common).parent if common else experiments.parent
    folder = root / WORKTREES
    found = {p.name for p in folder.iterdir() if p.is_dir()} if folder.is_dir() else set()

    # unmerged experiment branches
    branches = _git(experiments, "for-each-ref", "--no-merged=HEAD", "--format=%(refname:short)", *EXPERIMENT_BRANCHES)
    for name in branches.splitlines():
        _, _, slug = name.partition(EXPERIMENT_PREFIX)
        if slug:
            found.add(slug)
    return frozenset(found)


# ===============================
# Lists
# ===============================


def overview_items(site: Site, since: datetime.date | None, changed: list[Model], running: frozenset[str]) -> list[dict]:
    """The Overview list: its landing head, the programmes, then Changed since."""
    items = [landing_head("overview")]
    programme_models = site.of_kind("programme")
    if programme_models:
        items.append(group_head(PROGRAMMES, "o-programmes"))
        for programme in programme_models:
            names = programme_names(programme)
            items.append(item(names["pid"], names["short"], names["title"], group="o-programmes"))
    if changed:
        items.append(group_head(f"Changed since {since.isoformat()[5:]}", "o-changed", title=f"concluded after {since.isoformat()}"))
        for record in changed:
            short, title = record_names(record)
            items.append(
                item(records.record_slug(record), short, title, word=rail_word(record, running), group="o-changed", new=True)
            )
    return items


def research_items(site: Site, running: frozenset[str], new: set[str]) -> list[dict]:
    """The Research list: its landing head, each programme's members in `members:`
    order and its planned experiments, then the records no programme lists."""
    by_path = {r.document.path.resolve(): r for r in site.of_kind("record")}
    slugs = {records.record_slug(r) for r in by_path.values()}

    def experiment(record: Model, group: str) -> dict:
        short, title = record_names(record)
        slug = records.record_slug(record)
        return item(slug, short, title, word=rail_word(record, running), group=group, new=slug in new, experiment=True)

    # each programme: members, then the planned experiments no record has taken
    items, listed = [landing_head("research")], set()
    for programme in site.of_kind("programme"):
        names = programme_names(programme)
        items.append(group_head(names["short"], names["pid"], names["pid"], names["title"]))
        member_records = [by_path[d.path.resolve()] for _, d in programme_checks.members(programme.document, site.corpus)]
        member_records = [r for n, r in enumerate(member_records) if r not in member_records[:n]]
        items += [experiment(r, names["pid"]) for r in member_records]
        listed |= {r.document.path.resolve() for r in member_records}
        for planned in programmes.read_planned(programme, slugs):
            word = RUNNING if planned.slug in running else PLANNED
            hover = planned.title or planned.short
            items.append(item(names["pid"], planned.short, hover[:1].upper() + hover[1:], word=word, group=names["pid"]))

    # the records no programme lists, in conclusion order
    loose = sorted(
        (r for path, r in by_path.items() if path not in listed),
        key=lambda r: (r.value("concluded") is None, r.value("concluded") or datetime.date.min, records.record_slug(r)),
    )
    if loose:
        items.append(group_head(NOT_IN_A_PROGRAMME if site.of_kind("programme") else EXPERIMENTS, "r-loose"))
        items += [experiment(r, "r-loose") for r in loose]
    return items


def reference_items(site: Site) -> list[dict]:
    """The Reference list: its landing head, then every topic, shut, with its entries."""
    entries = sorted(site.of_kind("entry"), key=lambda m: m.id)
    topic_models = {m.id: m for m in site.of_kind("topic")}
    topics = sorted(set(topic_models) | {m.id.split("/", 1)[0] for m in entries})
    items = [landing_head("reference")]
    for topic in topics:
        model = topic_models.get(topic)
        title = str((model.value("title") if model else None) or topic.replace("-", " ").capitalize())
        pid = f"topic-{topic}"
        items.append(group_head(title, pid, pid, shut=True))
        for entry in (m for m in entries if m.id.startswith(f"{topic}/")):
            name = entry_title(entry)
            square = ENTRY_STATUS.get(entry.value("status") or "draft", ENTRY_STATUS["draft"])[0]
            items.append(item(entry_page_id(entry.id), name, str(entry.value("summary") or name), square=square, group=pid))
    return items


def glossary_items(site: Site) -> list[dict]:
    """The Glossary list: the Glossary head, then Metrics, each term group and Other."""
    glossary = next(iter(site.of_kind("glossary")), None)
    if glossary is None:
        return []
    items = [group_head("Glossary", GLOSSARY_GROUP, "glossary")]
    items += [item(section_id(name), name, name, group=GLOSSARY_GROUP) for name in section_counts(terms(glossary.document))]
    return items


# ===============================
# Marks and names
# ===============================


def rail_word(record: Model, running: frozenset[str]) -> str:
    """The word a record's mark shows: dropped for a dropped exploration; its
    outcome; running while it runs; planned while a draft; "" when it ended
    with no outcome word.

    Args:
        record: The record.
        running: The slugs of the experiments that run.

    Returns:
        A word `OUTCOME` maps to a mark, or "".
    """
    if record.stage == records.DROPPED:
        return records.DROPPED
    if outcome := records.record_outcome(record):
        return outcome if outcome in OUTCOME else ""
    # an abandoned record keeps its exp/ branch, so the end wins over running
    status = record.value("status")
    if status in ENDED:
        return ""
    if records.record_slug(record) in running or status == RUNNING:
        return RUNNING
    return PLANNED


def changed_since(record_models: tuple[Model, ...] | list[Model]) -> tuple[datetime.date | None, list[Model]]:
    """The new results: the records concluded in the RAIL_NEW_DAYS before the
    newest conclusion, oldest first.

    Args:
        record_models: Every record.

    Returns:
        The day they count from, and the records; None and [] when none concluded.
    """
    dated = sorted(
        ((r.value("concluded"), records.record_slug(r), r) for r in record_models if r.value("concluded")),
        key=lambda x: x[:2],
    )
    if not dated:
        return None, []
    since = dated[-1][0] - datetime.timedelta(days=RAIL_NEW_DAYS)
    return since, [r for day, _, r in dated if day > since]


def record_names(record: Model) -> tuple[str, str]:
    """A record's short name and its title for the hover."""
    title = str(record.value("title") or records.record_slug(record))
    return str(record.value("short") or title), title


def programme_names(programme: Model) -> dict:
    """A programme's page id, short name and title."""
    slug = str(programme.value("slug") or programme.document.path.stem)
    title = str(programme.value("title") or slug)
    return {"pid": f"programme-{slug}", "short": str(programme.value("short") or title), "title": title}


# ===============================
# Helper functions
# ===============================


def landing_head(tab: str) -> dict:
    # a list's first head: the tab's title, linking its landing page
    return {"kind": "landing", "label": TABS[tab][0], "href": TABS[tab][2]}


def group_head(label: str, group: str, href: str = "", title: str = "", shut: bool = False) -> dict:
    # a group's head: a link when it names a page; its chevron opens and closes the group
    return {"kind": "group", "label": label, "group": group, "href": href, "title": title or label, "shut": shut}


def item(
    href: str,
    label: str,
    title: str,
    *,
    word: str = "",
    square: str = "",
    group: str = "",
    new: bool = False,
    experiment: bool = False,
) -> dict:
    # a link: its mark, its label cut to one line, the full title on hover, the new-result dot
    return {
        "kind": "item",
        "href": href,
        "label": label,
        "title": title or label,
        "word": word,
        "square": square,
        "group": group,
        "new": new,
        "experiment": experiment,
    }


def _git(folder: Path, *args: str) -> str:
    # a git answer, or "" outside a checkout or without git
    try:
        result = subprocess.run(["git", *args], cwd=folder, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()

