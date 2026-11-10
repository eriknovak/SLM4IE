"""The golden harness: each report page's structure against a stored golden (ADR-0005).

The harness builds a report with `build_report.py` as it stands, offline, and
reduces every page to its structure: the order of its sections, the count of
each component class and the count of each element. It compares that with
the golden stored in `golden/<case>.yaml`, so a builder change that moves,
drops or adds a component shows up page by page.

A case is the bundled fixture (`golden/fixture.yaml`, always available) or a
snapshot of an experiments repo, pinned in its golden by `repo` and `commit`.
The harness finds the repo's clone in a folder named by `LABFLOW_GOLDEN_REPOS`
(several folders joined by `:`), extracts that commit with `git archive` and
builds it; a case whose clone or commit is absent is skipped, with the reason.

    cd claude/skills/labflow/skills/report/references
    uv run --no-project --with pyyaml --with jinja2 --with markdown --with latex2mathml \\
        python -m labflow_report.golden check [CASE ...] [--screenshots DIR]
    ... python -m labflow_report.golden regenerate [CASE ...] [--commit SHA]

`regenerate` rewrites the goldens from the current output; with `--commit` it
pins one repo case to a new commit, or adds it. `--screenshots DIR` writes a
PNG of every page to `DIR/<case>/<page id>.png`, to set beside its design
board, when a headless Chromium is found (`LABFLOW_BROWSER` names one).
Exit codes: 0 every available case matches, 1 a page differs, 2 cannot run.
"""

from __future__ import annotations

import argparse
import datetime
import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

import yaml

from . import ship

# ===============================
# Defaults
# ===============================

REFERENCES = Path(__file__).resolve().parents[1]
GOLDEN = REFERENCES / "golden"
FIXTURE = REFERENCES / "fixture" / "experiments"
BUILD_REPORT = REFERENCES / "build_report.py"
FIXTURE_CASE = "fixture"
GOLDEN_HEADER = (
    "# written by `python -m labflow_report.golden regenerate`; do not edit\n"
)

REPOS_ENV = "LABFLOW_GOLDEN_REPOS"
BROWSER_ENV = "LABFLOW_BROWSER"
BROWSER_NAMES = (
    "chromium",
    "chromium-browser",
    "google-chrome",
    "google-chrome-stable",
    "chrome-headless-shell",
)
PLAYWRIGHT_BROWSERS = (
    "chromium-*/chrome-linux*/chrome",
    "chromium_headless_shell-*/chrome-headless-shell-*/chrome-headless-shell",
)
SCREEN_SIZE = "1440,1000"
# seconds a browser may take to load the report and screenshot one page
SCREENSHOT_TIMEOUT = 60

# build_report.py exits 1 when a record has a hole; the report is written anyway
BUILD_EXIT_CODES = (0, 1)

# elements whose insides are drawings or equations, not page structure
OPAQUE = ("svg", "math")
SECTION_CLASS = "sec"
SECTION_HEADING = "h2"
PAGE_CLASS = "page"
# the rail's root element: `nav.side`, or `div.sidecol2` in the redesigned report
RAIL_ROOTS = (("nav", "side"), ("div", "sidecol2"))
RAIL = "rail"
# the page kind an id names, for pages without `data-page`
PAGE_KINDS = {"overview": "overview", "glossary": "glossary"}
KIND_PREFIXES = (("programme-", "programme"), ("topic-", "topic"), ("ref-", "entry"))
RECORD = "record"


class Unavailable(Exception):
    """A case's snapshot cannot be found here; the message says why."""


# ===============================
# Golden classes
# ===============================


@dataclass(frozen=True)
class Page:
    """The structure of one page: section order, component classes, element counts."""

    id: str
    kind: str
    sections: tuple[str, ...] = ()
    classes: Mapping[str, int] = field(default_factory=dict)
    elements: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Golden:
    """One case: where its experiments come from, and the structure of each page.

    A case with no `repo` is the bundled fixture.
    """

    case: str
    repo: str | None = None
    commit: str | None = None
    pages: tuple[Page, ...] = ()


# ===============================
# Main functions
# ===============================


def page_structure(html: str) -> tuple[Page, ...]:
    """Reduces a built report to the structure of each page.

    A page is each `<section class="page">`, in report order, then the rail
    (`<nav class="side">` or `<div class="sidecol2">`) as the page `rail`. Sections are the ids of
    `h2` headings and `.sec` blocks, without the page id prefix. The insides
    of `svg` and `math` elements are not counted.

    Args:
        html: The report's HTML.

    Returns:
        The pages, in report order, the rail last.
    """
    parser = _StructureParser()
    parser.feed(html)
    parser.close()
    pages = parser.pages
    return tuple(p for p in pages if p.id != RAIL) + tuple(
        p for p in pages if p.id == RAIL
    )


def compare(expected: Sequence[Page], actual: Sequence[Page]) -> list[str]:
    """Lists every structural difference between two reports, one line each.

    Args:
        expected: The golden's pages.
        actual: The pages of the report just built.

    Returns:
        Lines shaped `<page id>: <what differs>`; empty when they match.
    """
    differences = []
    have = {p.id: p for p in actual}
    want = {p.id: p for p in expected}

    # the page set and its order
    differences += [f"{i}: page missing" for i in want if i not in have]
    differences += [f"{i}: page not in the golden" for i in have if i not in want]
    common_want = [i for i in want if i in have]
    common_have = [i for i in have if i in want]
    if common_want != common_have:
        differences.append(
            f"pages: order {', '.join(common_want)} → {', '.join(common_have)}"
        )

    # each page present in both
    for page_id in common_want:
        old, new = want[page_id], have[page_id]
        if old.kind != new.kind:
            differences.append(f"{page_id}: kind {old.kind} → {new.kind}")
        if old.sections != new.sections:
            differences.append(
                f"{page_id}: sections {', '.join(old.sections) or '(none)'}"
                f" → {', '.join(new.sections) or '(none)'}"
            )
        differences += _count_differences(page_id, "class", old.classes, new.classes)
        differences += _count_differences(
            page_id, "element", old.elements, new.elements
        )
    return differences


def read_golden(path: Path) -> Golden:
    """Reads a golden file.

    Args:
        path: `golden/<case>.yaml`; the case is its stem.

    Returns:
        The golden.

    Raises:
        ValueError: If the file names a repo without a commit.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if data.get("repo") and not data.get("commit"):
        raise ValueError(f"golden {path} names repo '{data['repo']}' but no commit")
    pages = tuple(
        Page(
            id=p["id"],
            kind=p["kind"],
            sections=tuple(p.get("sections") or ()),
            classes=dict(p.get("classes") or {}),
            elements=dict(p.get("elements") or {}),
        )
        for p in data.get("pages") or ()
    )
    return Golden(path.stem, data.get("repo"), data.get("commit"), pages)


def write_golden(golden: Golden, folder: Path) -> Path:
    """Writes a golden to `<folder>/<case>.yaml`, keys sorted so diffs stay small.

    Args:
        golden: The golden to write.
        folder: The golden folder.

    Returns:
        The path written.
    """
    data = {}
    if golden.repo:
        data |= {"repo": golden.repo, "commit": golden.commit}
    data["pages"] = [
        {
            "id": p.id,
            "kind": p.kind,
            "sections": list(p.sections),
            "classes": dict(sorted(p.classes.items())),
            "elements": dict(sorted(p.elements.items())),
        }
        for p in golden.pages
    ]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{golden.case}.yaml"
    text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100)
    path.write_text(GOLDEN_HEADER + text, encoding="utf-8")
    return path


def build_case(golden: Golden, workdir: Path, search: str | None) -> Path:
    """Builds a case's report offline in a working folder outside any checkout.

    The fixture is copied; a repo case is extracted at its pinned commit from
    the first clone found under `search`. `build_report.py` builds the folder
    it sits in, so the builder (`build_report.py` and `labflow_report/`) is
    shipped into the case's `experiments/` over whatever copy it holds.

    Args:
        golden: The case; only its name, repo and commit are read.
        workdir: An empty folder the build may write to.
        search: Folders holding clones named by repo, joined by `os.pathsep`
            (the value of `LABFLOW_GOLDEN_REPOS`), or None.

    Returns:
        The built report's path.

    Raises:
        Unavailable: If the repo's clone or commit is not found.
        RuntimeError: If `build_report.py` fails or writes no report.
    """
    # place the experiments folder under a folder named for the case
    root = workdir / golden.case
    if golden.repo is None:
        shutil.copytree(
            FIXTURE,
            root / "experiments",
            ignore=shutil.ignore_patterns("report.html", "__pycache__"),
        )
    else:
        clone = find_clone(golden.repo, golden.commit, search)
        archive = subprocess.run(
            ["git", "-C", str(clone), "archive", "--format=tar", golden.commit],
            capture_output=True,
            check=True,
        ).stdout
        root.mkdir(parents=True)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(root, filter="data")
    experiments = root / "experiments"
    if not experiments.is_dir():
        raise Unavailable(f"{golden.case}: commit {golden.commit} has no experiments/")

    # ship the builder into the experiments folder, which it reads; a copy the
    # snapshot holds goes first, so an edited one never wins
    if (experiments / ship.PACKAGE).is_dir():
        shutil.rmtree(experiments / ship.PACKAGE)
    ship.ship(REFERENCES, experiments, datetime.date.today())
    script = experiments / BUILD_REPORT.name
    out = workdir / f"{golden.case}.html"
    result = subprocess.run(
        [sys.executable, str(script), "--out", str(out), "--offline"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in BUILD_EXIT_CODES or not out.is_file():
        raise RuntimeError(
            f"{golden.case}: build_report.py exited {result.returncode}\n"
            f"{result.stderr[-2000:]}"
        )
    return out


def find_clone(repo: str, commit: str, search: str | None) -> Path:
    """Finds a clone of `repo` that holds `commit`.

    Args:
        repo: The repo's folder name.
        commit: The pinned commit sha.
        search: Folders to look in, joined by `os.pathsep`, or None.

    Returns:
        The clone's path.

    Raises:
        Unavailable: If no folder holds a clone with that commit.
    """
    if not search:
        raise Unavailable(f"{repo}: set {REPOS_ENV} to a folder holding a clone")
    folders = [Path(f).expanduser() for f in search.split(os.pathsep) if f]
    clones = [f / repo for f in folders if (f / repo).is_dir()]
    for clone in clones:
        found = subprocess.run(
            ["git", "-C", str(clone), "cat-file", "-e", f"{commit}^{{commit}}"],
            capture_output=True,
            check=False,
        )
        if found.returncode == 0:
            return clone
    if clones:
        where = ", ".join(map(str, clones))
        raise Unavailable(f"{repo}: no clone holds commit {commit} ({where})")
    raise Unavailable(f"{repo}: no clone under {REPOS_ENV}={search}")


def find_browser(environ: Mapping[str, str] = os.environ) -> str | None:
    """Finds a Chromium that can screenshot headless.

    Looks at `LABFLOW_BROWSER`, then the PATH, then Playwright's browser cache.

    Args:
        environ: The environment to read.

    Returns:
        The browser's path, or None when there is none.
    """
    if browser := environ.get(BROWSER_ENV):
        return browser
    for name in BROWSER_NAMES:
        if found := shutil.which(name, path=environ.get("PATH")):
            return found
    cache = Path(
        environ.get("PLAYWRIGHT_BROWSERS_PATH")
        or Path.home() / ".cache" / "ms-playwright"
    )
    for pattern in PLAYWRIGHT_BROWSERS:
        found = sorted(p for p in cache.glob(pattern) if os.access(p, os.X_OK))
        if found:
            return str(found[-1])
    return None


def screenshot(report: Path, page_id: str, out: Path, browser: str) -> Path:
    """Screenshots one page of a report at the design boards' width.

    Args:
        report: The built report.
        page_id: The page to show, routed by the report's `#<page id>`.
        out: The PNG to write.
        browser: A Chromium binary.

    Returns:
        The PNG's path.

    Raises:
        RuntimeError: If the browser writes no PNG.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            browser,
            "--headless",
            "--no-sandbox",  # the report is a local file the harness built
            "--disable-gpu",
            "--hide-scrollbars",
            f"--window-size={SCREEN_SIZE}",
            "--virtual-time-budget=3000",  # let the script route to the page
            f"--screenshot={out}",
            f"{report.resolve().as_uri()}#{page_id}",
        ],
        capture_output=True,
        text=True,
        timeout=SCREENSHOT_TIMEOUT,
        check=False,
    )
    if result.returncode != 0 or not out.is_file():
        raise RuntimeError(
            f"{browser} wrote no screenshot of {page_id} (exit {result.returncode})"
            f"\n{result.stderr[-1000:]}"
        )
    return out


# ===============================
# Helper functions
# ===============================


class _StructureParser(HTMLParser):
    """Collects each page's sections and its class and element counts."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.pages: list[Page] = []
        self._page: dict | None = None
        self._root = ""
        self._depth = 0
        self._opaque = 0

    def handle_starttag(self, tag, attrs):
        self._open(tag, dict(attrs), closed=False)

    def handle_startendtag(self, tag, attrs):
        self._open(tag, dict(attrs), closed=True)

    def handle_endtag(self, tag):
        if self._page is None:
            return
        if self._opaque:
            if tag in OPAQUE:
                self._opaque -= 1
            return
        if tag == self._root:
            self._depth -= 1
            if self._depth == 0:
                self._close()

    def _open(self, tag: str, attrs: dict, closed: bool) -> None:
        classes = (attrs.get("class") or "").split()

        # start a page at its root element
        if self._page is None:
            if tag == "section" and PAGE_CLASS in classes:
                page_id = attrs.get("id") or ""
                self._start(tag, page_id, attrs.get("data-page") or _kind(page_id))
            elif any(tag == name and rail in classes for name, rail in RAIL_ROOTS):
                self._start(tag, RAIL, RAIL)
            return

        # skip the insides of drawings and equations
        if self._opaque:
            if tag in OPAQUE and not closed:
                self._opaque += 1
            return
        if tag in OPAQUE and not closed:
            self._opaque = 1

        # count the element and note a section
        if tag == self._root and not closed:
            self._depth += 1
        self._page["elements"][tag] += 1
        self._page["classes"].update(classes)
        element_id = attrs.get("id")
        if element_id and (tag == SECTION_HEADING or SECTION_CLASS in classes):
            prefix = f"{self._page['id']}/"
            self._page["sections"].append(element_id.removeprefix(prefix))

    def _start(self, tag: str, page_id: str, kind: str) -> None:
        self._root, self._depth = tag, 1
        self._page = {
            "id": page_id,
            "kind": kind,
            "sections": [],
            "classes": Counter(),
            "elements": Counter(),
        }

    def _close(self) -> None:
        page = self._page
        self.pages.append(
            Page(
                id=page["id"],
                kind=page["kind"],
                sections=tuple(page["sections"]),
                classes=dict(page["classes"]),
                elements=dict(page["elements"]),
            )
        )
        self._page = None


def _kind(page_id: str) -> str:
    """The page kind an id names, for a page without `data-page`."""
    if page_id in PAGE_KINDS:
        return PAGE_KINDS[page_id]
    return next(
        (kind for prefix, kind in KIND_PREFIXES if page_id.startswith(prefix)), RECORD
    )


def _count_differences(
    page_id: str, what: str, old: Mapping[str, int], new: Mapping[str, int]
) -> list[str]:
    """One line per name whose count differs: `<page>: <what> <name> <old> → <new>`."""
    names = sorted(set(old) | set(new))
    return [
        f"{page_id}: {what} {name} {old.get(name, 0)} → {new.get(name, 0)}"
        for name in names
        if old.get(name, 0) != new.get(name, 0)
    ]


def _cases(names: Iterable[str], folder: Path) -> list[Golden]:
    """The goldens to run: the named ones, or every golden in the folder."""
    names = list(names)
    if not names:
        return [read_golden(p) for p in sorted(folder.glob("*.yaml"))]
    goldens = []
    for name in names:
        path = folder / f"{name}.yaml"
        if path.is_file():
            goldens.append(read_golden(path))
        elif name == FIXTURE_CASE:
            goldens.append(Golden(FIXTURE_CASE))
        else:
            raise FileNotFoundError(f"no golden {path}; pin it with --commit")
    return goldens


def _screenshots(report: Path, golden: Golden, pages, folder: Path) -> None:
    """Writes a PNG of every page to `<folder>/<case>/`, when a browser is found."""
    browser = find_browser()
    if browser is None:
        print(f"  {golden.case}: no headless browser, screenshots skipped")
        return
    for page in pages:
        screenshot(report, page.id, folder / golden.case / f"{page.id}.png", browser)
    print(f"  {golden.case}: {len(pages)} screenshots in {folder / golden.case}")


# ===============================
# Main function
# ===============================


def get_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m labflow_report.golden",
        description="Compare each report page's structure with its stored golden.",
    )
    parser.add_argument(
        "command",
        choices=("check", "regenerate"),
        help="check: compare with the goldens; regenerate: rewrite them",
    )
    parser.add_argument(
        "cases",
        nargs="*",
        help="case names, the golden file stems (default: every golden)",
    )
    parser.add_argument(
        "--commit",
        help="regenerate only: pin the one named repo case to this commit sha",
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=GOLDEN,
        help=f"the golden folder (default: {GOLDEN})",
    )
    parser.add_argument(
        "--screenshots",
        type=Path,
        help="write a PNG of every page to DIR/<case>/<page id>.png",
    )
    return parser.parse_args(argv)


def main(args: argparse.Namespace) -> int:
    if args.commit and (args.command != "regenerate" or len(args.cases) != 1):
        print("--commit needs `regenerate` and exactly one case", file=sys.stderr)
        return 2

    # read the cases to run
    try:
        if args.commit:
            name = args.cases[0]
            goldens = [Golden(name, repo=name, commit=args.commit)]
        else:
            goldens = _cases(args.cases, args.golden)
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(error, file=sys.stderr)
        return 2
    if not goldens:
        print(f"no goldens in {args.golden}", file=sys.stderr)
        return 2

    # build each case and compare or store its structure
    search = os.environ.get(REPOS_ENV)
    failed = False
    for golden in goldens:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                report = build_case(golden, Path(tmp), search)
                pages = page_structure(report.read_text(encoding="utf-8"))
                if args.screenshots:
                    _screenshots(report, golden, pages, args.screenshots)
            except Unavailable as reason:
                print(f"  skip {reason}")
                continue
            except (RuntimeError, subprocess.SubprocessError) as error:
                print(f"  FAIL {golden.case}: {error}", file=sys.stderr)
                return 2
        if args.command == "regenerate":
            path = write_golden(
                Golden(golden.case, golden.repo, golden.commit, pages), args.golden
            )
            print(f"  wrote {path}: {len(pages)} pages")
            continue
        differences = compare(golden.pages, pages)
        failed |= bool(differences)
        print(f"  {'FAIL' if differences else 'ok  '} {golden.case}")
        for line in differences:
            print(f"    {line}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(get_args()))
