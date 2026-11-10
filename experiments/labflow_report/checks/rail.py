"""Checks the rail spec (`specs/rail.yaml`) names: the brand in the repo's CLAUDE.md.

The top bar shows `brand:` from the `## labflow` section of CLAUDE.md, else
the folder name. A brand is at most ten characters, so a folder name longer
than that needs one. The folder name is the repo folder holding
`experiments/`; a linked worktree, whose `.git` file points into
`<repo>/.git/worktrees/`, goes by its main checkout's folder, as the top bar
shows the repo's name there and not the worktree's suffix. `brand_line` reads
the first `brand:` line of the section that holds a value; an empty `brand:`
line names no brand.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

from ..document import Document
from ..rules import Corpus, check
from ..spec import Rule, Section

# ===============================
# Defaults
# ===============================

BRAND = re.compile(r"^brand:[ \t]*(\S.*?)[ \t]*$")
BRAND_CHARACTERS = 10
GITDIR = "gitdir:"
# a linked worktree's gitdir is `<repo>/.git/worktrees/<name>`
WORKTREE_PARTS = (".git", "worktrees")

# ===============================
# Brand checks
# ===============================


@check("brand-needed")
def brand_needed(
    document: Document, part: Section, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """A folder name longer than ten characters needs a `brand:` line in the section."""
    folder = folder_name(document.path.resolve().parent)
    if len(folder) <= BRAND_CHARACTERS or brand_line(document, part):
        return
    section = document.sections.get(part.heading)
    yield (section.line if section else 1), f"folder '{folder}' is {len(folder)} characters"


@check("brand-length")
def brand_length(
    document: Document, part: Section, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """The brand is at most ten characters."""
    found = brand_line(document, part)
    if found and len(found[1]) > BRAND_CHARACTERS:
        yield found[0], f"'{found[1]}' is {len(found[1])} characters"


# ===============================
# Helper functions
# ===============================


def brand_line(document: Document, part: Section) -> tuple[int, str] | None:
    """The first `brand:` line of the section that names a brand.

    Args:
        document: The CLAUDE.md.
        part: The `labflow` section the spec declares.

    Returns:
        The line number and the brand, or None when the section names none.
    """
    section = document.sections.get(part.heading)
    if section is None:
        return None
    for number, line in enumerate(section.body, start=section.line + 1):
        if match := BRAND.match(line):
            return number, match.group(1)
    return None


def folder_name(folder: Path) -> str:
    """The repo folder's name; a linked worktree's is its main checkout's.

    Args:
        folder: The folder holding CLAUDE.md and `experiments/`.

    Returns:
        The folder name the top bar falls back to.
    """
    git = folder / ".git"
    if not git.is_file():
        return folder.name
    text = git.read_text(encoding="utf-8").strip()
    if not text.startswith(GITDIR):
        return folder.name
    gitdir = (folder / text[len(GITDIR) :].strip()).resolve()
    if gitdir.parts[-3:-1] == WORKTREE_PARTS:
        return gitdir.parts[-4]
    return folder.name
