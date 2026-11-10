"""Ship the report builder into a repo's `experiments/` with a version stamp (ADR-0005).

    python3 ~/.claude/skills/labflow/skills/report/references/labflow_report/ship.py experiments

The builder is `build_report.py` plus the `labflow_report/` package beside
it. Shipping copies both from the skill's references folder into the repo's
experiments folder and writes `labflow_report/shipped.toml`, whose `version`
is a hash of the shipped files. One command serves `labflow:init` and
`labflow:report`:

- no copy yet: install it, replacing an unstamped `build_report.py`;
- a copy equal to the skill's: leave it, stamping it when the stamp is gone;
- a copy that still hashes to its stamp: refresh it, since it was never edited;
- any other copy: leave it alone and warn, since it was edited.

The checker calls `stale_warning` to warn on a copy that is edited or behind.
This file uses the standard library only, so it runs as a plain script.

Exit codes: 0 shipped, current or left alone with a warning, 2 cannot run.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import tomllib
from argparse import ArgumentParser
from datetime import date
from pathlib import Path

# ===============================
# Defaults
# ===============================

BUILD_REPORT = "build_report.py"
PACKAGE = "labflow_report"
STAMP = "shipped.toml"
# caches stay behind: __pycache__, *.pyc and any dot folder or file (.ruff_cache)
NOT_SHIPPED = ("__pycache__", "*.pyc", ".*", STAMP)
SKILL_REFERENCES = Path(".claude/skills/labflow/skills/report/references")
SHORT_VERSION = 12
EXIT_ERROR = 2

INSTALLED = "installed"
REFRESHED = "refreshed"
CURRENT = "current"
EDITED = "edited"
ABSENT = "absent"
UNEDITED = "unedited"

# ===============================
# Main functions
# ===============================


def ship(source: Path, experiments: Path, today: date) -> tuple[str, str]:
    """Installs or refreshes the builder copy in an experiments folder.

    Args:
        source: The folder holding the shipped `build_report.py` and package.
        experiments: The repo's experiments folder.
        today: The date written into a new stamp.

    Returns:
        The action (`INSTALLED`, `REFRESHED`, `CURRENT` or `EDITED`) and a
        one-line message for the session; an `EDITED` message is a warning.

    Raises:
        FileNotFoundError: If the experiments folder does not exist.
        ValueError: If the experiments folder is the source itself.
    """
    if not experiments.is_dir():
        raise FileNotFoundError(f"no experiments folder at {experiments}")
    if experiments.resolve() == source.resolve():
        raise ValueError(f"{experiments} is its own source; pass a repo's experiments")

    version = shipped_hash(source)
    state = copy_state(experiments, version)
    package = experiments / PACKAGE

    if state == CURRENT:
        if read_version(experiments) != version:
            _write_stamp(experiments, version, today)
        return (
            CURRENT,
            f"the report builder in {experiments} is current ({version[:SHORT_VERSION]})",
        )
    if state == EDITED:
        return EDITED, (
            f"warning: {package} was edited after it was shipped, so it was left "
            f"alone; to take the shipped version, move the edits into the skill, "
            f"delete {package} and rerun"
        )

    # copy the builder over the absent or unedited copy, the stamp last
    old_version = read_version(experiments)
    has_old_entry = (experiments / BUILD_REPORT).is_file()
    if package.exists():
        shutil.rmtree(package)
    shutil.copytree(
        source / PACKAGE, package, ignore=shutil.ignore_patterns(*NOT_SHIPPED)
    )
    shutil.copy2(source / BUILD_REPORT, experiments / BUILD_REPORT)
    _write_stamp(experiments, version, today)

    new = version[:SHORT_VERSION]
    if state == UNEDITED:
        return REFRESHED, (
            f"refreshed the report builder in {experiments}: "
            f"{old_version[:SHORT_VERSION]} -> {new}"
        )
    replaced = " (replaced an unstamped build_report.py)" if has_old_entry else ""
    return INSTALLED, f"installed the report builder {new} into {experiments}{replaced}"


def stale_warning(experiments: Path, source: Path) -> str | None:
    """Says whether a repo's builder copy is edited or behind the skill's.

    Args:
        experiments: The repo's experiments folder.
        source: The skill's references folder; when it holds no builder (no
            dotfiles on this machine), only an edited copy is reported.

    Returns:
        A one-line warning, or None for no copy, a current copy, or an
        unedited copy with no source to compare against.
    """
    if not (experiments / PACKAGE).is_dir():
        return None

    has_source = (source / BUILD_REPORT).is_file() and (source / PACKAGE).is_dir()
    version = shipped_hash(source) if has_source else None
    state = copy_state(experiments, version)
    package = experiments / PACKAGE

    if state == EDITED:
        return (
            f"warning: {package} was edited after it was shipped, so "
            f"labflow:report will not refresh it and it goes stale"
        )
    if state == UNEDITED and version is not None:
        old = read_version(experiments)[:SHORT_VERSION]
        return (
            f"warning: {package} is stale (shipped {old}, current "
            f"{version[:SHORT_VERSION]}); labflow:report refreshes it"
        )
    return None


def copy_state(experiments: Path, source_version: str | None) -> str:
    """Classifies the builder copy in an experiments folder.

    Args:
        experiments: The repo's experiments folder.
        source_version: The hash of the skill's builder, or None if unknown.

    Returns:
        `ABSENT` (no package), `CURRENT` (equal to the source), `UNEDITED`
        (equal to its own stamp) or `EDITED` (anything else).
    """
    if not (experiments / PACKAGE).is_dir():
        return ABSENT
    version = shipped_hash(experiments)
    if version == source_version:
        return CURRENT
    if version == read_version(experiments):
        return UNEDITED
    return EDITED


def shipped_hash(folder: Path) -> str:
    """Hashes the shipped files in a folder: the entry script and the package.

    Args:
        folder: A skill references folder or a repo's experiments folder.

    Returns:
        The SHA-256 hex digest over each file's relative path and content.
    """
    files = [folder / BUILD_REPORT] if (folder / BUILD_REPORT).is_file() else []
    files += sorted(
        path
        for path in (folder / PACKAGE).rglob("*")
        if path.is_file() and _is_shipped(path.relative_to(folder))
    )
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(folder).as_posix().encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def read_version(experiments: Path) -> str | None:
    """Reads the version from a copy's stamp.

    Args:
        experiments: The repo's experiments folder.

    Returns:
        The stamped hash, or None when the stamp is missing or unreadable,
        so the copy counts as edited.
    """
    stamp = experiments / PACKAGE / STAMP
    if not stamp.is_file():
        return None
    try:
        version = tomllib.loads(stamp.read_text(encoding="utf-8")).get("version")
    except (tomllib.TOMLDecodeError, UnicodeDecodeError):
        return None
    return version if isinstance(version, str) else None


def skill_references() -> Path:
    """The skill's references folder under the current home directory."""
    return Path.home() / SKILL_REFERENCES


# ===============================
# Helper functions
# ===============================


def _is_shipped(relative: Path) -> bool:
    # the same names shutil.ignore_patterns(*NOT_SHIPPED) skips when copying
    return not (
        "__pycache__" in relative.parts
        or any(part.startswith(".") for part in relative.parts)
        or relative.suffix == ".pyc"
        or relative.name == STAMP
    )


def _write_stamp(experiments: Path, version: str, today: date) -> None:
    stamp = (
        "# written by labflow_report/ship.py; the copy is refreshed only while\n"
        "# its files still hash to this version\n"
        f'version = "{version}"\n'
        f"shipped = {today.isoformat()}\n"
    )
    (experiments / PACKAGE / STAMP).write_text(stamp, encoding="utf-8")


# ===============================
# Main function
# ===============================


def get_args(argv: list[str] | None):
    parser = ArgumentParser(
        prog="ship.py",
        description="Install or refresh the labflow report builder in a repo",
    )
    parser.add_argument("experiments", type=Path, help="the repo's experiments folder")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="the folder holding build_report.py and labflow_report/ (default: the one this file sits in)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = get_args(argv)
    try:
        action, message = ship(args.source, args.experiments, date.today())
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"ship: {error}", file=sys.stderr)
        return EXIT_ERROR
    print(message, file=sys.stderr if action == EDITED else sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
