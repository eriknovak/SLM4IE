"""The report checker: run every spec's rules and print `file:line RULE-ID message`.

    python -m labflow_report check [paths] [--experiments DIR] [--specs DIR]
        [--skip-stage STAGES] [--gate NAME]

Run it from the folder that holds `labflow_report/`. With paths, only those
files are reported; the whole experiments folder is still read, so links
resolve. Without paths, every document a spec declares is checked. Each
document is held to the rules of its current stage only, so a plan-stage
record passes on plan-stage rules. `--skip-stage concluded,abandoned` leaves
out documents at those stages; the report-guard hook passes it, so a finished
record is not re-checked on every edit. `--gate data` also holds each
document to the rules on the parts its spec's `gates: {data: [...]}` names,
at any stage; labflow:build runs it on a draft record before final/ is fixed.

Exit codes: 0 clean, 1 a breach, 2 the check could not run (a missing path,
no experiments folder, no spec files, a broken spec, an unknown check or
gate). A
stage gate passes only on 0.

When the experiments folder holds a shipped copy of the builder that was
edited, or that is behind the skill's, a warning goes to stderr; it never
changes the exit code.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import yaml

from .rules import Breach, document_stage, read_corpus, run_rules
from .ship import skill_references, stale_warning
from .spec import SpecError, load_specs

# ===============================
# Defaults
# ===============================

SPEC_FOLDER = Path(__file__).resolve().parent / "specs"
EXPERIMENTS_NAME = "experiments"
EXIT_CLEAN = 0
EXIT_BREACH = 1
EXIT_ERROR = 2

# ===============================
# Errors
# ===============================


class CheckError(ValueError):
    """A check that cannot run, named with the path or folder at fault."""


# ===============================
# Main functions
# ===============================


def find_breaches(
    paths: list[Path],
    experiments: Path,
    spec_folder: Path,
    skip_stages: frozenset[str] = frozenset(),
    gate: str = "",
) -> tuple[list[Breach], list[Path]]:
    """Runs every spec's rules over the documents named, or over all of them.

    Args:
        paths: The files to report on; empty means every declared document.
        experiments: The experiments folder the specs' globs start from.
        spec_folder: The folder of `*.yaml` specs.
        skip_stages: Stages whose documents are left out.
        gate: A gate whose targets' rules hold at any stage; empty for none.

    Returns:
        The breaches, sorted by file and line, and the paths no spec declares.

    Raises:
        CheckError: If the experiments folder or a path does not exist, the
            folder holds no spec, a spec is not valid YAML or UTF-8, a
            document is not UTF-8, a skip stage is no spec's stage, or the
            gate is no spec's gate.
        SpecError: If a spec fails to load.
        OSError: If a file cannot be read.
        KeyError: If a rule names a check nobody registered.
    """
    if not experiments.is_dir():
        raise CheckError(f"no experiments folder at {experiments}")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise CheckError(f"no such file: {', '.join(map(str, missing))}")
    if not any(spec_folder.glob("*.yaml")):
        raise CheckError(f"no spec files (*.yaml) in {spec_folder}")

    # read every document, then keep the ones asked for
    try:
        specs = load_specs(spec_folder)
    except yaml.YAMLError as error:
        culprit = _culprit(sorted(spec_folder.glob("*.yaml")), _read_yaml)
        raise CheckError(f"{culprit}: spec is not valid YAML: {error}") from None
    except UnicodeDecodeError as error:
        culprit = _culprit(sorted(spec_folder.glob("*.yaml")), _read_text)
        raise CheckError(f"{culprit}: spec is not UTF-8: {error}") from None
    known = {s for spec in specs.values() for s in (*spec.stages, *spec.side_stages)}
    unknown = sorted(skip_stages - known)
    if unknown:
        raise CheckError(f"skip stage(s) {unknown} are not stages any spec declares")
    gates = sorted({name for spec in specs.values() for name, _ in spec.gates})
    if gate and gate not in gates:
        raise CheckError(f"gate '{gate}' is no gate a spec declares; known: {gates}")
    try:
        documents, corpus = read_corpus(experiments, specs)
    except UnicodeDecodeError as error:
        culprit = _culprit(_declared_files(experiments, specs), _read_text)
        raise CheckError(f"{culprit}: document is not UTF-8: {error}") from None
    wanted = {path.resolve() for path in paths}
    if wanted:
        documents = [d for d in documents if d.path.resolve() in wanted]
    declared = {d.path.resolve() for d in documents}
    skipped = [path for path in paths if path.resolve() not in declared]
    documents = [
        d for d in documents if document_stage(d, specs[d.kind]) not in skip_stages
    ]

    breaches = []
    for document in documents:
        breaches += run_rules(document, specs[document.kind], corpus, gate)
    return sorted(breaches), skipped


def main(
    paths: list[Path],
    experiments: Path | None,
    spec_folder: Path,
    skip_stages: frozenset[str] = frozenset(),
    gate: str = "",
) -> int:
    """Prints each breach on stdout, each problem and a stale copy on stderr.

    Args:
        paths: The files to report on; empty means every declared document.
        experiments: The experiments folder, or None to take the nearest
            folder named `experiments` above the first path, else
            `./experiments`, else the current folder when it is one.
        spec_folder: The folder of `*.yaml` specs.
        skip_stages: Stages whose documents are left out.
        gate: A gate whose targets' rules hold at any stage; empty for none.

    Returns:
        The exit code: 0 clean, 1 a breach, 2 the check could not run.
    """
    try:
        folder = experiments or _experiments_folder(paths)
        breaches, skipped = find_breaches(paths, folder, spec_folder, skip_stages, gate)
        warning = stale_warning(folder, skill_references())
    except (CheckError, SpecError, OSError) as error:
        print(f"labflow_report check: {error}", file=sys.stderr)
        return EXIT_ERROR
    except KeyError as error:
        print(f"labflow_report check: {error.args[0]}", file=sys.stderr)
        return EXIT_ERROR

    if warning:
        print(f"labflow_report check: {warning}", file=sys.stderr)
    for path in skipped:
        print(f"{path}: no spec declares this file, skipped", file=sys.stderr)
    for breach in breaches:
        print(breach)
    return EXIT_BREACH if breaches else EXIT_CLEAN


# ===============================
# Helper functions
# ===============================


def _experiments_folder(paths: list[Path]) -> Path:
    if not paths:
        # ./experiments from the repo root, or the current folder when run inside it
        for folder in (Path(EXPERIMENTS_NAME), Path.cwd()):
            if folder.resolve().name == EXPERIMENTS_NAME and folder.is_dir():
                return folder.resolve()
        raise CheckError(f"no experiments folder at {Path.cwd()}; pass --experiments")
    for parent in paths[0].resolve().parents:
        if parent.name == EXPERIMENTS_NAME:
            return parent
    raise CheckError(f"no experiments folder above {paths[0]}; pass --experiments")


def _culprit(paths: list[Path], read: Callable[[Path], object]) -> Path | str:
    # the error names no file, so find the first one that fails the same read
    for path in paths:
        try:
            read(path)
        except (yaml.YAMLError, UnicodeDecodeError):
            return path
    return "a file"


def _read_yaml(path: Path) -> object:
    return yaml.safe_load(_read_text(path))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _declared_files(experiments: Path, specs: dict) -> list[Path]:
    files = []
    for spec in specs.values():
        base = experiments / spec.root if spec.root else experiments
        files += sorted(base.glob(spec.path))
    return files
