"""The report spec: one YAML file per source document (ADR-0005).

A spec file describes one document kind. Every key below is read by the
loader; an unknown key is an error, so a typo never passes silently.

```yaml
document: record                 # the document kind, unique across specs
path: "*/*/README.md"            # glob of its files, relative to root
root: ""                         # folder under experiments/ that ids start from
stage_key: status                # frontmatter key holding the stage; omit if none
stages: [draft, running, reporting, concluded]   # ordered: a rule from one stage holds at every later one
side_stages: [abandoned]         # reached only by rules that list them
derived_stages:                  # a stage no frontmatter names: the document reaches it
  reporting: Findings            # from the stage before it once this entries section holds a written entry
gates:                           # a gate a skill runs `check --gate <name>` at
  data: [section.Datasets]       # parts whose rules hold at every stage then
variant_key: kind                # frontmatter key naming a variant; omit if none
variants:                        # a variant the key may name: the rules it switches off
  measured: [section.Rounds, REC-125]   # a part (and every part under it) or a rule id
frontmatter:
  - key: short
    type: str                    # str | int | bool | date | list | map | any
    values: []                   # allowed values (each item, for a list)
    words: [2, 5]                # a cap, or [min, max]
    links_to: record             # the document kind each value names
    shows: the title in pills and the sidebar
    writer: labflow:start step 4
sections:
  - heading: Hypothesis
    form: rows                   # prose | rows | bullets | steps | table | entries | any
    words: 300
    shows: the record's answer card
    writer: labflow:start step 4
    rows:
      - label: Question
        words: 30
        values: []               # allowed lead words: the text before ` — ` or `:`
        shows: the question above the answer
        writer: labflow:start step 4
artifacts:
  - name: clause-readings
    kind: table                  # table | figure
    pattern: tables/clause-readings.csv
    columns: [clause, reading]
    generator: analysis.py
    writer: labflow:report step 3
rules:
  - id: REC-001
    check: required              # a built-in check or a registered custom one
    target: frontmatter.short    # frontmatter.<key> | section.<heading>[.<row>] | artifact.<name> | document
    at: draft                    # a stage (and every later one) or a list of stages; omit for always
    message: the record has no short title
```

A derived stage lets a section's rules wait until the section is written:
a record at `running` is held to `reporting` rules once Findings holds an
entry whose title is no longer template guidance.

A gate holds a document to some rules before their stage: `check --gate
data` runs every rule on a gate target (or on a row under a target section)
at any stage in `stages`, so labflow:build checks a draft record's Datasets
rows before final/ is fixed.

A variant switches rules off for the documents whose variant key names it:
a record with `kind: measured` is held to no rule on a listed part, on a
part under it, or with a listed id, at any stage or gate. A document without
the key keeps every rule. Each variant must be one of the key's `values`, so
the key's values rule reports a variant the spec does not declare.

A part that declares a shape (`type` other than `any`, `values`, `words`,
`form` other than `any`, `links_to`) must have a rule with that check on it,
so every constraint the spec states carries a rule id.

A page spec describes report pages that no document of their own feeds, such
as the landing pages. It holds `pages` instead of `document`; `load_specs`
skips it and `load_pages` reads it. Each part names the rule ids of the keys
it reads, and every id must be a rule some document spec declares.

```yaml
pages:
  - page: research               # the page id, unique across page specs
    tab: Research                # the rail tab whose click opens the page
    shows: every experiment, one programme at a time
    parts:
      - part: planned row        # unique within the page
        shows: a dashed row per planned experiment
        reads: [PRG-031, PRG-034]   # the rules that hold the keys it reads
```
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

# ===============================
# Defaults
# ===============================

TYPES = ("str", "int", "bool", "date", "list", "map", "any")
FORMS = ("prose", "rows", "bullets", "steps", "table", "entries", "any")
ARTIFACT_KINDS = ("table", "figure")
WRITER_PREFIX = "labflow:"

DOCUMENT_KEYS = {
    "document",
    "path",
    "root",
    "stage_key",
    "stages",
    "side_stages",
    "derived_stages",
    "gates",
    "variant_key",
    "variants",
    "frontmatter",
    "sections",
    "artifacts",
    "rules",
}
FIELD_KEYS = {"key", "type", "values", "words", "links_to", "shows", "writer"}
SECTION_KEYS = {"heading", "form", "words", "shows", "writer", "rows"}
ROW_KEYS = {"label", "words", "values", "shows", "writer"}
ARTIFACT_KEYS = {"name", "kind", "pattern", "columns", "generator", "writer"}
RULE_KEYS = {"id", "check", "target", "at", "message"}
TEXT_KEYS = ("document", "path", "root", "stage_key", "variant_key")
PAGE_FILE_KEYS = {"pages"}
PAGE_KEYS = {"page", "tab", "shows", "parts"}
PART_KEYS = {"part", "shows", "reads"}

# the checks a declared shape implies, by the attribute that declares it
SHAPE_CHECKS = {
    "type": "type",
    "values": "values",
    "words": "words",
    "form": "form",
    "links_to": "link",
}

# ===============================
# Spec classes
# ===============================


class SpecError(ValueError):
    """A spec file that cannot be loaded, named with its path."""


@dataclass(frozen=True)
class Field:
    """A frontmatter key."""

    key: str
    type: str = "any"
    values: tuple = ()
    words: tuple = ()
    links_to: str = ""
    shows: str = ""
    writer: str = ""


@dataclass(frozen=True)
class Row:
    """A `- **Label**: text` row inside a section.

    `heading` names the section; `in_entries` is set when that section's
    form is `entries`, so the row belongs to each `###` entry, not the section.
    """

    label: str
    heading: str = ""
    in_entries: bool = False
    words: tuple = ()
    values: tuple = ()
    shows: str = ""
    writer: str = ""


@dataclass(frozen=True)
class Section:
    """A `## Heading` section and the rows it holds."""

    heading: str
    form: str = "any"
    words: tuple = ()
    shows: str = ""
    writer: str = ""
    rows: tuple[Row, ...] = ()


@dataclass(frozen=True)
class Artifact:
    """A table or figure file the document's folder holds."""

    name: str
    kind: str
    pattern: str
    columns: tuple = ()
    generator: str = ""
    writer: str = ""


@dataclass(frozen=True)
class Rule:
    """One check on one part, with the id every report of it carries."""

    id: str
    check: str
    target: str
    message: str
    from_stage: str = ""
    at: tuple = ()


@dataclass(frozen=True)
class DocumentSpec:
    """Everything one document kind must hold, and the rules that say so."""

    document: str
    path: str
    root: str = ""
    stage_key: str = ""
    stages: tuple = ()
    side_stages: tuple = ()
    derived_stages: tuple[tuple[str, str], ...] = ()
    gates: tuple[tuple[str, tuple[str, ...]], ...] = ()
    variant_key: str = ""
    variants: tuple[tuple[str, tuple[str, ...]], ...] = ()
    frontmatter: tuple[Field, ...] = ()
    sections: tuple[Section, ...] = ()
    artifacts: tuple[Artifact, ...] = ()
    rules: tuple[Rule, ...] = ()

    def part(self, target: str) -> Field | Section | Row | Artifact | None:
        """Returns the part a rule target names, or None for `document`.

        Raises:
            KeyError: If the target names no part of this spec.
        """
        if target == "document":
            return None
        kind, _, name = target.partition(".")
        if kind == "frontmatter":
            part = next((f for f in self.frontmatter if f.key == name), None)
        elif kind == "artifact":
            part = next((a for a in self.artifacts if a.name == name), None)
        else:
            part = None
        if part is not None:
            return part
        if kind == "section":
            for section in self.sections:
                if name == section.heading:
                    return section
                for row in section.rows:
                    if name == f"{section.heading}.{row.label}":
                        return row
        raise KeyError(target)

    def applies(
        self, rule: Rule, stage: str | None, gate: str = "", variant: str = ""
    ) -> bool:
        """Whether a rule holds for a document at this stage and variant.

        A rule the document's variant switches off never holds. Otherwise a
        rule with no `at` always holds. `at: <stage>` (`from_stage`) means
        that stage and every later one in `stages`; `at: [<stage>, …]` means
        exactly those stages. With a gate, a rule on one of the gate's
        targets, or on a row under a target section, holds at every stage in
        `stages`; a gate this spec does not declare changes nothing, and
        neither does a variant.
        """
        switched_off = dict(self.variants).get(variant, ())
        if rule.id in switched_off or any(
            rule.target == item or rule.target.startswith(f"{item}.")
            for item in switched_off
        ):
            return False
        targets = dict(self.gates).get(gate, ())
        if stage in self.stages and any(
            rule.target == target or rule.target.startswith(f"{target}.")
            for target in targets
        ):
            return True
        if rule.from_stage:
            return stage in self.stages and self.stages.index(
                stage
            ) >= self.stages.index(rule.from_stage)
        if rule.at:
            return stage in rule.at
        return True


@dataclass(frozen=True)
class PagePart:
    """One part of a page, with the rule ids of the keys it reads."""

    part: str
    shows: str = ""
    reads: tuple = ()


@dataclass(frozen=True)
class PageSpec:
    """A report page no document of its own feeds, and the parts it draws."""

    page: str
    tab: str = ""
    shows: str = ""
    parts: tuple[PagePart, ...] = ()


# ===============================
# Loader
# ===============================


def load_spec(path: Path) -> DocumentSpec:
    """Reads one spec file and checks it against itself.

    Args:
        path: The YAML spec file.

    Returns:
        The document spec.

    Raises:
        SpecError: If a key is unknown or missing, a value is out of range, a
            rule names a part that does not exist, or a declared shape has no
            rule.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    where = str(path)
    _keys(data, DOCUMENT_KEYS, {"document", "path"}, where)
    for key in TEXT_KEYS:
        if not isinstance(data.get(key, ""), str):
            raise SpecError(f"{where}: {key} must be text, got {data[key]!r}")

    # build the parts
    frontmatter = tuple(
        _field(item, where) for item in _list(data, "frontmatter", where)
    )
    sections = tuple(_section(item, where) for item in _list(data, "sections", where))
    artifacts = tuple(
        _artifact(item, where) for item in _list(data, "artifacts", where)
    )
    rules = tuple(_rule(item, where) for item in _list(data, "rules", where))
    derived = data.get("derived_stages") or {}
    if not isinstance(derived, dict) or not all(
        isinstance(item, str) for pair in derived.items() for item in pair
    ):
        raise SpecError(
            f"{where}: derived_stages must map a stage to a heading, got {derived!r}"
        )
    gates = data.get("gates") or {}
    if not isinstance(gates, dict) or not all(
        isinstance(name, str)
        and isinstance(targets, list)
        and all(isinstance(t, str) for t in targets)
        for name, targets in gates.items()
    ):
        raise SpecError(
            f"{where}: gates must map a gate name to a list of targets, got {gates!r}"
        )
    variants = data.get("variants") or {}
    if not isinstance(variants, dict) or not all(
        isinstance(name, str)
        and isinstance(items, list)
        and all(isinstance(i, str) for i in items)
        for name, items in variants.items()
    ):
        raise SpecError(
            f"{where}: variants must map a variant to a list of parts or rule ids, got {variants!r}"
        )
    spec = DocumentSpec(
        document=data["document"],
        path=data["path"],
        root=data.get("root", ""),
        stage_key=data.get("stage_key", ""),
        stages=tuple(_list(data, "stages", where)),
        side_stages=tuple(_list(data, "side_stages", where)),
        derived_stages=tuple(derived.items()),
        gates=tuple((name, tuple(targets)) for name, targets in gates.items()),
        variant_key=data.get("variant_key", ""),
        variants=tuple((name, tuple(items)) for name, items in variants.items()),
        frontmatter=frontmatter,
        sections=sections,
        artifacts=artifacts,
        rules=rules,
    )

    # check names are unique within the spec
    _unique([f.key for f in frontmatter], "frontmatter key", where)
    _unique([s.heading for s in sections], "section heading", where)
    _unique([a.name for a in artifacts], "artifact name", where)
    _unique([r.id for r in rules], "rule id", where)
    if spec.stage_key and spec.stage_key not in {f.key for f in frontmatter}:
        raise SpecError(
            f"{where}: stage_key '{spec.stage_key}' is not a frontmatter key"
        )

    # check a derived stage follows a written stage and waits on an entries section
    forms = {s.heading: s.form for s in sections}
    for stage, heading in spec.derived_stages:
        if stage not in spec.stages or spec.stages.index(stage) == 0:
            raise SpecError(
                f"{where}: derived stage '{stage}' must be a stage after the first in stages"
            )
        if forms.get(heading) != "entries":
            raise SpecError(
                f"{where}: derived stage '{stage}' waits on '{heading}', which is no entries section"
            )

    # check every rule names a part and a known stage
    known_stages = set(spec.stages) | set(spec.side_stages)
    for rule in rules:
        try:
            spec.part(rule.target)
        except KeyError:
            raise SpecError(
                f"{where}: rule {rule.id} targets '{rule.target}', which the spec does not declare"
            ) from None
        if rule.from_stage in spec.side_stages:
            raise SpecError(
                f"{where}: rule {rule.id} starts at side stage '{rule.from_stage}'; list a side stage instead"
            )
        unknown = [
            s for s in (*rule.at, rule.from_stage) if s and s not in known_stages
        ]
        if unknown:
            raise SpecError(f"{where}: rule {rule.id} names unknown stage(s) {unknown}")

    # check every gate target names a part
    for name, targets in spec.gates:
        for target in targets:
            try:
                is_part = spec.part(target) is not None
            except KeyError:
                is_part = False
            if not is_part:
                raise SpecError(
                    f"{where}: gate '{name}' targets '{target}', which names no part of the spec"
                )

    # check every variant is a value of its key and names a part or a rule
    key_field = next((f for f in frontmatter if f.key == spec.variant_key), None)
    if (spec.variant_key or spec.variants) and key_field is None:
        raise SpecError(
            f"{where}: variant_key '{spec.variant_key}' is not a frontmatter key"
        )
    rule_ids = {r.id for r in rules}
    for name, items in spec.variants:
        if name not in key_field.values:
            raise SpecError(
                f"{where}: variant '{name}' is not one of {spec.variant_key}'s values {list(key_field.values)}"
            )
        for item in items:
            try:
                is_part = item in rule_ids or spec.part(item) is not None
            except KeyError:
                is_part = False
            if not is_part:
                raise SpecError(
                    f"{where}: variant '{name}' switches off '{item}', which names no part or rule of the spec"
                )

    # check every declared shape carries a rule
    covered = {(r.target, r.check) for r in rules}
    for target, part in _targets(spec):
        for attribute, check in SHAPE_CHECKS.items():
            value = getattr(part, attribute, None)
            if value and value != "any" and (target, check) not in covered:
                raise SpecError(
                    f"{where}: {target} declares {attribute} but no rule checks '{check}' on it"
                )
    return spec


def load_specs(folder: Path) -> dict[str, DocumentSpec]:
    """Reads every `*.yaml` document spec in a folder and checks them against each other.

    A page spec (`pages` instead of `document`) is skipped; `load_pages`
    reads it.

    Args:
        folder: The spec folder.

    Returns:
        The specs by document kind.

    Raises:
        SpecError: If a spec fails `load_spec`, two specs share a kind or a
            rule id, or a `links_to` names a kind no spec declares.
    """
    specs: dict[str, DocumentSpec] = {}
    owner: dict[str, str] = {}
    for path in sorted(folder.glob("*.yaml")):
        if _is_page_file(path):
            continue
        spec = load_spec(path)
        if spec.document in specs:
            raise SpecError(
                f"{path}: document kind '{spec.document}' is declared twice"
            )
        specs[spec.document] = spec
        for rule in spec.rules:
            if rule.id in owner:
                raise SpecError(
                    f"{path}: rule id {rule.id} is also used by the '{owner[rule.id]}' spec"
                )
            owner[rule.id] = spec.document

    # check links name a declared kind
    for spec in specs.values():
        for target, part in _targets(spec):
            linked = getattr(part, "links_to", "")
            if linked and linked not in specs:
                raise SpecError(
                    f"{folder}: {spec.document} {target} links to '{linked}', which no spec declares"
                )
    return specs


def load_pages(folder: Path, specs: dict[str, DocumentSpec]) -> dict[str, PageSpec]:
    """Reads every `*.yaml` page spec in a folder and checks its reads.

    Args:
        folder: The spec folder.
        specs: The document specs `load_specs` read from the same folder.

    Returns:
        The page specs by page id, in file order, then declaration order.

    Raises:
        SpecError: If a key is unknown or missing, a page or a part is
            declared twice, a page declares no part, or a part reads a rule
            id no document spec declares.
    """
    declared = {rule.id for spec in specs.values() for rule in spec.rules}
    pages: dict[str, PageSpec] = {}
    for path in sorted(folder.glob("*.yaml")):
        if not _is_page_file(path):
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        where = str(path)
        _keys(data, PAGE_FILE_KEYS, PAGE_FILE_KEYS, where)
        for item in _list(data, "pages", where):
            page = _page(item, where)
            if page.page in pages:
                raise SpecError(f"{where}: page '{page.page}' is declared twice")
            for part in page.parts:
                unknown = [i for i in part.reads if i not in declared]
                if unknown:
                    raise SpecError(
                        f"{where}: {page.page} {part.part} reads {', '.join(unknown)}, which no spec declares"
                    )
            pages[page.page] = page
    return pages


# ===============================
# Helper functions
# ===============================


def _keys(data: dict, allowed: set, required: set, where: str) -> None:
    if not isinstance(data, dict):
        raise SpecError(f"{where}: expected a mapping, got {data!r}")
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise SpecError(f"{where}: unknown key(s) {unknown}")
    missing = sorted(required - set(data))
    if missing:
        raise SpecError(f"{where}: missing key(s) {missing}")


def _words(value, where: str) -> tuple:
    if value is None:
        return ()
    if isinstance(value, int):
        return (0, value)
    if (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(v, int) for v in value)
        and value[0] <= value[1]
    ):
        return tuple(value)
    raise SpecError(f"{where}: words must be a cap or [min, max], got {value!r}")


def _writer(value: str, where: str) -> str:
    if value and not value.startswith(WRITER_PREFIX):
        raise SpecError(
            f"{where}: writer '{value}' must name a skill as '{WRITER_PREFIX}<skill> …'"
        )
    return value


def _field(data: dict, where: str) -> Field:
    _keys(data, FIELD_KEYS, {"key"}, where)
    here = f"{where}: frontmatter.{data['key']}"
    kind = data.get("type", "any")
    if kind not in TYPES:
        raise SpecError(f"{here}: type '{kind}' is not one of {TYPES}")
    return Field(
        key=data["key"],
        type=kind,
        values=tuple(_list(data, "values", here)),
        words=_words(data.get("words"), here),
        links_to=data.get("links_to", ""),
        shows=data.get("shows", ""),
        writer=_writer(data.get("writer", ""), here),
    )


def _section(data: dict, where: str) -> Section:
    _keys(data, SECTION_KEYS, {"heading"}, where)
    here = f"{where}: section.{data['heading']}"
    form = data.get("form", "any")
    if form not in FORMS:
        raise SpecError(f"{here}: form '{form}' is not one of {FORMS}")
    rows = []
    for item in _list(data, "rows", here):
        _keys(item, ROW_KEYS, {"label"}, here)
        row_here = f"{here}.{item['label']}"
        rows.append(
            Row(
                label=item["label"],
                heading=data["heading"],
                in_entries=form == "entries",
                words=_words(item.get("words"), row_here),
                values=tuple(_list(item, "values", row_here)),
                shows=item.get("shows", ""),
                writer=_writer(item.get("writer", ""), row_here),
            )
        )
    _unique([r.label for r in rows], "row label", here)
    return Section(
        heading=data["heading"],
        form=form,
        words=_words(data.get("words"), here),
        shows=data.get("shows", ""),
        writer=_writer(data.get("writer", ""), here),
        rows=tuple(rows),
    )


def _artifact(data: dict, where: str) -> Artifact:
    _keys(data, ARTIFACT_KEYS, {"name", "kind", "pattern"}, where)
    here = f"{where}: artifact.{data['name']}"
    if data["kind"] not in ARTIFACT_KINDS:
        raise SpecError(f"{here}: kind '{data['kind']}' is not one of {ARTIFACT_KINDS}")
    return Artifact(
        name=data["name"],
        kind=data["kind"],
        pattern=data["pattern"],
        columns=tuple(_list(data, "columns", here)),
        generator=data.get("generator", ""),
        writer=_writer(data.get("writer", ""), here),
    )


def _rule(data: dict, where: str) -> Rule:
    _keys(data, RULE_KEYS, {"id", "check", "target", "message"}, where)
    at = data.get("at")
    if at is not None and not isinstance(at, (str, list)):
        raise SpecError(
            f"{where}: rule {data['id']} has at {at!r}; give a stage or a list of stages"
        )
    return Rule(
        id=data["id"],
        check=data["check"],
        target=data["target"],
        message=data["message"],
        from_stage=at if isinstance(at, str) else "",
        at=tuple(at) if isinstance(at, list) else (),
    )


def _is_page_file(path: Path) -> bool:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return isinstance(data, dict) and "pages" in data and "document" not in data


def _page(data: dict, where: str) -> PageSpec:
    _keys(data, PAGE_KEYS, {"page"}, where)
    here = f"{where}: {data['page']}"
    parts = []
    for item in _list(data, "parts", here):
        _keys(item, PART_KEYS, {"part"}, here)
        parts.append(
            PagePart(
                part=item["part"],
                shows=item.get("shows", ""),
                reads=tuple(_list(item, "reads", f"{here} {item['part']}")),
            )
        )
    if not parts:
        raise SpecError(f"{here} declares no part")
    _unique([p.part for p in parts], "part", here)
    return PageSpec(
        page=data["page"],
        tab=data.get("tab", ""),
        shows=data.get("shows", ""),
        parts=tuple(parts),
    )


def _list(data: dict, key: str, where: str) -> list:
    value = data.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise SpecError(f"{where}: {key} must be a list, got {value!r}")
    return value


def _unique(names: list, what: str, where: str) -> None:
    seen = set()
    for name in names:
        if name in seen:
            raise SpecError(f"{where}: {what} '{name}' is declared twice")
        seen.add(name)


def _targets(spec: DocumentSpec):
    for field in spec.frontmatter:
        yield f"frontmatter.{field.key}", field
    for section in spec.sections:
        yield f"section.{section.heading}", section
        for row in section.rows:
            yield f"section.{section.heading}.{row.label}", row
    for artifact in spec.artifacts:
        yield f"artifact.{artifact.name}", artifact
