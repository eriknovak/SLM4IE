"""Build the open Slovene data landscape catalogue and its per-domain supply tables.

The survey's evidence is one JSON file per searched source family under
`data/experiments/data/data-landscape-slovenian/raw/`, each a list of rows on the
schema fixed in the record's D4. This script merges them into the committed
catalogue, flags what the project already downloads, reads how each dataset's
Slovene text came to exist, scores each row against KPI 2 and KPI 4, and
aggregates per domain under the three access filters of D8.

Run it from anywhere; `--mlflow` also records the pass as a lineage run in the
experiment's MLflow experiment, with the tables and figures as artifacts:

    uv run --group analysis python experiments/data/data-landscape-slovenian/analysis.py [--mlflow]
"""

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import yaml
from datachart.charts import BarChart, DumbbellChart
from datachart.constants import BAR_MODE, DUMBBELL_SORT_KEY, EMPHASIS, FIG_SIZE, LEGEND_LOCATION, ORIENTATION, SORT

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from report_figures import save_figure  # noqa: E402  — path set above so the vendored helper resolves

from slm4ie.utils import mlflow as ml  # noqa: E402

MLFLOW_EXPERIMENT = "slm4ie/data/data-landscape-slovenian"

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RAW_DIR = ROOT / "data" / "experiments" / "data" / "data-landscape-slovenian" / "raw"
TABLES_DIR = HERE / "tables"
FIGURES_DIR = HERE / "figures"
REGISTRY = ROOT / "configs" / "data" / "download.yaml"
# What the pretraining build measured per registry source after curation.
CORPUS_STATISTICS = ROOT / "data" / "pretrain" / "07_statistics" / "aggregate.json"
# What size_medical_sources.py counted for the new native medical sources.
SIZES_TABLE = HERE / "tables" / "native-medical-sizes.csv"
# Each non-native row's provenance as read from its card or paper (M7).
PROVENANCE_CHECK = RAW_DIR / "provenance-check.json"

# One file per source family searched (D2), in the order that decides which
# copy of a dataset catalogued twice is kept: the repository that publishes it
# beats the aggregator that mirrors it.
FAMILY_FILES: List[Tuple[str, str]] = [
    ("clarin-si", "clarin-si.json"),
    ("lindat", "lindat.json"),
    ("zenodo", "zenodo.json"),
    ("huggingface", "huggingface.json"),
    ("huggingface", "huggingface-recheck.json"),
    ("medical-publishers", "medical-publishers.json"),
    ("elg", "elg.json"),
]

# The shared domain taxonomy (D5); the headline science figure is scientific + academic.
DOMAINS: List[str] = [
    "medical",
    "scientific",
    "academic",
    "legal",
    "news",
    "parliamentary",
    "encyclopedic",
    "forum/social",
    "general-web",
    "finance",
    "other",
]
KPI2_DOMAINS: Tuple[str, ...] = ("medical", "scientific", "academic")
# The Predictions' domains, each the taxonomy domains it spans; a row counts once.
CLAUSE_DOMAINS: List[Tuple[str, Tuple[str, ...]]] = [
    ("medicine", ("medical",)),
    ("science", ("scientific", "academic")),
]

# What each access filter of D8 admits: what anyone can download, what a
# registered researcher can, and what exists at all.
ACCESS_FILTERS: List[Tuple[str, Tuple[str, ...]]] = [
    ("open", ("open",)),
    ("open+login", ("open", "login")),
    ("all", ("open", "login", "gated", "not_downloadable")),
]

# Annotation types that make a row a source of information-extraction examples
# rather than of pretraining text (D7's second reading of KPI 2).
IE_ANNOTATIONS: Tuple[str, ...] = ("IE spans", "IE spans,labels", "labels")

# How a dataset's Slovene text came to exist (D11), read off the name and notes.
# `generated` is tested first: a set that was translated and then automatically
# annotated is machine output either way.
GENERATED_TERMS: Tuple[str, ...] = (
    r"synthetic",
    r"generated",
    r"instruction[- ]following",
    r"instruct\b",
    r"\bsft\b",
    r"auto(matically)?[- ]annotat",
    r"\bllm\b",
    r"\bgpt\b",
)
TRANSLATED_TERMS: Tuple[str, ...] = (
    r"translat",
    r"parallel corpus",
    r"parallel translation",
    r"bilingual",
    r"\btmx\b",
    r"\bmt\b",
)

# A checked provenance class, folded into the three-way reading the Predictions use.
PROVENANCE_OF_CLASS: Dict[str, str] = {
    "native": "native",
    "human_translated": "translated",
    "machine_translated": "translated",
    "mixed": "translated",
    "bilingual_resource": "translated",
    "synthetic": "generated",
}
PROVENANCE_CLASSES: List[str] = [*PROVENANCE_OF_CLASS, "unchecked"]

# The sizing pass's basis, as the catalogue's `words_basis` records it (D6).
SIZE_BASIS: Dict[str, str] = {"full": "counted", "partial": "partial", "sampled": "sampled"}

TOKENS_PER_WORD = 2.0  # D6: an estimate until the project tokenizer exists
KPI2_EXAMPLES = 10_000
KPI4_TOKENS = 500_000

CATALOGUE_COLUMNS: List[str] = [
    "name",
    "url",
    "licence",
    "access",
    "languages",
    "domains",
    "documents",
    "documents_counted",
    "words",
    "words_basis",
    "words_corpus",
    "documents_corpus",
    "tokens_estimated",
    "size_verified",
    "annotation",
    "length_class",
    "format",
    "provenance",
    "provenance_class",
    "provenance_checked",
    "source_language",
    "translation_system",
    "generator_model",
    "in_registry",
    "registry_key",
    "kpi2_fit",
    "kpi4_fit",
    "family",
    "mirrors",
    "notes",
]

# Registry entries fetched by hand, so they name no URL or Hub repo: matched on
# a fragment of the catalogue's own name for the same corpus instead.
REGISTRY_NAME_FRAGMENTS: Dict[str, str] = {
    "gigafida22": "gigafida",
    "metafida10": "metafida",
    "slovenetrendi": "trendi",
}

# Registry entries the catalogue found at a different address than the one the
# project downloads from: the Hub repo that publishes the same corpus.
REGISTRY_REPO_ALIASES: Dict[str, str] = {"statmt/cc100": "cc100"}


def normalise(text: str) -> str:
    """Reduces a dataset name to a comparison key.

    Args:
        text: The name as the source catalogue prints it.

    Returns:
        The name lowercased with every non-alphanumeric character dropped.
    """
    return re.sub(r"[^a-z0-9]", "", text.lower())


def number(value: object) -> Optional[int]:
    """Reads a count from a catalogue field that may be blank.

    Args:
        value: The field as read from the source JSON.

    Returns:
        The count, or None when the source reported none.
    """
    text = str(value or "").strip()
    return int(text) if text.isdigit() else None


def load_rows() -> List[Dict[str, str]]:
    """Reads every source family's rows, tagged with the family it came from.

    Returns:
        Every catalogued row, in family order.

    Raises:
        FileNotFoundError: If a family's evidence file is missing from `RAW_DIR`.
    """
    rows: List[Dict[str, str]] = []
    for family, filename in FAMILY_FILES:
        path = RAW_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"missing survey evidence: {path}")
        for row in json.loads(path.read_text(encoding="utf-8")):
            rows.append({**row, "family": family})
    return rows


def merge_duplicates(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    """Collapses datasets catalogued by more than one source family.

    The first row seen wins, because `FAMILY_FILES` is ordered publisher before
    aggregator; the duplicates only fill its blank fields and leave their URLs
    behind in `mirrors`, so nothing found is silently dropped.

    Args:
        rows: Rows from every family, in family order.

    Returns:
        One row per dataset, each carrying a `mirrors` field.
    """
    merged: Dict[str, Dict[str, str]] = {}
    for row in rows:
        key = normalise(row["name"])
        kept = merged.get(key)
        if kept is None:
            merged[key] = {**row, "mirrors": ""}
            continue
        for field, value in row.items():
            if str(value or "").strip() and not str(kept.get(field, "") or "").strip():
                kept[field] = value
        mirrors = [m for m in (kept["mirrors"], row["url"]) if m]
        kept["mirrors"] = " ".join(mirrors)
    return list(merged.values())


def registry_index() -> Dict[str, Set[str]]:
    """Collects, per registry entry, the identifiers a catalogue row can match.

    An entry is identified by its CLARIN or LINDAT handle or its Hugging Face
    repo, plus the Hub aliases of `REGISTRY_REPO_ALIASES`; the entries fetched
    by hand carry neither and are matched on the name instead, by
    `registry_match`.

    Returns:
        Registry key to the identifiers `row_keys` can produce for it.
    """
    registry = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))["datasets"]
    index: Dict[str, Set[str]] = {key: set() for key in registry}
    for key, entry in registry.items():
        if entry.get("repo_id"):
            index[key].add(str(entry["repo_id"]).lower())
        for url in entry.get("urls") or []:
            index[key] |= url_keys(str(url))
    for repo, key in REGISTRY_REPO_ALIASES.items():
        if key in index:
            index[key].add(repo)
    return index


def registry_match(row: Dict[str, str], index: Dict[str, Set[str]]) -> Optional[str]:
    """Names the registry entry the project downloads this dataset under.

    Args:
        row: A merged catalogue row.
        index: The identifiers per entry, from `registry_index`.

    Returns:
        The registry key, or None when the project does not download it.
    """
    keys = row_keys(row)
    for key, identifiers in index.items():
        if keys & identifiers:
            return key
    name = normalise(row["name"])
    return next((key for fragment, key in REGISTRY_NAME_FRAGMENTS.items() if fragment in name), None)


def survey_sizes() -> Dict[str, Dict[str, str]]:
    """Reads what the sizing pass counted for the new native medical sources.

    Returns:
        Normalised catalogue name to its line of the sizes table; empty until
        `size_medical_sources.py` has run.
    """
    if not SIZES_TABLE.exists():
        return {}
    with SIZES_TABLE.open(encoding="utf-8") as handle:
        return {normalise(r["catalogue_name"]): r for r in csv.DictReader(handle)}


def provenance_checks() -> Dict[str, Dict[str, str]]:
    """Reads the checked provenance of the rows the keyword reading marked non-native.

    Returns:
        Normalised catalogue name to its checked entry; empty until the check exists.
    """
    if not PROVENANCE_CHECK.exists():
        return {}
    return {normalise(c["name"]): c for c in json.loads(PROVENANCE_CHECK.read_text(encoding="utf-8"))}


def corpus_counts() -> Dict[str, Dict[str, int]]:
    """Reads what the pretraining build measured for each registry source.

    Returns:
        Registry key to its curated `doc_count` and `word_count`; empty when
        the build's statistics stage has not run.
    """
    if not CORPUS_STATISTICS.exists():
        return {}
    return json.loads(CORPUS_STATISTICS.read_text(encoding="utf-8"))["by_dataset"]


def url_keys(url: str) -> Set[str]:
    """Extracts the identifiers a URL carries for registry matching.

    Args:
        url: A dataset landing page, download link, or Hub URL.

    Returns:
        The handle and Hub repo found in the URL; empty when it carries neither.
    """
    keys: Set[str] = set()
    handle = re.search(r"(?:11356|11234)/\d+", url)
    if handle:
        keys.add(handle.group(0))
    repo = re.search(r"huggingface\.co/datasets/([\w.-]+/[\w.-]+)", url)
    if repo:
        keys.add(repo.group(1).lower())
    statmt = re.search(r"data\.statmt\.org/(cc-100)", url)
    if statmt:
        keys.add(statmt.group(1))
    return keys


def row_keys(row: Dict[str, str]) -> Set[str]:
    """Extracts the identifiers a catalogue row offers for registry matching.

    Args:
        row: A merged catalogue row.

    Returns:
        Identifiers from the row's own URL, its mirrors, and its name.
    """
    keys: Set[str] = {normalise(row["name"])}
    for url in [row["url"], *str(row.get("mirrors", "")).split()]:
        keys |= url_keys(url)
    return keys


def provenance(row: Dict[str, str]) -> str:
    """Says how the Slovene text in a dataset came to exist.

    The hypothesis asks whether open sources cover the KPIs *without* synthetic
    data, so a document count means little until machine output and translation
    are separated from prose written in Slovene. The reading comes from what the
    source says about itself, in the name and the notes, and never from the
    domain or the size.

    Args:
        row: A merged catalogue row.

    Returns:
        `generated` for machine-written or automatically annotated text,
        `translated` for text carried into Slovene from another language, and
        `native` for the rest, which is text written in Slovene.
    """
    text = f"{row['name']} {row.get('notes', '')}".lower()
    if any(re.search(term, text) for term in GENERATED_TERMS):
        return "generated"
    if row["annotation"] == "parallel" or any(re.search(term, text) for term in TRANSLATED_TERMS):
        return "translated"
    return "native"


def kpi_fit(row: Dict[str, str]) -> Tuple[str, str]:
    """Scores one row against KPI 2 and KPI 4.

    KPI 2 asks for more than 10k examples in medicine and in science, so it is
    scored only for rows carrying one of those domains. KPI 4 asks for more than
    500k tokens, read off the reported word count at `TOKENS_PER_WORD`. A size
    the source never reported is `unknown` rather than `no`, because D6 forbids
    reading a missing number as a small one.

    Args:
        row: A merged catalogue row.

    Returns:
        The KPI 2 and KPI 4 verdicts, each `yes`, `no`, `unknown`, or — for
        KPI 2 outside medicine and science — `n/a`.
    """
    domains = {d.strip() for d in str(row["domains"]).split(",") if d.strip()}
    documents, words = number(row["documents"]), number(row["words"])

    if not domains & set(KPI2_DOMAINS):
        kpi2 = "n/a"
    elif documents is None:
        kpi2 = "unknown"
    else:
        kpi2 = "yes" if documents >= KPI2_EXAMPLES else "no"

    if words is None:
        kpi4 = "unknown"
    else:
        kpi4 = "yes" if words * TOKENS_PER_WORD >= KPI4_TOKENS else "no"
    return kpi2, kpi4


def annotate(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    """Adds the derived catalogue columns to every row.

    A size the source reported is kept as reported. A row whose source
    reported none takes, in this order, what the sizing pass counted over its
    downloaded text (`words_basis` `counted`, or `sampled` when extrapolated
    from a sample) or what the pretraining build measured for a registry
    source (`corpus`) — each a count over downloaded text, the verification
    D6 asks for. `documents` takes the sizing pass's item count wherever it
    exists, since it counts in the unit the catalogue means and a publisher's
    figure may not; `documents_counted` carries it beside the result.

    Args:
        rows: Merged catalogue rows.

    Returns:
        The same rows, each with the registry, corpus, provenance, token and
        KPI columns filled in.
    """
    index = registry_index()
    measured = corpus_counts()
    surveyed = survey_sizes()
    checks = provenance_checks()
    annotated: List[Dict[str, str]] = []
    for row in rows:
        key = registry_match(row, index)
        counts = measured.get(key or "", {})
        size = surveyed.get(normalise(row["name"]), {})
        candidates = [
            ("reported", number(row["words"])),
            (SIZE_BASIS.get(size.get("basis", ""), "counted"), number(size.get("words_estimated"))),
            ("corpus", number(counts.get("word_count"))),
        ]
        basis, words = next(((b, w) for b, w in candidates if w is not None), ("", None))
        # the sizing pass counted the source's items itself, in the unit the
        # catalogue means; a publisher's figure stands only where it did not
        documents = number(size.get("items_total"))
        if documents is None:
            documents = number(row["documents"])
        sized = {
            **row,
            "words": words if words is not None else "",
            "documents": documents if documents is not None else "",
        }
        kpi2, kpi4 = kpi_fit(sized)
        check = checks.get(normalise(row["name"]), {})
        read = provenance(row)
        # an unchecked keyword-native row stays native; any other unchecked row says so
        klass = check.get("provenance_class") or ("native" if read == "native" else "unchecked")
        annotated.append(
            {
                **row,
                "documents": str(documents) if documents is not None else "",
                "documents_counted": str(size.get("items_total", "")),
                "words": str(words) if words is not None else "",
                "words_basis": basis,
                "words_corpus": str(counts.get("word_count", "")),
                "documents_corpus": str(counts.get("doc_count", "")),
                "tokens_estimated": str(int(words * TOKENS_PER_WORD)) if words is not None else "",
                "provenance": PROVENANCE_OF_CLASS[klass] if check else read,
                "provenance_class": klass,
                "provenance_checked": "yes" if check else "no",
                "source_language": check.get("source_language", ""),
                "translation_system": check.get("translation_system", ""),
                "generator_model": check.get("generator_model", ""),
                "in_registry": "yes" if key else "no",
                "registry_key": key or "",
                "kpi2_fit": kpi2,
                "kpi4_fit": kpi4,
            }
        )
    return annotated


def write_table(path: Path, columns: List[str], rows: Iterable[Dict[str, object]]) -> None:
    """Writes one results table as CSV.

    Args:
        path: Destination file; its parent is created if missing.
        columns: Column order, also the set of fields written.
        rows: The rows to write, keyed by column name.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def survey_statistics(raw: List[Dict[str, str]], rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    """Counts, per source family, what it returned and what survived the merge.

    Args:
        raw: Every row as the families returned it, before merging.
        rows: The merged, annotated catalogue, Slovene and fallback rows together.

    Returns:
        One line per family plus a `total` line.
    """
    lines: List[Dict[str, object]] = []
    for family in dict.fromkeys(f for f, _ in FAMILY_FILES):
        kept = [r for r in rows if r["family"] == family]
        lines.append(
            {
                "family": family,
                "rows_returned": sum(1 for r in raw if r["family"] == family),
                "rows_kept": len(kept),
                "rows_slovene": sum(1 for r in kept if r.get("part") != "B"),
                "rows_medical_other_languages": sum(1 for r in kept if r.get("part") == "B"),
                "rows_in_registry": sum(1 for r in kept if r["in_registry"] == "yes"),
            }
        )
    total = {"family": "total"}
    total.update({k: sum(int(line[k]) for line in lines) for k in lines[0] if k != "family"})
    return [*lines, total]


def verdict(total: int, threshold: int, uncounted: int) -> str:
    """Reads a per-domain total against a KPI threshold.

    A total that clears the threshold is `yes` however many sizes are missing,
    since the missing ones could only raise it. A total that falls short while
    the domain still holds datasets nobody sized is `unknown`, not `no` — D6
    forbids reading an unreported size as a small one.

    Args:
        total: What the sized datasets in the domain add up to.
        threshold: The KPI's threshold.
        uncounted: Datasets in the domain whose source reported no size.

    Returns:
        `yes`, `no`, or `unknown`.
    """
    if total >= threshold:
        return "yes"
    return "unknown" if uncounted else "no"


def summarise(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    """Totals the catalogue per domain under each access filter.

    Totals cover only the rows whose size the source reported; the two
    `datasets_without_*` columns say how much of the domain each total leaves
    out, and the verdicts read a shortfall behind missing sizes as `unknown`.

    Args:
        rows: The annotated Slovene catalogue.

    Returns:
        One entry per domain and access filter, in taxonomy order.
    """
    summary: List[Dict[str, object]] = []
    for domain in DOMAINS:
        in_domain = [r for r in rows if domain in {d.strip() for d in str(r["domains"]).split(",")}]
        for label, admitted in ACCESS_FILTERS:
            selected = [r for r in in_domain if r["access"] in admitted]
            annotated_rows = [r for r in selected if r["annotation"] in IE_ANNOTATIONS]
            native_rows = [r for r in selected if r["provenance"] == "native"]
            words = sum(number(r["words"]) or 0 for r in selected)
            documents = sum(number(r["documents"]) or 0 for r in selected)
            annotated = sum(number(r["documents"]) or 0 for r in annotated_rows)
            native = sum(number(r["documents"]) or 0 for r in native_rows)
            tokens = int(words * TOKENS_PER_WORD)
            no_words = sum(1 for r in selected if number(r["words"]) is None)
            corpus_sized = sum(1 for r in selected if r["words_basis"] == "corpus")
            survey_sized = sum(1 for r in selected if r["words_basis"] in SIZE_BASIS.values())
            no_documents = sum(1 for r in selected if number(r["documents"]) is None)
            no_annotated = sum(1 for r in annotated_rows if number(r["documents"]) is None)
            no_native = sum(1 for r in native_rows if number(r["documents"]) is None)
            summary.append(
                {
                    "domain": domain,
                    "access_filter": label,
                    "datasets": len(selected),
                    "datasets_new": sum(1 for r in selected if r["in_registry"] == "no"),
                    "datasets_native": len(native_rows),
                    "datasets_without_word_count": no_words,
                    "datasets_sized_by_corpus": corpus_sized,
                    "datasets_sized_by_survey": survey_sized,
                    "datasets_without_document_count": no_documents,
                    "words": words,
                    "tokens_estimated": tokens,
                    "documents": documents,
                    "documents_native": native,
                    "annotated_examples": annotated,
                    "meets_kpi2_documents": verdict(documents, KPI2_EXAMPLES, no_documents),
                    "meets_kpi2_native": verdict(native, KPI2_EXAMPLES, no_native),
                    "meets_kpi2_annotated": verdict(annotated, KPI2_EXAMPLES, no_annotated),
                    "meets_kpi4_tokens": verdict(tokens, KPI4_TOKENS, no_words),
                }
            )
    return summary


def provenance_summary(rows: List[Dict[str, str]]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Counts datasets per checked provenance class, and machine-translated ones per system.

    Args:
        rows: The annotated Slovene catalogue.

    Returns:
        One line per provenance class, and one line per translation system
        behind the machine-translated and mixed rows.
    """
    classes = [
        {
            "provenance_class": klass,
            "datasets": sum(1 for r in rows if r["provenance_class"] == klass),
            "datasets_checked": sum(
                1 for r in rows if r["provenance_class"] == klass and r["provenance_checked"] == "yes"
            ),
            "documents": sum(number(r["documents"]) or 0 for r in rows if r["provenance_class"] == klass),
            "words": sum(number(r["words"]) or 0 for r in rows if r["provenance_class"] == klass),
        }
        for klass in PROVENANCE_CLASSES
    ]
    # a mixed row counts only when machine translation is part of the mix
    machine = [
        r
        for r in rows
        if r["provenance_class"] == "machine_translated"
        or (r["provenance_class"] == "mixed" and r["translation_system"] not in ("", "human"))
    ]
    systems = [
        {
            "translation_system": system,
            "datasets": len(named),
            "names": "; ".join(sorted(r["name"] for r in named)),
        }
        for system in sorted({r["translation_system"] or "unknown" for r in machine})
        for named in [[r for r in machine if (r["translation_system"] or "unknown") == system]]
    ]
    return classes, sorted(systems, key=lambda line: -int(line["datasets"]))


def clause_supply(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    """Totals what the Predictions count: native rows outside the registry.

    Each clause domain is read once per access filter, a row spanning two of
    its taxonomy domains counted once, so these totals are the ones H1 to H4
    are decided on.

    Args:
        rows: The annotated Slovene catalogue.

    Returns:
        One entry per clause domain and access filter.
    """
    lines: List[Dict[str, object]] = []
    for name, spans in CLAUSE_DOMAINS:
        in_domain = [r for r in rows if set(spans) & {d.strip() for d in str(r["domains"]).split(",")}]
        for label, admitted in ACCESS_FILTERS:
            counted = [
                r
                for r in in_domain
                if r["access"] in admitted and r["provenance"] == "native" and r["in_registry"] == "no"
            ]
            documents = sum(number(r["documents"]) or 0 for r in counted)
            tokens = int(sum(number(r["words"]) or 0 for r in counted) * TOKENS_PER_WORD)
            no_documents = sum(1 for r in counted if number(r["documents"]) is None)
            no_words = sum(1 for r in counted if number(r["words"]) is None)
            lines.append(
                {
                    "domain": name,
                    "access_filter": label,
                    "datasets": len(counted),
                    "documents": documents,
                    "tokens_estimated": tokens,
                    "datasets_without_document_count": no_documents,
                    "datasets_without_word_count": no_words,
                    "meets_kpi2": verdict(documents, KPI2_EXAMPLES, no_documents),
                    "meets_kpi4": verdict(tokens, KPI4_TOKENS, no_words),
                }
            )
    return lines


def access_gain(summary: List[Dict[str, object]], quantity: str) -> List[Dict[str, object]]:
    """Pairs each domain's open total with its all-access total for a dumbbell.

    A domain that has no sized dataset at either end is left out: a zero has
    no place on a log axis, and the reported-sizes figure accounts for it.
    Rows where loosening access more than doubles the total are highlighted,
    because that gap — not either endpoint — is what the figure is for.

    Args:
        summary: The per-domain summary.
        quantity: The summary field to pair.

    Returns:
        One `{label, start, end, emphasis}` record per drawable domain.
    """
    totals = {(r["domain"], r["access_filter"]): r[quantity] for r in summary}
    records: List[Dict[str, object]] = []
    for domain in DOMAINS:
        start, end = totals[(domain, "open")], totals[(domain, "all")]
        if start <= 0 or end <= 0:
            continue
        records.append(
            {
                "label": domain,
                "start": start,
                "end": end,
                "emphasis": EMPHASIS.HIGHLIGHT if end / start > 2 else None,
            }
        )
    return records


def draw_figures(summary: List[Dict[str, object]]) -> None:
    """Draws the four supply figures the record links.

    Two dumbbells show what loosening access buys each domain, in tokens and
    in documents, against the KPI thresholds. Two sorted bars carry the
    medical result: the share of each domain written in Slovene, and the share
    of each domain's datasets that publish a size at all.

    Args:
        summary: The per-domain summary.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    # Below the axes: inside, the legend would sit on the smallest domains' rows.
    legend_out = {"location": LEGEND_LOCATION.OUTSIDE_BOTTOM}

    tokens = access_gain(summary, "tokens_estimated")
    save_figure(
        lambda: DumbbellChart(
            tokens,
            title="Estimated tokens by access",
            xlabel="estimated tokens (log scale)",
            ylabel="domain",
            start_name="open",
            end_name="all",
            scaley="log",
            sort=SORT.DESCENDING,
            sort_by=DUMBBELL_SORT_KEY.END,
            show_direction=True,
            show_legend=True,
            legend=legend_out,
            vlines=[{"x": KPI4_TOKENS, "label": "KPI 4 threshold"}],
            figsize=FIG_SIZE.FULL_MEDIUM,
        ),
        FIGURES_DIR / "tokens-by-domain-and-access.svg",
    )

    documents = access_gain(summary, "documents")
    save_figure(
        lambda: DumbbellChart(
            documents,
            title="Documents by access",
            xlabel="documents (log scale)",
            ylabel="domain",
            start_name="open",
            end_name="all",
            scaley="log",
            sort=SORT.DESCENDING,
            sort_by=DUMBBELL_SORT_KEY.END,
            show_direction=True,
            show_legend=True,
            legend=legend_out,
            vlines=[{"x": KPI2_EXAMPLES, "label": "KPI 2 threshold"}],
            figsize=FIG_SIZE.FULL_MEDIUM,
        ),
        FIGURES_DIR / "documents-by-domain-and-access.svg",
    )

    # Medicine is the subject of F1 and F2, so its row is bolded in both bar
    # figures. The others stay unmuted: muting drops their value labels and,
    # in the stack, the very segment the figure is about.
    def emphasis(domain: str) -> Optional[str]:
        return EMPHASIS.HIGHLIGHT if domain == "medical" else None

    opened = {r["domain"]: r for r in summary if r["access_filter"] == "open"}
    native_share = [
        {
            "label": f"{d}  ({opened[d]['documents']:,} docs)",
            "y": 100 * opened[d]["documents_native"] / opened[d]["documents"] if opened[d]["documents"] else 0,
            "emphasis": emphasis(d),
        }
        for d in DOMAINS
    ]
    save_figure(
        lambda: BarChart(
            native_share,
            title="Share of documents written in Slovene",
            xlabel="documents written in Slovene (%)",
            ylabel="domain (open documents)",
            orientation=ORIENTATION.HORIZONTAL,
            sort=SORT.ASCENDING,
            show_values=True,
            value_format="{x:.0f}%",
            xmin=0,
            xmax=100,
            figsize=FIG_SIZE.FULL_MEDIUM,
        ),
        FIGURES_DIR / "documents-by-domain-and-provenance.svg",
    )

    totals = {r["domain"]: r for r in summary if r["access_filter"] == "all"}
    sized: List[Dict[str, object]] = []
    unsized: List[Dict[str, object]] = []
    for d in DOMAINS:
        count = totals[d]["datasets"]
        reported = (
            count
            - totals[d]["datasets_without_word_count"]
            - totals[d]["datasets_sized_by_corpus"]
            - totals[d]["datasets_sized_by_survey"]
        )
        label = f"{d}  ({reported} of {count})"
        sized.append({"label": label, "y": 100 * reported / count, "emphasis": emphasis(d)})
        unsized.append({"label": label, "y": 100 * (count - reported) / count, "emphasis": emphasis(d)})
    save_figure(
        lambda: BarChart(
            [sized, unsized],
            title="Datasets with a reported size",
            xlabel="datasets (%)",
            ylabel="domain (sized of catalogued)",
            subtitle=["size reported", "no size reported"],
            bar_mode=BAR_MODE.STACK,
            orientation=ORIENTATION.HORIZONTAL,
            sort=SORT.ASCENDING,
            sort_by="size reported",
            show_legend=True,
            legend=legend_out,
            xmin=0,
            xmax=100,
            figsize=FIG_SIZE.FULL_MEDIUM,
        ),
        FIGURES_DIR / "reported-sizes-by-domain.svg",
    )


def git_output(*args: str) -> str:
    """Runs one git query in the experiment folder's repository.

    Args:
        *args: The git subcommand and its arguments.

    Returns:
        The command's trimmed output, or an empty string outside a repository.
    """
    try:
        return subprocess.run(["git", *args], cwd=HERE, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def log_lineage(rows: List[Dict[str, str]], summary: List[Dict[str, object]]) -> None:
    """Records this pass as a lineage run, with every table and figure as an artifact.

    The run is tagged the way `labflow:run` asks — tier, branch, base commit,
    type — so `labflow:analyze` can find it, and carries the catalogue's
    headline counts as metrics so the run list reads without opening files.

    Args:
        rows: The annotated Slovene catalogue.
        summary: The per-domain summary.
    """
    if not ml.ensure_experiment(MLFLOW_EXPERIMENT):
        print("mlflow unavailable; lineage not logged")
        return
    tags = {
        "tier": "tracked",
        "branch": git_output("rev-parse", "--abbrev-ref", "HEAD"),
        "base_commit": git_output("rev-parse", "--short", "HEAD"),
        "type": "lineage",
        "language": "sl",
    }
    medical = next(r for r in summary if r["domain"] == "medical" and r["access_filter"] == "open")
    with ml.mlflow_run("catalogue", tags=tags) as run:
        ml.log_params(
            {
                "families": len({f for f, _ in FAMILY_FILES}),
                "tokens_per_word": TOKENS_PER_WORD,
                "kpi2_examples": KPI2_EXAMPLES,
                "kpi4_tokens": KPI4_TOKENS,
            }
        )
        ml.log_metrics(
            {
                "datasets": len(rows),
                "datasets_new": sum(1 for r in rows if r["in_registry"] == "no"),
                "datasets_sized": sum(1 for r in rows if r["words_basis"]),
                "medical_documents_open": int(medical["documents"]),
                "medical_documents_native_open": int(medical["documents_native"]),
                "medical_tokens_open": int(medical["tokens_estimated"]),
            }
        )
        ml.log_artifacts(TABLES_DIR, artifact_path="tables")
        ml.log_artifacts(FIGURES_DIR, artifact_path="figures")
        print(f"mlflow run {run.info.run_id} in {MLFLOW_EXPERIMENT}")


def main() -> None:
    """Builds every table and figure the record links, and optionally logs the pass."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mlflow", action="store_true", help="also record this pass as an MLflow lineage run")
    args = parser.parse_args()

    raw = load_rows()
    rows = annotate(merge_duplicates(raw))
    slovene = [r for r in rows if r.get("part") != "B"]
    fallback = [r for r in rows if r.get("part") == "B"]

    statistics = survey_statistics(raw, rows)
    write_table(TABLES_DIR / "dataset-survey-rows-statistics.csv", list(statistics[0].keys()), statistics)

    write_table(TABLES_DIR / "catalogue.csv", CATALOGUE_COLUMNS, slovene)
    write_table(TABLES_DIR / "catalogue-medical-other-languages.csv", CATALOGUE_COLUMNS, fallback)

    summary = summarise(slovene)
    clauses = clause_supply(slovene)
    classes, systems = provenance_summary(slovene)
    write_table(TABLES_DIR / "provenance-by-class.csv", list(classes[0].keys()), classes)
    write_table(TABLES_DIR / "translation-systems.csv", list(systems[0].keys()), systems)
    write_table(TABLES_DIR / "supply-by-clause.csv", list(clauses[0].keys()), clauses)
    write_table(TABLES_DIR / "supply-by-domain-and-access.csv", list(summary[0].keys()), summary)
    draw_figures(summary)

    print(f"catalogue: {len(slovene)} Slovene rows, {len(fallback)} medical rows in other languages")
    print(f"tables -> {TABLES_DIR}")
    print(f"figures -> {FIGURES_DIR}")
    if args.mlflow:
        log_lineage(slovene, summary)


if __name__ == "__main__":
    main()
