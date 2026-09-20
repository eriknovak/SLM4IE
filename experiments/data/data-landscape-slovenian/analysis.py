"""Build the open Slovene data landscape catalogue and its per-domain supply tables.

The survey's evidence is one JSON file per searched source family under
`data/experiments/data/data-landscape-slovenian/raw/`, each a list of rows on the
schema fixed in the record's D4. This script merges them into the committed
catalogue, flags what the project already downloads, reads how each dataset's
Slovene text came to exist, scores each row against KPI 2 and KPI 4, and
aggregates per domain under the three access filters of D8.

Run it from anywhere:

    uv run --group analysis python experiments/data/data-landscape-slovenian/analysis.py
"""

import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import yaml
from datachart.charts import BarChart, DumbbellChart
from datachart.constants import BAR_MODE, DUMBBELL_SORT_KEY, EMPHASIS, FIG_SIZE, LEGEND_LOCATION, ORIENTATION, SORT

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from report_figures import save_figure  # noqa: E402  — path set above so the vendored helper resolves

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RAW_DIR = ROOT / "data" / "experiments" / "data" / "data-landscape-slovenian" / "raw"
TABLES_DIR = HERE / "tables"
FIGURES_DIR = HERE / "figures"
REGISTRY = ROOT / "configs" / "data" / "download.yaml"

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
    "words",
    "tokens_estimated",
    "size_verified",
    "annotation",
    "length_class",
    "format",
    "provenance",
    "in_registry",
    "kpi2_fit",
    "kpi4_fit",
    "family",
    "mirrors",
    "notes",
]

# Registry entries fetched by hand, so they name no URL or Hub repo: matched on
# a fragment of the catalogue's own name for the same corpus instead.
REGISTRY_NAME_FRAGMENTS: Tuple[str, ...] = ("gigafida22", "metafida10", "slovenetrendi")

# Registry entries the catalogue found at a different address than the one the
# project downloads from, keyed by the Hub repo that publishes the same corpus.
REGISTRY_REPO_ALIASES: Tuple[str, ...] = ("statmt/cc100",)


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


def registry_keys() -> Set[str]:
    """Collects the identifiers of every dataset the project already downloads.

    A registry entry is identified by its CLARIN or LINDAT handle or its Hugging
    Face repo; the entries fetched by hand carry neither and are matched on the
    name instead, by `in_registry`.

    Returns:
        Identifiers comparable with what `row_keys` reads off a catalogue row.
    """
    registry = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))["datasets"]
    keys: Set[str] = set(REGISTRY_REPO_ALIASES)
    for entry in registry.values():
        if entry.get("repo_id"):
            keys.add(str(entry["repo_id"]).lower())
        for url in entry.get("urls") or []:
            keys |= url_keys(str(url))
    return keys


def in_registry(row: Dict[str, str], keys: Set[str]) -> bool:
    """Decides whether the project already downloads this dataset.

    Args:
        row: A merged catalogue row.
        keys: The registry identifiers from `registry_keys`.

    Returns:
        True when the row's address or name matches a registry entry.
    """
    if row_keys(row) & keys:
        return True
    name = normalise(row["name"])
    return any(fragment in name for fragment in REGISTRY_NAME_FRAGMENTS)


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

    Args:
        rows: Merged catalogue rows.

    Returns:
        The same rows, each with `tokens_estimated`, `in_registry`, `kpi2_fit`
        and `kpi4_fit` filled in.
    """
    known = registry_keys()
    annotated: List[Dict[str, str]] = []
    for row in rows:
        words = number(row["words"])
        kpi2, kpi4 = kpi_fit(row)
        annotated.append(
            {
                **row,
                "tokens_estimated": str(int(words * TOKENS_PER_WORD)) if words is not None else "",
                "provenance": provenance(row),
                "in_registry": "yes" if in_registry(row, known) else "no",
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
        count, missing = totals[d]["datasets"], totals[d]["datasets_without_word_count"]
        label = f"{d}  ({count - missing} of {count})"
        sized.append({"label": label, "y": 100 * (count - missing) / count, "emphasis": emphasis(d)})
        unsized.append({"label": label, "y": 100 * missing / count, "emphasis": emphasis(d)})
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


def main() -> None:
    """Builds every table and figure the record links."""
    rows = annotate(merge_duplicates(load_rows()))
    slovene = [r for r in rows if r.get("part") != "B"]
    fallback = [r for r in rows if r.get("part") == "B"]

    write_table(TABLES_DIR / "catalogue.csv", CATALOGUE_COLUMNS, slovene)
    write_table(TABLES_DIR / "catalogue-medical-other-languages.csv", CATALOGUE_COLUMNS, fallback)

    summary = summarise(slovene)
    write_table(TABLES_DIR / "supply-by-domain-and-access.csv", list(summary[0].keys()), summary)
    draw_figures(summary)

    print(f"catalogue: {len(slovene)} Slovene rows, {len(fallback)} medical rows in other languages")
    print(f"tables -> {TABLES_DIR}")
    print(f"figures -> {FIGURES_DIR}")


if __name__ == "__main__":
    main()
