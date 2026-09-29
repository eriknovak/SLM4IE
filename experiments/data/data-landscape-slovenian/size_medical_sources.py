"""Size the new native Slovene medical sources by downloading and counting them.

The catalogue's medical rows outside the download registry publish no word
count (F2). This script fetches each source — whole where it is small or
reachable through an API, a fixed-seed sample where it is one file per item —
and counts words, writing `tables/native-medical-sizes.csv` for `analysis.py`
to join back into the catalogue as a third `words_basis`.

Everything fetched lands under `data/experiments/data/data-landscape-slovenian/
interim/<source>/` and is reused on a rerun, so the script is resumable and a
second run re-counts without re-downloading. Requests carry a contact address
and are spaced out; a source is never hit faster than one request per two
seconds except the Wikipedia API.

    uv run --group analysis python experiments/data/data-landscape-slovenian/size_medical_sources.py
        [--only zdravniski-vestnik,clinical-guidelines] [--force]
"""

import argparse
import csv
import json
import random
import re
import statistics
import time
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional

from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
INTERIM = ROOT / "data" / "experiments" / "data" / "data-landscape-slovenian" / "interim"
SIZES_TABLE = HERE / "tables" / "native-medical-sizes.csv"

USER_AGENT = "SLM4IE-survey/0.1 (research corpus survey; novak.erik@gmail.com)"
POLITE_DELAY = 2.0  # seconds between requests to a publisher or university site
API_DELAY = 0.2  # the Wikipedia API tolerates far more; this keeps well under it
SAMPLE_SIZE = 40
SEED = 20260921  # the day the sample was drawn, so a rerun draws the same items

# The catalogue row each size belongs to, by the row's exact name.
CATALOGUE_NAMES: Dict[str, str] = {
    "clinical-case-reports": "Slovenian Relation Extraction (clinical case reports)",
    "wikipedia-medicina": "Slovenska Wikipedija — Kategorija:Medicina",
    "zdravniski-vestnik": "Zdravniški vestnik — official archive",
    "clinical-guidelines": "Slovenian clinical guidelines list (UL Medical Faculty)",
}

HF_SPLITS = "https://datasets-server.huggingface.co/splits?dataset=NLP-FBK/Slovenian_Relation_Extraction"
HF_ROWS = "https://datasets-server.huggingface.co/rows?dataset=NLP-FBK/Slovenian_Relation_Extraction&config=default"
WIKI_API = "https://sl.wikipedia.org/w/api.php"
WIKI_ROOT_CATEGORY = "Kategorija:Medicina"
WIKI_DEPTH = 2
VESTNIK_OAI = "https://vestnik.szd.si/index.php/ZdravVest/oai"
VESTNIK_TOTAL_HINT = 2549  # what ListRecords reported on 2026-09-21; the harvest recounts it
GUIDELINES_PAGE = "https://libguides.mf.uni-lj.si/c.php?g=221983&p=1469272"
GUIDELINES_TOTAL = 112  # the count the page states; links that resolve to a PDF are what gets counted
GUIDELINE_HOSTS = ("mail.szd.si", "vestnik.szd.si", "ricinus2.mf.uni-lj.si")

SIZE_COLUMNS: List[str] = [
    "source",
    "catalogue_name",
    "items_total",
    "items_counted",
    "words_counted",
    "words_estimated",
    "basis",
    "method",
    "sized_on",
]


def fetch(url: str, delay: float, binary: bool = False) -> Optional[object]:
    """Fetches one URL politely, returning None on any HTTP or network failure.

    Args:
        url: What to fetch.
        delay: Seconds to sleep after the request, whatever its outcome.
        binary: Return raw bytes rather than decoded text.

    Returns:
        The body as text or bytes, or None when the request failed.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read()
            kind = response.headers.get("Content-Type", "")
    except Exception as error:  # noqa: BLE001 — one bad item must not stop a survey
        print(f"  failed {url}: {error}")
        return None
    finally:
        time.sleep(delay)
    if binary:
        return body if "pdf" in kind or body[:5] == b"%PDF-" else None
    return body.decode("utf-8", errors="ignore")


def count_words(text: str) -> int:
    """Counts whitespace-delimited words, the unit the catalogue's `words` uses.

    Args:
        text: Extracted plain text.

    Returns:
        The word count.
    """
    return len(text.split())


def pdf_words(path: Path) -> int:
    """Extracts a PDF's text and counts its words; an unreadable PDF counts zero.

    Args:
        path: The PDF file.

    Returns:
        The word count.
    """
    try:
        reader = PdfReader(str(path))
        return sum(count_words(page.extract_text() or "") for page in reader.pages)
    except Exception as error:  # noqa: BLE001 — a scanned or broken PDF is a zero, not a crash
        print(f"  unreadable {path.name}: {error}")
        return 0


def size_row(source: str, total: int, counted: int, words: int, basis: str, method: str) -> Dict[str, object]:
    """Builds one line of the sizes table.

    A `sampled` source's estimate is the mean words per counted item times the
    item total. A `full` source's estimate is its count; when fewer items
    resolved than the source holds — dead links, mostly — the basis becomes
    `partial` and the count stands as a lower bound rather than an estimate.

    Args:
        source: The interim folder name.
        total: How many items the source holds.
        counted: How many were fetched and counted.
        words: Words counted over the fetched items.
        basis: `full` or `sampled`.
        method: One sentence on how the items were obtained.

    Returns:
        The row, keyed by `SIZE_COLUMNS`.
    """
    if basis == "full" and counted < total:
        basis = "partial"
    estimated = words if basis != "sampled" else (round(words / counted * total) if counted else 0)
    return {
        "source": source,
        "catalogue_name": CATALOGUE_NAMES[source],
        "items_total": total,
        "items_counted": counted,
        "words_counted": words,
        "words_estimated": estimated,
        "basis": basis,
        "method": method,
        "sized_on": date.today().isoformat(),
    }


def size_clinical_case_reports(force: bool) -> Dict[str, object]:
    """Counts every row of the NLP-FBK clinical case report set through the Hub rows API.

    Args:
        force: Re-fetch even when the rows are already saved.

    Returns:
        The sizes-table row.
    """
    folder = INTERIM / "clinical-case-reports"
    folder.mkdir(parents=True, exist_ok=True)
    saved = folder / "rows.jsonl"
    if force or not saved.exists():
        listing = fetch(HF_SPLITS, API_DELAY)
        splits = [s["split"] for s in json.loads(listing or '{"splits": []}')["splits"] if s["config"] == "default"]
        with saved.open("w", encoding="utf-8") as out:
            for split in splits:
                offset = 0
                while True:
                    body = fetch(f"{HF_ROWS}&split={split}&offset={offset}&length=100", API_DELAY)
                    if body is None:
                        break
                    rows = json.loads(body).get("rows", [])
                    for item in rows:
                        out.write(json.dumps({"split": split, **item["row"]}, ensure_ascii=False) + "\n")
                    if len(rows) < 100:
                        break
                    offset += 100
    rows = [json.loads(line) for line in saved.read_text(encoding="utf-8").splitlines()]
    # one row is one sentence of a case report; the longest string field is the sentence itself
    words = sum(max((count_words(v) for v in r.values() if isinstance(v, str)), default=0) for r in rows)
    reports = len({r.get("text_id") for r in rows})
    method = f"every row via the Hub rows API; {len(rows)} sentences over {reports} case reports"
    return size_row("clinical-case-reports", reports, reports, words, "full", method)


def wiki_query(params: Dict[str, str]) -> Iterator[Dict]:
    """Runs one Wikipedia API query, following its continuation.

    Args:
        params: The query parameters, without format and action.

    Yields:
        Each page of the response's `query` block.
    """
    base = {"format": "json", "action": "query", **params}
    continuation: Dict[str, str] = {}
    while True:
        body = fetch(f"{WIKI_API}?{urllib.parse.urlencode({**base, **continuation})}", API_DELAY)
        if body is None:
            return
        data = json.loads(body)
        yield data.get("query", {})
        if "continue" not in data:
            return
        continuation = data["continue"]


def size_wikipedia_medicina(force: bool) -> Dict[str, object]:
    """Counts the plain text of every article within two levels of the medicine category.

    Args:
        force: Re-fetch even when the articles are already saved.

    Returns:
        The sizes-table row.
    """
    folder = INTERIM / "wikipedia-medicina"
    folder.mkdir(parents=True, exist_ok=True)
    saved = folder / "articles.jsonl"
    if force or not saved.exists():
        pages: Dict[int, str] = {}
        queue: List[tuple] = [(WIKI_ROOT_CATEGORY, 0)]
        seen = set()
        while queue:
            category, depth = queue.pop(0)
            if category in seen or depth > WIKI_DEPTH:
                continue
            seen.add(category)
            for block in wiki_query({"list": "categorymembers", "cmtitle": category, "cmlimit": "500"}):
                for member in block.get("categorymembers", []):
                    if member["ns"] == 14:
                        queue.append((member["title"], depth + 1))
                    elif member["ns"] == 0:
                        pages[member["pageid"]] = member["title"]
        print(f"  {len(pages)} articles under {len(seen)} categories")
        with saved.open("w", encoding="utf-8") as out:
            for index, (pageid, title) in enumerate(sorted(pages.items()), 1):
                text = ""
                for block in wiki_query({"prop": "extracts", "explaintext": "1", "pageids": str(pageid)}):
                    text = block.get("pages", {}).get(str(pageid), {}).get("extract", "")
                out.write(json.dumps({"pageid": pageid, "title": title, "words": count_words(text)}) + "\n")
                if index % 200 == 0:
                    print(f"  {index}/{len(pages)} articles fetched")
    articles = [json.loads(line) for line in saved.read_text(encoding="utf-8").splitlines()]
    words = sum(a["words"] for a in articles)
    method = f"category walk to depth {WIKI_DEPTH}, plain-text extracts via the API"
    return size_row("wikipedia-medicina", len(articles), len(articles), words, "full", method)


def harvest_vestnik(saved: Path) -> None:
    """Harvests every Zdravniški vestnik record through OAI-PMH into one JSONL file.

    Args:
        saved: Where the records go, one per line with identifier, language and title.
    """
    token: Optional[str] = None
    with saved.open("w", encoding="utf-8") as out:
        while True:
            query = (
                f"verb=ListRecords&resumptionToken={urllib.parse.quote(token)}"
                if token
                else "verb=ListRecords&metadataPrefix=oai_dc"
            )
            body = fetch(f"{VESTNIK_OAI}?{query}", POLITE_DELAY)
            if body is None:
                break
            for record in re.findall(r"<record>(.*?)</record>", body, re.S):
                identifier = re.search(r"<dc:identifier>(https?://[^<]+)</dc:identifier>", record)
                language = re.search(r"<dc:language>([^<]+)</dc:language>", record)
                title = re.search(r"<dc:title[^>]*>([^<]*)</dc:title>", record)
                if identifier:
                    out.write(
                        json.dumps(
                            {
                                "identifier": identifier.group(1),
                                "language": language.group(1) if language else "",
                                "title": title.group(1) if title else "",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            match = re.search(r"<resumptionToken[^>]*>([^<]+)</resumptionToken>", body)
            if not match:
                break
            token = match.group(1)


def size_zdravniski_vestnik(force: bool) -> Dict[str, object]:
    """Samples the journal's Slovene-language articles and extrapolates from their PDFs.

    Args:
        force: Re-harvest and re-fetch even when the files are already saved.

    Returns:
        The sizes-table row.
    """
    folder = INTERIM / "zdravniski-vestnik"
    pdfs = folder / "pdf"
    pdfs.mkdir(parents=True, exist_ok=True)
    records_file = folder / "records.jsonl"
    if force or not records_file.exists():
        harvest_vestnik(records_file)
    records = [json.loads(line) for line in records_file.read_text(encoding="utf-8").splitlines()]
    slovene = [r for r in records if r["language"].lower().startswith("sl")]
    print(f"  {len(records)} records harvested, {len(slovene)} in Slovene")
    sample = random.Random(SEED).sample(slovene, min(SAMPLE_SIZE, len(slovene)))
    counts: List[int] = []
    for record in sample:
        article_id = record["identifier"].rstrip("/").split("/")[-1]
        target = pdfs / f"{article_id}.pdf"
        if force or not target.exists():
            page = fetch(record["identifier"], POLITE_DELAY)
            galley = re.search(
                r'href="([^"]*/article/(?:view|download)/' + re.escape(article_id) + r'/\d+)"', page or ""
            )
            if galley:
                body = fetch(galley.group(1).replace("/article/view/", "/article/download/"), POLITE_DELAY, binary=True)
                if body:
                    target.write_bytes(body)
        if target.exists():
            counts.append(pdf_words(target))
    words = sum(counts)
    method = f"OAI-PMH harvest, {len(sample)} Slovene articles drawn with seed {SEED}, PDF text via pypdf"
    print(f"  {len(counts)} PDFs counted, median {statistics.median(counts) if counts else 0} words")
    return size_row("zdravniski-vestnik", len(slovene), len(counts), words, "sampled", method)


def size_clinical_guidelines(force: bool) -> Dict[str, object]:
    """Fetches every guideline the faculty's list links to a PDF and counts them all.

    Args:
        force: Re-fetch even when the PDFs are already saved.

    Returns:
        The sizes-table row.
    """
    folder = INTERIM / "clinical-guidelines"
    pdfs = folder / "pdf"
    pdfs.mkdir(parents=True, exist_ok=True)
    links_file = folder / "links.json"
    if force or not links_file.exists():
        page = fetch(GUIDELINES_PAGE, POLITE_DELAY) or ""
        links = []
        for href in re.findall(r'href="(https?://[^"]+)"', page):
            host = urllib.parse.urlparse(href).netloc
            if href.lower().endswith(".pdf") or host in GUIDELINE_HOSTS:
                links.append(href)
        links = sorted(set(links))
        links_file.write_text(json.dumps(links, indent=1), encoding="utf-8")
    links = json.loads(links_file.read_text(encoding="utf-8"))
    print(f"  {len(links)} candidate links")
    counts: List[int] = []
    for index, href in enumerate(links, 1):
        target = pdfs / f"{index:03d}.pdf"
        if force or not target.exists():
            body = fetch(href, POLITE_DELAY, binary=True)
            if body:
                target.write_bytes(body)
        if target.exists():
            counts.append(pdf_words(target))
    words = sum(counts)
    method = f"every link on the faculty list that resolves to a PDF ({len(counts)} of {len(links)} candidates)"
    return size_row("clinical-guidelines", GUIDELINES_TOTAL, len(counts), words, "full", method)


SOURCES: Dict[str, Callable[[bool], Dict[str, object]]] = {
    "clinical-case-reports": size_clinical_case_reports,
    "wikipedia-medicina": size_wikipedia_medicina,
    "zdravniski-vestnik": size_zdravniski_vestnik,
    "clinical-guidelines": size_clinical_guidelines,
}


def main() -> None:
    """Sizes the selected sources and writes the sizes table."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", help="comma-separated subset of: " + ", ".join(SOURCES))
    parser.add_argument("--force", action="store_true", help="re-download instead of reusing interim files")
    args = parser.parse_args()
    selected = args.only.split(",") if args.only else list(SOURCES)

    existing: Dict[str, Dict[str, object]] = {}
    if SIZES_TABLE.exists():
        with SIZES_TABLE.open(encoding="utf-8") as handle:
            existing = {r["source"]: r for r in csv.DictReader(handle)}
    for name in selected:
        print(f"{name}:")
        existing[name] = SOURCES[name](args.force)
        print(f"  -> {existing[name]['words_estimated']:,} words ({existing[name]['basis']})")

    SIZES_TABLE.parent.mkdir(parents=True, exist_ok=True)
    with SIZES_TABLE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SIZE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(existing[name] for name in SOURCES if name in existing)
    print(f"sizes -> {SIZES_TABLE}")


if __name__ == "__main__":
    main()
