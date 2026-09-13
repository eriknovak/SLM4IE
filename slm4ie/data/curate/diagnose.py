"""Read-only diagnostics over an already-built pretraining corpus.

The corpus top-200 word table surfaces English function words (`the`, `of`,
`and`, ...) at under 1% of mass. The `language` stage already drops whole
foreign documents, so the question is where the residue comes from:

* whole foreign documents that slipped past the document-level filter, which a
  config change can tighten, or
* English passages embedded inside Slovenian-dominant documents, which the
  project accepts as-is.

`diagnose_language_leakage` samples documents from the corpus on disk,
classifies each whole document and each of its newline-delimited paragraphs
with lingua, and reports the whole-document-vs-embedded split, the foreign
character fraction, a foreign-language histogram, per-dataset embedded rates,
and where the English leakage tokens concentrate. Nothing under the data tree
is written.

The detector mirrors `LinguaLanguageFilter` but omits the
minimum-relative-distance gate, so every unit gets a best-guess label;
uncommitted units would otherwise be invisible to the report.
"""

import gzip
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from slm4ie.data.curate.stages import final_corpus_dir
from slm4ie.data.io_utils import resolve_project_path

logger = logging.getLogger(__name__)

#: Candidate set used when the curation config cannot be read. Mirrors the
#: `language` stage's European candidate list.
FALLBACK_CANDIDATES: List[str] = [
    "be",
    "bg",
    "bs",
    "ca",
    "cs",
    "cy",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "eu",
    "fi",
    "fr",
    "ga",
    "hr",
    "hu",
    "is",
    "it",
    "lt",
    "lv",
    "mk",
    "nb",
    "nl",
    "nn",
    "pl",
    "pt",
    "ro",
    "ru",
    "sk",
    "sl",
    "sq",
    "sr",
    "sv",
    "tr",
    "uk",
]

#: English function words whose corpus-wide frequency flagged the leakage.
#: Counted per paragraph to locate where they live (foreign vs target blocks).
LEAKAGE_TOKENS = frozenset({"the", "of", "and", "more", "de", "to", "in", "a", "is", "for"})

#: In-language code: documents and paragraphs in this language are not foreign.
TARGET = "sl"


def load_language_config(config_path: Path) -> Tuple[List[str], bool]:
    """Read the candidate language set and accuracy mode from a curation config.

    Args:
        config_path: Path to the curation config, e.g. configs/data/curate.yaml.

    Returns:
        Tuple `(candidates, low_accuracy)`. Falls back to `FALLBACK_CANDIDATES`
        and `True` when the file or its `language` keys are absent.
    """
    try:
        with config_path.open() as fh:
            cfg = yaml.safe_load(fh) or {}
    except OSError:
        return list(FALLBACK_CANDIDATES), True
    lang_cfg = cfg.get("language") or {}
    candidates = lang_cfg.get("candidates") or list(FALLBACK_CANDIDATES)
    if TARGET not in candidates:
        candidates = [*candidates, TARGET]
    low_accuracy = bool(lang_cfg.get("low_accuracy", True))
    return [str(code).lower() for code in candidates], low_accuracy


def resolve_corpus_dir(config_path: Path, base_dir: Optional[Path] = None) -> Path:
    """Resolve the corpus folder to sample from.

    Args:
        config_path: Path to the curation config, read for its `output_dir`.
        base_dir: Explicit corpus root, which wins over the config.

    Returns:
        Path to the folder holding `<dataset>/<shard>.jsonl.gz`.

    Raises:
        FileNotFoundError: If neither `base_dir` nor the config's `output_dir`
            is set.
    """
    if base_dir is not None:
        return base_dir
    try:
        with config_path.open() as fh:
            cfg = yaml.safe_load(fh) or {}
    except OSError as exc:
        raise FileNotFoundError(f"curation config not readable: {config_path}") from exc
    output_dir = cfg.get("output_dir")
    if output_dir is None:
        raise FileNotFoundError(f"no corpus root: pass --base-dir or set output_dir in {config_path}.")
    return resolve_project_path(output_dir) / final_corpus_dir()


def build_detector(candidates: List[str], low_accuracy: bool) -> Any:
    """Build a lingua detector over the given candidate languages.

    Args:
        candidates: ISO 639-1 candidate codes.
        low_accuracy: Use lingua's faster trigram-only model.

    Returns:
        A built `lingua.LanguageDetector`.

    Raises:
        ValueError: If a candidate code is unknown to lingua.
    """
    from lingua import Language, LanguageDetectorBuilder

    code_to_language = {lang.iso_code_639_1.name.lower(): lang for lang in Language.all()}
    try:
        languages = [code_to_language[code] for code in candidates]
    except KeyError as exc:
        raise ValueError(f"Unknown lingua language code: {exc.args[0]!r}") from exc
    builder = LanguageDetectorBuilder.from_languages(*languages).with_preloaded_language_models()
    if low_accuracy:
        builder = builder.with_low_accuracy_mode()
    return builder.build()


def sample_documents(base_dir: Path, per_dataset: int, max_shards: int) -> Tuple[List[Tuple[str, dict]], List[str]]:
    """Sample documents across every dataset folder under `base_dir`.

    Args:
        base_dir: Corpus root holding `<dataset>/<shard>.jsonl.gz` files.
        per_dataset: Max documents to read per dataset.
        max_shards: Max shards to scan per dataset.

    Returns:
        Tuple `(records, datasets)` where `records` is a list of
        `(dataset, record)` pairs and `datasets` is the dataset names seen.

    Raises:
        FileNotFoundError: If `base_dir` does not exist.
    """
    if not base_dir.is_dir():
        raise FileNotFoundError(f"corpus dir not found: {base_dir}")
    records: List[Tuple[str, dict]] = []
    datasets: List[str] = []
    for dataset_dir in sorted(path for path in base_dir.iterdir() if path.is_dir()):
        shards = sorted(dataset_dir.glob("*.jsonl.gz"))[:max_shards]
        if not shards:
            continue
        datasets.append(dataset_dir.name)
        taken = 0
        for shard in shards:
            if taken >= per_dataset:
                break
            with gzip.open(shard, "rt", encoding="utf-8") as fh:
                for line in fh:
                    records.append((dataset_dir.name, json.loads(line)))
                    taken += 1
                    if taken >= per_dataset:
                        break
    return records, datasets


def analyze_leakage(
    records: List[Tuple[str, dict]],
    detector: Any,
    min_unit_chars: int = 50,
    whole_doc_chars: int = 4000,
    max_paragraphs: int = 40,
) -> Dict[str, Any]:
    """Classify sampled documents and accumulate leakage statistics.

    Args:
        records: `(dataset, record)` pairs to analyze.
        detector: A built lingua detector.
        min_unit_chars: Paragraphs shorter than this are not classified
            (insufficient signal) and counted as undecided.
        whole_doc_chars: Truncate document text to this many chars for the
            whole-document classification.
        max_paragraphs: Classify at most this many paragraphs per document, to
            bound per-document cost on very long pages.

    Returns:
        A dict of aggregate counters and breakdowns, consumed by
        `format_report`.
    """
    doc_class: Counter = Counter()  # pure_target | embedded | whole_foreign
    foreign_lang_chars: Counter = Counter()
    target_chars = foreign_chars = undecided_chars = 0
    tokens_in_foreign: Counter = Counter()
    tokens_in_target: Counter = Counter()
    per_dataset_embedded: Counter = Counter()
    per_dataset_total: Counter = Counter()
    whole_foreign_examples: List[Tuple[str, str, str]] = []
    embedded_examples: List[Tuple[str, str, str]] = []

    for index, (dataset, record) in enumerate(records):
        if index and index % 200 == 0:
            logger.info("[diagnose] %s/%s docs", f"{index:,}", f"{len(records):,}")
        text = record.get("text", "") or ""
        per_dataset_total[dataset] += 1
        whole_lang = _classify(detector, text[:whole_doc_chars]) if text.strip() else None

        has_foreign_paragraph = False
        for paragraph in _paragraphs(text)[:max_paragraphs]:
            if len(paragraph) < min_unit_chars:
                undecided_chars += len(paragraph)
                continue
            lang = _classify(detector, paragraph)
            if lang is None:
                undecided_chars += len(paragraph)
            elif lang == TARGET:
                target_chars += len(paragraph)
                tokens_in_target += _count_tokens(paragraph)
            else:
                foreign_chars += len(paragraph)
                foreign_lang_chars[lang] += len(paragraph)
                tokens_in_foreign += _count_tokens(paragraph)
                has_foreign_paragraph = True

        if whole_lang is not None and whole_lang != TARGET:
            doc_class["whole_foreign"] += 1
            if len(whole_foreign_examples) < 8:
                whole_foreign_examples.append((dataset, whole_lang, text[:120].replace("\n", " ")))
        elif has_foreign_paragraph:
            doc_class["embedded"] += 1
            per_dataset_embedded[dataset] += 1
            if len(embedded_examples) < 8:
                embedded_examples.append((dataset, whole_lang or "?", text[:120].replace("\n", " ")))
        else:
            doc_class["pure_target"] += 1

    return {
        "n_docs": len(records),
        "doc_class": doc_class,
        "foreign_lang_chars": foreign_lang_chars,
        "target_chars": target_chars,
        "foreign_chars": foreign_chars,
        "undecided_chars": undecided_chars,
        "tokens_in_foreign": tokens_in_foreign,
        "tokens_in_target": tokens_in_target,
        "per_dataset_embedded": per_dataset_embedded,
        "per_dataset_total": per_dataset_total,
        "whole_foreign_examples": whole_foreign_examples,
        "embedded_examples": embedded_examples,
    }


def format_report(stats: Dict[str, Any]) -> str:
    """Render the aggregate statistics as a human-readable report.

    Args:
        stats: The aggregate dict returned by `analyze_leakage`.

    Returns:
        The report text, without a trailing newline.
    """
    n_docs = int(stats["n_docs"])
    doc_class = stats["doc_class"]
    target_chars = int(stats["target_chars"])
    foreign_chars = int(stats["foreign_chars"])
    undecided_chars = int(stats["undecided_chars"])
    classified = target_chars + foreign_chars

    lines: List[str] = [
        "=" * 72,
        f"LANGUAGE-LEAKAGE DIAGNOSTIC  —  {n_docs:,} docs sampled",
        "=" * 72,
        "",
        "Document classification:",
    ]
    for key in ("pure_target", "embedded", "whole_foreign"):
        lines.append(f"  {key:14s} {doc_class.get(key, 0):>8,}  ({_pct(doc_class.get(key, 0), n_docs)})")

    lines += [
        "",
        "Character mass (classified paragraphs only):",
        f"  {TARGET:<15s}{target_chars:>14,}  ({_pct(target_chars, classified)} of classified)",
        f"  foreign        {foreign_chars:>14,}  ({_pct(foreign_chars, classified)} of classified)",
        f"  undecided/short{undecided_chars:>14,}  (excluded from the ratio)",
        "",
        "Foreign-character mass by language (top 10):",
    ]
    for lang, chars in stats["foreign_lang_chars"].most_common(10):
        lines.append(f"  {lang:6s} {chars:>12,}  ({_pct(chars, foreign_chars)} of foreign)")

    in_foreign = sum(stats["tokens_in_foreign"].values())
    in_target = sum(stats["tokens_in_target"].values())
    lines += [
        "",
        "Leakage-token location (the/of/and/...):",
        f"  in foreign paragraphs: {in_foreign:>10,}",
        f"  in {TARGET} paragraphs:      {in_target:>10,}",
        f"  -> share inside foreign blocks: {_pct(in_foreign, in_foreign + in_target)}",
        "",
        "Per-dataset embedded-leakage rate (top 12 by rate, min 50 docs):",
    ]
    totals = stats["per_dataset_total"]
    embedded = stats["per_dataset_embedded"]
    rates = [(name, embedded.get(name, 0), total) for name, total in totals.items() if total >= 50]
    for name, hits, total in sorted(rates, key=lambda row: -(row[1] / row[2]))[:12]:
        lines.append(f"  {name:18s} {hits:>6,}/{total:<6,}  ({_pct(hits, total)})")

    lines += ["", "Example WHOLE-FOREIGN docs (slipped past the doc-level filter):"]
    if not stats["whole_foreign_examples"]:
        lines.append("  (none in sample)")
    for dataset, lang, text in stats["whole_foreign_examples"]:
        lines.append(f"  [{dataset}/{lang}] {text!r}")

    lines += ["", f"Example EMBEDDED-leakage docs ({TARGET} overall, foreign paragraphs):"]
    if not stats["embedded_examples"]:
        lines.append("  (none in sample)")
    for dataset, lang, text in stats["embedded_examples"]:
        lines.append(f"  [{dataset}/{lang}] {text!r}")

    n_embedded = doc_class.get("embedded", 0)
    n_whole_foreign = doc_class.get("whole_foreign", 0)
    if n_whole_foreign <= n_embedded * 0.1:
        verdict = (
            "Leakage is dominated by EMBEDDED foreign passages. Per the project "
            "decision, this is acceptable as-is — no scrubber/config change."
        )
    else:
        verdict = (
            "A non-trivial share is WHOLE FOREIGN docs slipping past the filter. "
            "Consider config-only tightening (minimum_relative_distance / max_chars)."
        )
    lines += ["", "-" * 72, f"VERDICT: {verdict}", "-" * 72]
    return "\n".join(lines)


def diagnose_language_leakage(
    *,
    pretrain_config: Path,
    base_dir: Optional[Path] = None,
    per_dataset: int = 2000,
    max_shards: int = 2,
    min_unit_chars: int = 50,
    whole_doc_chars: int = 4000,
    max_paragraphs: int = 40,
    candidates: Optional[List[str]] = None,
) -> str:
    """Sample the corpus, classify it, and return the leakage report.

    Args:
        pretrain_config: Path to the curation config, read for the corpus root
            and the detector's candidate set and accuracy mode.
        base_dir: Corpus root override, winning over the config's `output_dir`.
        per_dataset: Max documents sampled per dataset.
        max_shards: Max shards scanned per dataset.
        min_unit_chars: Paragraphs shorter than this are counted as undecided.
        whole_doc_chars: Chars used for the whole-document classification.
        max_paragraphs: Max paragraphs classified per document.
        candidates: ISO 639-1 candidate override. A smaller focused set is much
            faster and still separates the target language from English.

    Returns:
        The report text produced by `format_report`.

    Raises:
        FileNotFoundError: If the corpus root cannot be resolved or is missing.
        ValueError: If a candidate code is unknown to lingua.
    """
    config_candidates, low_accuracy = load_language_config(pretrain_config)
    if candidates:
        config_candidates = [code.strip().lower() for code in candidates if code.strip()]
        if TARGET not in config_candidates:
            config_candidates.append(TARGET)
    logger.info("[diagnose] detector: %d candidates, low_accuracy=%s", len(config_candidates), low_accuracy)

    corpus_dir = resolve_corpus_dir(pretrain_config, base_dir)
    detector = build_detector(config_candidates, low_accuracy)
    records, datasets = sample_documents(corpus_dir, per_dataset, max_shards)
    logger.info("[diagnose] sampled %s docs from %d datasets under %s", f"{len(records):,}", len(datasets), corpus_dir)

    stats = analyze_leakage(records, detector, min_unit_chars, whole_doc_chars, max_paragraphs)
    return format_report(stats)


def _classify(detector: Any, text: str) -> Optional[str]:
    """Return lingua's best-guess ISO 639-1 code for `text`, or None.

    Args:
        detector: A built lingua detector.
        text: The text to classify.

    Returns:
        Lowercased ISO 639-1 code, or `None` when lingua does not commit.
    """
    predicted = detector.detect_language_of(text)
    return predicted.iso_code_639_1.name.lower() if predicted is not None else None


def _paragraphs(text: str) -> List[str]:
    """Split text into non-empty newline-delimited paragraphs.

    Args:
        text: Document text.

    Returns:
        List of stripped, non-empty paragraphs.
    """
    return [line.strip() for line in text.split("\n") if line.strip()]


def _count_tokens(text: str) -> Counter:
    """Count whole-word occurrences of the leakage tokens in `text`.

    Args:
        text: Text to scan, in any case.

    Returns:
        Counter mapping each present leakage token to its occurrence count.
    """
    counts: Counter = Counter()
    for word in re.findall(r"[a-zA-Zčšžćđ]+", text.lower()):
        if word in LEAKAGE_TOKENS:
            counts[word] += 1
    return counts


def _pct(part: int, whole: int) -> str:
    """Format `part/whole` as a percentage string.

    Args:
        part: Numerator.
        whole: Denominator.

    Returns:
        A percentage with two decimals, or `0.00%` when `whole` is 0.
    """
    return f"{(100.0 * part / whole) if whole else 0.0:.2f}%"
