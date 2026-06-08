"""Diagnose foreign-language leakage in the pretraining corpus (read-only).

The corpus top-200 word table surfaces English function words (`the`,
`of`, `and`, ...) at <1% of mass. The document-level lingua filter already
drops whole foreign documents, so this script measures whether the residual
leakage comes from:

  * whole foreign documents that slipped past the doc-level filter
    (a config-tightenable problem), or
  * English passages embedded inside Slovenian-dominant documents
    (accepted as-is per project decision).

It samples documents from the existing on-disk corpus, classifies each
whole document and each of its newline-delimited paragraphs with lingua
(reusing the candidate set from `configs/data/pretrain.yaml::language`),
and reports the whole-doc-vs-embedded split, the foreign-character
fraction, a foreign-language histogram, per-dataset embedded rates, and
where the English leakage tokens concentrate.

This script is strictly read-only: it reads corpus shards and the config,
and writes nothing to the data tree.

Example:
    uv run python scripts/analysis/diagnose_language_leakage.py
    uv run python scripts/analysis/diagnose_language_leakage.py \
        --base-dir /vault/data/SLM4IE/pretrain/04_2_dedup --per-dataset 3000
"""

import argparse
import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

#: Default location of the final corpus on disk. Still the pre-renumber
#: folder name; the code-level rename to `05_2_dedup` only lands after the
#: next pipeline run.
DEFAULT_BASE_DIR = Path("/vault/data/SLM4IE/pretrain/04_2_dedup")

#: Default path to the pipeline config, used to mirror the detector's
#: candidate language set and accuracy mode.
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "data" / "pretrain.yaml"

#: Fallback candidate set if the config cannot be read (mirrors the
#: language stage's European candidate list).
FALLBACK_CANDIDATES: List[str] = [
    "be", "bg", "bs", "ca", "cs", "cy", "da", "de", "el", "en", "es", "et",
    "eu", "fi", "fr", "ga", "hr", "hu", "is", "it", "lt", "lv", "mk", "nb",
    "nl", "nn", "pl", "pt", "ro", "ru", "sk", "sl", "sq", "sr", "sv", "tr", "uk",
]

#: English function words whose corpus-wide frequency flagged the leakage.
#: Counted per-paragraph to locate where they live (foreign vs sl blocks).
LEAKAGE_TOKENS = frozenset({"the", "of", "and", "more", "de", "to", "in", "a", "is", "for"})

#: Target (in-language) code: documents/paragraphs in this language are
#: not foreign.
TARGET = "sl"


def load_language_config(config_path: Path) -> Tuple[List[str], bool]:
    """Read the candidate language set and accuracy mode from the config.

    Args:
        config_path: Path to `pretrain.yaml`.

    Returns:
        Tuple `(candidates, low_accuracy)`. Falls back to
        `FALLBACK_CANDIDATES` and `True` when the file or keys are absent.
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
    return [str(c).lower() for c in candidates], low_accuracy


def build_detector(candidates: List[str], low_accuracy: bool):
    """Build a lingua detector over the given candidate languages.

    Mirrors `slm4ie.data.curate.language.LinguaLanguageFilter._ensure_detector`
    but omits the minimum-relative-distance gate so every unit gets a
    best-guess label (uncommitted units would otherwise be invisible).

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


def classify(detector, text: str) -> Optional[str]:
    """Return lingua's best-guess ISO 639-1 code for `text`, or None.

    Args:
        detector: A built lingua detector.
        text: The text to classify.

    Returns:
        Lowercased ISO 639-1 code, or `None` when lingua does not commit.
    """
    predicted = detector.detect_language_of(text)
    return predicted.iso_code_639_1.name.lower() if predicted is not None else None


def iter_sampled_docs(
    base_dir: Path, per_dataset: int, max_shards: int
) -> Tuple[List[Tuple[str, dict]], List[str]]:
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
    for ds_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        shards = sorted(ds_dir.glob("*.jsonl.gz"))[:max_shards]
        if not shards:
            continue
        datasets.append(ds_dir.name)
        taken = 0
        for shard in shards:
            if taken >= per_dataset:
                break
            with gzip.open(shard, "rt", encoding="utf-8") as fh:
                for line in fh:
                    records.append((ds_dir.name, json.loads(line)))
                    taken += 1
                    if taken >= per_dataset:
                        break
    return records, datasets


def paragraphs(text: str) -> List[str]:
    """Split text into non-empty newline-delimited paragraphs.

    Args:
        text: Document text.

    Returns:
        List of stripped, non-empty lines/paragraphs.
    """
    return [p.strip() for p in text.split("\n") if p.strip()]


def count_tokens(text: str) -> Counter:
    """Count whole-word occurrences of the leakage tokens in `text`.

    Args:
        text: Text to scan (any case).

    Returns:
        Counter mapping each present leakage token to its occurrence count.
    """
    out: Counter = Counter()
    for word in re.findall(r"[a-zA-Zčšžćđ]+", text.lower()):
        if word in LEAKAGE_TOKENS:
            out[word] += 1
    return out


def analyze(
    records: List[Tuple[str, dict]],
    detector,
    min_unit_chars: int,
    whole_doc_chars: int,
    max_paragraphs: int = 40,
) -> Dict[str, object]:
    """Classify sampled docs and accumulate leakage statistics.

    Args:
        records: `(dataset, record)` pairs to analyze.
        detector: A built lingua detector.
        min_unit_chars: Paragraphs shorter than this are not classified
            (insufficient signal) and counted as undecided.
        whole_doc_chars: Truncate doc text to this many chars for the
            whole-document classification.
        max_paragraphs: Classify at most this many paragraphs per doc to
            bound per-doc cost on very long pages.

    Returns:
        A dict of aggregate counters and breakdowns for reporting.
    """
    doc_class = Counter()  # pure_sl | embedded | whole_foreign
    foreign_lang_chars: Counter = Counter()
    sl_chars = foreign_chars = undecided_chars = 0
    tokens_in_foreign: Counter = Counter()
    tokens_in_sl: Counter = Counter()
    per_dataset_embedded = Counter()
    per_dataset_total = Counter()
    whole_foreign_examples: List[Tuple[str, str, str]] = []
    embedded_examples: List[Tuple[str, str, str]] = []

    for i, (dataset, rec) in enumerate(records):
        if i and i % 200 == 0:
            print(f"  ... {i:,}/{len(records):,} docs", file=sys.stderr, flush=True)
        text = rec.get("text", "") or ""
        per_dataset_total[dataset] += 1
        whole_lang = classify(detector, text[:whole_doc_chars]) if text.strip() else None

        has_foreign_para = False
        for para in paragraphs(text)[:max_paragraphs]:
            if len(para) < min_unit_chars:
                undecided_chars += len(para)
                continue
            lang = classify(detector, para)
            if lang is None:
                undecided_chars += len(para)
            elif lang == TARGET:
                sl_chars += len(para)
                tokens_in_sl += count_tokens(para)
            else:
                foreign_chars += len(para)
                foreign_lang_chars[lang] += len(para)
                tokens_in_foreign += count_tokens(para)
                has_foreign_para = True

        if whole_lang is not None and whole_lang != TARGET:
            doc_class["whole_foreign"] += 1
            if len(whole_foreign_examples) < 8:
                whole_foreign_examples.append((dataset, whole_lang, text[:120].replace("\n", " ")))
        elif has_foreign_para:
            doc_class["embedded"] += 1
            per_dataset_embedded[dataset] += 1
            if len(embedded_examples) < 8:
                embedded_examples.append((dataset, whole_lang or "?", text[:120].replace("\n", " ")))
        else:
            doc_class["pure_sl"] += 1

    return {
        "n_docs": len(records),
        "doc_class": doc_class,
        "foreign_lang_chars": foreign_lang_chars,
        "sl_chars": sl_chars,
        "foreign_chars": foreign_chars,
        "undecided_chars": undecided_chars,
        "tokens_in_foreign": tokens_in_foreign,
        "tokens_in_sl": tokens_in_sl,
        "per_dataset_embedded": per_dataset_embedded,
        "per_dataset_total": per_dataset_total,
        "whole_foreign_examples": whole_foreign_examples,
        "embedded_examples": embedded_examples,
    }


def _pct(part: int, whole: int) -> str:
    """Format `part/whole` as a percentage string.

    Args:
        part: Numerator.
        whole: Denominator.

    Returns:
        A percentage with two decimals, or `0.00%` when `whole` is 0.
    """
    return f"{(100.0 * part / whole) if whole else 0.0:.2f}%"


def print_report(stats: Dict[str, object]) -> None:
    """Print the human-readable diagnostic report.

    Args:
        stats: The aggregate dict returned by `analyze`.
    """
    n = int(stats["n_docs"])
    dc = stats["doc_class"]
    sl_c = int(stats["sl_chars"])
    fo_c = int(stats["foreign_chars"])
    un_c = int(stats["undecided_chars"])
    classified = sl_c + fo_c

    print("=" * 72)
    print(f"LANGUAGE-LEAKAGE DIAGNOSTIC  —  {n:,} docs sampled")
    print("=" * 72)
    print("\nDocument classification:")
    for key in ("pure_sl", "embedded", "whole_foreign"):
        print(f"  {key:14s} {dc.get(key, 0):>8,}  ({_pct(dc.get(key, 0), n)})")

    print("\nCharacter mass (classified paragraphs only):")
    print(f"  slovenian      {sl_c:>14,}  ({_pct(sl_c, classified)} of classified)")
    print(f"  foreign        {fo_c:>14,}  ({_pct(fo_c, classified)} of classified)")
    print(f"  undecided/short{un_c:>14,}  (excluded from the ratio)")

    print("\nForeign-character mass by language (top 10):")
    for lang, c in stats["foreign_lang_chars"].most_common(10):
        print(f"  {lang:6s} {c:>12,}  ({_pct(c, fo_c)} of foreign)")

    tf = stats["tokens_in_foreign"]
    ts = stats["tokens_in_sl"]
    tot_f = sum(tf.values())
    tot_s = sum(ts.values())
    print("\nLeakage-token location (the/of/and/...):")
    print(f"  in foreign paragraphs: {tot_f:>10,}")
    print(f"  in slovenian paragraphs:{tot_s:>9,}")
    print(f"  -> share inside foreign blocks: {_pct(tot_f, tot_f + tot_s)}")

    print("\nPer-dataset embedded-leakage rate (top 12 by rate, min 50 docs):")
    pdt = stats["per_dataset_total"]
    pde = stats["per_dataset_embedded"]
    rates = [
        (ds, pde.get(ds, 0), tot)
        for ds, tot in pdt.items()
        if tot >= 50
    ]
    for ds, emb, tot in sorted(rates, key=lambda x: -(x[1] / x[2]))[:12]:
        print(f"  {ds:18s} {emb:>6,}/{tot:<6,}  ({_pct(emb, tot)})")

    print("\nExample WHOLE-FOREIGN docs (slipped past the doc-level filter):")
    if not stats["whole_foreign_examples"]:
        print("  (none in sample)")
    for ds, lang, txt in stats["whole_foreign_examples"]:
        print(f"  [{ds}/{lang}] {txt!r}")

    print("\nExample EMBEDDED-leakage docs (sl overall, foreign paragraphs):")
    for ds, lang, txt in stats["embedded_examples"]:
        print(f"  [{ds}/{lang}] {txt!r}")

    print("\n" + "-" * 72)
    embedded = dc.get("embedded", 0)
    whole_foreign = dc.get("whole_foreign", 0)
    if whole_foreign <= embedded * 0.1:
        verdict = (
            "Leakage is dominated by EMBEDDED foreign passages. Per the project "
            "decision, this is acceptable as-is — no scrubber/config change."
        )
    else:
        verdict = (
            "A non-trivial share is WHOLE FOREIGN docs slipping past the filter. "
            "Consider config-only tightening (minimum_relative_distance / max_chars)."
        )
    print(f"VERDICT: {verdict}")
    print("-" * 72)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to `sys.argv`).

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR,
                        help="Corpus root with <dataset>/<shard>.jsonl.gz.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="pretrain.yaml to read the candidate set from.")
    parser.add_argument("--per-dataset", type=int, default=2000,
                        help="Max docs sampled per dataset.")
    parser.add_argument("--max-shards-per-dataset", type=int, default=2,
                        help="Max shards scanned per dataset.")
    parser.add_argument("--min-unit-chars", type=int, default=50,
                        help="Paragraphs shorter than this are not classified.")
    parser.add_argument("--whole-doc-chars", type=int, default=4000,
                        help="Chars used for the whole-document classification.")
    parser.add_argument("--max-paragraphs", type=int, default=40,
                        help="Classify at most this many paragraphs per doc.")
    parser.add_argument(
        "--candidates", type=str, default=None,
        help=(
            "Comma-separated ISO 639-1 candidate override (e.g. "
            "'sl,en,de,hr,sr,it,fr'). Defaults to the config's full set; a "
            "smaller focused set is much faster and still separates sl from English."
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    """Entry point: sample the corpus, classify, and print the report."""
    args = parse_args()
    if args.candidates:
        candidates = [c.strip().lower() for c in args.candidates.split(",") if c.strip()]
        if TARGET not in candidates:
            candidates.append(TARGET)
        _, low_accuracy = load_language_config(args.config)
    else:
        candidates, low_accuracy = load_language_config(args.config)
    print(f"Detector: {len(candidates)} candidates, low_accuracy={low_accuracy}")
    detector = build_detector(candidates, low_accuracy)
    records, datasets = iter_sampled_docs(
        args.base_dir, args.per_dataset, args.max_shards_per_dataset
    )
    print(f"Sampled {len(records):,} docs from {len(datasets)} datasets under {args.base_dir}")
    stats = analyze(
        records, detector, args.min_unit_chars, args.whole_doc_chars, args.max_paragraphs
    )
    print_report(stats)


if __name__ == "__main__":
    main()
