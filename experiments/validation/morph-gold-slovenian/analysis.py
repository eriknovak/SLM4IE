"""Build the morphological golds the tokenizers are ranked on, and table the data.

Three golds cut each inflected form into morphemes. The Sloleks gold reads the
stem-ending split the Sloleks authors released (D1). The heuristic gold aligns
each form to its lemma with the word-class tags corrected to the Slovene set.
The June gold is the heuristic as the June sweep saved it, before the tag fix,
read beside the clauses only. Every gold is cut down to the held-out forms all
three cover (D4), so each tokenizer is scored on the same forms under each
gold; each scored form carries the held-out lexeme a bootstrap resamples
(D6). It counts the scored forms in the evaluation sample for the weighted
score (D9) and builds the MorphScore v2 Slovene gold (D10). The cut golds,
the counts and the MorphScore gold are written to the experiment's `final/`
folder. Beside them it tables the corpus split and its two samples (D3), the
lexeme split and the MorphScore items, the statistics behind the record's
`## Datasets` rows.

Run it with:

    uv run --group analysis python experiments/validation/morph-gold-slovenian/analysis.py
"""

import csv
import dataclasses
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

from tqdm import tqdm

from slm4ie.tokenizers.config import SweepConfig, load_sweep_config
from slm4ie.tokenizers.corpus import SPLIT_SIDES, iter_sample, load_corpus_split
from slm4ie.tokenizers.morphology import (
    LexemeSplit,
    MorphLexicon,
    build_inflection_lexicon,
    build_morph_lexicon,
    build_morphscore_lexicon,
    is_morphscore_scored,
    load_lexicon,
    save_lexicon,
    split_lexemes,
)
from slm4ie.utils.io import open_output, open_text_stream

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TABLES_DIR = HERE / "tables"
DATA = ROOT / "data"
SWEEP_CONFIG = HERE / "configs" / "sweep-vocab-size.yaml"
FINAL = DATA / "experiments" / "validation" / "morph-gold-slovenian" / "final"
FINAL_DIR = FINAL / "morph-golds-heldout-forms"
FORM_COUNTS = FINAL / "heldout-form-counts" / "eval-sample.jsonl.gz"
MORPHSCORE_GOLD = FINAL / "morphscore-slovene-items" / "morphscore.jsonl.gz"
MORPHSCORE_ITEMS = DATA / "tokenization" / "morphscore_slovenian.jsonl.gz"
DATASET_STATISTICS = DATA / "pretrain" / "07_statistics" / "per_dataset"
JUNE_GOLD = DATA / "tokenizers" / "morph_lexicon_infl.jsonl.gz"

# JOS category letters of the word classes Sloleks 3.2 splits.
WORD_CLASSES = {"S": "noun", "P": "adjective", "G": "verb", "R": "adverb"}

# A corpus word for the D9 counts: a maximal run of letters, compared lowercased.
WORD = re.compile(r"[^\W\d_]+")


def build_golds(cfg: SweepConfig) -> Dict[str, MorphLexicon]:
    """Build or load the three golds from the shared tokenization files.

    Args:
        cfg: The sweep dial, naming the lexicon files and the stem-length setting.

    Returns:
        Dict[str, MorphLexicon]: The `sloleks`, `heuristic` and `june` golds.
    """
    return {
        "sloleks": build_inflection_lexicon(cfg.inflection_path),
        "heuristic": build_morph_lexicon(cfg.sloleks_path, min_stem_len=cfg.min_stem_len),
        "june": load_lexicon(JUNE_GOLD),
    }


def restrict_to_scored_forms(golds: Dict[str, MorphLexicon], lexemes: LexemeSplit) -> Dict[str, MorphLexicon]:
    """Cut every gold down to the held-out forms all golds cover, in the Sloleks gold's order.

    The manual flag and the tag of each form are taken from the Sloleks gold,
    so the manual slice and the word class are the same under every gold.
    Each form carries the held-out lexeme it is resampled under (D6).

    Args:
        golds: The full golds, keyed by name; must include `sloleks`.
        lexemes: The lexeme split (D4).

    Returns:
        Dict[str, MorphLexicon]: One lexicon per gold over the scored forms.
    """
    reference = golds["sloleks"].by_form
    heldout = lexemes.heldout_forms
    scored = [form for form in reference if form in heldout and all(form in gold.by_form for gold in golds.values())]
    cut: Dict[str, MorphLexicon] = {}
    for name, gold in golds.items():
        lexicon = MorphLexicon()
        for form in scored:
            lexicon.by_form[form] = dataclasses.replace(
                gold.by_form[form],
                verified=reference[form].verified,
                msd=reference[form].msd,
                lexeme=lexemes.lexeme_of[form],
            )
        cut[name] = lexicon
    logger.info("Scored forms: %d of %d in the Sloleks gold", len(scored), len(reference))
    return cut


def count_forms_in_sample(texts: Iterable[str], forms: Iterable[str]) -> Dict[str, int]:
    """Count each form's occurrences as a corpus word in `texts` (D9).

    A corpus word is a maximal run of letters; words and forms are compared
    lowercased, so forms differing only in case share one count, and a form
    holding a non-letter never matches.

    Args:
        texts: The documents to count in.
        forms: The forms to count.

    Returns:
        Dict[str, int]: Every form to its count, zero when absent.
    """
    forms = list(forms)
    keys = {form.lower() for form in forms}
    counts: Counter = Counter()
    for text in texts:
        counts.update(word for word in WORD.findall(text.lower()) if word in keys)
    return {form: counts[form.lower()] for form in forms}


def save_form_counts(counts: Dict[str, int], path: Path) -> Path:
    """Write one `{"form", "count"}` line per form.

    Args:
        counts: Form to its count in the evaluation sample.
        path: Destination `.jsonl.gz` path.

    Returns:
        Path: `path`, for chaining.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open_output(path) as out:
        for form, count in counts.items():
            out.write(json.dumps({"form": form, "count": count}, ensure_ascii=False) + "\n")
    return path


def statistics_rows(
    full: Dict[str, MorphLexicon], cut: Dict[str, MorphLexicon], counts: Dict[str, int]
) -> List[Dict[str, Any]]:
    """Count forms per gold, slice and word class, before and after the cut.

    Args:
        full: The full golds.
        cut: The golds over the scored forms.
        counts: Each scored form's count in the evaluation sample.

    Returns:
        List[Dict[str, Any]]: One row per gold, slice and word class with the
            forms in the full gold, the scored forms, their lexemes, the
            scored forms the gold splits, those seen in the evaluation sample,
            those holding a non-letter (never counted), and the occurrences
            there, each case group summed once.
    """
    rows: List[Dict[str, Any]] = []
    for name, lexicon in cut.items():
        for slice_name in ("all", "manual"):
            scored: Counter = Counter()
            split: Counter = Counter()
            seen: Counter = Counter()
            uncountable: Counter = Counter()
            # Keyed by lowercased form: case variants share one count, summed once.
            occurrences: Dict[str, Dict[str, int]] = {}
            units: Dict[str, Set[str]] = {}
            for segmentation in lexicon.by_form.values():
                if slice_name == "manual" and not segmentation.verified:
                    continue
                word_class = WORD_CLASSES.get((segmentation.msd or "?")[:1], "other")
                scored[word_class] += 1
                split[word_class] += bool(segmentation.boundaries())
                seen[word_class] += counts[segmentation.form] > 0
                uncountable[word_class] += not WORD.fullmatch(segmentation.form)
                occurrences.setdefault(word_class, {})[segmentation.form.lower()] = counts[segmentation.form]
                units.setdefault(word_class, set()).add(segmentation.lexeme)
            for word_class in [*WORD_CLASSES.values(), "other"]:
                if not scored[word_class]:
                    continue
                rows.append(
                    {
                        "gold": name,
                        "slice": slice_name,
                        "word_class": word_class,
                        "gold_forms": len(full[name].by_form),
                        "scored_forms": scored[word_class],
                        "scored_lexemes": len(units[word_class]),
                        "scored_forms_split": split[word_class],
                        "share_split": round(split[word_class] / scored[word_class], 4),
                        "scored_forms_in_eval_sample": seen[word_class],
                        "scored_forms_uncountable": uncountable[word_class],
                        "eval_sample_occurrences": sum(occurrences[word_class].values()),
                    }
                )
    return rows


def lexeme_split_rows(split: LexemeSplit) -> List[Dict[str, Any]]:
    """Name the word classes of the lexeme split's per-stratum counts.

    Args:
        split: The lexeme split.

    Returns:
        List[Dict[str, Any]]: One row per side, word class and status.
    """
    return [{**row, "word_class": WORD_CLASSES.get(row["word_class"], row["word_class"])} for row in split.counts]


def morphscore_rows(items_path: Path, gold: MorphLexicon) -> List[Dict[str, Any]]:
    """Table the MorphScore v2 Slovene items through each default filter, then per part of speech.

    Args:
        items_path: The converted `morphscore_slovenian` file.
        gold: The MorphScore gold built from it.

    Returns:
        List[Dict[str, Any]]: Rows of `cut`, `pos`, `items` and
            `items_with_prefix`: one per filter over all parts of speech, then
            one per part of speech of the gold.
    """
    funnel: Counter = Counter()
    with open_text_stream(items_path) as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            funnel["treebank rows"] += 1
            funnel["first occurrence"] += item["is_unique"]
            funnel["stem equals lemma"] += item["is_unique"] and item["stem"] == item["lemma"]
            funnel["no digit"] += is_morphscore_scored(item)
    rows: List[Dict[str, Any]] = [
        {"cut": cut, "pos": "all", "items": n, "items_with_prefix": ""} for cut, n in funnel.items()
    ]
    rows.append({"cut": "one cut per form", "pos": "all", "items": len(gold.by_form), "items_with_prefix": ""})
    by_pos: Counter = Counter()
    prefixed: Counter = Counter()
    for segmentation in gold.by_form.values():
        by_pos[segmentation.msd] += 1
        prefixed[segmentation.msd] += segmentation.labels[0] == "prefix"
    for pos, n in by_pos.most_common():
        rows.append({"cut": "one cut per form", "pos": pos, "items": n, "items_with_prefix": prefixed[pos]})
    return rows


def source_domains() -> Dict[str, str]:
    """Read each source's domain from the corpus statistics stage.

    Returns:
        Dict[str, str]: Source name to its domain label.
    """
    return {
        path.stem: json.loads(path.read_text(encoding="utf-8"))["domain"]
        for path in sorted(DATASET_STATISTICS.glob("*.json"))
    }


def corpus_split_rows(cfg: SweepConfig) -> List[Dict[str, Any]]:
    """Table documents, words and bytes per source and side, for the split and both samples.

    Args:
        cfg: The sweep dial, naming the split and the two sample files.

    Returns:
        List[Dict[str, Any]]: One row per set (`split` or `sample`), side and
            source, plus an `all` row per set and side.
    """
    split = load_corpus_split(cfg.corpus_split_path)
    domains = source_domains()
    samples = {"train": cfg.corpus_sample_path, "eval": cfg.eval_sample_path}
    rows: List[Dict[str, Any]] = []
    for side in SPLIT_SIDES:
        sources_file = samples[side].with_name(samples[side].name + ".sources.json")
        sample_counts = json.loads(sources_file.read_text(encoding="utf-8"))["sources"]
        split_counts = {source: counts[side] for source, counts in split.counts.items()}
        for set_name, counts in (("split", split_counts), ("sample", sample_counts)):
            total = {"documents": 0, "words": 0, "bytes": 0}
            for source in sorted(counts):
                rows.append({"set": set_name, "side": side, "source": source, "domain": domains.get(source, "")})
                rows[-1].update(counts[source])
                for key in total:
                    total[key] += counts[source][key]
            rows.append({"set": set_name, "side": side, "source": "all", "domain": "", **total})
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> Path:
    """Write rows as a CSV with the first row's keys as header.

    Args:
        path: Destination file.
        rows: Rows sharing one key set.

    Returns:
        Path: `path`, for chaining.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    """Build the golds, write the scored-form golds and the dataset statistics."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_sweep_config(SWEEP_CONFIG)
    lexemes = split_lexemes(cfg.inflection_path, cfg.heldout_lexeme_share)
    full = build_golds(cfg)
    cut = restrict_to_scored_forms(full, lexemes)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    for name, lexicon in cut.items():
        save_lexicon(lexicon, FINAL_DIR / f"{name}.jsonl.gz")
    texts = tqdm(iter_sample(cfg.eval_sample_path), desc="count held-out forms", unit="doc")
    counts = count_forms_in_sample(texts, cut["sloleks"].by_form)
    save_form_counts(counts, FORM_COUNTS)
    morphscore = build_morphscore_lexicon(MORPHSCORE_ITEMS)
    MORPHSCORE_GOLD.parent.mkdir(parents=True, exist_ok=True)
    save_lexicon(morphscore, MORPHSCORE_GOLD)
    write_csv(TABLES_DIR / "dataset-morph-golds-statistics.csv", statistics_rows(full, cut, counts))
    write_csv(TABLES_DIR / "dataset-morphscore-statistics.csv", morphscore_rows(MORPHSCORE_ITEMS, morphscore))
    write_csv(TABLES_DIR / "dataset-lexeme-split-statistics.csv", lexeme_split_rows(lexemes))
    write_csv(TABLES_DIR / "dataset-corpus-split-statistics.csv", corpus_split_rows(cfg))


if __name__ == "__main__":
    main()
