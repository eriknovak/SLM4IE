"""Build the morphological golds, score the 24 tokenizers on them, and table the results.

`data` builds the golds and the dataset statistics. Three golds cut each
inflected form into morphemes. The Sloleks gold reads the stem-ending split
the Sloleks authors released (D1). The heuristic gold aligns each form to its
lemma with the word-class tags corrected to the Slovene set. The June gold is
the heuristic as the June sweep saved it, before the tag fix, read beside the
clauses only. Every gold is cut down to the held-out forms all three cover
(D4), so each tokenizer is scored on the same forms under each gold; each
scored form carries the held-out lexeme a bootstrap resamples (D6). It counts
the scored forms in the evaluation sample for the weighted score (D9) and
builds the MorphScore v2 Slovene gold (D10). The cut golds, the counts and the
MorphScore gold are written to the experiment's `final/` folder. Beside them
it tables the corpus split and its two samples (D3), the lexeme split and the
MorphScore items, the statistics behind the record's `## Datasets` rows.

`scores` scores every tokenizer of the sweep on those frozen golds and on both
corpus samples, then ranks the tokenizers per gold and slice and computes
Kendall's tau between the rankings with a bootstrap interval over lexemes
(H1, H2), the metric curves against vocabulary size (Q1), and the readings
beside the clauses (D5, D7–D10, D12, unknown tokens, training cost). It checks
that every tokenizer comes from a finished tracked run and logs the figures to
the sweep's parent run.

Run it with:

    uv run --group analysis python experiments/validation/morph-gold-slovenian/analysis.py data
    ~/.claude/skills/exp/scripts/tracking-env.sh --experiment slm4ie/validation/morph-gold-slovenian -- \
        uv run --group analysis python experiments/validation/morph-gold-slovenian/analysis.py scores
"""

import argparse
import csv
import dataclasses
import hashlib
import inspect
import json
import logging
import os
import re
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
from datachart.charts import LineChart, ScatterChart
from datachart.constants import FIG_SIZE, LINE_STYLE, SHOW_GRID
from datachart.utils import Grid
from mlflow.tracking import MlflowClient
from scipy.stats import kendalltau
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from report_figures import save_figure  # noqa: E402  — path set above so the shared helper resolves

from slm4ie.tokenizers import stats as st  # noqa: E402
from slm4ie.tokenizers.base import split_words  # noqa: E402
from slm4ie.tokenizers.config import SweepConfig, load_sweep_config
from slm4ie.tokenizers.corpus import SPLIT_SIDES, iter_sample, load_corpus_split
from slm4ie.tokenizers.evaluate import load_tokenizer_artifact
from slm4ie.tokenizers.metrics import (
    METRIC_DIRECTIONS,
    morph_consistency_over,
    morph_form_stats,
    renyi_efficiency,
)
from slm4ie.tokenizers.morphology import (
    LexemeSplit,
    MorphemeSegmentation,
    MorphLexicon,
    build_inflection_lexicon,
    build_morph_lexicon,
    build_morphscore_lexicon,
    is_morphscore_scored,
    load_lexicon,
    save_lexicon,
    split_lexemes,
)
from slm4ie.tokenizers.train import MLFLOW_LINK_FILENAME, parse_sweep_run_key
from slm4ie.utils.config import load_yaml
from slm4ie.utils.io import open_output, open_text_stream
from slm4ie.utils.parallel import run_parallel
from slm4ie.utils.versioning import file_sha256

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
TABLES_DIR = HERE / "tables"
FIGURES_DIR = HERE / "figures"
SWEEP_CONFIG = HERE / "configs" / "sweep-vocab-size.yaml"
ANALYSIS_CONFIG = HERE / "configs" / "analysis.yaml"

# JOS category letters of the word classes Sloleks 3.2 splits.
WORD_CLASSES = {"S": "noun", "P": "adjective", "G": "verb", "R": "adverb"}

# A corpus word for the D9 counts: a maximal run of letters, compared lowercased.
WORD = re.compile(r"[^\W\d_]+")

GOLDS = ("sloleks", "heuristic", "june")
# The algorithms that train on the morpheme table, left out of the D5 and D10 readings.
TABLE_ALGORITHMS = frozenset({"morphbpe", "morphpiece"})
SAMPLE_NAMES = {"train": "training sample", "eval": "evaluation sample"}
H1_BAR = 0.8
H2_BAR = 0.9
# Bootstrap draws per matrix product; bounds the draw matrix at about 50 MB.
BOOTSTRAP_BATCH = 100
# The corpus metrics Q1 reads, all on the evaluation sample.
CORPUS_METRICS = ("fertility", "tokens_per_byte", "chars_per_token", "renyi_efficiency")


@dataclasses.dataclass(frozen=True)
class AnalysisConfig:
    """The data files and scoring settings of `configs/analysis.yaml`.

    Attributes:
        golds_dir: The three golds cut to the held-out forms.
        form_counts: Each scored form's count in the evaluation sample.
        morphscore_gold: The MorphScore v2 Slovene gold.
        morphscore_items: The converted MorphScore v2 Slovene items.
        dataset_statistics: The corpus statistics stage's per-source files.
        june_gold: The heuristic gold the June sweep saved.
        scores_cache: One file pair of per-tokenizer counts per tokenizer.
        max_workers: Tokenizers scored at once.
        count_bins: Lower edges of the D12 bins of vocabulary-token counts.
        rare_below: A vocabulary token rarer than this in a sample is rare.
        levelled_ratio: Share of the 16k→32k gain under which a 64k→128k gain
            counts as levelled off.
    """

    golds_dir: Path
    form_counts: Path
    morphscore_gold: Path
    morphscore_items: Path
    dataset_statistics: Path
    june_gold: Path
    scores_cache: Path
    max_workers: int
    count_bins: Tuple[int, ...]
    rare_below: int
    levelled_ratio: float


def load_analysis_config(path: Path) -> AnalysisConfig:
    """Load `configs/analysis.yaml`.

    Args:
        path: The analysis config.

    Returns:
        AnalysisConfig: Its paths and scoring settings.

    Raises:
        KeyError: If a section or key is missing.
    """
    raw = load_yaml(path)
    paths = {key: Path(value) for key, value in raw["paths"].items()}
    scoring = raw["scoring"]
    return AnalysisConfig(
        **paths,
        max_workers=int(scoring["max_workers"]),
        count_bins=tuple(int(edge) for edge in scoring["count_bins"]),
        rare_below=int(scoring["rare_below"]),
        levelled_ratio=float(scoring["levelled_ratio"]),
    )


# --- Golds and dataset statistics (`data`) -------------------------------


def build_golds(cfg: SweepConfig, june_gold: Path) -> Dict[str, MorphLexicon]:
    """Build or load the three golds from the shared tokenization files.

    Args:
        cfg: The sweep dial, naming the lexicon files and the stem-length setting.
        june_gold: The heuristic gold the June sweep saved.

    Returns:
        Dict[str, MorphLexicon]: The `sloleks`, `heuristic` and `june` golds.
    """
    return {
        "sloleks": build_inflection_lexicon(cfg.inflection_path),
        "heuristic": build_morph_lexicon(cfg.sloleks_path, min_stem_len=cfg.min_stem_len),
        "june": load_lexicon(june_gold),
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


def load_form_counts(path: Path) -> Dict[str, int]:
    """Read the counts `save_form_counts` wrote.

    Args:
        path: The `.jsonl.gz` counts file.

    Returns:
        Dict[str, int]: Form to its count in the evaluation sample.
    """
    counts: Dict[str, int] = {}
    with open_text_stream(path) as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                counts[row["form"]] = row["count"]
    return counts


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


def source_domains(statistics_dir: Path) -> Dict[str, str]:
    """Read each source's domain from the corpus statistics stage.

    Args:
        statistics_dir: The statistics stage's per-source folder.

    Returns:
        Dict[str, str]: Source name to its domain label.
    """
    return {
        path.stem: json.loads(path.read_text(encoding="utf-8"))["domain"]
        for path in sorted(statistics_dir.glob("*.json"))
    }


def corpus_split_rows(cfg: SweepConfig, statistics_dir: Path) -> List[Dict[str, Any]]:
    """Table documents, words and bytes per source and side, for the split and both samples.

    Args:
        cfg: The sweep dial, naming the split and the two sample files.
        statistics_dir: The statistics stage's per-source folder.

    Returns:
        List[Dict[str, Any]]: One row per set (`split` or `sample`), side and
            source, plus an `all` row per set and side.
    """
    split = load_corpus_split(cfg.corpus_split_path)
    domains = source_domains(statistics_dir)
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


def rounded(rows: List[Dict[str, Any]], digits: int) -> List[Dict[str, Any]]:
    """Round every float in `rows` for a table.

    Args:
        rows: The rows.
        digits: Decimal places kept.

    Returns:
        List[Dict[str, Any]]: New rows with the floats rounded.
    """
    return [{k: round(v, digits) if isinstance(v, float) else v for k, v in row.items()} for row in rows]


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
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def build_data(cfg: SweepConfig, analysis: AnalysisConfig) -> None:
    """Build the golds, write the scored-form golds and the dataset statistics.

    Args:
        cfg: The sweep dial.
        analysis: The analysis paths.
    """
    lexemes = split_lexemes(cfg.inflection_path, cfg.heldout_lexeme_share)
    full = build_golds(cfg, analysis.june_gold)
    cut = restrict_to_scored_forms(full, lexemes)
    analysis.golds_dir.mkdir(parents=True, exist_ok=True)
    for name, lexicon in cut.items():
        save_lexicon(lexicon, analysis.golds_dir / f"{name}.jsonl.gz")
    texts = tqdm(iter_sample(cfg.eval_sample_path), desc="count held-out forms", unit="doc")
    counts = count_forms_in_sample(texts, cut["sloleks"].by_form)
    save_form_counts(counts, analysis.form_counts)
    morphscore = build_morphscore_lexicon(analysis.morphscore_items)
    analysis.morphscore_gold.parent.mkdir(parents=True, exist_ok=True)
    save_lexicon(morphscore, analysis.morphscore_gold)
    write_csv(TABLES_DIR / "dataset-morph-golds-statistics.csv", statistics_rows(full, cut, counts))
    write_csv(TABLES_DIR / "dataset-morphscore-statistics.csv", morphscore_rows(analysis.morphscore_items, morphscore))
    write_csv(TABLES_DIR / "dataset-lexeme-split-statistics.csv", lexeme_split_rows(lexemes))
    write_csv(TABLES_DIR / "dataset-corpus-split-statistics.csv", corpus_split_rows(cfg, analysis.dataset_statistics))


# --- Per-tokenizer counts (`scores`) -------------------------------------

# Read-only inputs shared with the scoring workers by fork, never pickled per task.
_SCORED: Dict[str, List[MorphemeSegmentation]] = {}
_MORPHSCORE: List[MorphemeSegmentation] = []
_WORD_TYPES: Dict[str, Tuple[List[str], List[int]]] = {}


def _count_chunk(texts: List[str]) -> Counter:
    """Count the `split_words` tokens of a chunk of documents.

    Args:
        texts: The documents.

    Returns:
        Counter: Word token to its count.
    """
    counts: Counter = Counter()
    for text in texts:
        counts.update(split_words(text))
    return counts


def count_word_types(path: Path, workers: int, chunk_docs: int = 5000) -> Tuple[List[str], List[int]]:
    """Count every word and punctuation token of a sample, as the corpus metrics split it.

    Args:
        path: The sample file, one document per line.
        workers: Processes counting chunks at once.
        chunk_docs: Documents per chunk.

    Returns:
        Tuple[List[str], List[int]]: The distinct tokens and their counts, in
            the same order.
    """

    def chunks() -> Iterable[List[str]]:
        chunk: List[str] = []
        for text in iter_sample(path):
            chunk.append(text)
            if len(chunk) == chunk_docs:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

    total: Counter = Counter()
    with Pool(workers) as pool:
        for counts in tqdm(pool.imap_unordered(_count_chunk, chunks()), desc=f"count words in {path.name}"):
            total.update(counts)
    words = list(total)
    return words, [total[word] for word in words]


def count_bin_labels(edges: Sequence[int]) -> List[str]:
    """Name the count bins whose lower edges are `edges`.

    Args:
        edges: Increasing lower edges, the first 0.

    Returns:
        List[str]: One label per bin, such as `0`, `1–9` and `1000+`.
    """
    labels = []
    for low, high in zip(edges, [*edges[1:], None]):
        if high is None:
            labels.append(f"{low}+")
        elif high - low == 1:
            labels.append(str(low))
        else:
            labels.append(f"{low}–{high - 1}")
    return labels


def vocabulary_bins(token_counts: Counter, vocab: Iterable[str], edges: Sequence[int]) -> List[Dict[str, Any]]:
    """Bin a vocabulary by how often each token occurs in a sample (D12).

    Args:
        token_counts: Token to its count in the sample.
        vocab: The tokens binned, special tokens left out.
        edges: Increasing lower edges of the bins, the first 0.

    Returns:
        List[Dict[str, Any]]: One row per bin with its `count_bin` label, the
            share of the vocabulary in it and the share of the sample's tokens
            those vocabulary tokens make up.
    """
    vocab = list(vocab)
    in_bin = np.searchsorted(np.asarray(edges), [token_counts.get(token, 0) for token in vocab], side="right") - 1
    occurrences = np.asarray([token_counts.get(token, 0) for token in vocab], dtype=np.int64)
    # The shares are over the binned tokens' occurrences, so they sum to one without the special tokens.
    binned_tokens = int(occurrences.sum())
    return [
        {
            "count_bin": label,
            "vocab_share": float(np.mean(in_bin == index)),
            "token_share": float(occurrences[in_bin == index].sum() / binned_tokens) if binned_tokens else 0.0,
        }
        for index, label in enumerate(count_bin_labels(edges))
    ]


def _sample_counts(tokenizer: Any, side: str, *, specials: Set[str], edges: Sequence[int], alpha: float):
    """Encode one sample's word types once and accumulate every count the scores need.

    Each word token is encoded on its own, as the shared corpus metrics do, so
    a word type is encoded once and weighted by its count. A token is unknown
    when it is the unknown token or not in the vocabulary: SentencePiece emits
    an unknown character as its own surface text.

    Args:
        tokenizer: The tokenizer under evaluation.
        side: `train` or `eval`, a key of `_WORD_TYPES`.
        specials: Its special tokens, left out of the bins.
        edges: Lower edges of the D12 bins.
        alpha: The Rényi order.

    Returns:
        Dict[str, Any]: Token totals, unknown tokens, the D12 bins and, over
            the word tokens holding a letter, the four corpus metrics.
    """
    vocab = tokenizer.vocab
    unk = tokenizer.unk_token
    token_counts: Counter = Counter()
    word_freqs: Counter = Counter()
    totals: Counter = Counter()
    for word, count in zip(*_WORD_TYPES[side]):
        pieces = tokenizer.encode(word)
        unknown = sum(piece == unk or piece not in vocab for piece in pieces)
        is_word = any(ch.isalpha() for ch in word)
        for piece in pieces:
            token_counts[piece] += count
            if is_word:
                word_freqs[piece] += count
        totals["tokens"] += len(pieces) * count
        totals["unk_tokens"] += unknown * count
        totals["words"] += count
        totals["words_with_unk"] += bool(unknown) * count
        if is_word:
            totals["word_tokens"] += len(pieces) * count
            totals["word_words"] += count
            totals["word_chars"] += len(word) * count
            totals["word_bytes"] += len(word.encode("utf-8")) * count
    return {
        "tokens": totals["tokens"],
        "unk_tokens": totals["unk_tokens"],
        "words": totals["words"],
        "words_with_unk": totals["words_with_unk"],
        "bins": vocabulary_bins(token_counts, (token for token in vocab if token not in specials), edges),
        "fertility": totals["word_tokens"] / totals["word_words"],
        "tokens_per_byte": totals["word_tokens"] / totals["word_bytes"],
        "chars_per_token": totals["word_chars"] / totals["word_tokens"],
        "renyi_efficiency": renyi_efficiency(word_freqs, alpha),
    }


def scoring_digest(analysis: AnalysisConfig, cfg: SweepConfig) -> str:
    """Hash everything but the tokenizer that a cached score depends on.

    Covers the golds, the MorphScore gold, both samples' sidecars, the scoring
    settings and the scoring code, so an edit to any of them rescores.

    Args:
        analysis: The analysis paths and settings.
        cfg: The sweep dial.

    Returns:
        str: A SHA-256 hex digest.
    """
    parts = [file_sha256(analysis.golds_dir / f"{gold}.jsonl.gz") for gold in GOLDS]
    parts.append(file_sha256(analysis.morphscore_gold))
    for sample in (cfg.corpus_sample_path, cfg.eval_sample_path):
        parts.append(file_sha256(sample.with_name(sample.name + ".sources.json")))
    parts.append(json.dumps([cfg.special_tokens, list(analysis.count_bins), cfg.renyi_alpha]))
    parts.append(inspect.getsource(score_tokenizer) + inspect.getsource(_sample_counts))
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def cache_key(digest: str, artifact_dir: Path) -> str:
    """Name the inputs of one tokenizer's cached score: the shared digest and its files.

    Args:
        digest: The `scoring_digest`.
        artifact_dir: The tokenizer's folder.

    Returns:
        str: A SHA-256 hex digest over the digest and each file's name, size
            and modification time.
    """
    listing = [(path.name, path.stat().st_size, path.stat().st_mtime_ns) for path in sorted(artifact_dir.iterdir())]
    return hashlib.sha256(f"{digest}\n{listing}".encode("utf-8")).hexdigest()


def is_cached(key: str, cache_dir: Path, expected: str) -> bool:
    """Tell whether a tokenizer's cached score was made from the inputs `expected` names.

    Args:
        key: Run key.
        cache_dir: The scores cache.
        expected: The `cache_key` of the current inputs.

    Returns:
        bool: True when both cache files exist and the summary carries `expected`.
    """
    summary = cache_dir / f"{key}.json"
    if not summary.exists() or not (cache_dir / f"{key}.npz").exists():
        return False
    return json.loads(summary.read_text(encoding="utf-8")).get("cache_key") == expected


def score_tokenizer(
    key: str,
    *,
    output_root: Path,
    cache_dir: Path,
    specials: List[str],
    edges: List[int],
    alpha: float,
    digest: str,
) -> str:
    """Score one tokenizer on the frozen golds and both samples, and cache the counts.

    Writes `<key>.npz` with per-form counts aligned to the Sloleks gold's
    order (the number of tokens, the unknown tokens among them, and per gold the
    boundary counts, edit distance and whether the tokens tile the form) and
    `<key>.json` with Morph-Consistency, the corpus metrics, the unknown-token
    counts, the D12 bins and the `cache_key` of the inputs it was scored from.

    Args:
        key: Run key (`<algorithm>-<vocab size>`).
        output_root: The sweep's tokenizer folders.
        cache_dir: Where the two files go.
        specials: The sweep's special tokens.
        edges: Lower edges of the D12 bins.
        alpha: The Rényi order.
        digest: The `scoring_digest` of the shared inputs.

    Returns:
        str: The key, once both files are written.
    """
    tokenizer = load_tokenizer_artifact(output_root / key)
    vocab = tokenizer.vocab
    unk = tokenizer.unk_token
    reference = _SCORED["sloleks"]
    pieces = [tokenizer.encode(segmentation.form) for segmentation in reference]
    arrays: Dict[str, np.ndarray] = {
        "n_tokens": np.asarray([len(form_pieces) for form_pieces in pieces], dtype=np.int16),
        "unk_tokens": np.asarray(
            [sum(piece == unk or piece not in vocab for piece in form_pieces) for form_pieces in pieces], dtype=np.int16
        ),
    }
    for gold, segmentations in [*_SCORED.items(), ("morphscore", _MORPHSCORE)]:
        form_stats = morph_form_stats(tokenizer, segmentations)
        for stat, values in form_stats.items():
            arrays[f"{gold}_{stat}"] = np.asarray(values, dtype=np.int16)
    manual = [segmentation for segmentation in reference if segmentation.verified]
    name, vocab_size = parse_sweep_run_key(key)
    summary = {
        "key": key,
        "algorithm": name,
        "vocab_size": vocab_size,
        "cache_key": cache_key(digest, output_root / key),
        "scored_forms": len(reference),
        "morph_consistency": {
            "all": morph_consistency_over(tokenizer, reference),
            "manual": morph_consistency_over(tokenizer, manual),
        },
        "samples": {
            side: _sample_counts(tokenizer, side, specials=set(specials), edges=edges, alpha=alpha)
            for side in _WORD_TYPES
        },
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_dir / f"{key}.npz", **arrays)
    (cache_dir / f"{key}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return key


# --- Rankings and rank agreement (`scores`) ------------------------------


@dataclasses.dataclass
class SweepScores:
    """The cached per-tokenizer counts of the whole sweep, stacked.

    Attributes:
        keys: Run keys, one per tokenizer, in sweep order.
        algorithms: Each key's algorithm.
        vocab_sizes: Each key's vocabulary size.
        forms: Per-form count name to a `(tokenizers, forms)` array.
        summaries: Each key's cached JSON summary.
    """

    keys: List[str]
    algorithms: List[str]
    vocab_sizes: List[int]
    forms: Dict[str, np.ndarray]
    summaries: List[Dict[str, Any]]


@dataclasses.dataclass
class Reading:
    """One way of scoring every tokenizer on boundary F1, ranked as one ranking.

    Attributes:
        name: The ranking's name in the tables.
        gold: The gold whose boundary counts it sums.
        mask: `(tokenizers, forms)`, the forms each tokenizer is scored on.
        weights: Per-form weight (one, or the D9 count plus one).
        units: Per-form index of the unit a bootstrap resamples.
    """

    name: str
    gold: str
    mask: np.ndarray
    weights: np.ndarray
    units: np.ndarray


def load_scores(keys: List[str], cache_dir: Path, n_forms: int) -> SweepScores:
    """Stack the cached counts of every tokenizer.

    Args:
        keys: Run keys in sweep order.
        cache_dir: The scores cache.
        n_forms: The scored forms every cached array must be aligned to.

    Returns:
        SweepScores: The stacked counts.

    Raises:
        ValueError: If a cached array does not hold one count per scored form.
    """
    arrays = [np.load(cache_dir / f"{key}.npz") for key in keys]
    summaries = [json.loads((cache_dir / f"{key}.json").read_text(encoding="utf-8")) for key in keys]
    for key, array in zip(keys, arrays):
        if len(array["n_tokens"]) != n_forms:
            raise ValueError(f"cached scores of {key} hold {len(array['n_tokens'])} forms, the golds {n_forms}")
    return SweepScores(
        keys=keys,
        algorithms=[summary["algorithm"] for summary in summaries],
        vocab_sizes=[summary["vocab_size"] for summary in summaries],
        forms={name: np.stack([array[name] for array in arrays]) for name in arrays[0].files},
        summaries=summaries,
    )


def define_readings(
    scores: SweepScores, reference: List[MorphemeSegmentation], counts: Dict[str, int]
) -> List[Reading]:
    """Define every ranking the clauses and the readings beside them compare.

    Every reading scores only the forms every tokenizer's tokens tile, so all
    tokenizers are ranked on the same forms (D7); the multi-token reading then
    keeps, per tokenizer, the forms it splits.

    Args:
        scores: The stacked counts.
        reference: The scored forms in the Sloleks gold's order.
        counts: Each scored form's count in the evaluation sample (D9).

    Returns:
        List[Reading]: The Sloleks and heuristic readings, the June reading and
            the MorphScore reading, in table order.

    Raises:
        ValueError: If a gold's forms are not tiled exactly as the Sloleks
            gold's, which means the golds are misaligned.
    """
    forms = scores.forms
    tiled = forms["sloleks_valid"].all(axis=0).astype(bool)
    for gold in GOLDS[1:]:
        if not np.array_equal(forms[f"{gold}_valid"], forms["sloleks_valid"]):
            raise ValueError(f"the {gold} gold's forms are not tiled as the Sloleks gold's: the golds are misaligned")
    every = np.broadcast_to(tiled, forms["n_tokens"].shape)
    manual = np.asarray([bool(segmentation.verified) for segmentation in reference])
    known = ~(forms["unk_tokens"] > 0).any(axis=0)
    _, lexemes = np.unique([segmentation.lexeme for segmentation in reference], return_inverse=True)
    ones = np.ones(len(reference))
    weighted = np.asarray([counts[segmentation.form] + 1 for segmentation in reference], dtype=float)
    readings: List[Reading] = []
    for gold, label in (("sloleks", "Sloleks gold"), ("heuristic", "heuristic gold")):
        readings += [
            Reading(f"{label}, all forms", gold, every, ones, lexemes),
            Reading(f"{label}, manual forms", gold, every & manual, ones, lexemes),
            Reading(f"{label}, multi-token forms", gold, every & (forms["n_tokens"] >= 2), ones, lexemes),
            Reading(f"{label}, forms with an ending", gold, every & (forms[f"{gold}_gold"] > 0), ones, lexemes),
            Reading(f"{label}, weighted by count", gold, every, weighted, lexemes),
            Reading(f"{label}, forms without unknown tokens", gold, every & known, ones, lexemes),
            Reading(
                f"{label}, manual multi-token forms", gold, every & manual & (forms["n_tokens"] >= 2), ones, lexemes
            ),
        ]
    # The manual slice has a different word-class mix, so H2 is also read within each class.
    word_classes = np.asarray([WORD_CLASSES.get((segmentation.msd or "?")[:1], "other") for segmentation in reference])
    for word_class in ("noun", "adjective", "verb"):
        in_class = every & (word_classes == word_class)
        readings += [
            Reading(f"Sloleks gold, all {word_class}s", "sloleks", in_class, ones, lexemes),
            Reading(f"Sloleks gold, manual {word_class}s", "sloleks", in_class & manual, ones, lexemes),
        ]
    readings.append(Reading("June gold, all forms", "june", every, ones, lexemes))
    items = forms["morphscore_valid"].all(axis=0).astype(bool)
    n_items = len(items)
    readings.append(
        Reading(
            "MorphScore v2 items",
            "morphscore",
            np.broadcast_to(items, forms["morphscore_valid"].shape),
            np.ones(n_items),
            np.arange(n_items),
        )
    )
    logger.info(
        "Forms every tokenizer tiles: %d of %d; MorphScore items: %d of %d",
        tiled.sum(),
        len(tiled),
        items.sum(),
        n_items,
    )
    return readings


def _f1(sums: np.ndarray) -> np.ndarray:
    """Turn interleaved (true positive, predicted plus gold) sums into micro F1.

    Args:
        sums: `(..., 2 * tokenizers)` sums, true positives at even columns.

    Returns:
        np.ndarray: `(..., tokenizers)` F1, zero where nothing was counted.
    """
    true_positive, denominator = sums[..., 0::2], sums[..., 1::2]
    return np.divide(2 * true_positive, denominator, out=np.zeros_like(true_positive), where=denominator > 0)


def f1_distributions(
    readings: List[Reading], forms: Dict[str, np.ndarray], n_resamples: int, seed: int
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Score every tokenizer under every reading, with a bootstrap over its units (D6).

    Each form's counts are summed per unit, and each draw reweights the units
    by a multinomial count, so all readings sharing a unit set share the
    draws: the manual and all-forms F1 of one draw come from the same
    resampled lexemes.

    Args:
        readings: The readings.
        forms: The stacked per-form counts.
        n_resamples: Bootstrap draws.
        seed: Seed of the draws.

    Returns:
        Dict[str, Tuple[np.ndarray, np.ndarray]]: Reading name to its point F1
            per tokenizer and its `(draws, tokenizers)` bootstrap F1.
    """
    rng = np.random.default_rng(seed)
    groups: Dict[int, List[Reading]] = {}
    for reading in readings:
        groups.setdefault(id(reading.units), []).append(reading)
    result: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for group in groups.values():
        units = group[0].units
        n_units = int(units.max()) + 1
        columns = []
        for reading in group:
            true_positive = forms[f"{reading.gold}_tp"]
            denominator = forms[f"{reading.gold}_predicted"] + forms[f"{reading.gold}_gold"]
            weight = reading.mask * reading.weights
            for index in range(len(true_positive)):
                columns.append(np.bincount(units, weights=true_positive[index] * weight[index], minlength=n_units))
                columns.append(np.bincount(units, weights=denominator[index] * weight[index], minlength=n_units))
        sums = np.stack(columns, axis=1)
        draws = np.empty((n_resamples, sums.shape[1]))
        for start in range(0, n_resamples, BOOTSTRAP_BATCH):
            size = min(BOOTSTRAP_BATCH, n_resamples - start)
            draws[start : start + size] = rng.multinomial(n_units, np.full(n_units, 1 / n_units), size=size) @ sums
        width = sums.shape[1] // len(group)
        for position, reading in enumerate(group):
            block = slice(width * position, width * (position + 1))
            result[reading.name] = (_f1(sums.sum(axis=0)[block]), _f1(draws[:, block]))
    return result


# (clause, ranking a, ranking b, tokenizers, bar): the rank agreements tabled, clauses first.
COMPARISONS = [
    ("H1", "heuristic gold, all forms", "Sloleks gold, all forms", "all", H1_BAR),
    ("H2", "Sloleks gold, manual forms", "Sloleks gold, all forms", "all", H2_BAR),
    ("H1 beside (D5)", "heuristic gold, all forms", "Sloleks gold, all forms", "without table", H1_BAR),
    ("H2 beside (D5)", "Sloleks gold, manual forms", "Sloleks gold, all forms", "without table", H2_BAR),
    ("H1 beside", "heuristic gold, manual forms", "Sloleks gold, manual forms", "all", H1_BAR),
    ("H1 beside (D7)", "heuristic gold, multi-token forms", "Sloleks gold, multi-token forms", "all", H1_BAR),
    ("H1 beside (D8)", "heuristic gold, forms with an ending", "Sloleks gold, forms with an ending", "all", H1_BAR),
    ("H1 beside (D9)", "heuristic gold, weighted by count", "Sloleks gold, weighted by count", "all", H1_BAR),
    (
        "H1 beside",
        "heuristic gold, forms without unknown tokens",
        "Sloleks gold, forms without unknown tokens",
        "all",
        H1_BAR,
    ),
    (
        "H1 beside (D7)",
        "heuristic gold, manual multi-token forms",
        "Sloleks gold, manual multi-token forms",
        "all",
        H1_BAR,
    ),
    ("H2 beside (D7)", "Sloleks gold, manual multi-token forms", "Sloleks gold, multi-token forms", "all", H2_BAR),
    ("H2 beside", "Sloleks gold, manual nouns", "Sloleks gold, all nouns", "all", H2_BAR),
    ("H2 beside", "Sloleks gold, manual adjectives", "Sloleks gold, all adjectives", "all", H2_BAR),
    ("H2 beside", "Sloleks gold, manual verbs", "Sloleks gold, all verbs", "all", H2_BAR),
    ("beside", "Sloleks gold, all nouns", "Sloleks gold, all forms", "all", None),
    ("beside", "Sloleks gold, all adjectives", "Sloleks gold, all forms", "all", None),
    ("beside", "Sloleks gold, all verbs", "Sloleks gold, all forms", "all", None),
    ("beside", "June gold, all forms", "Sloleks gold, all forms", "all", None),
    ("beside (D10)", "MorphScore v2 items", "Sloleks gold, all forms", "without table", None),
]


def rank_agreement_rows(
    scores: SweepScores, f1: Dict[str, Tuple[np.ndarray, np.ndarray]], ci_level: float
) -> List[Dict[str, Any]]:
    """Kendall's tau between two rankings for every comparison, with its bootstrap interval.

    Draw `b` of one ranking is paired with draw `b` of the other; for the
    MorphScore comparison the two draws are independent, as its items are.

    Args:
        scores: The stacked counts, for the algorithm of each tokenizer.
        f1: Each reading's point and bootstrap F1.
        ci_level: Central mass of the interval.

    Returns:
        List[Dict[str, Any]]: One row per comparison in `COMPARISONS`.
    """
    without_table = np.asarray([algorithm not in TABLE_ALGORITHMS for algorithm in scores.algorithms])
    rows = []
    for clause, first, second, population, bar in COMPARISONS:
        keep = without_table if population == "without table" else np.ones_like(without_table)
        (point_a, draws_a), (point_b, draws_b) = f1[first], f1[second]
        draws = np.asarray([kendalltau(a[keep], b[keep]).statistic for a, b in zip(draws_a, draws_b)])
        low, high = st.percentile_ci(draws, ci_level)
        rows.append(
            {
                "clause": clause,
                "ranking_a": first,
                "ranking_b": second,
                "tokenizers": int(keep.sum()),
                "bar": "" if bar is None else bar,
                "tau": round(float(kendalltau(point_a[keep], point_b[keep]).statistic), 4),
                "ci_low": round(float(low), 4),
                "ci_high": round(float(high), 4),
            }
        )
    return rows


def boundary_f1_rows(
    scores: SweepScores, readings: List[Reading], f1: Dict[str, Tuple[np.ndarray, np.ndarray]], ci_level: float
) -> List[Dict[str, Any]]:
    """Table each tokenizer's boundary F1 under every reading, best first within a reading.

    Args:
        scores: The stacked counts.
        readings: The readings, in table order.
        f1: Each reading's point and bootstrap F1.
        ci_level: Central mass of the interval.

    Returns:
        List[Dict[str, Any]]: One row per reading and tokenizer.
    """
    rows = []
    for reading in readings:
        point, draws = f1[reading.name]
        for index in np.argsort(-point, kind="stable"):
            low, high = st.percentile_ci(draws[:, index], ci_level)
            rows.append(
                {
                    "algorithm": scores.algorithms[index],
                    "vocab_size": scores.vocab_sizes[index],
                    "ranking": reading.name,
                    "forms": int(reading.mask[index].sum()),
                    "f1": round(float(point[index]), 4),
                    "ci_low": round(float(low), 4),
                    "ci_high": round(float(high), 4),
                }
            )
    return rows


# --- Metric curves, levelling and the readings beside Q1 -----------------


def metric_rows(
    scores: SweepScores, readings: List[Reading], f1: Dict[str, Tuple[np.ndarray, np.ndarray]], ci_level: float
) -> List[Dict[str, Any]]:
    """Table every Q1 metric per tokenizer: the corpus metrics and the Sloleks morph metrics.

    Morph-Edit-Distance is the mean edit distance over the slice's forms every
    tokenizer tiles; Morph-Consistency is a point estimate (D11).

    Args:
        scores: The stacked counts.
        readings: The readings, for the Sloleks masks.
        f1: Each reading's point and bootstrap F1.
        ci_level: Central mass of the F1 interval.

    Returns:
        List[Dict[str, Any]]: One row per tokenizer, metric and slice.
    """
    by_name = {reading.name: reading for reading in readings}
    rows = []
    for index, summary in enumerate(scores.summaries):
        base = {"algorithm": summary["algorithm"], "vocab_size": summary["vocab_size"]}
        for metric in CORPUS_METRICS:
            value = summary["samples"]["eval"][metric]
            rows.append(
                {**base, "metric": metric, "slice": "evaluation sample", "value": value, "ci_low": "", "ci_high": ""}
            )
        for slice_name in ("all", "manual"):
            reading = by_name[f"Sloleks gold, {slice_name} forms"]
            point, draws = f1[reading.name]
            low, high = st.percentile_ci(draws[:, index], ci_level)
            edit = float(scores.forms["sloleks_edit"][index][reading.mask[index]].mean())
            consistency = summary["morph_consistency"][slice_name]
            label = f"{slice_name} forms"
            rows += [
                {**base, "metric": "morph_score_f1", "slice": label, "value": float(point[index]),
                 "ci_low": float(low), "ci_high": float(high)},
                {**base, "metric": "morph_edit_distance", "slice": label, "value": edit, "ci_low": "", "ci_high": ""},
                {**base, "metric": "morph_consistency", "slice": label, "value": consistency,
                 "ci_low": "", "ci_high": ""},
            ]  # fmt: skip
    return rows


def levelling_rows(metrics: List[Dict[str, Any]], vocab_sizes: List[int], ratio: float) -> List[Dict[str, Any]]:
    """Read Q1's rule per metric, slice and algorithm: has the metric levelled off by 128k?

    A gain is the change in the metric's better direction. The metric has
    levelled off when its 64k→128k gain is under `ratio` of its 16k→32k gain,
    and falls when that gain is negative; with no 16k→32k gain the rule does
    not apply.

    Args:
        metrics: The rows of `metric_rows`.
        vocab_sizes: The four vocabulary sizes, ascending.
        ratio: The levelling share.

    Returns:
        List[Dict[str, Any]]: One row per metric, slice and algorithm.

    Raises:
        ValueError: If the sweep does not have exactly four vocabulary sizes.
    """
    if len(vocab_sizes) != 4:
        raise ValueError(f"Q1's rule reads four vocabulary sizes, not {vocab_sizes}")
    values: Dict[Tuple[str, str, str], Dict[int, float]] = {}
    for row in metrics:
        values.setdefault((row["metric"], row["slice"], row["algorithm"]), {})[row["vocab_size"]] = row["value"]
    small, second, third, large = vocab_sizes
    rows = []
    for (metric, slice_name, algorithm), curve in values.items():
        sign = 1 if METRIC_DIRECTIONS[metric] == "higher" else -1
        early = sign * (curve[second] - curve[small])
        late = sign * (curve[large] - curve[third])
        if early <= 0:
            share, reading = "", "no early gain"
        else:
            share = round(late / early, 3)
            reading = "falls" if late < 0 else "levelled off" if late < ratio * early else "still improving"
        rows.append(
            {
                "metric": metric,
                "slice": slice_name,
                "algorithm": algorithm,
                "gain_16k_to_32k": round(early, 4),
                "gain_64k_to_128k": round(late, 4),
                "late_to_early": share,
                "reading": reading,
            }
        )
    return rows


def vocabulary_count_rows(scores: SweepScores) -> List[Dict[str, Any]]:
    """Table each vocabulary's count bins in both samples (D12), training sample first.

    Args:
        scores: The stacked counts.

    Returns:
        List[Dict[str, Any]]: One row per tokenizer, sample and bin.
    """
    rows = []
    for side in ("train", "eval"):
        for summary in scores.summaries:
            sample = summary["samples"][side]
            for row in sample["bins"]:
                rows.append(
                    {
                        "algorithm": summary["algorithm"],
                        "vocab_size": summary["vocab_size"],
                        "sample": SAMPLE_NAMES[side],
                        "count_bin": row["count_bin"],
                        "vocab_share": round(row["vocab_share"], 4),
                        "token_share": round(row["token_share"], 6),
                        "sample_tokens": sample["tokens"],
                    }
                )
    return rows


def unknown_token_rows(scores: SweepScores) -> List[Dict[str, Any]]:
    """Table the share of unknown tokens per tokenizer on both samples and the scored forms, worst first.

    Args:
        scores: The stacked counts.

    Returns:
        List[Dict[str, Any]]: One row per tokenizer and set; on the scored
            forms a word is a form.
    """
    rows = []
    for index, summary in enumerate(scores.summaries):
        base = {"algorithm": summary["algorithm"], "vocab_size": summary["vocab_size"]}
        for side in ("train", "eval"):
            sample = summary["samples"][side]
            rows.append(
                {
                    **base,
                    "set": SAMPLE_NAMES[side],
                    "tokens": sample["tokens"],
                    "unk_token_share": sample["unk_tokens"] / sample["tokens"],
                    "words_with_unk_share": sample["words_with_unk"] / sample["words"],
                }
            )
        unknown = scores.forms["unk_tokens"][index]
        tokens = int(scores.forms["n_tokens"][index].sum())
        rows.append(
            {
                **base,
                "set": "scored forms",
                "tokens": tokens,
                "unk_token_share": unknown.sum() / tokens,
                "words_with_unk_share": float((unknown > 0).mean()),
            }
        )
    rows.sort(key=lambda row: -row["unk_token_share"])
    return [
        {
            **row,
            "unk_token_share": round(row["unk_token_share"], 6),
            "words_with_unk_share": round(row["words_with_unk_share"], 6),
        }
        for row in rows
    ]


def training_cost_rows(scores: SweepScores, output_root: Path, runs: Dict[str, Dict[str, str]]) -> List[Dict[str, Any]]:
    """Table each tokenizer's training time and peak memory with its run, largest vocabulary first.

    Args:
        scores: The stacked counts, for the keys.
        output_root: The sweep's tokenizer folders.
        runs: Each key's training run, from `tracked_train_runs`.

    Returns:
        List[Dict[str, Any]]: One row per tokenizer.
    """
    rows = []
    for key in scores.keys:
        stats = json.loads((output_root / key / "train_stats.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "algorithm": stats["tokenizer"],
                "vocab_size": stats["vocab_size"],
                "tier": runs[key]["tier"],
                "train_hours": round(stats["train_seconds"] / 3600, 2),
                "peak_memory_gb": round(stats["peak_rss_bytes"] / 1e9, 2),
                "run_id": runs[key]["run_id"],
            }
        )
    rows.sort(key=lambda row: (-row["vocab_size"], -row["train_hours"]))
    return rows


# --- Figures -------------------------------------------------------------


def _vocab_ticks(vocab_sizes: List[int]) -> Dict[str, Any]:
    """Place the vocabulary sizes at even steps, labelled in thousands.

    Args:
        vocab_sizes: The sweep's vocabulary sizes, ascending.

    Returns:
        Dict[str, Any]: The `xticks` and `xticklabels` of a chart.
    """
    return {"xticks": list(range(len(vocab_sizes))), "xticklabels": [f"{size // 1000}k" for size in vocab_sizes]}


def _curves(rows: List[Dict[str, Any]], algorithms: List[str], vocab_sizes: List[int], value: str) -> List[List[Dict]]:
    """Turn rows of one metric into one line per algorithm over the vocabulary positions.

    Args:
        rows: Rows holding `algorithm`, `vocab_size` and `value`.
        algorithms: The algorithms, in legend order.
        vocab_sizes: The vocabulary sizes, ascending.
        value: The key of the plotted value.

    Returns:
        List[List[Dict]]: One `{"x", "y"}` record list per algorithm.
    """
    position = {size: index for index, size in enumerate(vocab_sizes)}
    return [
        sorted(
            ({"x": position[row["vocab_size"]], "y": row[value]} for row in rows if row["algorithm"] == algorithm),
            key=lambda point: point["x"],
        )
        for algorithm in algorithms
    ]


def rank_agreement_figure(rows: List[Dict[str, Any]]) -> Any:
    """Draw each Kendall's tau with its interval against the two bars.

    Args:
        rows: The rows of `rank_agreement_rows`.

    Returns:
        Any: The datachart figure.
    """
    ordered = list(reversed(rows))
    points = [
        {"x": row["tau"], "y": index, "xerr": [row["tau"] - row["ci_low"], row["ci_high"] - row["tau"]]}
        for index, row in enumerate(ordered)
    ]
    labels = [f"{row['clause']}: {row['ranking_a']} vs {row['ranking_b']} ({row['tokenizers']})" for row in ordered]
    bars = [
        {"x": H1_BAR, "label": "H1 bar", "style": {"plot_vline_style": LINE_STYLE.DASHED}},
        {"x": H2_BAR, "label": "H2 bar", "style": {"plot_vline_style": LINE_STYLE.DOTTED}},
    ]
    return ScatterChart(
        points,
        show_yerr=False,
        vlines=bars,
        title="Rank agreement between rankings",
        xlabel="Kendall's τ over the tokenizers",
        ylabel="Ranking pair",
        yticks=list(range(len(ordered))),
        yticklabels=labels,
        figsize=FIG_SIZE.A4_LANDSCAPE,
        show_grid=SHOW_GRID.X,
    )


def boundary_f1_figure(rows: List[Dict[str, Any]], algorithms: List[str]) -> Any:
    """Draw each tokenizer's boundary F1 under the heuristic gold against the Sloleks gold.

    Args:
        rows: The rows of `boundary_f1_rows`.
        algorithms: The algorithms, in legend order.

    Returns:
        Any: The datachart figure.
    """
    f1 = {(row["ranking"], row["algorithm"], row["vocab_size"]): row["f1"] for row in rows}
    sizes = sorted({row["vocab_size"] for row in rows})
    series = [
        [
            {
                "x": f1[("heuristic gold, all forms", algorithm, size)],
                "y": f1[("Sloleks gold, all forms", algorithm, size)],
            }
            for size in sizes
        ]
        for algorithm in algorithms
    ]
    return ScatterChart(
        series,
        subtitle=algorithms,
        title="Boundary F1 under two golds",
        xlabel="Boundary F1, heuristic gold",
        ylabel="Boundary F1, Sloleks gold",
        figsize=FIG_SIZE.FULL_MEDIUM,
        show_grid=SHOW_GRID.BOTH,
        show_legend=True,
    )


METRIC_LABELS = {
    "fertility": "Fertility (tokens per word)",
    "tokens_per_byte": "Tokens per byte",
    "chars_per_token": "Characters per token",
    "renyi_efficiency": "Rényi efficiency",
    "morph_score_f1": "Boundary F1",
    "morph_edit_distance": "Morph-Edit-Distance",
    "morph_consistency": "Morph-Consistency",
}


def metric_curves_figure(
    rows: List[Dict[str, Any]], panels: List[Tuple[str, str]], algorithms: List[str], vocab_sizes: List[int], title: str
) -> Any:
    """Draw one panel of metric-against-vocabulary curves per (metric, slice), one line per algorithm.

    Args:
        rows: The rows of `metric_rows`.
        panels: The (metric, slice) pairs, in grid order, three per row.
        algorithms: The algorithms, in legend order.
        vocab_sizes: The vocabulary sizes, ascending.
        title: The grid's title.

    Returns:
        Any: The datachart grid.
    """
    charts = []
    for position, (metric, slice_name) in enumerate(panels):
        selected = [row for row in rows if row["metric"] == metric and row["slice"] == slice_name]
        # The y label names the metric; three panels per row leave room only for the slice.
        charts.append(
            LineChart(
                _curves(selected, algorithms, vocab_sizes, "value"),
                subtitle=algorithms,
                title=slice_name.capitalize(),
                xlabel="Vocabulary size",
                ylabel=METRIC_LABELS[metric],
                show_grid=SHOW_GRID.Y,
                show_legend=position == 0,
                **_vocab_ticks(vocab_sizes),
            )
        )
    return Grid(charts, max_cols=3, title=title, figsize=FIG_SIZE.FULL_TALL)


def rare_vocabulary_figure(
    rows: List[Dict[str, Any]], rare_labels: Set[str], algorithms: List[str], vocab_sizes: List[int]
) -> Any:
    """Draw the share of each vocabulary that is rare in each sample, one line per algorithm (D12).

    Args:
        rows: The rows of `vocabulary_count_rows`.
        rare_labels: The count bins below the rare threshold.
        algorithms: The algorithms, in legend order.
        vocab_sizes: The vocabulary sizes, ascending.

    Returns:
        Any: The datachart grid.
    """
    charts = []
    for position, sample in enumerate(SAMPLE_NAMES.values()):
        shares: Dict[Tuple[str, int], float] = Counter()
        for row in rows:
            if row["sample"] == sample and row["count_bin"] in rare_labels:
                shares[(row["algorithm"], row["vocab_size"])] += row["vocab_share"]
        points = [{"algorithm": name, "vocab_size": size, "share": share} for (name, size), share in shares.items()]
        charts.append(
            LineChart(
                _curves(points, algorithms, vocab_sizes, "share"),
                subtitle=algorithms,
                title=f"Rare tokens, {sample}",
                xlabel="Vocabulary size",
                ylabel="Share of the vocabulary",
                show_grid=SHOW_GRID.Y,
                show_legend=position == 0,
                **_vocab_ticks(vocab_sizes),
            )
        )
    return Grid(charts, max_cols=2, title="Rare vocabulary tokens", figsize=FIG_SIZE.FULL_MEDIUM)


# --- Lineage --------------------------------------------------------------


def tracked_train_runs(client: Any, keys: List[str], output_root: Path) -> Dict[str, Dict[str, str]]:
    """Check that every tokenizer was trained by a finished tracked run, all on one commit.

    Args:
        client: An MLflow client on the tracking server.
        keys: Run keys.
        output_root: The sweep's tokenizer folders, each holding its run link.

    Returns:
        Dict[str, Dict[str, str]]: Each key's `run_id`, `parent_run_id`,
            `tier` and `base_commit`.

    Raises:
        RuntimeError: If a tokenizer comes from a scratch or unfinished run, or
            the runs span several commits.
    """
    runs: Dict[str, Dict[str, str]] = {}
    for key in keys:
        link = json.loads((output_root / key / MLFLOW_LINK_FILENAME).read_text(encoding="utf-8"))
        run = client.get_run(link["run_id"])
        tier = run.data.tags.get("tier", "")
        if tier != "tracked" or run.info.status != "FINISHED":
            raise RuntimeError(
                f"tokenizer {key} comes from run {link['run_id']} (tier {tier!r}, status {run.info.status}),"
                " not a finished tracked run"
            )
        runs[key] = {
            "run_id": link["run_id"],
            "parent_run_id": link["parent_run_id"],
            "tier": tier,
            "base_commit": run.data.tags.get("base_commit", ""),
        }
    commits = {run["base_commit"] for run in runs.values()}
    if len(commits) != 1:
        raise RuntimeError(f"the tokenizers were trained on several commits: {sorted(commits)}")
    return runs


def build_scores(cfg: SweepConfig, analysis: AnalysisConfig, force: bool) -> None:
    """Score every tokenizer, rank them, and write the tables and figures of step 8.

    Args:
        cfg: The sweep dial.
        analysis: The analysis paths and settings.
        force: Rescore tokenizers already in the scores cache.

    Raises:
        RuntimeError: If MLflow is not configured or a tokenizer fails to score.
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        raise RuntimeError("MLFLOW_TRACKING_URI is unset: run `scores` through tracking-env.sh")
    client = MlflowClient(uri)
    keys = [f"{algorithm}-{size}" for algorithm in cfg.tokenizers for size in sorted(cfg.vocab_sizes)]
    runs = tracked_train_runs(client, keys, cfg.output_root)

    for gold in GOLDS:
        _SCORED[gold] = list(load_lexicon(analysis.golds_dir / f"{gold}.jsonl.gz").by_form.values())
    if any([s.form for s in _SCORED[gold]] != [s.form for s in _SCORED["sloleks"]] for gold in GOLDS):
        raise RuntimeError(f"the golds under {analysis.golds_dir} do not list the same forms in the same order")
    reference = _SCORED["sloleks"]
    digest = scoring_digest(analysis, cfg)
    todo = [
        key
        for key in keys
        if force or not is_cached(key, analysis.scores_cache, cache_key(digest, cfg.output_root / key))
    ]
    if todo:
        _MORPHSCORE[:] = list(load_lexicon(analysis.morphscore_gold).by_form.values())
        samples = {"train": cfg.corpus_sample_path, "eval": cfg.eval_sample_path}
        for side, path in samples.items():
            _WORD_TYPES[side] = count_word_types(path, analysis.max_workers)
        settings = {
            "output_root": cfg.output_root,
            "cache_dir": analysis.scores_cache,
            "specials": cfg.special_tokens,
            "edges": list(analysis.count_bins),
            "alpha": cfg.renyi_alpha,
            "digest": digest,
        }
        _, failures = run_parallel(
            score_tokenizer,
            todo,
            max_workers=analysis.max_workers,
            desc="score tokenizers",
            kwargs_for=lambda _key: settings,
        )
        if failures:
            raise RuntimeError(f"tokenizers failed to score: {[key for key, _error in failures]}") from failures[0][1]

    scores = load_scores(keys, analysis.scores_cache, len(reference))
    readings = define_readings(scores, reference, load_form_counts(analysis.form_counts))
    f1 = f1_distributions(readings, scores.forms, cfg.stats_n_resamples, cfg.stats_seed)
    agreement = rank_agreement_rows(scores, f1, cfg.stats_ci_level)
    boundary = boundary_f1_rows(scores, readings, f1, cfg.stats_ci_level)
    metrics = metric_rows(scores, readings, f1, cfg.stats_ci_level)
    vocabulary = vocabulary_count_rows(scores)
    vocab_sizes = sorted(cfg.vocab_sizes)
    write_csv(TABLES_DIR / "sweep-rank-agreement-by-ranking-pair.csv", agreement)
    write_csv(TABLES_DIR / "sweep-boundary-f1-by-tokenizer.csv", boundary)
    # Q1's rule reads the unrounded values; the table rounds them.
    write_csv(
        TABLES_DIR / "sweep-levelling-by-algorithm.csv", levelling_rows(metrics, vocab_sizes, analysis.levelled_ratio)
    )
    write_csv(TABLES_DIR / "sweep-metrics-by-vocab.csv", rounded(metrics, 4))
    write_csv(TABLES_DIR / "sweep-vocabulary-tokens-by-count.csv", vocabulary)
    write_csv(TABLES_DIR / "sweep-unknown-tokens-by-tokenizer.csv", unknown_token_rows(scores))
    write_csv(TABLES_DIR / "sweep-training-cost-by-tokenizer.csv", training_cost_rows(scores, cfg.output_root, runs))

    algorithms = list(cfg.tokenizers)
    edges = list(analysis.count_bins)
    rare = {label for label, low in zip(count_bin_labels(edges), edges) if low < analysis.rare_below}
    morph_panels = [(metric, f"{slice_name} forms") for slice_name in ("all", "manual")
                    for metric in ("morph_score_f1", "morph_edit_distance", "morph_consistency")]  # fmt: skip
    corpus_panels = [(metric, "evaluation sample") for metric in CORPUS_METRICS]
    figures = {
        "sweep-rank-agreement-by-ranking-pair.svg": lambda: rank_agreement_figure(agreement),
        "sweep-boundary-f1-by-gold.svg": lambda: boundary_f1_figure(boundary, algorithms),
        "sweep-morph-metrics-by-vocab.svg": lambda: metric_curves_figure(
            metrics, morph_panels, algorithms, vocab_sizes, "Morph metrics against vocabulary size"
        ),
        "sweep-corpus-metrics-by-vocab.svg": lambda: metric_curves_figure(
            metrics, corpus_panels, algorithms, vocab_sizes, "Corpus metrics against vocabulary size"
        ),
        "sweep-rare-vocabulary-share-by-vocab.svg": lambda: rare_vocabulary_figure(
            vocabulary, rare, algorithms, vocab_sizes
        ),
    }
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    parent_run = {run["parent_run_id"] for run in runs.values()}
    for name, build in figures.items():
        save_figure(build, FIGURES_DIR / name)
        for run_id in parent_run:
            client.log_artifact(run_id, str(FIGURES_DIR / name), artifact_path="analysis")
    # The settings behind the intervals, on the sweep's parent run beside the figures.
    for run_id in parent_run:
        for name, value in {
            "analysis_n_resamples": cfg.stats_n_resamples,
            "analysis_ci_level": cfg.stats_ci_level,
            "analysis_bootstrap_seed": cfg.stats_seed,
            "analysis_scoring_digest": digest,
        }.items():
            client.log_param(run_id, name, value)


def main() -> None:
    """Run the `data` or the `scores` step."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    steps = parser.add_subparsers(dest="step", required=True)
    steps.add_parser("data", help="build the golds, the form counts and the dataset statistics")
    scores = steps.add_parser("scores", help="score every tokenizer and write the step-8 tables and figures")
    scores.add_argument("--force", action="store_true", help="rescore tokenizers already in the scores cache")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_sweep_config(SWEEP_CONFIG)
    analysis = load_analysis_config(ANALYSIS_CONFIG)
    if args.step == "data":
        build_data(cfg, analysis)
    else:
        build_scores(cfg, analysis, args.force)


if __name__ == "__main__":
    main()
