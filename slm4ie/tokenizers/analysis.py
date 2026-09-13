"""Evaluate trained tokenizers and build the comparison report.

`evaluate_artifact` loads one trained tokenizer, runs the six metrics over the
grouped held-out documents and the shared Sloleks-derived morph sample, and
persists per-unit sufficient statistics to an `eval_units.npz` sidecar.
`augment_with_statistics` reads those sidecars to attach bootstrap confidence
intervals and paired significance tests (per vocab) to the five decomposable
metrics. `build_report` aggregates the per-run metrics into a Markdown table and
a JSON payload, and `log_results_to_mlflow` records the sweep as a parent run
with one nested child per tokenizer x vocab-size run. `evaluate_sweep` ties
those together: it materializes the shared evaluation inputs, runs the metrics
over every selected artifact in parallel, and writes the report.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np

import slm4ie.tokenizers.backends  # noqa: F401  (registers backends on import)
from slm4ie.data.parallel import (
    configure_script_logging,
    cpu_default,
    resolve_workers,
    run_parallel,
)
from slm4ie.tokenizers import stats as st
from slm4ie.tokenizers.corpus import iter_sample_cache, sample_corpus, write_sample_cache
from slm4ie.tokenizers.metrics import (
    METRIC_DIRECTIONS,
    corpus_doc_stats,
    iter_words,
    morph_consistency_over,
    morph_form_stats,
    renyi_efficiency,
)
from slm4ie.tokenizers.morphology import (
    MorphemeSegmentation,
    build_derivational_lexicon,
    build_morph_lexicon,
    load_lexicon,
    sample_segmentations,
    save_lexicon,
)
from slm4ie.tokenizers.registry import get_tokenizer
from slm4ie.tokenizers.train import MLFLOW_LINK_FILENAME, parse_run_key
from slm4ie.utils import mlflow as ml
from slm4ie.utils.cli import stamped_log_dir
from slm4ie.tokenizers.config import TokenizerSweepConfig

logger = logging.getLogger(__name__)

#: Conservative upper bound on parallel eval workers so a large box is not
#: saturated by default; raise it explicitly with the worker argument.
DEFAULT_MAX_WORKERS = 8

#: Heavy, read-only eval inputs shared with process-pool workers via fork
#: inheritance (copy-on-write) instead of being pickled per task. Populated by
#: `evaluate_sweep` before the pool is created; read by `_evaluate_worker`.
_EVAL_DOCS: List[List[str]] = []
_MORPH_SAMPLE: List[MorphemeSegmentation] = []
_DERIV_SAMPLE: List[MorphemeSegmentation] = []

#: Metric columns shown in the report, in order.
_REPORT_COLUMNS = [
    "fertility",
    "tokens_per_byte",
    "chars_per_token",
    "renyi_efficiency",
    "morph_score_f1",
    "morph_edit_distance",
    "morph_consistency",
]

#: Derivational morph columns, scored against the Sloleks word-relations gold.
#: Shown after the base columns when any run carries them. The two boundary
#: metrics get bootstrap CIs + paired significance (like the inflectional track);
#: `morph_consistency_deriv` stays a point estimate.
_DERIV_COLUMNS = [
    "morph_score_f1_deriv",
    "morph_edit_distance_deriv",
    "morph_consistency_deriv",
]

#: Decomposable inflectional/corpus metrics that get bootstrap CIs and paired
#: significance tests. Each reduces to per-unit sufficient statistics that sum
#: across the resampling unit (documents for corpus metrics, forms for morph).
_DECOMPOSABLE_METRICS = [
    "fertility",
    "tokens_per_byte",
    "chars_per_token",
    "morph_score_f1",
    "morph_edit_distance",
]

#: Decomposable derivational morph metrics; resampled over the derivational
#: gold forms. Only processed when a run carries the derivational per-form
#: arrays (i.e. a derivational gold was configured at evaluation time).
_DERIV_DECOMPOSABLE_METRICS = [
    "morph_score_f1_deriv",
    "morph_edit_distance_deriv",
]

#: Filename of the per-run sufficient-statistics sidecar consumed by the
#: aggregation. Stored as compact integer arrays under each artifact dir.
EVAL_UNITS_FILENAME = "eval_units.npz"

#: Arrow shown next to each metric header indicating the better direction.
_DIRECTION_ARROW = {"higher": "↑", "lower": "↓"}


def _safe_div(numerator: Any, denominator: Any) -> np.ndarray:
    """Divide elementwise, returning 0 where the denominator is 0.

    Args:
        numerator (Any): Summed numerator (scalar or bootstrap-axis array).
        denominator (Any): Summed denominator (scalar or bootstrap-axis array).

    Returns:
        np.ndarray: The elementwise ratio with zeros where the denominator is 0.
    """
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    return np.divide(num, den, out=np.zeros_like(num), where=den != 0)


def _f1_combine(true_positive: Any, predicted: Any, gold: Any) -> np.ndarray:
    """Combine summed boundary counts into the MorphScore F1.

    Args:
        true_positive (Any): Summed correct boundaries.
        predicted (Any): Summed predicted boundaries.
        gold (Any): Summed gold boundaries.

    Returns:
        np.ndarray: The F1, elementwise over the bootstrap axis.
    """
    precision = _safe_div(true_positive, predicted)
    recall = _safe_div(true_positive, gold)
    return _safe_div(2.0 * precision * recall, precision + recall)


def _morph_point_estimates(stats: Dict[str, List[int]]) -> Dict[str, float]:
    """Aggregate per-form morph statistics into point estimates.

    Used for the derivational track, which reports point estimates only (no
    bootstrap CIs), so it does not persist per-form arrays the way the
    inflectional track does.

    Args:
        stats (Dict[str, List[int]]): Per-form arrays from `morph_form_stats`
            (`tp`, `predicted`, `gold`, `edit`, `valid`).

    Returns:
        Dict[str, float]: `f1`, `precision`, `recall` (boundary scores),
            `edit` (mean segment-edit distance), and `coverage` (fraction of
            forms the tokenizer tiled).
    """
    valid = np.asarray(stats["valid"], dtype=bool)
    n_valid = int(valid.sum())
    tp = int(np.asarray(stats["tp"], dtype=np.int64)[valid].sum())
    predicted = int(np.asarray(stats["predicted"], dtype=np.int64)[valid].sum())
    gold = int(np.asarray(stats["gold"], dtype=np.int64)[valid].sum())
    edit = int(np.asarray(stats["edit"], dtype=np.int64)[valid].sum())
    precision = tp / predicted if predicted else 0.0
    recall = tp / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "edit": edit / n_valid if n_valid else 0.0,
        "coverage": n_valid / len(valid) if len(valid) else 0.0,
    }


def load_tokenizer_artifact(artifact_dir: Path):
    """Load a trained tokenizer from its artifact directory.

    Args:
        artifact_dir (Path): Directory holding `metadata.json` and the model.

    Returns:
        BaseTokenizer: The reconstructed tokenizer.

    Raises:
        FileNotFoundError: If no metadata sidecar is present.
    """
    meta_path = artifact_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No tokenizer metadata under {artifact_dir}")
    name = json.loads(meta_path.read_text(encoding="utf-8"))["name"]
    return get_tokenizer(name).load(artifact_dir)


def evaluate_artifact(
    key: str,
    *,
    output_root: Path,
    eval_docs: Sequence[Sequence[str]],
    morph_sample: Sequence[MorphemeSegmentation],
    deriv_sample: Sequence[MorphemeSegmentation] = (),
    alpha: float = 2.5,
) -> Optional[Dict[str, Any]]:
    """Run all six metrics for one trained tokenizer run.

    Encodes the held-out documents and the shared morph sample once, derives the
    point estimates from the per-unit sufficient statistics, and persists those
    statistics to an `eval_units.npz` sidecar so the aggregation can bootstrap
    confidence intervals and paired tests. Corpus point estimates use every
    document; morph point estimates use the forms in the sample that the
    tokenizer tiled.

    Args:
        key (str): Run key (`<name>-<vocab>`).
        output_root (Path): Directory holding per-run artifact subdirs.
        eval_docs (Sequence[Sequence[str]]): Held-out evaluation word tokens,
            kept grouped per document (the corpus resampling unit).
        morph_sample (Sequence[MorphemeSegmentation]): The shared, deterministic
            sample of gold segmentations (the morph resampling unit).
        deriv_sample (Sequence[MorphemeSegmentation]): The derivational gold
            sample; when non-empty, derivational point-estimate columns
            (`morph_score_f1_deriv` etc.) are added. Empty skips the track.
        alpha (float): Renyi order.

    Returns:
        Optional[Dict[str, Any]]: A flat metrics record, or None when the
            artifact is missing (skipped).
    """
    artifact_dir = output_root / key
    if not (artifact_dir / "metadata.json").exists():
        logger.warning("No artifact for %s; skipping.", key)
        return None

    name, vocab_size = parse_run_key(key)
    tokenizer = load_tokenizer_artifact(artifact_dir)

    corpus = corpus_doc_stats(tokenizer, eval_docs)
    doc_tokens = np.asarray(corpus["tokens"], dtype=np.int64)
    doc_words = np.asarray(corpus["words"], dtype=np.int64)
    doc_chars = np.asarray(corpus["chars"], dtype=np.int64)
    doc_bytes = np.asarray(corpus["bytes"], dtype=np.int64)

    morph = morph_form_stats(tokenizer, morph_sample)
    form_tp = np.asarray(morph["tp"], dtype=np.int32)
    form_pred = np.asarray(morph["predicted"], dtype=np.int32)
    form_gold = np.asarray(morph["gold"], dtype=np.int32)
    form_edit = np.asarray(morph["edit"], dtype=np.int32)
    form_valid = np.asarray(morph["valid"], dtype=np.uint8)

    valid = form_valid.astype(bool)
    tp_sum = int(form_tp[valid].sum())
    pred_sum = int(form_pred[valid].sum())
    gold_sum = int(form_gold[valid].sum())
    n_valid = int(valid.sum())
    precision = tp_sum / pred_sum if pred_sum else 0.0
    recall = tp_sum / gold_sum if gold_sum else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    record: Dict[str, Any] = {
        "run_key": key,
        "tokenizer": name,
        "vocab_size": vocab_size,
        "vocab_used": len(tokenizer.vocab),
        "fertility": _safe_div(doc_tokens.sum(), doc_words.sum()).item(),
        "tokens_per_byte": _safe_div(doc_tokens.sum(), doc_bytes.sum()).item(),
        "chars_per_token": _safe_div(doc_chars.sum(), doc_tokens.sum()).item(),
        "ctc_total": float(doc_tokens.sum()),
        "renyi_efficiency": renyi_efficiency(cast(Dict[str, int], corpus["freqs"]), alpha),
        "morph_score_f1": f1,
        "morph_score_precision": precision,
        "morph_score_recall": recall,
        "morph_coverage": n_valid / len(form_valid) if len(form_valid) else 0.0,
        "morph_edit_distance": float(form_edit[valid].sum()) / n_valid if n_valid else 0.0,
        "morph_consistency": morph_consistency_over(tokenizer, morph_sample),
    }

    arrays: Dict[str, np.ndarray] = {
        "doc_tokens": doc_tokens,
        "doc_words": doc_words,
        "doc_chars": doc_chars,
        "doc_bytes": doc_bytes,
        "form_tp": form_tp,
        "form_predicted": form_pred,
        "form_gold": form_gold,
        "form_edit": form_edit,
        "form_valid": form_valid,
    }

    if deriv_sample:
        deriv_stats = morph_form_stats(tokenizer, deriv_sample)
        deriv = _morph_point_estimates(deriv_stats)
        record.update(
            {
                "morph_score_f1_deriv": deriv["f1"],
                "morph_score_precision_deriv": deriv["precision"],
                "morph_score_recall_deriv": deriv["recall"],
                "morph_edit_distance_deriv": deriv["edit"],
                "morph_coverage_deriv": deriv["coverage"],
                "morph_consistency_deriv": morph_consistency_over(tokenizer, deriv_sample),
            }
        )
        # Persist the derivational per-form arrays so the aggregation can attach
        # bootstrap CIs + paired significance to the derivational boundary metrics.
        arrays.update(
            {
                "form_tp_deriv": np.asarray(deriv_stats["tp"], dtype=np.int32),
                "form_predicted_deriv": np.asarray(deriv_stats["predicted"], dtype=np.int32),
                "form_gold_deriv": np.asarray(deriv_stats["gold"], dtype=np.int32),
                "form_edit_deriv": np.asarray(deriv_stats["edit"], dtype=np.int32),
                "form_valid_deriv": np.asarray(deriv_stats["valid"], dtype=np.uint8),
            }
        )

    (artifact_dir / "metrics.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(artifact_dir / EVAL_UNITS_FILENAME, **arrays)
    return record


def _load_units(output_root: Path, run_key: str) -> Dict[str, np.ndarray]:
    """Load a run's per-unit sufficient-statistics sidecar.

    Args:
        output_root (Path): Directory holding per-run artifact subdirs.
        run_key (str): Run key (`<name>-<vocab>`).

    Returns:
        Dict[str, np.ndarray]: The arrays stored in `eval_units.npz`.

    Raises:
        FileNotFoundError: If the sidecar is missing.
    """
    path = output_root / run_key / EVAL_UNITS_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Missing {EVAL_UNITS_FILENAME} for {run_key}; re-run evaluation.")
    with np.load(path) as handle:
        return {name: handle[name] for name in handle.files}


def _components_for(
    metric: str,
    units: Dict[str, np.ndarray],
    contexts: Dict[str, Any],
) -> Tuple[Tuple[np.ndarray, ...], st.CombineFn, np.ndarray]:
    """Return the per-unit components, combine, and resample indices for a metric.

    Morph metrics carry a `_deriv` suffix when they score the derivational gold;
    they read the `*_deriv` per-form arrays and resample over the derivational
    track's own valid-form mask. Both morph tracks share the dispatch below.

    Args:
        metric (str): A decomposable metric name.
        units (Dict[str, np.ndarray]): The run's per-unit arrays.
        contexts (Dict[str, Any]): Resample contexts: `corpus` maps to the
            document-resample indices; `infl` and `deriv` each map to a
            `(valid_pos, morph_idx)` tuple for that morph track.

    Returns:
        Tuple[Tuple[np.ndarray, ...], st.CombineFn, np.ndarray]: The components,
            the combine callable, and the resample-index matrix to use.

    Raises:
        ValueError: If `metric` is not decomposable.
    """
    if metric == "fertility":
        return (units["doc_tokens"], units["doc_words"]), _safe_div, contexts["corpus"]
    if metric == "tokens_per_byte":
        return (units["doc_tokens"], units["doc_bytes"]), _safe_div, contexts["corpus"]
    if metric == "chars_per_token":
        return (units["doc_chars"], units["doc_tokens"]), _safe_div, contexts["corpus"]
    if metric in ("morph_score_f1", "morph_score_f1_deriv"):
        suffix = "_deriv" if metric.endswith("_deriv") else ""
        valid_pos, morph_idx = contexts["deriv" if suffix else "infl"]
        tp = units[f"form_tp{suffix}"][valid_pos]
        predicted = units[f"form_predicted{suffix}"][valid_pos]
        gold = units[f"form_gold{suffix}"][valid_pos]
        return (tp, predicted, gold), _f1_combine, morph_idx
    if metric in ("morph_edit_distance", "morph_edit_distance_deriv"):
        suffix = "_deriv" if metric.endswith("_deriv") else ""
        valid_pos, morph_idx = contexts["deriv" if suffix else "infl"]
        edit = units[f"form_edit{suffix}"][valid_pos]
        return (edit, np.ones_like(edit)), _safe_div, morph_idx
    raise ValueError(f"Not a decomposable metric: {metric}")


def _vocab_significance(
    keys: List[str],
    names: Dict[str, str],
    points: Dict[str, float],
    dists: Dict[str, np.ndarray],
    *,
    direction: str,
    ci_level: float,
) -> Dict[str, Any]:
    """Build the paired-significance block for one metric within one vocab.

    Runs every pairwise paired-bootstrap difference, Holm-corrects the family of
    p-values, and produces a best-to-worst ranking with compact-letter groups
    (tokenizers sharing a letter are not significantly different).

    Args:
        keys (List[str]): Run keys compared in this vocab.
        names (Dict[str, str]): Run key to tokenizer name.
        points (Dict[str, float]): Run key to point estimate.
        dists (Dict[str, np.ndarray]): Run key to bootstrap distribution (drawn
            from the shared resample indices).
        direction (str): `higher` or `lower` (which is better for the metric).
        ci_level (float): Confidence level for the difference CIs.

    Returns:
        Dict[str, Any]: A block with `ranking` and `pairs` entries.
    """
    pairs_raw: List[Tuple[str, str, float, Tuple[float, float], float]] = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            diff = dists[keys[i]] - dists[keys[j]]
            pairs_raw.append(
                (
                    keys[i],
                    keys[j],
                    float(np.median(diff)),
                    st.percentile_ci(diff, ci_level),
                    st.bootstrap_p_value(diff),
                )
            )
    adjusted = st.holm_correction([pr[4] for pr in pairs_raw])

    pairs: List[Dict[str, Any]] = []
    significant_pair: Dict[Tuple[str, str], bool] = {}
    for (left, right, diff_median, diff_ci, p_raw), p_adj in zip(pairs_raw, adjusted):
        significant = p_adj < 0.05
        significant_pair[(left, right)] = significant
        significant_pair[(right, left)] = significant
        pairs.append(
            {
                "a": names[left],
                "b": names[right],
                "diff_median": diff_median,
                "diff_ci": [diff_ci[0], diff_ci[1]],
                "p_raw": p_raw,
                "p_adj": p_adj,
                "significant": significant,
            }
        )

    ordered = sorted(keys, key=lambda k: points[k], reverse=(direction == "higher"))
    n = len(ordered)
    not_different = [
        [(i == j) or (not significant_pair.get((ordered[i], ordered[j]), False)) for j in range(n)] for i in range(n)
    ]
    letters = st.compact_letters(n, not_different)
    ranking = [{"tokenizer": names[ordered[i]], "value": points[ordered[i]], "letters": letters[i]} for i in range(n)]
    return {"ranking": ranking, "pairs": pairs}


def _morph_context(
    units: Dict[str, Dict[str, np.ndarray]],
    keys: List[str],
    valid_key: str,
    n_resamples: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a morph track's cross-run valid mask and shared resample indices.

    Args:
        units (Dict[str, Dict[str, np.ndarray]]): Per-run unit arrays by run key.
        keys (List[str]): Run keys compared in this vocab.
        valid_key (str): Per-form validity array name (`form_valid` or
            `form_valid_deriv`).
        n_resamples (int): Bootstrap resamples (B).
        seed (int): Seed for this track's resample indices.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The form positions valid across every
            compared run, and the resample-index matrix over those positions.
    """
    valid_stack = np.vstack([units[key][valid_key].astype(bool) for key in keys])
    valid_pos = np.nonzero(valid_stack.all(axis=0))[0]
    idx = st.make_resample_indices(len(valid_pos), n_resamples, seed)
    return valid_pos, idx


def augment_with_statistics(
    records: List[Dict[str, Any]],
    output_root: Path,
    *,
    n_resamples: int,
    ci_level: float,
    seed: int,
    morph_form_sample: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Add bootstrap CIs to records and compute paired significance per vocab.

    For each vocab size, all tokenizers share one set of resample indices per
    track (documents for corpus metrics, inflectional forms, and derivational
    forms) so that paired differences are computed on the same resamples. Each
    morph track is bootstrapped over the forms valid across every compared
    tokenizer. Each record gains `<metric>_ci` and `<metric>_std` for every
    decomposable metric present (corpus + inflectional morph, plus derivational
    morph when the `*_deriv` per-form arrays exist), whose point estimate is also
    recomputed from the per-unit arrays so it matches the CI; `renyi_efficiency`,
    `morph_consistency`, and `morph_consistency_deriv` stay bare point estimates.

    Args:
        records (List[Dict[str, Any]]): Per-run metric records (mutated in place).
        output_root (Path): Directory holding per-run artifact subdirs.
        n_resamples (int): Bootstrap resamples (B).
        ci_level (float): Confidence level for the CIs.
        seed (int): Base seed for the shared resample indices.
        morph_form_sample (int): Size of the shared morph form sample (recorded).

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: The `significance` section keyed
            `[vocab][metric]` and the `stats_config` section.
    """
    by_vocab: Dict[int, List[Dict[str, Any]]] = {}
    for record in records:
        by_vocab.setdefault(int(record["vocab_size"]), []).append(record)

    significance: Dict[str, Any] = {}
    for vocab, recs in sorted(by_vocab.items()):
        keys = [r["run_key"] for r in recs]
        names = {r["run_key"]: r["tokenizer"] for r in recs}
        rec_by_key = {r["run_key"]: r for r in recs}
        units = {key: _load_units(output_root, key) for key in keys}

        # Per-track resample contexts. corpus uses base; inflectional uses base+1;
        # derivational uses base+500_000 so its seed stays disjoint from both for
        # every vocab size (vocab*2 stays well under 500_000).
        base = seed * 1_000_003 + vocab * 2
        contexts: Dict[str, Any] = {
            "corpus": st.make_resample_indices(len(units[keys[0]]["doc_tokens"]), n_resamples, base),
            "infl": _morph_context(units, keys, "form_valid", n_resamples, base + 1),
        }
        metrics = list(_DECOMPOSABLE_METRICS)
        if all("form_valid_deriv" in units[key] for key in keys):
            contexts["deriv"] = _morph_context(units, keys, "form_valid_deriv", n_resamples, base + 500_000)
            metrics += _DERIV_DECOMPOSABLE_METRICS

        vocab_block: Dict[str, Any] = {}
        for metric in metrics:
            points: Dict[str, float] = {}
            dists: Dict[str, np.ndarray] = {}
            for key in keys:
                components, combine, idx = _components_for(metric, units[key], contexts)
                dist = st.bootstrap_distribution(idx, components, combine)
                point = st.point_estimate(components, combine)
                low, high = st.percentile_ci(dist, ci_level)
                rec_by_key[key][metric] = point
                rec_by_key[key][f"{metric}_ci"] = [low, high]
                rec_by_key[key][f"{metric}_std"] = float(dist.std(ddof=1)) if dist.size > 1 else 0.0
                points[key] = point
                dists[key] = dist
            vocab_block[metric] = _vocab_significance(
                keys, names, points, dists, direction=METRIC_DIRECTIONS[metric], ci_level=ci_level
            )
        significance[str(vocab)] = vocab_block

    stats_config = {
        "n_resamples": n_resamples,
        "ci_level": ci_level,
        "seed": seed,
        "morph_form_sample": morph_form_sample,
        "units": {"corpus": "documents", "morph": "forms", "morph_deriv": "forms"},
        "point_only": ["renyi_efficiency", "morph_consistency", "morph_consistency_deriv"],
    }
    return significance, stats_config


def _format_cell(value: Any) -> str:
    """Format a metric value for the Markdown table.

    Args:
        value (Any): A metric value.

    Returns:
        str: A 4-decimal float, or the string form for non-numbers.
    """
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _format_metric_cell(record: Dict[str, Any], metric: str) -> str:
    """Format a metric cell, appending its bootstrap CI when available.

    Args:
        record (Dict[str, Any]): A per-run record.
        metric (str): The metric column name.

    Returns:
        str: The point estimate, with a `[lo, hi]` 95%-CI suffix for the five
            decomposable metrics that carry one.
    """
    value = _format_cell(record.get(metric, ""))
    ci = record.get(f"{metric}_ci")
    if ci is not None:
        value = f"{value} [{ci[0]:.4f}, {ci[1]:.4f}]"
    return value


def build_report(
    results: List[Dict[str, Any]],
    significance: Optional[Dict[str, Any]] = None,
    stats_config: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Build the Markdown comparison table and JSON payload.

    Args:
        results (List[Dict[str, Any]]): Per-run metric records.
        significance (Optional[Dict[str, Any]]): Paired-significance section
            keyed `[vocab][metric]`, or None when statistics were not computed.
        stats_config (Optional[Dict[str, Any]]): The bootstrap configuration
            recorded alongside the results, or None.

    Returns:
        Tuple[str, Dict[str, Any]]: The Markdown report and a JSON payload
            carrying the results, directions, significance, and stats config.
    """
    ordered = sorted(results, key=lambda r: (r["tokenizer"], r["vocab_size"]))
    columns = _REPORT_COLUMNS + [col for col in _DERIV_COLUMNS if any(col in r for r in results)]
    headers = ["tokenizer", "vocab"] + [
        f"{col} {_DIRECTION_ARROW.get(METRIC_DIRECTIONS.get(col, ''), '')}".strip() for col in columns
    ]
    lines = [
        "# Tokenizer comparison (Slovenian)",
        "",
        "Base morph metrics use a Sloleks-derived silver-gold segmentation "
        "(inflectional only); `*_deriv` columns use the Sloleks word-relations "
        "derivational silver-gold (lemmas). Treat both as relative comparators, "
        "not absolute morphological accuracy.",
        "",
        "Decomposable metrics carry a 95% bootstrap CI in brackets (documents "
        "resampled for corpus metrics, forms for the morph metrics, including "
        "the `morph_score_f1_deriv` / `morph_edit_distance_deriv` derivational "
        "metrics). `renyi_efficiency`, `morph_consistency`, and "
        "`morph_consistency_deriv` are point estimates only.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for record in ordered:
        row = [record["tokenizer"], str(record["vocab_size"])]
        row += [_format_metric_cell(record, col) for col in columns]
        lines.append("| " + " | ".join(row) + " |")

    payload = {
        "results": ordered,
        "directions": METRIC_DIRECTIONS,
        "significance": significance or {},
        "stats_config": stats_config or {},
    }
    return "\n".join(lines) + "\n", payload


def write_report(
    results: List[Dict[str, Any]],
    report_dir: Path,
    significance: Optional[Dict[str, Any]] = None,
    stats_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Path, Path]:
    """Write the Markdown and JSON reports under `report_dir`.

    Args:
        results (List[Dict[str, Any]]): Per-run metric records.
        report_dir (Path): Destination directory, created if missing.
        significance (Optional[Dict[str, Any]]): Paired-significance section.
        stats_config (Optional[Dict[str, Any]]): Bootstrap configuration.

    Returns:
        Tuple[Path, Path]: The Markdown and JSON report paths.
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    markdown, payload = build_report(results, significance, stats_config)
    md_path = report_dir / "report.md"
    json_path = report_dir / "report.json"
    md_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return md_path, json_path


def _train_link_tags(artifact_dir: Path) -> Dict[str, str]:
    """Build cross-link tags pointing an eval run at its training run.

    Reads the `mlflow_train.json` sidecar written by the training logger and
    returns tags that surface the linked training run in the MLflow UI. Returns
    an empty mapping when no sidecar exists (training was not MLflow-logged).

    Args:
        artifact_dir (Path): The run's artifact directory.

    Returns:
        Dict[str, str]: Tags keyed `train_run_id` / `train_parent_run_id` /
            `train_run_name`, or an empty mapping when no linkage is present.
    """
    link_path = artifact_dir / MLFLOW_LINK_FILENAME
    if not link_path.exists():
        return {}
    link = json.loads(link_path.read_text(encoding="utf-8"))
    return {
        "train_run_id": link.get("run_id") or "unknown",
        "train_parent_run_id": link.get("parent_run_id") or "unknown",
        "train_run_name": link.get("run_name") or "unknown",
    }


def log_results_to_mlflow(
    results: List[Dict[str, Any]],
    cfg: TokenizerSweepConfig,
    report_paths: Tuple[Path, Path],
) -> None:
    """Log the sweep to MLflow as a parent run with nested per-run children.

    Each child logs its parameters and metrics under the `phase=eval` tag, and
    is cross-linked back to the training run that produced its artifact (via the
    `mlflow_train.json` sidecar) through `train_run_id` tags. The parent records
    the best MorphScore F1 and the report artifacts. A no-op when tracking is
    disabled.

    Args:
        results (List[Dict[str, Any]]): Per-run metric records.
        cfg (TokenizerSweepConfig): The resolved sweep configuration.
        report_paths (Tuple[Path, Path]): The Markdown and JSON report paths.
    """
    if not cfg.mlflow_enabled:
        return
    if not ml.ensure_experiment(cfg.mlflow_experiment, tracking_uri=cfg.mlflow_tracking_uri):
        logger.warning("MLflow unavailable; skipping logging.")
        return

    commit = ml.git_commit()
    metric_keys = [
        *_REPORT_COLUMNS,
        *_DERIV_COLUMNS,
        "morph_score_precision",
        "morph_score_recall",
        "morph_score_precision_deriv",
        "morph_score_recall_deriv",
        "morph_coverage_deriv",
        "vocab_used",
    ]
    with ml.mlflow_run("sweep-eval", tags={"run_type": "sweep", "phase": "eval"}):
        for record in results:
            with ml.mlflow_run(
                record["run_key"],
                nested=True,
                tags={
                    "model_type": record["tokenizer"],
                    "model_version": str(record["vocab_size"]),
                    "run_type": "sweep",
                    "phase": "eval",
                    "git_commit": commit or "unknown",
                },
            ):
                ml.log_params(
                    {
                        "tokenizer": record["tokenizer"],
                        "vocab_size": record["vocab_size"],
                        "seed": cfg.train_budget.seed,
                        "renyi_alpha": cfg.renyi_alpha,
                    }
                )
                ml.log_metrics({k: record[k] for k in metric_keys if k in record})
                ci_metrics: Dict[str, float] = {}
                for metric in (*_DECOMPOSABLE_METRICS, *_DERIV_DECOMPOSABLE_METRICS):
                    bounds = record.get(f"{metric}_ci")
                    if bounds is not None:
                        ci_metrics[f"{metric}_ci_low"] = bounds[0]
                        ci_metrics[f"{metric}_ci_high"] = bounds[1]
                ml.log_metrics(ci_metrics)
                ml.set_tags(_train_link_tags(cfg.output_root / record["run_key"]))

        best = max(results, key=lambda r: r["morph_score_f1"], default=None)
        if best is not None:
            ml.log_metrics({"best_morph_score_f1": best["morph_score_f1"]})
            ml.set_tags({"best_run_key": best["run_key"]})
        for path in report_paths:
            ml.log_artifact(path)


@dataclass
class EvaluationSummary:
    """Outcome of an evaluation sweep.

    Attributes:
        records: Per-run metric records, one per evaluated artifact.
        failed: Run keys whose evaluation raised.
        report_paths: The Markdown and JSON report paths, or None when no run
            produced a record.
    """

    records: List[Dict[str, Any]] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)
    report_paths: Optional[Tuple[Path, Path]] = None


def _evaluate_worker(key: str, *, output_root: Path, alpha: float) -> Optional[Dict[str, Any]]:
    """Evaluate one run, reading the shared eval docs and morph sample globals.

    The grouped evaluation documents and the shared morph and derivational form
    samples live in module globals so a forked process-pool worker inherits them
    copy-on-write, rather than receiving them through pickled keyword arguments
    (which would copy the multi-million-form inputs once per task).

    Args:
        key (str): Run key (`<name>-<vocab>`).
        output_root (Path): Directory holding per-run artifact subdirs.
        alpha (float): Renyi order.

    Returns:
        Optional[Dict[str, Any]]: The metrics record, or None when skipped.
    """
    return evaluate_artifact(
        key,
        output_root=output_root,
        eval_docs=_EVAL_DOCS,
        morph_sample=_MORPH_SAMPLE,
        deriv_sample=_DERIV_SAMPLE,
        alpha=alpha,
    )


def prepare_eval_inputs(
    cfg: TokenizerSweepConfig,
    force: bool,
    morph_form_sample: int,
) -> Tuple[List[List[str]], List[MorphemeSegmentation], List[MorphemeSegmentation]]:
    """Materialize the grouped eval docs and the shared morph form samples.

    The evaluation documents are kept grouped (one word list per document) so the
    corpus metrics can resample documents. The morph sample is a deterministic
    draw from the Sloleks-derived lexicon, identical across runs so per-form morph
    statistics align by index for the paired tests.

    Args:
        cfg (TokenizerSweepConfig): The resolved sweep configuration.
        force (bool): Rebuild the eval sample and lexicons even if present.
        morph_form_sample (int): Size of the shared morph form sample.

    Returns:
        Tuple[List[List[str]], List[MorphemeSegmentation], List[MorphemeSegmentation]]:
            The grouped eval documents, the shared inflectional morph form
            sample, and the derivational form sample (empty when no derivational
            gold is configured).
    """
    eval_path = cfg.eval_sample_path
    if force or not eval_path.exists():
        logger.info("Sampling held-out evaluation corpus -> %s", eval_path)
        write_sample_cache(sample_corpus(cfg.corpus_root, cfg.eval_budget), eval_path)
    eval_docs = [iter_words(line) for line in iter_sample_cache(eval_path)]
    logger.info("Evaluation sample: %d documents, %d word tokens", len(eval_docs), sum(len(d) for d in eval_docs))

    infl_path = cfg.infl_lexicon_path
    if force or not infl_path.exists():
        logger.info("Deriving inflectional gold lexicon from %s", cfg.sloleks_path)
        save_lexicon(build_morph_lexicon(cfg.sloleks_path, min_stem_len=cfg.min_stem_len), infl_path)
    lexicon = load_lexicon(infl_path)
    morph_sample = sample_segmentations(lexicon, morph_form_sample, cfg.stats_seed)
    logger.info("Inflectional gold: %d forms; morph sample: %d forms", len(lexicon.by_form), len(morph_sample))

    deriv_sample: List[MorphemeSegmentation] = []
    if cfg.needs_derivational():
        deriv_path = cfg.deriv_lexicon_path
        if force or not deriv_path.exists():
            logger.info("Deriving derivational gold lexicon from %s", cfg.sloleks_relations_path)
            save_lexicon(build_derivational_lexicon(cfg.sloleks_relations_path), deriv_path)
        deriv_lexicon = load_lexicon(deriv_path)
        deriv_sample = sample_segmentations(deriv_lexicon, morph_form_sample, cfg.stats_seed)
        logger.info(
            "Derivational gold: %d forms; deriv sample: %d forms", len(deriv_lexicon.by_form), len(deriv_sample)
        )

    return eval_docs, morph_sample, deriv_sample


def evaluate_sweep(
    cfg: TokenizerSweepConfig,
    keys: List[str],
    force: bool = False,
    morph_form_sample: Optional[int] = None,
    n_resamples: Optional[int] = None,
    max_workers: int = 0,
) -> EvaluationSummary:
    """Evaluate trained tokenizer artifacts and write the comparison report.

    Materializes the shared evaluation inputs, scores every selected artifact in
    a process pool, attaches bootstrap CIs and paired significance, writes the
    Markdown and JSON reports, and logs the sweep to MLflow.

    Args:
        cfg (TokenizerSweepConfig): The resolved sweep configuration.
        keys (List[str]): Run keys (`<name>-<vocab>`) with trained artifacts.
        force (bool): Rebuild the evaluation sample and gold lexicons.
        morph_form_sample (Optional[int]): Override for the config's morph
            form-sample size, or None to use the configured value.
        n_resamples (Optional[int]): Override for the config's bootstrap
            resample count, or None to use the configured value.
        max_workers (int): Parallel evaluations. 0=auto, 1=serial, N=N workers.

    Returns:
        EvaluationSummary: The metric records, failed keys, and report paths.
    """
    # CPU-bound metrics: cap the auto default conservatively so a many-core box
    # is not saturated by default.
    workers = resolve_workers(max_workers, len(keys), min(DEFAULT_MAX_WORKERS, cpu_default(len(keys))))
    configure_script_logging(parallel=workers > 1, console_level=logging.INFO)

    form_sample = morph_form_sample if morph_form_sample is not None else cfg.stats_morph_form_sample
    resamples = n_resamples if n_resamples is not None else cfg.stats_n_resamples

    # Publish the heavy eval inputs to module globals BEFORE the process pool
    # forks, so workers inherit them copy-on-write rather than via pickling.
    global _EVAL_DOCS, _MORPH_SAMPLE, _DERIV_SAMPLE
    _EVAL_DOCS, _MORPH_SAMPLE, _DERIV_SAMPLE = prepare_eval_inputs(cfg, force=force, morph_form_sample=form_sample)

    def kwargs_for(_key: str) -> Dict[str, Any]:
        """Return per-run kwargs for `_evaluate_worker`.

        Only small, picklable values go here; the eval docs and morph sample are
        shared via module globals (see `_evaluate_worker`).

        Args:
            _key (str): Run key (unused; kwargs are identical per run).

        Returns:
            Dict[str, Any]: Keyword arguments for `_evaluate_worker`.
        """
        return {"output_root": cfg.output_root, "alpha": cfg.renyi_alpha}

    results, failures = run_parallel(
        _evaluate_worker,
        keys,
        max_workers=workers,
        desc="tokenizer-evaluate",
        pool="process",
        kwargs_for=kwargs_for,
        log_dir=stamped_log_dir("tokenizer-evaluate"),
    )

    records = [r for r in results.values() if r is not None]
    summary = EvaluationSummary(records=records, failed=[k for k, _ in failures])
    if records:
        logger.info("Computing bootstrap CIs + paired significance (B=%d) ...", resamples)
        significance, stats_config = augment_with_statistics(
            records,
            cfg.output_root,
            n_resamples=resamples,
            ci_level=cfg.stats_ci_level,
            seed=cfg.stats_seed,
            morph_form_sample=form_sample,
        )
        summary.report_paths = write_report(records, cfg.report_dir, significance, stats_config)
        logger.info("Wrote report: %s", summary.report_paths[0])
        log_results_to_mlflow(records, cfg, summary.report_paths)

    logger.info("Done. Evaluated %d, failed %s.", len(records), summary.failed or "none")
    return summary
