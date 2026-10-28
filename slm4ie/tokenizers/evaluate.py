"""Evaluate trained tokenizers and build the comparison report.

`evaluate_artifact` loads one trained tokenizer, runs the six metrics over the
grouped held-out documents and the shared Sloleks-derived morph sample, and
persists per-unit sufficient statistics to an `eval_units.npz` sidecar.
`augment_with_statistics` reads those sidecars to attach bootstrap confidence
intervals and paired significance tests (per vocab) to the five decomposable
metrics. `build_report` aggregates the per-run metrics into a Markdown table and
a JSON payload, and `log_evaluate_runs` records the sweep as a parent run
with one nested child per tokenizer x vocab-size run. `evaluate_sweep` ties
those together: it materializes the shared evaluation inputs, runs the metrics
over every selected artifact in parallel, and writes the report.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

import slm4ie.tokenizers.backends  # noqa: F401  (registers backends on import)
from slm4ie.tokenizers import stats as st
from slm4ie.tokenizers.config import SweepConfig
from slm4ie.tokenizers.corpus import check_sample_side, iter_sample, materialize_sample
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
    split_lexemes,
    without_forms,
)
from slm4ie.tokenizers.registry import get_tokenizer
from slm4ie.tokenizers.train import (
    MLFLOW_LINK_FILENAME,
    corpus_split_for,
    log_sweep_lineage,
    parse_sweep_run_key,
    run_tags,
    sweep_inputs,
)
from slm4ie.utils import mlflow as ml
from slm4ie.utils.cli import stamped_log_dir
from slm4ie.utils.parallel import (
    configure_script_logging,
    cpu_default,
    resolve_workers,
    run_parallel,
)

# ===============================
# Logging
# ===============================

logger = logging.getLogger(__name__)

# ===============================
# Defaults
# ===============================

#: Conservative upper bound on parallel eval workers so a large box is not
#: saturated by default; raise it explicitly with the worker argument.
DEFAULT_MAX_WORKERS = 8

#: Heavy, read-only eval inputs shared with process-pool workers via fork
#: inheritance (copy-on-write) instead of being pickled per task. Populated by
#: `evaluate_sweep` before the pool is created; read by `_evaluate_worker`.
_EVAL_DOCS: list[list[str]] = []
_MORPH_SAMPLE: list[MorphemeSegmentation] = []
_DERIV_SAMPLE: list[MorphemeSegmentation] = []

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


# ===============================
# Bootstrap helpers
# ===============================


def _safe_div(numerator: Any, denominator: Any) -> np.ndarray:
    """Divide elementwise, returning 0 where the denominator is 0.

    Args:
        numerator: Summed numerator (scalar or bootstrap-axis array).
        denominator: Summed denominator (scalar or bootstrap-axis array).

    Returns:
        The elementwise ratio with zeros where the denominator is 0.
    """
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    return np.divide(num, den, out=np.zeros_like(num), where=den != 0)


def _f1_combine(true_positive: Any, predicted: Any, gold: Any) -> np.ndarray:
    """Combine summed boundary counts into the MorphScore F1.

    Args:
        true_positive: Summed correct boundaries.
        predicted: Summed predicted boundaries.
        gold: Summed gold boundaries.

    Returns:
        The F1, elementwise over the bootstrap axis.
    """
    precision = _safe_div(true_positive, predicted)
    recall = _safe_div(true_positive, gold)
    return _safe_div(2.0 * precision * recall, precision + recall)


def _morph_point_estimates(stats: dict[str, list[int]]) -> dict[str, float]:
    """Aggregate per-form morph statistics into point estimates.

    Used for the derivational track, which reports point estimates only (no
    bootstrap CIs), so it does not persist per-form arrays the way the
    inflectional track does.

    Args:
        stats: Per-form arrays from `morph_form_stats`
            (`tp`, `predicted`, `gold`, `edit`, `valid`).

    Returns:
        `f1`, `precision`, `recall` (boundary scores),
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


# ===============================
# Per-run evaluation
# ===============================


def load_tokenizer_artifact(artifact_dir: Path):
    """Load a trained tokenizer from its artifact directory.

    Args:
        artifact_dir: Directory holding `metadata.json` and the model.

    Returns:
        The reconstructed tokenizer.

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
) -> dict[str, Any] | None:
    """Run all six metrics for one trained tokenizer run.

    Encodes the held-out documents and the shared morph sample once, derives the
    point estimates from the per-unit sufficient statistics, and persists those
    statistics to an `eval_units.npz` sidecar so the aggregation can bootstrap
    confidence intervals and paired tests. Corpus point estimates use every
    document; morph point estimates use the forms in the sample that the
    tokenizer tiled.

    Args:
        key: Run key (`<name>-<vocab>`).
        output_root: Directory holding per-run artifact subdirs.
        eval_docs: Held-out evaluation word tokens,
            kept grouped per document (the corpus resampling unit).
        morph_sample: The shared, deterministic
            sample of gold segmentations (the morph resampling unit).
        deriv_sample: The derivational gold
            sample; when non-empty, derivational point-estimate columns
            (`morph_score_f1_deriv` etc.) are added. Empty skips the track.
        alpha: Renyi order.

    Returns:
        A flat metrics record, or None when the
            artifact is missing (skipped).
    """
    artifact_dir = output_root / key
    if not (artifact_dir / "metadata.json").exists():
        logger.warning("No artifact for %s; skipping.", key)
        return None

    # load the tokenizer
    name, vocab_size = parse_sweep_run_key(key)
    tokenizer = load_tokenizer_artifact(artifact_dir)

    # corpus statistics per document
    corpus = corpus_doc_stats(tokenizer, eval_docs)
    doc_tokens = np.asarray(corpus["tokens"], dtype=np.int64)
    doc_words = np.asarray(corpus["words"], dtype=np.int64)
    doc_chars = np.asarray(corpus["chars"], dtype=np.int64)
    doc_bytes = np.asarray(corpus["bytes"], dtype=np.int64)

    # morph statistics per form
    morph = morph_form_stats(tokenizer, morph_sample)
    form_tp = np.asarray(morph["tp"], dtype=np.int32)
    form_pred = np.asarray(morph["predicted"], dtype=np.int32)
    form_gold = np.asarray(morph["gold"], dtype=np.int32)
    form_edit = np.asarray(morph["edit"], dtype=np.int32)
    form_valid = np.asarray(morph["valid"], dtype=np.uint8)

    # morph point estimates over the tiled forms
    valid = form_valid.astype(bool)
    tp_sum = int(form_tp[valid].sum())
    pred_sum = int(form_pred[valid].sum())
    gold_sum = int(form_gold[valid].sum())
    n_valid = int(valid.sum())
    precision = tp_sum / pred_sum if pred_sum else 0.0
    recall = tp_sum / gold_sum if gold_sum else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # assemble the metrics record
    record: dict[str, Any] = {
        "run_key": key,
        "tokenizer": name,
        "vocab_size": vocab_size,
        "vocab_used": len(tokenizer.vocab),
        "fertility": _safe_div(doc_tokens.sum(), doc_words.sum()).item(),
        "tokens_per_byte": _safe_div(doc_tokens.sum(), doc_bytes.sum()).item(),
        "chars_per_token": _safe_div(doc_chars.sum(), doc_tokens.sum()).item(),
        "ctc_total": float(doc_tokens.sum()),
        "renyi_efficiency": renyi_efficiency(cast(dict[str, int], corpus["freqs"]), alpha),
        "morph_score_f1": f1,
        "morph_score_precision": precision,
        "morph_score_recall": recall,
        "morph_coverage": n_valid / len(form_valid) if len(form_valid) else 0.0,
        "morph_edit_distance": float(form_edit[valid].sum()) / n_valid if n_valid else 0.0,
        "morph_consistency": morph_consistency_over(tokenizer, morph_sample),
    }

    # collect the per-unit arrays for the sidecar
    arrays: dict[str, np.ndarray] = {
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

    # add the derivational track
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
        # persist the per-form arrays so the aggregation can bootstrap the derivational metrics
        arrays.update(
            {
                "form_tp_deriv": np.asarray(deriv_stats["tp"], dtype=np.int32),
                "form_predicted_deriv": np.asarray(deriv_stats["predicted"], dtype=np.int32),
                "form_gold_deriv": np.asarray(deriv_stats["gold"], dtype=np.int32),
                "form_edit_deriv": np.asarray(deriv_stats["edit"], dtype=np.int32),
                "form_valid_deriv": np.asarray(deriv_stats["valid"], dtype=np.uint8),
            }
        )

    # write the record and the sidecar
    (artifact_dir / "metrics.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(artifact_dir / EVAL_UNITS_FILENAME, **arrays)
    return record


# ===============================
# Statistics aggregation
# ===============================


def _load_units(output_root: Path, run_key: str) -> dict[str, np.ndarray]:
    """Load a run's per-unit sufficient-statistics sidecar.

    Args:
        output_root: Directory holding per-run artifact subdirs.
        run_key: Run key (`<name>-<vocab>`).

    Returns:
        The arrays stored in `eval_units.npz`.

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
    units: dict[str, np.ndarray],
    contexts: dict[str, Any],
) -> tuple[tuple[np.ndarray, ...], st.CombineFn, np.ndarray]:
    """Return the per-unit components, combine, and resample indices for a metric.

    Morph metrics carry a `_deriv` suffix when they score the derivational gold;
    they read the `*_deriv` per-form arrays and resample over the derivational
    track's own valid-form mask. Both morph tracks share the dispatch below.

    Args:
        metric: A decomposable metric name.
        units: The run's per-unit arrays.
        contexts: Resample contexts: `corpus` maps to the
            document-resample indices; `infl` and `deriv` each map to a
            `(valid_pos, morph_idx)` tuple for that morph track.

    Returns:
        The components,
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
    keys: list[str],
    names: dict[str, str],
    points: dict[str, float],
    dists: dict[str, np.ndarray],
    *,
    direction: str,
    ci_level: float,
) -> dict[str, Any]:
    """Build the paired-significance block for one metric within one vocab.

    Runs every pairwise paired-bootstrap difference, Holm-corrects the family of
    p-values, and produces a best-to-worst ranking with compact-letter groups
    (tokenizers sharing a letter are not significantly different).

    Args:
        keys: Run keys compared in this vocab.
        names: Run key to tokenizer name.
        points: Run key to point estimate.
        dists: Run key to bootstrap distribution (drawn
            from the shared resample indices).
        direction: `higher` or `lower` (which is better for the metric).
        ci_level: Confidence level for the difference CIs.

    Returns:
        A block with `ranking` and `pairs` entries.
    """
    # paired bootstrap differences for every pair
    pairs_raw: list[tuple[str, str, float, tuple[float, float], float]] = []
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

    # Holm-correct the p-values
    adjusted = st.holm_correction([pr[4] for pr in pairs_raw])
    pairs: list[dict[str, Any]] = []
    significant_pair: dict[tuple[str, str], bool] = {}
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

    # rank best to worst with compact letters
    ordered = sorted(keys, key=lambda k: points[k], reverse=(direction == "higher"))
    n = len(ordered)
    not_different = [
        [(i == j) or (not significant_pair.get((ordered[i], ordered[j]), False)) for j in range(n)] for i in range(n)
    ]
    letters = st.compact_letters(n, not_different)
    ranking = [{"tokenizer": names[ordered[i]], "value": points[ordered[i]], "letters": letters[i]} for i in range(n)]
    return {"ranking": ranking, "pairs": pairs}


def _morph_context(
    units: dict[str, dict[str, np.ndarray]],
    keys: list[str],
    valid_key: str,
    n_resamples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a morph track's cross-run valid mask and shared resample indices.

    Args:
        units: Per-run unit arrays by run key.
        keys: Run keys compared in this vocab.
        valid_key: Per-form validity array name (`form_valid` or
            `form_valid_deriv`).
        n_resamples: Bootstrap resamples (B).
        seed: Seed for this track's resample indices.

    Returns:
        The form positions valid across every
            compared run, and the resample-index matrix over those positions.
    """
    valid_stack = np.vstack([units[key][valid_key].astype(bool) for key in keys])
    valid_pos = np.nonzero(valid_stack.all(axis=0))[0]
    idx = st.make_resample_indices(len(valid_pos), n_resamples, seed)
    return valid_pos, idx


def augment_with_statistics(
    records: list[dict[str, Any]],
    output_root: Path,
    *,
    n_resamples: int,
    ci_level: float,
    seed: int,
    morph_form_sample: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
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
        records: Per-run metric records (mutated in place).
        output_root: Directory holding per-run artifact subdirs.
        n_resamples: Bootstrap resamples (B).
        ci_level: Confidence level for the CIs.
        seed: Base seed for the shared resample indices.
        morph_form_sample: Size of the shared morph form sample (recorded).

    Returns:
        The `significance` section keyed
            `[vocab][metric]` and the `stats_config` section.
    """
    # group the records by vocab size
    by_vocab: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        by_vocab.setdefault(int(record["vocab_size"]), []).append(record)

    significance: dict[str, Any] = {}
    for vocab, recs in sorted(by_vocab.items()):
        keys = [r["run_key"] for r in recs]
        names = {r["run_key"]: r["tokenizer"] for r in recs}
        rec_by_key = {r["run_key"]: r for r in recs}
        units = {key: _load_units(output_root, key) for key in keys}

        # resample contexts: corpus uses base, inflectional base+1, derivational base+500_000
        # (disjoint seeds for every vocab size; vocab*2 stays well under 500_000)
        base = seed * 1_000_003 + vocab * 2
        contexts: dict[str, Any] = {
            "corpus": st.make_resample_indices(len(units[keys[0]]["doc_tokens"]), n_resamples, base),
            "infl": _morph_context(units, keys, "form_valid", n_resamples, base + 1),
        }
        metrics = list(_DECOMPOSABLE_METRICS)
        if all("form_valid_deriv" in units[key] for key in keys):
            contexts["deriv"] = _morph_context(units, keys, "form_valid_deriv", n_resamples, base + 500_000)
            metrics += _DERIV_DECOMPOSABLE_METRICS

        # bootstrap every metric and rank the tokenizers
        vocab_block: dict[str, Any] = {}
        for metric in metrics:
            points: dict[str, float] = {}
            dists: dict[str, np.ndarray] = {}
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


# ===============================
# Report
# ===============================


def _format_cell(value: Any) -> str:
    """Format a metric value for the Markdown table.

    Args:
        value: A metric value.

    Returns:
        A 4-decimal float, or the string form for non-numbers.
    """
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _format_metric_cell(record: dict[str, Any], metric: str) -> str:
    """Format a metric cell, appending its bootstrap CI when available.

    Args:
        record: A per-run record.
        metric: The metric column name.

    Returns:
        The point estimate, with a `[lo, hi]` 95%-CI suffix for the five
            decomposable metrics that carry one.
    """
    value = _format_cell(record.get(metric, ""))
    ci = record.get(f"{metric}_ci")
    if ci is not None:
        value = f"{value} [{ci[0]:.4f}, {ci[1]:.4f}]"
    return value


def build_report(
    results: list[dict[str, Any]],
    significance: dict[str, Any] | None = None,
    stats_config: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Build the Markdown comparison table and JSON payload.

    Args:
        results: Per-run metric records.
        significance: Paired-significance section
            keyed `[vocab][metric]`, or None when statistics were not computed.
        stats_config: The bootstrap configuration
            recorded alongside the results, or None.

    Returns:
        The Markdown report and a JSON payload
            carrying the results, directions, significance, and stats config.
    """
    # header and explanatory text
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

    # one table row per run
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
    results: list[dict[str, Any]],
    report_dir: Path,
    significance: dict[str, Any] | None = None,
    stats_config: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write the Markdown and JSON reports under `report_dir`.

    Args:
        results: Per-run metric records.
        report_dir: Destination directory, created if missing.
        significance: Paired-significance section.
        stats_config: Bootstrap configuration.

    Returns:
        The Markdown and JSON report paths.
    """
    # build the report
    report_dir.mkdir(parents=True, exist_ok=True)
    markdown, payload = build_report(results, significance, stats_config)

    # write the Markdown and JSON files
    md_path = report_dir / "report.md"
    json_path = report_dir / "report.json"
    md_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return md_path, json_path


# ===============================
# MLflow logging
# ===============================


def _train_link_tags(artifact_dir: Path) -> dict[str, str]:
    """Build cross-link tags pointing an eval run at its training run.

    Reads the `mlflow_train.json` sidecar written by the training logger and
    returns tags that surface the linked training run in the MLflow UI. Returns
    an empty mapping when no sidecar exists (training was not MLflow-logged).

    Args:
        artifact_dir: The run's artifact directory.

    Returns:
        Tags keyed `train_run_id` / `train_parent_run_id` /
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


def log_evaluate_runs(
    results: list[dict[str, Any]],
    cfg: SweepConfig,
    report_paths: tuple[Path, Path],
) -> None:
    """Log the sweep to MLflow as a parent run with nested per-run children.

    Each child logs its parameters and metrics under the `phase=eval` tag, and
    is cross-linked back to the training run that produced its artifact (via the
    `mlflow_train.json` sidecar) through `train_run_id` tags. The parent records
    the best MorphScore F1 and the report artifacts. A no-op when tracking is
    disabled.

    Args:
        results: Per-run metric records.
        cfg: The resolved sweep configuration.
        report_paths: The Markdown and JSON report paths.
    """
    if not cfg.mlflow_enabled:
        return
    if not ml.ensure_experiment(cfg.mlflow_experiment, tracking_uri=cfg.mlflow_tracking_uri):
        logger.warning("MLflow unavailable; skipping logging.")
        return

    # settings shared by every run
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
    tags = run_tags(cfg)
    gold_paths = {"sloleks-inflection": cfg.inflection_path} if cfg.inflection_path else {}
    inputs = sweep_inputs({"corpus-split": cfg.corpus_split_path, "eval-sample": cfg.eval_sample_path, **gold_paths})

    # log the parent run with one child per tokenizer
    with ml.mlflow_run("sweep-eval", tags={**tags, "run_type": "sweep", "phase": "eval"}):
        for record in results:
            with ml.mlflow_run(
                record["run_key"],
                nested=True,
                tags={
                    **tags,
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
                log_sweep_lineage(cfg, inputs, context="eval")
                ml.log_metrics({k: record[k] for k in metric_keys if k in record})
                ci_metrics: dict[str, float] = {}
                for metric in (*_DECOMPOSABLE_METRICS, *_DERIV_DECOMPOSABLE_METRICS):
                    bounds = record.get(f"{metric}_ci")
                    if bounds is not None:
                        ci_metrics[f"{metric}_ci_low"] = bounds[0]
                        ci_metrics[f"{metric}_ci_high"] = bounds[1]
                ml.log_metrics(ci_metrics)
                ml.set_tags(_train_link_tags(cfg.output_root / record["run_key"]))

        # best run and report artifacts
        best = max(results, key=lambda r: r["morph_score_f1"], default=None)
        if best is not None:
            ml.log_metrics({"best_morph_score_f1": best["morph_score_f1"]})
            ml.set_tags({"best_run_key": best["run_key"]})
        for path in report_paths:
            ml.log_artifact(path)


# ===============================
# Sweep orchestration
# ===============================


@dataclass
class EvaluateSummary:
    """Outcome of an evaluation sweep.

    Attributes:
        records: Per-run metric records, one per evaluated artifact.
        failed: Run keys whose evaluation raised.
        report_paths: The Markdown and JSON report paths, or None when no run
            produced a record.
    """

    records: list[dict[str, Any]] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    report_paths: tuple[Path, Path] | None = None


def _evaluate_worker(key: str, *, output_root: Path, alpha: float) -> dict[str, Any] | None:
    """Evaluate one run, reading the shared eval docs and morph sample globals.

    The grouped evaluation documents and the shared morph and derivational form
    samples live in module globals so a forked process-pool worker inherits them
    copy-on-write, rather than receiving them through pickled keyword arguments
    (which would copy the multi-million-form inputs once per task).

    Args:
        key: Run key (`<name>-<vocab>`).
        output_root: Directory holding per-run artifact subdirs.
        alpha: Renyi order.

    Returns:
        The metrics record, or None when skipped.
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
    cfg: SweepConfig,
    force: bool,
    morph_form_sample: int,
) -> tuple[list[list[str]], list[MorphemeSegmentation], list[MorphemeSegmentation]]:
    """Materialize the grouped eval docs and the shared morph form samples.

    The evaluation documents are kept grouped (one word list per document) so the
    corpus metrics can resample documents. The morph sample is a deterministic
    draw from the Sloleks-derived lexicon, identical across runs so per-form morph
    statistics align by index for the paired tests.

    Args:
        cfg: The resolved sweep configuration.
        force: Rebuild the gold lexicons even if present. The eval
            sample is drawn only when missing; `sample --force` redraws it.
        morph_form_sample: Size of the shared morph form sample.

    Returns:
        The grouped eval documents, the shared inflectional morph form
            sample, and the derivational form sample (empty when no derivational
            gold is configured).
    """
    # sample the held-out evaluation corpus
    eval_path = cfg.eval_sample_path
    if not eval_path.exists():
        logger.info("Sampling held-out evaluation corpus -> %s", eval_path)
        split = corpus_split_for(cfg)
        materialize_sample(cfg.corpus_root, cfg.eval_budget, eval_path, split, "eval" if split else None)
    elif cfg.corpus_train_share is not None:
        check_sample_side(eval_path, "eval")
    eval_docs = [iter_words(line) for line in iter_sample(eval_path)]
    logger.info("Evaluation sample: %d documents, %d word tokens", len(eval_docs), sum(len(d) for d in eval_docs))

    # build the inflectional gold and its sample
    infl_path = cfg.infl_lexicon_path
    if force or not infl_path.exists():
        logger.info("Deriving inflectional gold lexicon from %s", cfg.sloleks_path)
        save_lexicon(build_morph_lexicon(cfg.sloleks_path, min_stem_len=cfg.min_stem_len), infl_path)
    lexicon = load_lexicon(infl_path)
    heldout = None
    if cfg.heldout_lexeme_share is not None:
        # only held-out forms are scored: every other form is in the morph backends' table
        heldout = split_lexemes(cfg.inflection_path, cfg.heldout_lexeme_share).heldout_forms
        lexicon = without_forms(lexicon, frozenset(lexicon.by_form.keys() - heldout))
    morph_sample = sample_segmentations(lexicon, morph_form_sample, cfg.stats_seed)
    logger.info("Inflectional gold: %d forms; morph sample: %d forms", len(lexicon.by_form), len(morph_sample))

    # build the derivational gold sample
    deriv_sample: list[MorphemeSegmentation] = []
    if cfg.needs_derivational():
        deriv_path = cfg.deriv_lexicon_path
        if force or not deriv_path.exists():
            logger.info("Deriving derivational gold lexicon from %s", cfg.sloleks_relations_path)
            save_lexicon(build_derivational_lexicon(cfg.sloleks_relations_path), deriv_path)
        deriv_lexicon = load_lexicon(deriv_path)
        if heldout is not None:
            deriv_lexicon = without_forms(deriv_lexicon, frozenset(deriv_lexicon.by_form.keys() - heldout))
        deriv_sample = sample_segmentations(deriv_lexicon, morph_form_sample, cfg.stats_seed)
        logger.info(
            "Derivational gold: %d forms; deriv sample: %d forms", len(deriv_lexicon.by_form), len(deriv_sample)
        )

    return eval_docs, morph_sample, deriv_sample


def evaluate_sweep(
    cfg: SweepConfig,
    keys: list[str],
    force: bool = False,
    morph_form_sample: int | None = None,
    n_resamples: int | None = None,
    max_workers: int = 0,
) -> EvaluateSummary:
    """Evaluate trained tokenizer artifacts and write the comparison report.

    Materializes the shared evaluation inputs, scores every selected artifact in
    a process pool, attaches bootstrap CIs and paired significance, writes the
    Markdown and JSON reports, and logs the sweep to MLflow.

    Args:
        cfg: The resolved sweep configuration.
        keys: Run keys (`<name>-<vocab>`) with trained artifacts.
        force: Rebuild the gold lexicons; the evaluation sample is kept.
        morph_form_sample: Override for the config's morph
            form-sample size, or None to use the configured value.
        n_resamples: Override for the config's bootstrap
            resample count, or None to use the configured value.
        max_workers: Parallel evaluations. 0=auto, 1=serial, N=N workers.

    Returns:
        The metric records, failed keys, and report paths.
    """
    # cap the auto worker count for CPU-bound metrics so a many-core box is not saturated
    workers = resolve_workers(max_workers, len(keys), min(DEFAULT_MAX_WORKERS, cpu_default(len(keys))))
    configure_script_logging(parallel=workers > 1, console_level=logging.INFO)

    # resolve the overrides
    form_sample = morph_form_sample if morph_form_sample is not None else cfg.stats_morph_form_sample
    resamples = n_resamples if n_resamples is not None else cfg.stats_n_resamples

    # publish the eval inputs to globals before the pool forks, so workers inherit them copy-on-write
    global _EVAL_DOCS, _MORPH_SAMPLE, _DERIV_SAMPLE
    _EVAL_DOCS, _MORPH_SAMPLE, _DERIV_SAMPLE = prepare_eval_inputs(cfg, force=force, morph_form_sample=form_sample)

    def kwargs_for(_key: str) -> dict[str, Any]:
        """Return per-run kwargs for `_evaluate_worker`.

        Only small, picklable values go here; the eval docs and morph sample are
        shared via module globals (see `_evaluate_worker`).

        Args:
            _key: Run key (unused; kwargs are identical per run).

        Returns:
            Keyword arguments for `_evaluate_worker`.
        """
        return {"output_root": cfg.output_root, "alpha": cfg.renyi_alpha}

    # evaluate every run in parallel
    results, failures = run_parallel(
        _evaluate_worker,
        keys,
        max_workers=workers,
        desc="tokenizer-evaluate",
        pool="process",
        kwargs_for=kwargs_for,
        log_dir=stamped_log_dir("tokenizer-evaluate"),
    )

    # attach statistics, write the report and log it
    records = [r for r in results.values() if r is not None]
    summary = EvaluateSummary(records=records, failed=[k for k, _ in failures])
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
        log_evaluate_runs(records, cfg, summary.report_paths)

    logger.info("Done. Evaluated %d, failed %s.", len(records), summary.failed or "none")
    return summary
