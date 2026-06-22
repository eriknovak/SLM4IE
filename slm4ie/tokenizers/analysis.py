"""Evaluate trained tokenizers and build the comparison report.

`evaluate_artifact` loads one trained tokenizer, runs the six metrics over the
grouped held-out documents and the shared Sloleks-derived morph sample, and
persists per-unit sufficient statistics to an `eval_units.npz` sidecar.
`augment_with_statistics` reads those sidecars to attach bootstrap confidence
intervals and paired significance tests (per vocab) to the five decomposable
metrics. `build_report` aggregates the per-run metrics into a Markdown table and
a JSON payload, and `log_results_to_mlflow` records the sweep as a parent run
with one nested child per tokenizer x vocab-size run.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np

import slm4ie.tokenizers.backends  # noqa: F401  (registers backends on import)
from slm4ie.tokenizers import stats as st
from slm4ie.tokenizers.metrics import (
    METRIC_DIRECTIONS,
    corpus_doc_stats,
    morph_consistency_over,
    morph_form_stats,
    renyi_efficiency,
)
from slm4ie.tokenizers.morphology import MorphemeSegmentation
from slm4ie.tokenizers.registry import get_tokenizer
from slm4ie.tokenizers.train import MLFLOW_LINK_FILENAME, parse_run_key
from slm4ie.utils import mlflow as ml
from slm4ie.utils.config import TokenizerSweepConfig

logger = logging.getLogger(__name__)

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

#: Decomposable metrics that get bootstrap CIs and paired significance tests.
#: Each reduces to per-unit sufficient statistics that sum across the resampling
#: unit (documents for corpus metrics, forms for morph metrics).
_DECOMPOSABLE_METRICS = [
    "fertility",
    "tokens_per_byte",
    "chars_per_token",
    "morph_score_f1",
    "morph_edit_distance",
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
    (artifact_dir / "metrics.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(
        artifact_dir / EVAL_UNITS_FILENAME,
        doc_tokens=doc_tokens,
        doc_words=doc_words,
        doc_chars=doc_chars,
        doc_bytes=doc_bytes,
        form_tp=form_tp,
        form_predicted=form_pred,
        form_gold=form_gold,
        form_edit=form_edit,
        form_valid=form_valid,
    )
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
    valid_pos: np.ndarray,
    corpus_idx: np.ndarray,
    morph_idx: np.ndarray,
) -> Tuple[Tuple[np.ndarray, ...], st.CombineFn, np.ndarray]:
    """Return the per-unit components, combine, and resample indices for a metric.

    Args:
        metric (str): A decomposable metric name.
        units (Dict[str, np.ndarray]): The run's per-unit arrays.
        valid_pos (np.ndarray): Form positions valid across every compared run.
        corpus_idx (np.ndarray): Shared document-resample indices.
        morph_idx (np.ndarray): Shared form-resample indices over `valid_pos`.

    Returns:
        Tuple[Tuple[np.ndarray, ...], st.CombineFn, np.ndarray]: The components,
            the combine callable, and the resample-index matrix to use.

    Raises:
        ValueError: If `metric` is not decomposable.
    """
    if metric == "fertility":
        return (units["doc_tokens"], units["doc_words"]), _safe_div, corpus_idx
    if metric == "tokens_per_byte":
        return (units["doc_tokens"], units["doc_bytes"]), _safe_div, corpus_idx
    if metric == "chars_per_token":
        return (units["doc_chars"], units["doc_tokens"]), _safe_div, corpus_idx
    if metric == "morph_score_f1":
        tp = units["form_tp"][valid_pos]
        predicted = units["form_predicted"][valid_pos]
        gold = units["form_gold"][valid_pos]
        return (tp, predicted, gold), _f1_combine, morph_idx
    if metric == "morph_edit_distance":
        edit = units["form_edit"][valid_pos]
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

    For each vocab size, all tokenizers share one set of resample indices
    (documents for corpus metrics, forms for morph metrics) so that paired
    differences are computed on the same resamples. The morph metrics are
    bootstrapped over the forms valid across every compared tokenizer. Each
    record gains `<metric>_ci` and `<metric>_std` for the five decomposable
    metrics (whose point estimate is also recomputed from the per-unit arrays so
    it matches the CI), while `renyi_efficiency` and `morph_consistency` are left
    as bare point estimates.

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

        n_docs = len(units[keys[0]]["doc_tokens"])
        corpus_idx = st.make_resample_indices(n_docs, n_resamples, seed * 1_000_003 + vocab * 2)

        valid_stack = np.vstack([units[key]["form_valid"].astype(bool) for key in keys])
        valid_pos = np.nonzero(valid_stack.all(axis=0))[0]
        morph_idx = st.make_resample_indices(len(valid_pos), n_resamples, seed * 1_000_003 + vocab * 2 + 1)

        vocab_block: Dict[str, Any] = {}
        for metric in _DECOMPOSABLE_METRICS:
            points: Dict[str, float] = {}
            dists: Dict[str, np.ndarray] = {}
            for key in keys:
                components, combine, idx = _components_for(metric, units[key], valid_pos, corpus_idx, morph_idx)
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
        "units": {"corpus": "documents", "morph": "forms"},
        "point_only": ["renyi_efficiency", "morph_consistency"],
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
    headers = ["tokenizer", "vocab"] + [
        f"{col} {_DIRECTION_ARROW.get(METRIC_DIRECTIONS.get(col, ''), '')}".strip() for col in _REPORT_COLUMNS
    ]
    lines = [
        "# Tokenizer comparison (Slovenian)",
        "",
        "Morphological metrics use a Sloleks-derived silver-gold segmentation "
        "(inflectional only); treat them as relative comparators, not absolute "
        "morphological accuracy.",
        "",
        "Decomposable metrics carry a 95% bootstrap CI in brackets (documents "
        "resampled for corpus metrics, forms for morph metrics). "
        "`renyi_efficiency` and `morph_consistency` are point estimates only.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for record in ordered:
        row = [record["tokenizer"], str(record["vocab_size"])]
        row += [_format_metric_cell(record, col) for col in _REPORT_COLUMNS]
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
    metric_keys = [*_REPORT_COLUMNS, "morph_score_precision", "morph_score_recall", "vocab_used"]
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
                for metric in _DECOMPOSABLE_METRICS:
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
