"""Evaluate trained tokenizers and build the comparison report.

`evaluate_artifact` loads one trained tokenizer and runs the six metrics over
the held-out evaluation words and the Sloleks-derived gold lexicon.
`build_report` aggregates the per-run metrics into a Markdown table and a JSON
payload, and `log_results_to_mlflow` records the sweep as a parent run with one
nested child per tokenizer x vocab-size run.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import slm4ie.tokenizers.backends  # noqa: F401  (registers backends on import)
from slm4ie.tokenizers.metrics import (
    METRIC_DIRECTIONS,
    compression_stats,
    corpus_token_stats,
    fertility,
    morph_consistency_score,
    morph_edit_distance,
    morph_score,
    renyi_efficiency,
)
from slm4ie.tokenizers.morphology import MorphLexicon
from slm4ie.tokenizers.registry import get_tokenizer
from slm4ie.tokenizers.train import parse_run_key
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

#: Arrow shown next to each metric header indicating the better direction.
_DIRECTION_ARROW = {"higher": "↑", "lower": "↓"}


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
    lexicon: MorphLexicon,
    eval_words: List[str],
    alpha: float = 2.5,
    max_forms: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Run all six metrics for one trained tokenizer run.

    Args:
        key (str): Run key (`<name>-<vocab>`).
        output_root (Path): Directory holding per-run artifact subdirs.
        lexicon (MorphLexicon): The gold morpheme lexicon.
        eval_words (List[str]): Held-out evaluation word tokens.
        alpha (float): Renyi order.
        max_forms (Optional[int]): Cap on lexicon forms for the morph metrics.

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

    stats = corpus_token_stats(tokenizer, eval_words)
    compression = compression_stats(int(stats["n_tokens"]), int(stats["n_chars"]), int(stats["n_bytes"]))
    boundary = morph_score(tokenizer, lexicon, max_forms=max_forms)

    record: Dict[str, Any] = {
        "run_key": key,
        "tokenizer": name,
        "vocab_size": vocab_size,
        "vocab_used": len(tokenizer.vocab),
        "fertility": fertility(int(stats["n_tokens"]), int(stats["n_words"])),
        "tokens_per_byte": compression["tokens_per_byte"],
        "chars_per_token": compression["chars_per_token"],
        "ctc_total": compression["ctc_total"],
        "renyi_efficiency": renyi_efficiency(stats["freqs"], alpha),
        "morph_score_f1": boundary["f1"],
        "morph_score_precision": boundary["precision"],
        "morph_score_recall": boundary["recall"],
        "morph_coverage": boundary["coverage"],
        "morph_edit_distance": morph_edit_distance(tokenizer, lexicon, max_forms=max_forms),
        "morph_consistency": morph_consistency_score(tokenizer, lexicon, max_forms=max_forms),
    }
    (artifact_dir / "metrics.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


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


def build_report(results: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    """Build the Markdown comparison table and JSON payload.

    Args:
        results (List[Dict[str, Any]]): Per-run metric records.

    Returns:
        Tuple[str, Dict[str, Any]]: The Markdown report and a JSON payload
            carrying the results and metric directions.
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
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for record in ordered:
        row = [record["tokenizer"], str(record["vocab_size"])]
        row += [_format_cell(record.get(col, "")) for col in _REPORT_COLUMNS]
        lines.append("| " + " | ".join(row) + " |")

    payload = {"results": ordered, "directions": METRIC_DIRECTIONS}
    return "\n".join(lines) + "\n", payload


def write_report(results: List[Dict[str, Any]], report_dir: Path) -> Tuple[Path, Path]:
    """Write the Markdown and JSON reports under `report_dir`.

    Args:
        results (List[Dict[str, Any]]): Per-run metric records.
        report_dir (Path): Destination directory, created if missing.

    Returns:
        Tuple[Path, Path]: The Markdown and JSON report paths.
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    markdown, payload = build_report(results)
    md_path = report_dir / "report.md"
    json_path = report_dir / "report.json"
    md_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return md_path, json_path


def log_results_to_mlflow(
    results: List[Dict[str, Any]],
    cfg: TokenizerSweepConfig,
    report_paths: Tuple[Path, Path],
) -> None:
    """Log the sweep to MLflow as a parent run with nested per-run children.

    Each child logs its parameters and metrics; the parent records the best
    MorphScore F1 and the report artifacts. A no-op when tracking is disabled.

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
    with ml.mlflow_run("sweep-tokenizers", tags={"run_type": "sweep"}):
        for record in results:
            with ml.mlflow_run(
                record["run_key"],
                nested=True,
                tags={
                    "model_type": record["tokenizer"],
                    "model_version": str(record["vocab_size"]),
                    "run_type": "sweep",
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

        best = max(results, key=lambda r: r["morph_score_f1"], default=None)
        if best is not None:
            ml.log_metrics({"best_morph_score_f1": best["morph_score_f1"]})
            ml.set_tags({"best_run_key": best["run_key"]})
        for path in report_paths:
            ml.log_artifact(path)
