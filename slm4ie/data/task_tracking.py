"""Post-hoc MLflow tracking for the task converters.

Shared by the three task-family converters (`to_spans`, `to_sentiment`,
`to_superglue`). After a converter writes its selected `<task>/<dataset>`
entries, it logs one MLflow run per dataset under the `slm4ie/data/tasks`
experiment, keyed by a content digest of the dataset's split files and
upserted (skip unless `force`). One run per dataset -- rather than one per
converter invocation -- gives each task dataset its own digest and lineage
handle, which the contamination audit and future eval runs declare as an input.

The tracking is deliberately light: per-split row counts and a combined label
distribution as metrics, the entry metadata as params, a
`label_distribution.json` artifact, and the dataset as a `produced` dataset
input. It reads the split files from disk, so it reflects the dataset as it
currently exists regardless of whether the converter rewrote or skipped it.
"""

from __future__ import annotations

import json
import logging
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from slm4ie.data.curate import config_hash, corpus_digest
from slm4ie.data.tasks import TaskEntry, TasksConfig, resolve_output_dir
from slm4ie.data.task_writer import iter_jsonl, outputs_for_splits
from slm4ie.utils import mlflow as ml

logger = logging.getLogger(__name__)

#: Experiment name for task-conversion builds, following the project's
#: `slm4ie/<workstream>/<dataset>` convention.
DEFAULT_EXPERIMENT = "slm4ie/data/tasks"


def _record_labels(record: Dict[str, Any]) -> List[str]:
    """Extract label values from one task record across task families.

    Handles a top-level `label` (sentiment / NLI / QA / ...) and per-span
    labels (`spans` from the NER family, as `{"label": ...}` dicts or
    `[start, end, label]` triples).

    Args:
        record: A parsed task JSONL record.

    Returns:
        The label strings found in the record (possibly empty).
    """
    labels: List[str] = []
    if "label" in record and not isinstance(record["label"], (dict, list)):
        labels.append(str(record["label"]))
    for span in record.get("spans") or []:
        if isinstance(span, dict) and "label" in span:
            labels.append(str(span["label"]))
        elif isinstance(span, (list, tuple)) and len(span) >= 3:
            labels.append(str(span[2]))
    return labels


def profile_task_dataset(output_dir: Path, splits: Dict[str, str]) -> Dict[str, Any]:
    """Profile a converted dataset's split files from disk.

    Args:
        output_dir: The `<task>/<dataset>` output directory.
        splits: Mapping `{split_name: filename}` from the entry.

    Returns:
        A dict with `split_rows` (per-split row counts for splits present on
        disk), `total_rows`, and `label_distribution` (combined label counts).
    """
    split_rows: Dict[str, int] = {}
    labels: Counter = Counter()
    for split, path in outputs_for_splits(output_dir, splits).items():
        if not path.exists():
            continue
        rows = 0
        for record in iter_jsonl(path):
            rows += 1
            labels.update(_record_labels(record))
        split_rows[split] = rows
    return {
        "split_rows": split_rows,
        "total_rows": sum(split_rows.values()),
        "label_distribution": dict(labels),
    }


def log_task_dataset(
    key: str,
    entry: TaskEntry,
    roots: Any,
    *,
    enabled: bool = True,
    experiment: str = DEFAULT_EXPERIMENT,
    tracking_uri: Optional[str] = None,
    force: bool = False,
    artifact_dir: Optional[Path] = None,
) -> Optional[str]:
    """Upsert a single MLflow run describing one converted task dataset.

    Args:
        key: Entry key `"<task>/<dataset>"`, used as the run name.
        entry: The parsed task entry.
        roots: Filesystem roots (carrying `tasks`).
        enabled: When False, does nothing and returns None.
        experiment: MLflow experiment name.
        tracking_uri: Tracking URI override; resolved when None.
        force: Replace an existing run for the same digest instead of skipping.
        artifact_dir: Directory to write `label_distribution.json` into; a
            temporary directory is used when None.

    Returns:
        The digest that was logged, the existing digest that was skipped, or
        None when tracking is disabled/unavailable or the dataset has no splits
        on disk.
    """
    if not enabled:
        return None

    output_dir = resolve_output_dir(entry, roots)
    if not any(p.exists() for p in outputs_for_splits(output_dir, entry.splits).values()):
        logger.warning("No split files for %s under %s; skipping tracking.", key, output_dir)
        return None
    if not ml.ensure_experiment(experiment, tracking_uri=tracking_uri):
        logger.warning("MLflow unavailable; skipping task tracking.")
        return None

    digest = corpus_digest(output_dir)
    existing = ml.find_run_by_tag(experiment, "corpus_digest", digest, tracking_uri=tracking_uri)
    if existing is not None and not force:
        logger.info("Task dataset %s (%s) already tracked; pass --force to re-log.", key, digest)
        return digest
    if existing is not None:
        ml.delete_run(existing, tracking_uri=tracking_uri)

    profile = profile_task_dataset(output_dir, entry.splits)

    artifact_root = artifact_dir if artifact_dir is not None else Path(tempfile.mkdtemp(prefix="task-track-"))
    artifact_root.mkdir(parents=True, exist_ok=True)
    profile_path = artifact_root / "label_distribution.json"
    with profile_path.open("w", encoding="utf-8") as fh:
        json.dump(profile, fh, ensure_ascii=False, indent=2)

    metrics: Dict[str, float] = {f"rows__{split}": rows for split, rows in profile["split_rows"].items()}
    metrics["rows_total"] = profile["total_rows"]
    for label, count in profile["label_distribution"].items():
        metrics[f"label__{label}"] = count

    tags = {
        "run_type": "data_pipeline",
        "pipeline": "tasks",
        "task": entry.task,
        "dataset": entry.dataset,
        "role": entry.role,
        "corpus_digest": digest,
        "config_hash": config_hash(
            {"task": entry.task, "dataset": entry.dataset, "keys": entry.source.keys, "splits": entry.splits}
        ),
    }
    commit = ml.git_commit()
    if commit is not None:
        tags["git_commit"] = commit

    with ml.mlflow_run(key, tags=tags):
        ml.log_params(
            {
                "task": entry.task,
                "dataset": entry.dataset,
                "role": entry.role,
                "language": entry.language,
                "license": entry.license,
                "converter": entry.converter,
                "source_kind": entry.source.kind,
                "source_keys": json.dumps(entry.source.keys, ensure_ascii=False),
                "labels": json.dumps(entry.labels, ensure_ascii=False),
                "splits": json.dumps(sorted(entry.splits), ensure_ascii=False),
            }
        )
        ml.log_metrics(metrics)
        ml.log_artifact(profile_path)
        ml.log_dataset_input(f"tasks/{key}", digest, str(output_dir), context="produced")

    logger.info("Logged task dataset %s (%s) to %s.", key, digest, experiment)
    return digest


def log_task_runs(
    tasks_config: TasksConfig,
    by_key: Dict[str, TaskEntry],
    keys: List[str],
    *,
    mlflow_enabled: Optional[bool] = None,
    force: bool = False,
) -> None:
    """Log one MLflow run per processed task dataset.

    Resolves whether tracking is enabled (config default, overridden by
    `mlflow_enabled`) and logs each key in turn, reading its outputs from disk.

    Args:
        tasks_config: The parsed task registry (carrying roots + mlflow config).
        by_key: Mapping from entry key to its parsed `TaskEntry`.
        keys: Entry keys to log (typically the successfully-processed ones).
        mlflow_enabled: Tri-state override for tracking; None defers to
            `tasks.yaml::mlflow.enabled`.
        force: Replace existing runs for the same digest instead of skipping.
    """
    cfg = tasks_config.mlflow or {}
    enabled = bool(cfg.get("enabled", False)) if mlflow_enabled is None else mlflow_enabled
    if not enabled:
        return
    experiment = cfg.get("experiment", DEFAULT_EXPERIMENT)
    tracking_uri = cfg.get("tracking_uri")
    for key in keys:
        entry = by_key.get(key)
        if entry is None:
            continue
        log_task_dataset(
            key,
            entry,
            tasks_config.roots,
            experiment=experiment,
            tracking_uri=tracking_uri,
            force=force,
        )
