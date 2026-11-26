"""MLflow tracking for fine-tuning: one run per encoder, entry, learning rate and seed.

`track_finetune` opens one MLflow run, calls `finetune` inside it and logs what
the run used and what it measured. The params are the encoder, the entry and
every field of the run's settings, learning rate and seed among them. The
metrics are the test scores (`test_<name>`), the best epoch's val scores
(`val_<name>`), the best epoch, the parameter count, the peak GPU memory of
training and of inference (on a GPU only) and the fine-tuning wall time. The
entry's split folder is logged as a `training` dataset input, so the run links
to the task build it read.

Every run is tagged with `run_type`, `language`, `encoder`, `entry`, `role`,
`branch` and `base_commit`, the branch and base commit read from git. The
caller adds a `tier` tag, and any tag it passes overrides the default of the
same name. A run whose training raises is closed as failed in MLflow and the
error propagates.
"""

import dataclasses
import logging
from collections.abc import Mapping
from pathlib import PurePath

from slm4ie.data.tasks.config import TasksRoots, resolve_output_dir
from slm4ie.finetune.config import PlannedRun
from slm4ie.finetune.run import FinetuneResult, finetune
from slm4ie.utils import mlflow as ml
from slm4ie.utils.versioning import corpus_digest

# ===============================
# Logging
# ===============================

logger = logging.getLogger(__name__)

# ===============================
# Constants
# ===============================

RUN_TYPE = "finetune"
UNKNOWN_GIT = "unknown"

# ===============================
# Entry point
# ===============================


def track_finetune(
    planned: PlannedRun,
    roots: TasksRoots,
    experiment: str | None,
    tags: Mapping[str, str],
    *,
    device: str | None = None,
) -> FinetuneResult:
    """Fine-tune one planned run inside its own MLflow run and log its settings, scores and costs.

    Args:
        planned: The encoder, entry and settings of the run.
        roots: The task registry roots; the splits are read under `roots.tasks`.
        experiment: MLflow experiment name; the run is trained but not tracked when `None`.
        tags: Extra MLflow tags; `tier` is required when tracking, and any tag
            here overrides the default tag of the same name.
        device: The torch device to train on; `finetune` picks one when `None`.

    Returns:
        The result of `finetune`.

    Raises:
        ValueError: If tracking is on and `tags` names no `tier`, before any training.
    """
    if experiment is not None and not tags.get("tier"):
        raise ValueError(f"fine-tuning run '{run_name(planned)}' needs a 'tier' tag to log to '{experiment}'")

    # open the run with its tags and params
    entry = planned.entry
    tracking = experiment is not None and ml.ensure_experiment(experiment)
    run_tags = {
        "run_type": RUN_TYPE,
        "language": entry.language,
        "encoder": planned.encoder,
        "entry": entry.key,
        "role": entry.role,
        "branch": ml.git_branch() or UNKNOWN_GIT,
        "base_commit": ml.git_commit() or UNKNOWN_GIT,
        **tags,
    }
    with ml.mlflow_run(run_name(planned), enabled=tracking, tags=run_tags):
        ml.log_params(
            {
                "encoder": planned.encoder,
                "entry": entry.key,
                "role": entry.role,
                **dataclasses.asdict(planned.settings),
            },
            enabled=tracking,
        )
        if tracking:
            output_dir = resolve_output_dir(entry, roots)
            ml.log_dataset_input(f"tasks/{entry.key}", corpus_digest(output_dir), str(output_dir), context="training")

        # train, then log the scores and costs
        result = finetune(entry, planned.encoder, planned.settings, roots, device=device)
        ml.log_metrics(_result_metrics(result), enabled=tracking)
    logger.info("run=%s tracked=%s experiment=%s", run_name(planned), tracking, experiment)
    return result


def run_name(planned: PlannedRun) -> str:
    """Return the MLflow run name of a planned run.

    Args:
        planned: The encoder, entry and settings of the run.

    Returns:
        `<encoder>-<task>-<dataset>-lr<learning rate>-seed<seed>`, the encoder
        named by the last segment of its HuggingFace name or local path.
    """
    encoder = PurePath(planned.encoder.rstrip("/")).name or planned.encoder
    settings = planned.settings
    return f"{encoder}-{planned.entry.task}-{planned.entry.dataset}-lr{settings.learning_rate:g}-seed{settings.seed}"


# ===============================
# Helpers
# ===============================


def _result_metrics(result: FinetuneResult) -> dict[str, float]:
    """Return test and val scores, best epoch, parameters, wall time and GPU peaks as metrics."""
    metrics = {f"test_{name}": value for name, value in result.scores.items()}
    metrics.update({f"val_{name}": value for name, value in result.val_scores.items()})
    metrics.update(
        best_epoch=float(result.best_epoch),
        parameters=float(result.parameters),
        train_seconds=result.train_seconds,
    )
    # a CPU run measures no GPU memory, so it logs none
    if result.train_peak_gpu_memory_bytes is not None:
        metrics["train_peak_gpu_memory_bytes"] = float(result.train_peak_gpu_memory_bytes)
    if result.inference_peak_gpu_memory_bytes is not None:
        metrics["inference_peak_gpu_memory_bytes"] = float(result.inference_peak_gpu_memory_bytes)
    return metrics
