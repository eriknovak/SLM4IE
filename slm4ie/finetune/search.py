"""Hyperparameter search for fine-tuning one encoder on one task registry entry.

`search_finetune` runs one Optuna study per encoder and entry. Each trial draws
one setting from the config's search space, fine-tunes with the family's first
seed and returns the best epoch's val score on the family's checkpoint metric;
test is never scored inside the study. A `grid` search runs every combination
of the space (an ablation table); a `tpe` search draws a fixed number of trials
with the Tree-structured Parzen Estimator and, with pruning on, stops a trial
whose val score after an epoch falls below the median of earlier trials at that
epoch. The best trial's setting is then fine-tuned once per seed of the family
and scored on test.

The study is stored where `storage` points, so an interrupted search resumes:
finished, pruned and failed trials count toward the trial budget and are not
run again. A trial that runs out of GPU memory fails alone and the study goes
on; any other error stops it.

With an experiment, the study is one MLflow parent run (`run_type=search`) with
one child run per trial and one per final seed. A trial's child run logs its
settings, its val scores per epoch and a `trial_state` tag of `COMPLETE`,
`PRUNED` or `FAILED`. A final seed's child run is the run `track_finetune`
logs, tagged `stage=final`. The parent logs the space, the best trial's
settings and val score, its child run id, and the mean and standard deviation
of the final seeds' test scores.
"""

import dataclasses
import gc
import hashlib
import json
import logging
import statistics
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePath
from typing import Any

import optuna
import torch

from slm4ie.data.tasks.config import TaskEntry, TasksRoots
from slm4ie.finetune.config import (
    POSITIVE_INT_KEYS,
    FamilySettings,
    FinetuneSettings,
    PlannedRun,
    SearchConfig,
    SearchRange,
)
from slm4ie.finetune.heads import get_head
from slm4ie.finetune.run import FinetuneResult, finetune
from slm4ie.finetune.tracking import track_finetune
from slm4ie.utils import mlflow as ml

# ===============================
# Logging
# ===============================

logger = logging.getLogger(__name__)

# ===============================
# Constants
# ===============================

RUN_TYPE = "search"
TRIAL_RUN_TYPE = "trial"
FINAL_STAGE = "final"
COMPLETE, PRUNED, FAILED = "COMPLETE", "PRUNED", "FAILED"
RUN_ID_ATTR = "mlflow_run_id"
STUDY_HASH_LENGTH = 6

# a trial in one of these states counts toward the budget on a resumed study
FINISHED_STATES = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED, optuna.trial.TrialState.FAIL)

# ===============================
# Result
# ===============================


@dataclass(frozen=True)
class SearchResult:
    """The outcome of one search.

    Attributes:
        study_name: The Optuna study's name.
        best_settings: The best trial's settings, with the trials' seed.
        best_value: The best trial's val score on the family's checkpoint metric.
        finals: The test results of the best settings, one per seed of the family.
    """

    study_name: str
    best_settings: FinetuneSettings
    best_value: float
    finals: list[FinetuneResult]


# ===============================
# Entry point
# ===============================


def search_finetune(
    encoder: str,
    entry: TaskEntry,
    family: FamilySettings,
    search: SearchConfig,
    roots: TasksRoots,
    *,
    storage: str | None,
    experiment: str | None,
    tags: Mapping[str, str],
    device: str | None = None,
) -> SearchResult:
    """Search the settings of one encoder on one entry, then score the best one on test over the seeds.

    Args:
        encoder: The encoder, by HuggingFace name or local path.
        entry: The task registry entry; its role must keep a train split.
        family: The settings of the entry's family; a setting outside the
            search space keeps its value here, the learning rate its first one.
        search: The search: sampler, trial budget, sampler seed, pruning and space.
        roots: The task registry roots; the splits are read under `roots.tasks`.
        storage: Optuna storage URL, such as `sqlite:////abs/path/optuna.db`; the
            study lives in memory and cannot resume when `None`.
        experiment: MLflow experiment name; the search runs untracked when `None`.
        tags: Extra MLflow tags for every run; `tier` is required when tracking.
        device: The torch device to train on; `finetune` picks one when `None`.

    Returns:
        The best settings, their val score and their test results per seed.

    Raises:
        ValueError: If tracking is on and `tags` names no `tier`, before any
            training, or if no trial of the study completed.
    """
    name = study_name(encoder, entry, family, search)
    if experiment is not None and not tags.get("tier"):
        raise ValueError(f"search '{name}' needs a 'tier' tag to log to '{experiment}'")

    # load or create the study
    metric = get_head(entry.task).primary_metric
    # the first learning rate and seed of the family's grid
    base = family.grid()[0]
    study = optuna.create_study(
        study_name=name,
        storage=storage,
        sampler=_sampler(search, _stored_trials(name, storage)),
        pruner=optuna.pruners.MedianPruner() if search.pruning else optuna.pruners.NopPruner(),
        direction="maximize",
        load_if_exists=True,
    )
    tracking = experiment is not None and ml.ensure_experiment(experiment)
    run_tags = {
        "language": entry.language,
        "encoder": encoder,
        "entry": entry.key,
        "role": entry.role,
        "branch": ml.git_branch() or "unknown",
        "base_commit": ml.git_commit() or "unknown",
        **tags,
    }

    with ml.mlflow_run(name, enabled=tracking, tags={**run_tags, "run_type": RUN_TYPE}):
        ml.log_params(
            {
                "encoder": encoder,
                "entry": entry.key,
                "sampler": search.sampler,
                "trials": search.trials,
                "sampler_seed": search.seed,
                "pruning": search.pruning,
                "space": json.dumps(_space_params(search)),
            },
            enabled=tracking,
        )

        # run the trials the budget has left
        remaining = search.trials - sum(trial.state in FINISHED_STATES for trial in study.trials)
        if remaining > 0:
            objective = _objective(encoder, entry, base, search, roots, metric, tracking, run_tags, device)
            study.optimize(objective, n_trials=remaining, catch=(torch.OutOfMemoryError,))
        if not any(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials):
            raise ValueError(f"search '{name}' has no completed trial; see the failed trials in the log")
        best = study.best_trial
        best_settings = dataclasses.replace(base, **best.params)
        ml.log_params({f"best_{key}": value for key, value in best.params.items()}, enabled=tracking)
        ml.log_metrics({f"best_val_{metric}": best.value, "best_trial": float(best.number)}, enabled=tracking)
        if RUN_ID_ATTR in best.user_attrs:
            ml.set_tags({"best_child_run_id": best.user_attrs[RUN_ID_ATTR]}, enabled=tracking)

        # fine-tune the best settings once per seed and score test
        finals = [
            track_finetune(
                PlannedRun(encoder=encoder, entry=entry, settings=dataclasses.replace(best_settings, seed=seed)),
                roots,
                experiment,
                {**tags, "stage": FINAL_STAGE, "study": name},
                device=device,
                nested=True,
            )
            for seed in family.seeds
        ]
        ml.log_metrics(_final_metrics(finals, metric), enabled=tracking)
    logger.info("study=%s best_trial=%d best_val_%s=%.4f params=%s", name, best.number, metric, best.value, best.params)
    return SearchResult(study_name=name, best_settings=best_settings, best_value=best.value, finals=finals)


def study_name(encoder: str, entry: TaskEntry, family: FamilySettings, search: SearchConfig) -> str:
    """Return the Optuna study and MLflow parent run name of a search.

    The name ends in a hash of what makes two searches different: the full
    encoder name or path, the sampler, its seed, pruning, the space and the
    settings outside it. A changed config therefore starts a new study in the
    same storage, while a larger trial budget resumes the same one.

    Args:
        encoder: The encoder, by HuggingFace name or local path.
        entry: The task registry entry.
        family: The settings of the entry's family.
        search: The search.

    Returns:
        `<encoder>-<task>-<dataset>-<sampler>-<hash>`, the encoder named by the
        last segment of its HuggingFace name or local path, the hash 6 hex digits.
    """
    identity = {
        "encoder": encoder,
        "sampler": search.sampler,
        "seed": search.seed,
        "pruning": search.pruning,
        "space": _space_params(search),
        "base": dataclasses.asdict(family.grid()[0]),
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:STUDY_HASH_LENGTH]
    short = PurePath(encoder.rstrip("/")).name or encoder
    return f"{short}-{entry.task}-{entry.dataset}-{search.sampler}-{digest}"


# ===============================
# Helpers
# ===============================


def _stored_trials(name: str, storage: str | None) -> int:
    """Return how many trials a stored study of this name already holds; 0 when there is none."""
    if storage is None:
        return 0
    try:
        return len(optuna.load_study(study_name=name, storage=storage).trials)
    except KeyError:
        return 0


def _sampler(search: SearchConfig, stored_trials: int) -> optuna.samplers.BaseSampler:
    """Return the grid sampler over the space's lists, or a TPE sampler seeded past the stored trials."""
    if search.sampler == "grid":
        return optuna.samplers.GridSampler(
            {key: list(values) for key, values in search.space.items()}, seed=search.seed
        )
    # a resumed search seeded as before would redraw its first trials' settings
    return optuna.samplers.TPESampler(seed=search.seed + stored_trials)


def _suggest(trial: optuna.Trial, key: str, values: tuple[Any, ...] | SearchRange) -> Any:
    """Draw one setting: a choice among listed values, or a value in a range."""
    if not isinstance(values, SearchRange):
        return trial.suggest_categorical(key, list(values))
    if key in POSITIVE_INT_KEYS:
        return trial.suggest_int(key, values.low, values.high, log=values.log)
    return trial.suggest_float(key, float(values.low), float(values.high), log=values.log)


def _objective(
    encoder: str,
    entry: TaskEntry,
    base: FinetuneSettings,
    search: SearchConfig,
    roots: TasksRoots,
    metric: str,
    tracking: bool,
    run_tags: Mapping[str, str],
    device: str | None,
) -> Any:
    """Return the objective that fine-tunes one trial's settings and returns its val score."""

    def objective(trial: optuna.Trial) -> float:
        settings = dataclasses.replace(base, **{key: _suggest(trial, key, v) for key, v in search.space.items()})
        trial_tags = {**run_tags, "run_type": TRIAL_RUN_TYPE, "trial": str(trial.number)}

        # report each epoch's val score, so the pruner can stop a weak trial
        def report(epoch: int, scores: Mapping[str, float]) -> None:
            ml.log_metrics({f"val_{name}": value for name, value in scores.items()}, step=epoch, enabled=tracking)
            trial.report(scores[metric], epoch)
            if trial.should_prune():
                ml.log_metrics({"pruned_at_epoch": float(epoch)}, enabled=tracking)
                raise optuna.TrialPruned()

        # a pruned trial closes its run as finished, so only a real error reads as failed
        pruned = False
        try:
            with ml.mlflow_run(
                f"{trial.study.study_name}-trial{trial.number}", enabled=tracking, nested=True, tags=trial_tags
            ) as run:
                if run is not None:
                    trial.set_user_attr(RUN_ID_ATTR, run.info.run_id)
                ml.log_params(dataclasses.asdict(settings), enabled=tracking)
                try:
                    result = finetune(entry, encoder, settings, roots, device=device, score_test=False, on_epoch=report)
                except optuna.TrialPruned:
                    ml.set_tags({"trial_state": PRUNED}, enabled=tracking)
                    pruned = True
                except Exception:
                    ml.set_tags({"trial_state": FAILED}, enabled=tracking)
                    raise
                else:
                    ml.set_tags({"trial_state": COMPLETE}, enabled=tracking)
                    ml.log_metrics(
                        {"best_epoch": float(result.best_epoch), "train_seconds": result.train_seconds},
                        enabled=tracking,
                    )
        finally:
            _free_gpu_memory()
        if pruned:
            raise optuna.TrialPruned()
        return result.val_scores[metric]

    return objective


def _space_params(search: SearchConfig) -> dict[str, Any]:
    """Return the search space as JSON-ready values, a range as its low, high and log."""
    return {
        key: dataclasses.asdict(values) if isinstance(values, SearchRange) else list(values)
        for key, values in search.space.items()
    }


def _final_metrics(finals: list[FinetuneResult], metric: str) -> dict[str, float]:
    """Return the mean and the standard deviation over seeds of each test score of the final runs."""
    metrics = {}
    for name in finals[0].scores:
        values = [result.scores[name] for result in finals]
        metrics[f"final_test_{name}_mean"] = statistics.fmean(values)
        metrics[f"final_test_{name}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
    logger.info("final_test_%s_mean=%.4f seeds=%d", metric, metrics[f"final_test_{metric}_mean"], len(finals))
    return metrics


def _free_gpu_memory() -> None:
    """Release the GPU memory a finished trial left cached, so the next trial starts from empty."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
