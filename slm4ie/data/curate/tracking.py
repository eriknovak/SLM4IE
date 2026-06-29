"""Post-hoc MLflow tracking for the to_pretrain curation pipeline.

Reads the on-disk pipeline state after a full `--all` build and records a
single MLflow run per distinct corpus build under the `slm4ie/data/pretrain`
experiment. The run is keyed by a content digest of the final corpus
(`05_2_dedup/`) and upserted: a build whose digest already has a run is skipped
unless `force` is set. Like the rest of the pipeline, this is decoupled from
which stages actually ran on a given invocation -- it reports the corpus as it
currently exists on disk.

The per-stage funnel is read from the stage sentinels (`records_in` /
`records_out`), not by instrumenting datatrove live: scoped stages
(`convert`..`repetition`) carry per-dataset sentinels, corpus stages
(`exact_dedup`, `sentence_dedup`) a single corpus-level sentinel. Final
per-dataset / per-word statistics come from `06_statistics/aggregate.json`.

What is logged:

* A step-indexed funnel: `docs_remaining` (and per-source `docs_remaining__<key>`
  for scoped stages) and `drop_rate_stage`, indexed by stage position 0..7, so
  MLflow charts the survival curve across stages.
* Final-corpus scalars from `aggregate.json`: `final_docs`, `final_words`, and
  per-dataset `final_words__<key>` / `share_of_total_words__<key>` (the mixture
  shares that drive source-weighted sampling).
* The flattened resolved `pretrain.yaml` as params, and `aggregate.json`,
  `per_dataset/`, `_logs/`, plus a `funnel.json` as artifacts.
* The final corpus as a `produced` dataset input (name + digest) for lineage.

Per-stage word counts are not logged: only the final stats stage counts words,
so a per-stage word funnel does not exist upstream.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from slm4ie.data.curate import config_hash, corpus_digest
from slm4ie.data.curate.sentinel import read_sentinel
from slm4ie.data.curate.stages import SCOPED_STAGES, STAGE_DIRS, STAGE_NAMES
from slm4ie.utils import mlflow as ml

logger = logging.getLogger(__name__)

#: Experiment name for pretraining-corpus builds, following the project's
#: `slm4ie/<workstream>/<dataset>` convention.
DEFAULT_EXPERIMENT = "slm4ie/data/pretrain"

#: Folder (under the output dir) holding the final deduplicated corpus, used
#: for the build digest and lineage.
FINAL_CORPUS_DIR = STAGE_DIRS["sentence_dedup"]

#: Max characters kept for a flattened config param value (MLflow caps value
#: length; long lists/paths are truncated rather than dropped).
_MAX_PARAM_LEN = 500


def build_pretrain_funnel(output_dir: Path) -> List[Dict[str, Any]]:
    """Read the per-stage survival funnel from the pipeline sentinels.

    For each stage in `STAGE_NAMES`, reads `records_in` / `records_out` from its
    sentinel(s): scoped stages aggregate their per-dataset sentinels (keeping
    the per-dataset breakdown), corpus stages read the single corpus-level
    sentinel. Stages with no sentinel on disk are reported with null counts.

    Args:
        output_dir: The pretrain output root (holding `00_convert/` .. ).

    Returns:
        A list of per-stage dicts, ordered by stage position, each with
        `stage`, `index`, `scoped`, `records_in`, `records_out`, and (scoped
        only) `by_dataset` mapping each dataset to its in/out counts.
    """
    funnel: List[Dict[str, Any]] = []
    for index, stage in enumerate(STAGE_NAMES):
        stage_folder = output_dir / STAGE_DIRS[stage]
        entry: Dict[str, Any] = {
            "stage": stage,
            "index": index,
            "scoped": stage in SCOPED_STAGES,
            "records_in": None,
            "records_out": None,
            "by_dataset": {},
        }
        if stage in SCOPED_STAGES:
            total_in = total_out = 0
            found = False
            if stage_folder.is_dir():
                for sub in sorted(stage_folder.iterdir()):
                    if not sub.is_dir():
                        continue
                    sentinel = read_sentinel(sub)
                    if sentinel is None:
                        continue
                    found = True
                    entry["by_dataset"][sub.name] = {
                        "records_in": sentinel.records_in,
                        "records_out": sentinel.records_out,
                    }
                    total_in += sentinel.records_in
                    total_out += sentinel.records_out
            if found:
                entry["records_in"] = total_in
                entry["records_out"] = total_out
        else:
            sentinel = read_sentinel(stage_folder)
            if sentinel is not None:
                entry["records_in"] = sentinel.records_in
                entry["records_out"] = sentinel.records_out
        funnel.append(entry)
    return funnel


def read_aggregate_stats(output_dir: Path) -> Optional[Dict[str, Any]]:
    """Load the final-corpus aggregate stats, if present.

    Args:
        output_dir: The pretrain output root.

    Returns:
        The parsed `06_statistics/aggregate.json`, or None when it is absent.
    """
    path = output_dir / STAGE_DIRS["stats"] / "aggregate.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def flatten_config(config: Any, prefix: str = "") -> Dict[str, str]:
    """Flatten a nested config into dotted string params for MLflow.

    Nested dicts are flattened with dotted keys; lists and other leaves are
    serialized to a (length-capped) string. Used to record the resolved
    `pretrain.yaml` as run params.

    Args:
        config: The config value to flatten (dict, list, or scalar).
        prefix: Dotted key prefix accumulated during recursion.

    Returns:
        A flat mapping of dotted key to stringified value.
    """
    flat: Dict[str, str] = {}
    if isinstance(config, dict):
        for key, value in config.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            flat.update(flatten_config(value, child))
    elif prefix:
        value = config if isinstance(config, str) else json.dumps(config, ensure_ascii=False, default=str)
        flat[prefix] = value[:_MAX_PARAM_LEN]
    return flat


def _funnel_metrics(funnel: List[Dict[str, Any]]) -> List[tuple]:
    """Build (metric_name, value, step) tuples from the stage funnel.

    Args:
        funnel: The per-stage funnel from `build_pretrain_funnel`.

    Returns:
        A list of `(name, value, step)` tuples for step-indexed logging.
    """
    points: List[tuple] = []
    for entry in funnel:
        step = entry["index"]
        out = entry["records_out"]
        if out is None:
            continue
        points.append(("docs_remaining", float(out), step))
        in_ = entry["records_in"]
        if in_:
            points.append(("drop_rate_stage", (in_ - out) / in_, step))
        for key, counts in entry["by_dataset"].items():
            points.append((f"docs_remaining__{key}", float(counts["records_out"]), step))
    return points


def _aggregate_metrics(aggregate: Dict[str, Any]) -> Dict[str, float]:
    """Build final-corpus scalar metrics from the aggregate stats.

    Args:
        aggregate: The parsed `aggregate.json`.

    Returns:
        A flat mapping of metric name to value (corpus totals plus per-dataset
        word counts and word shares).
    """
    metrics: Dict[str, float] = {
        "final_docs": float(aggregate.get("total_docs", 0)),
        "final_words": float(aggregate.get("total_words", 0)),
    }
    for key, stats in (aggregate.get("by_dataset") or {}).items():
        metrics[f"final_words__{key}"] = float(stats.get("word_count", 0))
        metrics[f"share_of_total_words__{key}"] = float(stats.get("share_of_total_words", 0.0))
    return metrics


def log_pretrain_run(
    output_dir: Path,
    config: Dict[str, Any],
    *,
    enabled: bool = True,
    experiment: str = DEFAULT_EXPERIMENT,
    tracking_uri: Optional[str] = None,
    force: bool = False,
    artifact_dir: Optional[Path] = None,
) -> Optional[str]:
    """Upsert a single MLflow run describing the current pretrain build.

    Computes the digest of the final corpus and -- unless `force` is set --
    skips logging when a run already exists for that digest. Otherwise logs the
    step-indexed stage funnel, final-corpus scalars, flattened config params,
    stats/log artifacts, and the final corpus as a `produced` dataset input.

    Args:
        output_dir: The pretrain output root.
        config: The resolved `pretrain.yaml` as a dict (logged as params).
        enabled: When False, does nothing and returns None.
        experiment: MLflow experiment name.
        tracking_uri: Tracking URI override; resolved when None.
        force: Replace an existing run for the same digest instead of skipping.
        artifact_dir: Directory to write `funnel.json` into before logging; a
            temporary directory is used when None.

    Returns:
        The digest that was logged, the existing digest that was skipped, or
        None when tracking is disabled, unavailable, or the final corpus is
        absent.
    """
    if not enabled:
        return None

    final_corpus = output_dir / FINAL_CORPUS_DIR
    if not final_corpus.is_dir():
        logger.warning("Final corpus %s missing; skipping pretrain tracking.", final_corpus)
        return None
    if not ml.ensure_experiment(experiment, tracking_uri=tracking_uri):
        logger.warning("MLflow unavailable; skipping pretrain tracking.")
        return None

    digest = corpus_digest(final_corpus)
    existing = ml.find_run_by_tag(experiment, "corpus_digest", digest, tracking_uri=tracking_uri)
    if existing is not None and not force:
        logger.info("Pretrain build %s already tracked; pass --force to re-log.", digest)
        return digest
    if existing is not None:
        ml.delete_run(existing, tracking_uri=tracking_uri)

    funnel = build_pretrain_funnel(output_dir)
    aggregate = read_aggregate_stats(output_dir)

    artifact_root = artifact_dir if artifact_dir is not None else Path(tempfile.mkdtemp(prefix="pretrain-track-"))
    artifact_root.mkdir(parents=True, exist_ok=True)
    funnel_path = artifact_root / "funnel.json"
    with funnel_path.open("w", encoding="utf-8") as fh:
        json.dump(funnel, fh, ensure_ascii=False, indent=2)

    tags = {
        "run_type": "data_pipeline",
        "pipeline": "pretrain",
        "corpus_digest": digest,
        "config_hash": config_hash(config),
    }
    commit = ml.git_commit()
    if commit is not None:
        tags["git_commit"] = commit

    with ml.mlflow_run("pretrain-build", tags=tags):
        ml.log_params(flatten_config(config))
        ml.log_params({"output_dir": str(output_dir)})
        for name, value, step in _funnel_metrics(funnel):
            ml.log_metrics({name: value}, step=step)
        if aggregate is not None:
            ml.log_metrics(_aggregate_metrics(aggregate))
        ml.log_artifact(funnel_path)
        stats_dir = output_dir / STAGE_DIRS["stats"]
        if (stats_dir / "aggregate.json").exists():
            ml.log_artifact(stats_dir / "aggregate.json")
        ml.log_artifacts(stats_dir / "per_dataset", artifact_path="per_dataset")
        ml.log_artifacts(output_dir / "_logs", artifact_path="_logs")
        ml.log_dataset_input("pretrain/05_2_dedup", digest, str(final_corpus), context="produced")

    logger.info("Logged pretrain build %s to %s.", digest, experiment)
    return digest
