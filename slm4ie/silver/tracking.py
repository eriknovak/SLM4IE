"""Log one round's calls to MLflow: one run per screened model and one per judge.

A run holds what its model's stored calls cost and how many failed, read from
the model's record file, with the fixed sampling settings, the hashes of the
prompt templates and the hashes of the input files (passages, briefs, labels)
as parameters. Each run carries a `run_key` tag `<role>/<name>/<tier>/R<n>`,
such as `maker/claude-haiku-5-5/tracked/R1`, and logging the same key again
replaces the earlier run, so a resumed round still keeps one run per model and
a scratch run never replaces a tracked one. The run name is `<name>-R<n>`, with
`-scratch` added outside the tracked tier.
"""

import hashlib
import json
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from slm4ie.silver.make import model_settings, prompt_hash
from slm4ie.silver.rate import JUDGE
from slm4ie.utils import mlflow as ml

# ===============================
# Logging
# ===============================

logger = logging.getLogger(__name__)

# ===============================
# Defaults
# ===============================

MAKER = "maker"
# the tag that names the model of a run, per role
NAME_TAGS = {MAKER: "variant", JUDGE: "judge"}
RUN_KEY_TAG = "run_key"

# the tier of a run that counts; every other tier is kept apart as scratch
TRACKED_TIER = "tracked"
SCRATCH_SUFFIX = "scratch"

# hex characters of the sha256 kept as an input file's hash
FILE_HASH_CHARS = 12

# ===============================
# Metrics and Parameters
# ===============================


def call_metrics(rows: Iterable[Mapping[str, Any]], role: str) -> dict[str, float]:
    """Sum the calls, failures, tokens and cost of one model's stored rows.

    Args:
        rows: The model's stored records (`make_examples`) or ratings (`rate_outputs`).
        role: `maker` or `judge`; a judge's metrics add the ratings dropped as off the schema.

    Returns:
        `calls`, `failed_calls`, `input_tokens`, `output_tokens`, `cost_usd`, and the cost per call
        as `cost_per_example_usd` for a maker or `cost_per_rating_usd` for a judge.
    """
    rows = list(rows)
    calls = len(rows)
    cost = sum(row["cost_usd"] for row in rows)
    metrics = {
        "calls": float(calls),
        "failed_calls": float(sum(row["error"] is not None for row in rows)),
        "input_tokens": float(sum(row["input_tokens"] for row in rows)),
        "output_tokens": float(sum(row["output_tokens"] for row in rows)),
        "cost_usd": cost,
    }
    if role == MAKER:
        metrics["cost_per_example_usd"] = cost / calls if calls else 0.0
        return metrics
    metrics["cost_per_rating_usd"] = cost / calls if calls else 0.0
    metrics["dropped_ratings"] = float(sum(row["error"] is None and row["rating"] is None for row in rows))
    return metrics


def template_hashes(prompts: Mapping[str, Mapping[str, str]], names: Iterable[str]) -> dict[str, str]:
    """Hash the unfilled system and user templates of the named prompts.

    Args:
        prompts: Prompt name to its `system` and `user` template (`prompts-jobs.yaml`).
        names: The prompts a role uses, such as `annotate` and `write`.

    Returns:
        `prompt_hash_<name>` to the hash of that prompt's templates.
    """
    return {
        f"prompt_hash_{name}": prompt_hash(
            [{"role": "system", "content": prompts[name]["system"]}, {"role": "user", "content": prompts[name]["user"]}]
        )
        for name in names
    }


def file_hashes(files: Mapping[str, Path]) -> dict[str, str]:
    """Hash the bytes of the input files a run read, so a run names the exact inputs it saw.

    Args:
        files: Short name to file path, such as `passages` to the round's passages file.

    Returns:
        `<name>_sha256` to the first 12 hex characters of the file's sha256.

    Raises:
        FileNotFoundError: If a file does not exist.
    """
    return {
        f"{name}_sha256": hashlib.sha256(path.read_bytes()).hexdigest()[:FILE_HASH_CHARS]
        for name, path in files.items()
    }


def run_key(role: str, name: str, tags: Mapping[str, str]) -> str:
    """Build the key that identifies one model's run in one round and tier.

    Args:
        role: `maker` or `judge`.
        name: The variant or judge name.
        tags: Tags of the run, with `tier` and `round` (such as `R1`).

    Returns:
        `<role>/<name>/<tier>/<round>`, such as `maker/claude-haiku-5-5/tracked/R1`.

    Raises:
        KeyError: If `tags` lacks `tier` or `round`.
    """
    return f"{role}/{name}/{tags['tier']}/{tags['round']}"


# ===============================
# Logging Runs
# ===============================


def log_round_runs(
    experiment: str,
    role: str,
    specs: Mapping[str, Mapping[str, Any]],
    records_dir: Path,
    sampling: Mapping[str, Any],
    prompt_hashes: Mapping[str, str],
    input_hashes: Mapping[str, str],
    tags: Mapping[str, str],
) -> int:
    """Log one MLflow run per model from its record file, replacing a run of the same key (`run_key`).

    A model with no stored row yet logs no run.

    Args:
        experiment: The MLflow experiment name.
        role: `maker` for a screened model, `judge` for a judge model.
        specs: Model name to its spec (`screen-models.yaml`), with `model` and `family`.
        records_dir: Folder of the per-model files `<name>.jsonl`.
        sampling: The `sampling` block, logged per model as `model_settings` applies it.
        prompt_hashes: Parameter name to prompt hash (`template_hashes`).
        input_hashes: Parameter name to input file hash (`file_hashes`).
        tags: Tags of every run, among them `tier` and `round`.

    Returns:
        The number of runs logged; 0 when MLflow is unavailable.

    Raises:
        ValueError: If `role` is neither `maker` nor `judge`.
        KeyError: If `tags` lacks `tier` or `round`.
    """
    if role not in NAME_TAGS:
        raise ValueError(f"unknown role {role!r}; known: {sorted(NAME_TAGS)}")
    if not ml.ensure_experiment(experiment):
        return 0

    logged = 0
    for name, spec in specs.items():
        path = records_dir / f"{name}.jsonl"
        rows = _read_rows(path)
        if not rows:
            logger.warning("no stored calls for %s=%s in %s; no run logged", NAME_TAGS[role], name, records_dir)
            continue

        # replace an earlier run of this model in this round and tier
        key = run_key(role, name, tags)
        earlier = ml.find_run_by_tag(experiment, RUN_KEY_TAG, key)
        if earlier is not None:
            ml.delete_run(earlier)

        # log the settings, the sums and the record file
        params = {
            "model": spec["model"],
            "family": spec["family"],
            **{key: "none" if value is None else value for key, value in model_settings(sampling, spec).items()},
            **prompt_hashes,
            **input_hashes,
        }
        run_tags = {**tags, NAME_TAGS[role]: name, "role": role, RUN_KEY_TAG: key}
        suffix = "" if tags["tier"] == TRACKED_TIER else f"-{SCRATCH_SUFFIX}"
        with ml.mlflow_run(f"{name}-{tags['round']}{suffix}", tags=run_tags):
            ml.log_params(params)
            ml.log_metrics(call_metrics(rows, role))
            ml.log_artifact(path)
        logged += 1
    return logged


# ===============================
# Helpers
# ===============================


def _read_rows(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file; no rows when the file does not exist."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
