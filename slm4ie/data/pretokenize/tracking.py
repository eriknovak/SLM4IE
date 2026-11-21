"""MLflow lineage of the pretokenized outputs.

Each finished output gets one MLflow run under the `slm4ie/data/pretokenize`
experiment, keyed by a digest of its manifest without the last run's timings
and upserted (skipped when that digest is already tracked, replaced with
`force`). The run records the settings
as params, the per-split counts and the last run's throughput as metrics, the
manifest as an artifact, and two dataset inputs: the corpus stage it read and
the output it produced. A training project then declares the output's digest
as its input.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from slm4ie.data.pretokenize.config import PretokenizeEntry
from slm4ie.data.pretokenize.run import LAST_RUN_KEY, MANIFEST_FILE
from slm4ie.utils import mlflow as ml
from slm4ie.utils.versioning import config_hash, corpus_digest

# ===============================
# Logging
# ===============================

logger = logging.getLogger(__name__)

# ===============================
# Constants
# ===============================

DEFAULT_EXPERIMENT = "slm4ie/data/pretokenize"

# ===============================
# Run logging
# ===============================


def output_digest(manifest: dict[str, Any]) -> str:
    """Return the content digest of an output: its manifest without the last run's timings.

    A rebuild with the same settings and inputs writes the same samples, so it
    gets the same digest, and a forced rebuild replaces the run it upserts.

    Args:
        manifest: The parsed manifest.

    Returns:
        A `sha256:`-prefixed digest of the manifest without `last_run`.
    """
    return config_hash({key: value for key, value in manifest.items() if key != LAST_RUN_KEY})


def _metrics(manifest: dict[str, Any]) -> dict[str, float]:
    """Flatten a manifest's totals and last-run figures into MLflow metrics.

    Args:
        manifest: The parsed manifest.

    Returns:
        `<count>__<split>` per split total, plus the last run's seconds and
        tokens per second when recorded.
    """
    metrics = {
        f"{key}__{split}": value for split, counts in manifest["totals"].items() for key, value in counts.items()
    }
    for key in ("seconds", "tokens_per_second"):
        value = (manifest.get(LAST_RUN_KEY) or {}).get(key)
        if value is not None:
            metrics[f"last_run__{key}"] = value
    return metrics


def log_pretokenized(
    entry: PretokenizeEntry,
    *,
    enabled: bool = True,
    experiment: str = DEFAULT_EXPERIMENT,
    tracking_uri: str | None = None,
    force: bool = False,
) -> str | None:
    """Upsert the MLflow run describing one finished output.

    Args:
        entry: The conversion settings; its output must hold a manifest.
        enabled: When False, does nothing and returns None.
        experiment: MLflow experiment name.
        tracking_uri: Tracking URI override; resolved when None.
        force: Replace an existing run for the same manifest digest.

    Returns:
        The manifest digest logged or found already tracked, or None when
        tracking is off or unavailable, or the output has no manifest.
    """
    if not enabled:
        return None
    manifest_path = entry.output_dir / MANIFEST_FILE
    if not manifest_path.exists():
        logger.warning("No manifest for %s under %s; skipping tracking.", entry.key, entry.output_dir)
        return None
    if not ml.ensure_experiment(experiment, tracking_uri=tracking_uri):
        logger.warning("MLflow unavailable; skipping pretokenize tracking.")
        return None

    # skip or replace an already tracked output, keyed by its content, not its timings
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = output_digest(manifest)
    existing = ml.find_run_by_tag(experiment, "output_digest", digest, tracking_uri=tracking_uri)
    if existing is not None and not force:
        logger.info("Output %s (%s) already tracked; pass --force to re-log.", entry.key, digest)
        return digest
    if existing is not None:
        ml.delete_run(existing, tracking_uri=tracking_uri)

    # build the tags and params from the manifest
    settings = manifest["settings"]
    tags = {
        "run_type": "data_pipeline",
        "pipeline": "pretokenize",
        "language": entry.language,
        "output_digest": digest,
        "tokenizer_digest": settings["tokenizer_digest"],
        "corpus_digest": manifest["inputs"]["digest"],
    }
    commit = ml.git_commit()
    if commit is not None:
        tags["git_commit"] = commit
    params = {
        "key": entry.key,
        "corpus": settings["corpus"],
        "tokenizer": settings["tokenizer"],
        "max_seq_len": settings["max_seq_len"],
        "min_chunk_tokens": settings["min_chunk_tokens"],
        "formats": ",".join(settings["formats"]),
        "seed": settings["seed"],
        "validation_share_used": manifest["validation_share_used"],
        "input_shards": len(manifest["inputs"]["shards"]),
    }

    # log the run with the manifest and both dataset inputs
    with ml.mlflow_run(entry.key, tags=tags):
        ml.log_params(params)
        ml.log_metrics(_metrics(manifest))
        ml.log_artifact(manifest_path)
        # the corpus digest as the curation pipeline logs it, so the two runs link
        corpus = Path(settings["corpus"])
        ml.log_dataset_input(f"pretrain/{corpus.name}", corpus_digest(corpus), str(corpus), context="pretokenize")
        ml.log_dataset_input(f"pretokenized/{entry.key}", digest, str(entry.output_dir), context="produced")
    logger.info("Tracked %s as %s", entry.key, digest)
    return digest
