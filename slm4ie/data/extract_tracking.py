"""Post-hoc MLflow tracking for the extraction tier.

Reads the on-disk `extracted/` outputs after an extraction build and records a
single MLflow run per distinct build under the `slm4ie/data/extract`
experiment. The run is keyed by a content digest of the extracted tree
(`slm4ie.data.curate.corpus_digest`) and upserted: a build whose digest already
has a run is skipped unless `force` is set, in which case the prior run is
deleted and re-logged. This mirrors the rest of the pipeline's
skip-unless-`--force` convention and is decoupled from which sources were
(re-)extracted on any given invocation -- it always reports the corpus as it
currently exists on disk.

What is logged:

* Per-source `rows__<key>` and `empty_text_rate__<key>` metrics, plus corpus
  totals, so row counts and near-empty-doc rates are queryable per build.
* A `profile.json` artifact with the full per-source quality profile:
  required-field completeness, parse errors, annotation-sidecar presence and
  row alignment (the audit checks in TODO section 1).
* The extracted tree as a `produced` dataset input (name + digest) so
  downstream consumers can declare the exact extraction build they read.

A true per-source *skipped-record rate* is intentionally not logged here: the
extractors count only documents written, not input-vs-output, so there is no
denominator without instrumenting every extractor type. That remains a separate
follow-up.
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from slm4ie.data.curate import config_hash, corpus_digest
from slm4ie.data.io_utils import find_dataset_files
from slm4ie.utils import mlflow as ml

logger = logging.getLogger(__name__)

#: Experiment name for extraction builds, following the project's
#: `slm4ie/<workstream>/<dataset>` convention.
DEFAULT_EXPERIMENT = "slm4ie/data/extract"

#: Fields every extracted record must carry (see CLAUDE.md data layout). Their
#: per-source presence rate is part of the quality profile.
REQUIRED_FIELDS = ("text", "source", "domain", "doc_id", "metadata")


@dataclass
class SourceProfile:
    """Per-source quality profile derived from the extracted text JSONL.

    Attributes:
        key: Dataset key.
        domain: Configured provenance tag, or None when unknown.
        present: Whether `<key>.jsonl` exists on disk.
        rows: Number of records in `<key>.jsonl`.
        empty_text: Records whose `text` is missing or whitespace-only.
        parse_errors: Lines that failed JSON parsing.
        missing_fields: Per-required-field count of records lacking it.
        has_annotations: Whether an annotations sidecar exists.
        annotation_rows: Rows in the annotations sidecar (0 when absent).
        sidecar_aligned: True/False when a sidecar exists and its row count
            matches `rows`; None when there is no sidecar.
    """

    key: str
    domain: Optional[str] = None
    present: bool = False
    rows: int = 0
    empty_text: int = 0
    parse_errors: int = 0
    missing_fields: Dict[str, int] = field(default_factory=dict)
    has_annotations: bool = False
    annotation_rows: int = 0
    sidecar_aligned: Optional[bool] = None

    @property
    def empty_text_rate(self) -> float:
        """Return the fraction of records with empty/whitespace-only text."""
        return self.empty_text / self.rows if self.rows else 0.0


def _count_gz_rows(path: Path) -> int:
    """Count newline-delimited records in a gzipped file without parsing.

    Args:
        path: Path to a `.gz` file.

    Returns:
        The number of rows (newline count, with a trailing partial line
        counted).
    """
    import gzip

    count = 0
    last = b"\n"
    with gzip.open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            count += chunk.count(b"\n")
            last = chunk[-1:]
    if last not in (b"\n", b""):
        count += 1
    return count


def profile_source(key: str, text_path: Path, annotations_path: Optional[Path], domain: Optional[str]) -> SourceProfile:
    """Profile one extracted source by a single pass over its text JSONL.

    Args:
        key: Dataset key.
        text_path: Path to `<key>.jsonl`.
        annotations_path: Path to the annotations sidecar, or None.
        domain: Configured provenance tag for the source, or None.

    Returns:
        The populated `SourceProfile`.
    """
    profile = SourceProfile(key=key, domain=domain, present=True)
    profile.missing_fields = {f: 0 for f in REQUIRED_FIELDS}

    with text_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            profile.rows += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                profile.parse_errors += 1
                continue
            text = record.get("text")
            if not isinstance(text, str) or not text.strip():
                profile.empty_text += 1
            for required in REQUIRED_FIELDS:
                if record.get(required) in (None, ""):
                    profile.missing_fields[required] += 1

    if annotations_path is not None:
        profile.has_annotations = True
        profile.annotation_rows = _count_gz_rows(annotations_path)
        profile.sidecar_aligned = profile.annotation_rows == profile.rows

    return profile


def build_extraction_profile(output_base: Path, datasets: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Build the full per-source extraction profile from disk.

    Profiles every configured dataset that has a `<key>.jsonl` under
    `output_base`, reflecting the corpus as it currently exists regardless of
    which subset was extracted last. Configured sources without an output are
    recorded as absent.

    Args:
        output_base: The `extracted/` tier directory.
        datasets: The extraction config's dataset map (`<key>` ->
            spec dict carrying a `domain`).

    Returns:
        A dict with `by_source` (per-key profile dicts) and `totals`
        (corpus-wide row/empty/missing aggregates).
    """
    by_source: Dict[str, Dict[str, Any]] = {}
    total_rows = 0
    total_empty = 0
    present_count = 0

    for key in sorted(datasets):
        domain = datasets[key].get("domain")
        found = find_dataset_files(output_base, key)
        if found is None:
            by_source[key] = asdict(SourceProfile(key=key, domain=domain, present=False))
            continue
        text_path, annotations_path = found
        profile = profile_source(key, text_path, annotations_path, domain)
        by_source[key] = asdict(profile)
        total_rows += profile.rows
        total_empty += profile.empty_text
        present_count += 1

    totals = {
        "sources_configured": len(datasets),
        "sources_present": present_count,
        "rows": total_rows,
        "empty_text": total_empty,
        "empty_text_rate": (total_empty / total_rows) if total_rows else 0.0,
    }
    return {"by_source": by_source, "totals": totals}


def log_extraction_run(
    output_base: Path,
    datasets: Dict[str, Dict[str, Any]],
    *,
    enabled: bool = True,
    experiment: str = DEFAULT_EXPERIMENT,
    tracking_uri: Optional[str] = None,
    force: bool = False,
    artifact_dir: Optional[Path] = None,
) -> Optional[str]:
    """Upsert a single MLflow run describing the current extraction build.

    Computes the digest of the extracted tree, and -- unless `force` is set --
    skips logging when a run already exists for that digest. Otherwise logs
    per-source metrics, a `profile.json` artifact, and the extracted tree as a
    `produced` dataset input.

    Args:
        output_base: The `extracted/` tier directory to digest and profile.
        datasets: The extraction config's dataset map.
        enabled: When False, does nothing and returns None.
        experiment: MLflow experiment name.
        tracking_uri: Tracking URI override; resolved when None.
        force: Replace an existing run for the same digest instead of skipping.
        artifact_dir: Directory to write `profile.json` into before logging;
            a temporary directory is used when None so the data tier stays
            clean (MLflow keeps the canonical copy as a run artifact).

    Returns:
        The digest that was logged, the existing digest that was skipped, or
        None when tracking is disabled or unavailable.
    """
    if not enabled:
        return None
    if not ml.ensure_experiment(experiment, tracking_uri=tracking_uri):
        logger.warning("MLflow unavailable; skipping extraction tracking.")
        return None

    digest = corpus_digest(output_base)
    existing = ml.find_run_by_tag(experiment, "corpus_digest", digest, tracking_uri=tracking_uri)
    if existing is not None and not force:
        logger.info("Extraction build %s already tracked; pass --force to re-log.", digest)
        return digest
    if existing is not None:
        ml.delete_run(existing, tracking_uri=tracking_uri)

    profile = build_extraction_profile(output_base, datasets)

    artifact_root = artifact_dir if artifact_dir is not None else Path(tempfile.mkdtemp(prefix="extract-profile-"))
    artifact_root.mkdir(parents=True, exist_ok=True)
    profile_path = artifact_root / "profile.json"
    with profile_path.open("w", encoding="utf-8") as fh:
        json.dump(profile, fh, ensure_ascii=False, indent=2)

    metrics: Dict[str, float] = {}
    for key, source in profile["by_source"].items():
        if not source["present"]:
            continue
        rows = source["rows"]
        metrics[f"rows__{key}"] = rows
        metrics[f"empty_text_rate__{key}"] = source["empty_text"] / rows if rows else 0.0
    metrics["rows_total"] = profile["totals"]["rows"]
    metrics["sources_present"] = profile["totals"]["sources_present"]

    tags = {
        "run_type": "data_pipeline",
        "pipeline": "extract",
        "corpus_digest": digest,
        "config_hash": config_hash({"datasets": datasets}),
    }
    commit = ml.git_commit()
    if commit is not None:
        tags["git_commit"] = commit

    with ml.mlflow_run("extract-build", tags=tags):
        ml.log_params(
            {
                "output_base": str(output_base),
                "sources_configured": profile["totals"]["sources_configured"],
            }
        )
        ml.log_metrics(metrics)
        ml.log_artifact(profile_path)
        ml.log_dataset_input("extracted", digest, str(output_base), context="produced")

    logger.info("Logged extraction build %s to %s.", digest, experiment)
    return digest
