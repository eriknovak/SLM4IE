"""Run the pretraining-corpus curation pipeline stage by stage.

The eight stages, and the tree each writes under `<output_dir>/`:

    0. convert        -> 00_convert/
    1. language       -> 01_language/
    2. spam           -> 02_spam/
    3. quality        -> 03_quality/
    4. repetition     -> 04_repetition/
    5. exact_dedup    -> 05_exact_dedup/
    6. sentence_dedup -> 06_sentence_dedup/   (final corpus)
    7. statistics     -> 07_statistics/

Stage 0 turns the extraction-output `<key>.jsonl` files into datatrove-shaped
`<key>/<NNNNN>.jsonl.gz` shards.

Each stage writes a `.complete` sentinel into its output folder. On rerun a
stage's sentinel is compared against a fresh hash of its config slice; on
mismatch that stage and every downstream stage are invalidated and re-executed.

`curate` is the argv-free entry point; `recount` backfills per-source counts
onto sentinels written before that fix. `scripts/curate_pretraining_corpus.py` is
the CLI over both.
"""

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple, cast

import yaml

from datatrove.pipeline.dedup import SentDedupConfig

from slm4ie.data.curate import (
    STAGE_NAMES,
    cascade_from,
    config_hash,
    read_sentinel,
    upstream_stage,
    write_sentinel,
)
from slm4ie.data.curate.stages import CORPUS_STAGES, SCOPED_STAGES, STAGE_DIRS, STAGE_VERSIONS, is_scoped
from slm4ie.data.curate.overrides import effective_stage_config, validate_overrides
from slm4ie.data.curate.sentinel import (
    NOT_BUILT,
    SENTINEL_NAME,
    Sentinel,
    dataset_sentinel_path,
    invalidate_dataset_sentinels,
    stale_reason,
    update_dataset_sentinel_counts,
)
from slm4ie.data.curate.convert import (
    DEFAULT_ID_FIELD,
    DEFAULT_METADATA_FIELDS,
    DEFAULT_TEXT_FIELD,
    run_convert_stage,
)
from slm4ie.data.curate.dedup import make_exact_config
from slm4ie.data.curate.pipeline import (
    CuratePaths,
    QualityConfig,
    build_language_executors,
    build_quality_executors,
    build_repetition_executors,
    build_spam_executors,
    build_exact_dedup_executors,
    build_sentence_dedup_executors,
    build_statistics_executors,
    per_key_stage_counts,
    pipeline_io_counts,
    stage_io_counts,
)
from slm4ie.data.curate.spam import SpamAssets, SpamConfig, load_spam_assets
from slm4ie.data.io_utils import (
    DEFAULT_MAX_SHARD_BYTES,
    find_project_root as _find_project_root,
    resolve_project_path,
)
from slm4ie.data.stopwords import load_stopwords
from slm4ie.data.versioning import (
    UnitScan,
    check_integrity,
    combine_named_digests,
    file_sha256,
    merge_digests,
    merge_scans,
    read_lock,
    scan_documents,
    scan_files,
    shard_files,
    write_lock,
)

logger = logging.getLogger(__name__)


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Read a YAML file, returning an empty dict when the path does not exist.

    Args:
        path: Path to a YAML file.

    Returns:
        Parsed mapping, or `{}` when the file is missing.
    """
    if not path.exists():
        return {}
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def _list_datasets(extract_config: Path) -> List[str]:
    """Return the pretraining dataset keys declared in `extract.yaml`.

    Entries with `role: benchmark` are evaluation gold and never enter the
    corpus on an `--all` build; they can still be passed positionally.

    Args:
        extract_config: Path to the extraction config.

    Returns:
        Dataset keys with `role: pretrain` (the default), in declaration order.
    """
    cfg = _load_yaml(extract_config)
    datasets = cfg.get("datasets") or {}
    return [key for key, spec in datasets.items() if (spec or {}).get("role", "pretrain") == "pretrain"]


def _resolve_dirs(input_dir: Optional[Path], output_dir: Optional[Path], cfg: Dict[str, Any]) -> Tuple[Path, Path]:
    """Resolve input/output dirs from overrides or the pretrain config.

    Args:
        input_dir: Override for the pretrain config's `input_dir`, or None.
        output_dir: Override for the pretrain config's `output_dir`, or None.
        cfg: The parsed pretrain config.

    Returns:
        Tuple `(input_dir, output_dir)`.

    Raises:
        FileNotFoundError: If neither the override nor the YAML key is
            set on either side.
    """
    raw_input = input_dir if input_dir is not None else cfg.get("input_dir")
    raw_output = output_dir if output_dir is not None else cfg.get("output_dir")
    if raw_input is None or raw_output is None:
        raise FileNotFoundError(
            "Curation paths not set. Provide --input-dir/--output-dir or set "
            "the pretrain config's input_dir / output_dir."
        )
    resolved_input = Path(raw_input) if input_dir is not None else resolve_project_path(raw_input)
    resolved_output = Path(raw_output) if output_dir is not None else resolve_project_path(raw_output)
    return resolved_input, resolved_output


def _load_stopwords(cfg: Dict[str, Any]) -> Tuple[Set[str], bytes]:
    """Load the stopword set and return (set, raw_bytes_for_hashing).

    Thin wrapper over `slm4ie.data.stopwords.load_stopwords`. Reads the
    language code from `cfg['stopwords']`. A missing or empty key
    disables stopwords (returns an empty set and empty bytes, after
    logging a warning). An unknown code is propagated as `ValueError`
    so a config typo fails the run.

    Args:
        cfg: The parsed pretrain config.

    Returns:
        Tuple of `(stopword set, raw file bytes)`. The bytes are folded
        into the sentinel hash for stages that consume stopwords.

    Raises:
        ValueError: If `cfg['stopwords']` is set to a code that has no
            bundled list under `slm4ie/data/stopwords/`.
    """
    code = cfg.get("stopwords")
    if not code:
        logger.warning("stopwords code not configured; using empty set.")
        return set(), b""
    return load_stopwords(code)


def _load_spam_assets(cfg: Dict[str, Any]) -> SpamAssets:
    """Load the spam-filter lexicons and domain blocklist from config.

    Reads the languages and URL-blocklist toggle from `cfg['spam']`,
    then resolves the curated per-language lexicons plus the domain
    blocklist. The bundle's raw bytes are folded into the spam stage's
    sentinel hash so editing any list invalidates the stage.

    Args:
        cfg: The parsed pretrain config.

    Returns:
        A `SpamAssets` bundle (empty lexicons when no languages are
        configured).

    Raises:
        ValueError: If a configured language has no curated list under
            `slm4ie/data/spam/`.
    """
    scfg = cfg.get("spam") or {}
    languages = scfg.get("languages") or []
    url_blocklist = bool(scfg.get("url_blocklist", True))
    return load_spam_assets(languages, url_blocklist=url_blocklist)


def _filter_stage_subset(stage_dir: Path, keys: List[str], holder: Optional[Path] = None) -> Path:
    """Materialize a folder of symlinks restricted to *keys* under *stage_dir*.

    Args:
        stage_dir: A scoped stage's output folder (e.g.
            `<output_dir>/01_language/`).
        keys: Dataset keys to expose.
        holder: Folder to build the view in, replacing any previous view
            there; a fresh tempdir when omitted. A fixed path keeps a
            resumed corpus stage reading the same file paths.

    Returns:
        Path to the folder mirroring the requested keys via symlinks, so a
        downstream stage's reader walks only the subset's shards.

    Raises:
        FileNotFoundError: If any requested shard folder is missing or
            empty under *stage_dir*.
    """
    missing: List[str] = []
    for key in keys:
        src = stage_dir / key
        if not src.is_dir() or not any(src.glob("*.jsonl.gz")):
            missing.append(key)
    if missing:
        raise FileNotFoundError(
            f"No converted shard folder(s) under {stage_dir} for dataset(s): " + ", ".join(repr(k) for k in missing)
        )
    if holder is None:
        holder = Path(tempfile.mkdtemp(prefix="slm4ie-pretrain-subset-"))
    else:
        shutil.rmtree(holder, ignore_errors=True)
        holder.mkdir(parents=True)
    try:
        for key in keys:
            src = stage_dir / key
            holder_key = holder / key
            holder_key.mkdir()
            for shard in src.glob("*.jsonl.gz"):
                (holder_key / shard.name).symlink_to(shard.resolve())
    except BaseException:
        shutil.rmtree(holder, ignore_errors=True)
        raise
    return holder


def _has_stage_output(stage_dir: Path, key: str) -> bool:
    """Return True if *key* has shard output under *stage_dir*.

    Used to drop datasets that produced nothing upstream — declared in the
    roster but never downloaded, or fully filtered out by an earlier stage
    — before a scoped stage tries to read their (nonexistent) shards.

    Args:
        stage_dir: A stage's output folder (e.g. `<output_dir>/00_convert`).
        key: Dataset key to check.

    Returns:
        True if `<stage_dir>/<key>/` exists and holds `.jsonl.gz` shards.
    """
    folder = stage_dir / key
    return folder.is_dir() and any(folder.glob("*.jsonl.gz"))


#: File in a corpus stage folder recording what an unfinished run started with.
PROGRESS_NAME: str = ".in_progress.json"


def _input_fingerprint(view: Path) -> Tuple[int, str]:
    """Count and fingerprint the shards a corpus stage will read.

    Args:
        view: Symlink view of the stage's input, one folder per dataset.

    Returns:
        Tuple `(shard_count, digest)`; the digest covers each shard's relative
        path and size, which fix how datatrove assigns shards to tasks.
    """
    shards = sorted(view.glob("*/*.jsonl.gz"))
    digest = hashlib.sha256()
    for shard in shards:
        digest.update(f"{shard.relative_to(view)}\t{shard.stat().st_size}\n".encode("utf-8"))
    return len(shards), digest.hexdigest()


def _staging_dir(paths: CuratePaths, stage: str) -> Path:
    """Return the folder a stage writes into before its output is promoted.

    Args:
        paths: Resolved curation paths.
        stage: Stage name.

    Returns:
        `<output_dir>/_partial/<stage folder>`.
    """
    return paths.output_dir / "_partial" / STAGE_DIRS[stage]


def _swap_into_place(new: Path, final: Path) -> None:
    """Replace *final* with the finished folder *new* by renaming.

    The old output is renamed aside, the new one renamed in, then the old one
    removed; a crash in between leaves *final* missing (so it is rebuilt),
    never a mix of old and new files.

    Args:
        new: Finished folder, sentinel included.
        final: The unit's canonical output folder.
    """
    old = new.with_name(new.name + ".old")
    shutil.rmtree(old, ignore_errors=True)
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        os.rename(final, old)
    os.rename(new, final)
    shutil.rmtree(old, ignore_errors=True)


def _prepare_corpus_stage(paths: CuratePaths, stage: str, config_hash_value: str, tasks: int, inputs: str) -> bool:
    """Resume an unfinished corpus stage, or clear its staging folder for a fresh start.

    A corpus stage builds in its staging folder and is promoted only once it
    finishes and passes its integrity check, so the previous output stays in
    place until then. A run resumes only when the staging folder's progress
    file matches the current config hash, stage version, task count and
    inputs; datatrove then skips the tasks it marked complete. Otherwise the
    staging folder, the stage's logs (with the completion markers) and its
    dedup scratch are removed first, so nothing from an earlier run survives.

    Args:
        paths: Resolved curation paths.
        stage: Corpus stage name.
        config_hash_value: Hash of the stage's config slice and roster.
        tasks: Task count this run uses.
        inputs: The stage's input digest and shard fingerprint.

    Returns:
        True when resuming, False when the staging folder was cleared.
    """
    staging = _staging_dir(paths, stage)
    progress_file = staging / PROGRESS_NAME
    expected = {
        "config_hash": config_hash_value,
        "stage_version": STAGE_VERSIONS[stage],
        "tasks": tasks,
        "inputs": inputs,
    }
    if progress_file.is_file() and json.loads(progress_file.read_text(encoding="utf-8")) == expected:
        return True
    shutil.rmtree(staging, ignore_errors=True)
    shutil.rmtree(paths.logs_dir(stage), ignore_errors=True)
    if stage in ("exact_dedup", "sentence_dedup"):
        _purge_dedup_state(paths, stage)
    staging.mkdir(parents=True)
    progress_file.write_text(json.dumps(expected), encoding="utf-8")
    return False


def _starting_input_hint(paths: CuratePaths, stage: str) -> str:
    """Return an ` (input records ~N)` suffix for a stage's start log line.

    The approximate input size is read from the upstream stage's
    sentinel `records_out` field — already on disk from that stage's
    own run, so no shard scan is needed.

    Args:
        paths: Resolved curation paths.
        stage: Stage about to run.

    Returns:
        An ` (input records ~N)` suffix when the upstream stage's
        sentinel is present, or an empty string otherwise (the first
        stage, or an upstream stage that has not completed).
    """
    upstream = upstream_stage(stage)
    if upstream is None:
        return ""
    sentinel = read_sentinel(paths.stage_dir(upstream))
    if sentinel is None:
        return ""
    return f" (input records ~{sentinel.records_out})"


def _human_bytes(num: int) -> str:
    """Render a byte count as a human-readable string.

    Args:
        num: A non-negative byte count.

    Returns:
        The count scaled to the largest binary unit below 1024, with
        one decimal place for KiB and above (e.g. 1536 -> `"1.5 KiB"`)
        and no decimal for plain bytes (e.g. 512 -> `"512 B"`).
    """
    size = float(num)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if size < 1024.0:
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} EiB"


def _extracted_input_summary(input_dir: Path, keys: List[str]) -> Tuple[int, int]:
    """Summarize the convert stage's input without reading record contents.

    The convert stage performs a 1:1 format conversion, so its input
    record count is redundant with the output shard count taken after
    the run. Counting input lines would mean reading every `<key>.jsonl`
    in full; instead this reports the dataset count and the summed
    on-disk size via `stat` only.

    Args:
        input_dir: Directory holding extraction output JSONLs.
        keys: Dataset keys to include.

    Returns:
        Tuple `(num_datasets_present, total_bytes)`: the count of keys
        whose `<key>.jsonl` exists under *input_dir* and the summed
        on-disk size of those files. Keys with no `<key>.jsonl` are
        excluded from both figures (the convert stage reports them as
        skipped).
    """
    present = 0
    total_bytes = 0
    for key in keys:
        path = input_dir / f"{key}.jsonl"
        if not path.exists():
            continue
        present += 1
        total_bytes += path.stat().st_size
    return present, total_bytes


def _purge_dedup_state(paths: CuratePaths, which: str) -> None:
    """Purge the dedup scratch for *which* stage.

    Args:
        paths: Resolved curation paths.
        which: Either `"exact_dedup"` or `"sentence_dedup"`.
    """
    prefix = {"exact_dedup": "exact", "sentence_dedup": "sent"}[which]
    for sub in (paths.dedup_state_dir / f"{prefix}_sigs", paths.dedup_state_dir / f"{prefix}_dups"):
        if sub.exists():
            shutil.rmtree(sub, ignore_errors=True)


@dataclass(frozen=True)
class ConvertParams:
    """Resolved convert-stage parameters for one config bucket.

    Attributes:
        text_field: Record field copied into `text`.
        id_field: Record field kept as `doc_id`.
        metadata_fields: Record fields kept under `Document.metadata`.
        include_annotations: Whether to join the annotations sidecar.
        max_shard_bytes: Compressed-byte ceiling per output shard.
    """

    text_field: str
    id_field: str
    metadata_fields: List[str]
    include_annotations: bool
    max_shard_bytes: int


@dataclass(frozen=True)
class LanguageParams:
    """Resolved language-stage parameters for one config bucket.

    Attributes:
        target_languages: ISO 639-1 codes treated as in-language.
        candidate_languages: Candidate set lingua chooses among, or None.
        mode: `filter` drops out-of-target docs; `tag` only annotates.
        minimum_relative_distance: Confidence gap lingua needs to commit.
        low_accuracy: Use lingua's lighter trigram-only model.
        max_chars: Truncate doc text to this many chars, or None.
    """

    target_languages: List[str]
    candidate_languages: Optional[List[str]]
    mode: str
    minimum_relative_distance: float
    low_accuracy: bool
    max_chars: Optional[int]


def _build_convert_params(ccfg: Dict[str, Any]) -> ConvertParams:
    """Resolve a convert-stage config slice into typed parameters.

    Args:
        ccfg: The effective `convert` config slice for one bucket.

    Returns:
        The resolved `ConvertParams`, with defaults applied.
    """
    metadata_fields_raw = ccfg.get("metadata_fields")
    return ConvertParams(
        text_field=str(ccfg.get("text_field", DEFAULT_TEXT_FIELD)),
        id_field=str(ccfg.get("id_field", DEFAULT_ID_FIELD)),
        metadata_fields=(
            [str(f) for f in metadata_fields_raw] if metadata_fields_raw is not None else list(DEFAULT_METADATA_FIELDS)
        ),
        include_annotations=bool(ccfg.get("include_annotations", False)),
        max_shard_bytes=int(ccfg.get("max_shard_bytes", DEFAULT_MAX_SHARD_BYTES)),
    )


def _build_language_params(lang_cfg: Dict[str, Any]) -> LanguageParams:
    """Resolve a language-stage config slice into typed parameters.

    Args:
        lang_cfg: The effective `language` config slice for one bucket.

    Returns:
        The resolved `LanguageParams`, with defaults applied.
    """
    return LanguageParams(
        target_languages=lang_cfg.get("targets") or ["sl"],
        candidate_languages=lang_cfg.get("candidates"),
        mode=str(lang_cfg.get("mode", "filter")),
        minimum_relative_distance=float(lang_cfg.get("minimum_relative_distance", 0.0)),
        low_accuracy=bool(lang_cfg.get("low_accuracy", False)),
        max_chars=lang_cfg.get("max_chars"),
    )


def _build_spam_config(spcfg: Dict[str, Any]) -> SpamConfig:
    """Resolve a spam-stage config slice into a `SpamConfig`.

    Args:
        spcfg: The effective `spam` config slice for one bucket.

    Returns:
        The resolved `SpamConfig`, with defaults applied.

    Raises:
        ValueError: If `model` is set; no model resolver is wired.
    """
    if spcfg.get("model"):
        raise ValueError(
            "the pretrain config's spam.model is set, but no model resolver is "
            "configured. Leave it null, or wire a scorer before enabling it."
        )
    return SpamConfig(
        min_adult_hits=int(spcfg.get("min_adult_hits", 2)),
        min_spam_hits=int(spcfg.get("min_spam_hits", 2)),
        keep_fraction=float(spcfg.get("keep_fraction", 0.0)),
        default_language=str(spcfg.get("default_language", "sl")),
        url_blocklist=bool(spcfg.get("url_blocklist", True)),
        use_ldnoobw=bool(spcfg.get("use_ldnoobw", True)),
        model=spcfg.get("model"),
        model_threshold=float(spcfg.get("model_threshold", 0.5)),
    )


def _build_quality_config(qcfg: Dict[str, Any]) -> QualityConfig:
    """Resolve a quality-stage config slice into a `QualityConfig`.

    Args:
        qcfg: The effective `quality` config slice for one bucket.

    Returns:
        The resolved `QualityConfig`, with defaults applied.
    """
    return QualityConfig(
        min_doc_words=int(qcfg.get("min_doc_words", 50)),
        max_doc_words=int(qcfg.get("max_doc_words", 100_000)),
        min_avg_word_length=int(qcfg.get("min_avg_word_length", 3)),
        max_avg_word_length=int(qcfg.get("max_avg_word_length", 10)),
        max_symbol_word_ratio=float(qcfg.get("max_symbol_word_ratio", 0.1)),
        max_bullet_lines_ratio=float(qcfg.get("max_bullet_lines_ratio", 0.9)),
        max_ellipsis_lines_ratio=float(qcfg.get("max_ellipsis_lines_ratio", 0.3)),
        max_non_alpha_words_ratio=float(qcfg.get("max_non_alpha_words_ratio", 0.8)),
        min_stop_words=int(qcfg.get("min_stop_words", 2)),
    )


def _stage_runner(
    stage: str,
    paths: CuratePaths,
    cfg: Dict[str, Any],
    workers: int,
    stopwords: Set[str],
    spam_assets: SpamAssets,
    dataset_keys: List[str],
    input_view: Optional[Path] = None,
    log_dir: Optional[Path] = None,
    tasks: Optional[int] = None,
    output_folder: Optional[Path] = None,
) -> Callable[[], Tuple[int, int]]:
    """Return a zero-arg callable that runs *stage*'s executor chain.

    Args:
        stage: Stage name.
        paths: Resolved curation paths.
        cfg: The parsed pretrain config.
        workers: Resolved worker count.
        stopwords: Loaded stopword set (used by quality and statistics).
        spam_assets: Loaded spam lexicons and domain blocklist (used by
            the spam stage).
        dataset_keys: Dataset keys to process; consumed by the convert
            stage to know which `<key>.jsonl` files to read.
        input_view: Optional symlink view of the stage's upstream output,
            restricting the reader to the keys being (re)run for a scoped
            stage, or to the roster for a corpus stage. Ignored by convert
            (scoped by `dataset_keys`).
        log_dir: Optional directory for per-task log files (currently
            only consumed by the convert stage).
        tasks: Task count for the corpus stages; defaults to `workers`.
            The scoped stages run one task per worker.
        output_folder: Folder to write into instead of the stage's output
            folder; the runner passes the stage's staging folder.

    Returns:
        A callable that runs the stage when invoked and returns its
        `(records_in, records_out)` document counts. The convert stage
        sums the per-dataset counts `run_convert_stage` returns; the
        scoped datatrove stages read theirs from the run's `PipelineStats`
        via `pipeline_io_counts`, and the resumable corpus stages from
        every finished task's stats via `stage_io_counts`. The statistics
        stage reports `(records_in, 0)` since it emits a JSON bundle, not shards.

    Raises:
        ValueError: If *stage* is not a known stage name.
    """
    if stage == "convert":
        cparams = _build_convert_params(cfg.get("convert") or {})
        out = output_folder if output_folder is not None else paths.stage_dir("convert")

        def run() -> Tuple[int, int]:
            results = run_convert_stage(
                input_dir=paths.input_folder,
                output_dir=out,
                dataset_keys=dataset_keys,
                text_field=cparams.text_field,
                id_field=cparams.id_field,
                metadata_fields=cparams.metadata_fields,
                include_annotations=cparams.include_annotations,
                max_shard_bytes=cparams.max_shard_bytes,
                workers=workers,
                log_dir=log_dir,
            )
            # Convert is a 1:1 format conversion; the count it writes
            # out is also the count it read in. Datasets with no input
            # map to None (skipped) and contribute nothing.
            total = sum(n for n in results.values() if n is not None)
            return total, total

        return run

    if stage == "language":
        # Read through the upstream subset view when only some keys are
        # being (re)run, so the reader skips shards from currently
        # untouched keys.
        lparams = _build_language_params(cfg.get("language") or {})

        def run() -> Tuple[int, int]:
            execs = build_language_executors(
                paths,
                tasks=workers,
                target_languages=lparams.target_languages,
                candidate_languages=lparams.candidate_languages,
                lang_mode=lparams.mode,
                lang_minimum_relative_distance=lparams.minimum_relative_distance,
                lang_low_accuracy=lparams.low_accuracy,
                lang_max_chars=lparams.max_chars,
                input_override=input_view,
                output_override=output_folder,
            )
            return pipeline_io_counts(execs[-1].run())

        return run

    if stage == "spam":
        spam_config = _build_spam_config(cfg.get("spam") or {})

        def run() -> Tuple[int, int]:
            execs = build_spam_executors(
                paths,
                tasks=workers,
                spam_config=spam_config,
                adult_words=spam_assets.adult_words,
                spam_words=spam_assets.spam_words,
                domains=spam_assets.domains,
                input_override=input_view,
                output_override=output_folder,
            )
            return pipeline_io_counts(execs[-1].run())

        return run

    if stage == "quality":
        quality_config = _build_quality_config(cfg.get("quality") or {})

        def run() -> Tuple[int, int]:
            execs = build_quality_executors(
                paths,
                tasks=workers,
                quality_config=quality_config,
                stopwords=stopwords,
                input_override=input_view,
                output_override=output_folder,
            )
            return pipeline_io_counts(execs[-1].run())

        return run

    if stage == "repetition":

        def run() -> Tuple[int, int]:
            execs = build_repetition_executors(
                paths, tasks=workers, input_override=input_view, output_override=output_folder
            )
            return pipeline_io_counts(execs[-1].run())

        return run

    if stage == "exact_dedup":
        edcfg = cfg.get("exact_dedup") or {}
        raw_precision = int(edcfg.get("precision", 64))
        if raw_precision not in (32, 64):
            raise ValueError(f"the pretrain config's exact_dedup.precision must be 32 or 64, got {raw_precision}")
        raw_hash_fc = str(edcfg.get("hash_fc", "xxhash"))
        if raw_hash_fc not in ("sha1", "xxhash"):
            raise ValueError(
                f"the pretrain config's exact_dedup.hash_fc must be 'sha1' or 'xxhash', got {raw_hash_fc!r}"
            )
        # Both values are now narrowed by the runtime checks above; cast to
        # keep static type-checkers happy without losing the Literal contract.
        precision = cast(Literal[32, 64], raw_precision)
        hash_fc = cast(Literal["sha1", "xxhash"], raw_hash_fc)
        exact_cfg = make_exact_config(
            precision=precision,
            hash_fc=hash_fc,
            only_dedup_in_index=bool(edcfg.get("only_dedup_in_index", True)),
        )

        def run() -> Tuple[int, int]:
            execs = build_exact_dedup_executors(
                paths,
                tasks=tasks or workers,
                workers=workers,
                exact_config=exact_cfg,
                input_override=input_view,
                output_override=output_folder,
            )
            execs[-1].run()
            return stage_io_counts(paths.logs_dir("exact_dedup") / "3_filter")

        return run

    if stage == "sentence_dedup":
        scfg = cfg.get("sentence_dedup") or {}
        sent_cfg = SentDedupConfig(
            n_sentences=int(scfg.get("n_sentences", 3)),
            min_doc_words=int(scfg.get("min_doc_words", 50)),
            min_num_sentences=int(scfg.get("min_num_sentences", 2)),
            split_sentences=bool(scfg.get("split_sentences", True)),
        )

        def run() -> Tuple[int, int]:
            execs = build_sentence_dedup_executors(
                paths,
                tasks=tasks or workers,
                workers=workers,
                sentence_config=sent_cfg,
                input_override=input_view,
                output_override=output_folder,
            )
            execs[-1].run()
            return stage_io_counts(paths.logs_dir("sentence_dedup") / "3_filter")

        return run

    if stage == "statistics":
        stcfg = cfg.get("statistics") or {}

        def run() -> Tuple[int, int]:
            execs = build_statistics_executors(
                paths,
                tasks=tasks or workers,
                workers=workers,
                stopwords=stopwords,
                top_k_words=int(stcfg.get("top_k_words", 5_000)),
                input_override=input_view,
                output_override=output_folder,
            )
            execs[-1].run()
            # The statistics stage emits a JSON bundle, not shards: report the
            # documents its map tasks consumed and a zero output count.
            records_in, _ = stage_io_counts(paths.logs_dir("statistics") / "1_map")
            return records_in, 0

        return run

    raise ValueError(f"Unknown stage: {stage}")


def _stage_slice(stage: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return the config slice that drives *stage*'s sentinel hash.

    Args:
        stage: Stage name.
        cfg: The parsed pretrain config.

    Returns:
        The mapping under the stage's top-level YAML key, or `{}` if
        absent.
    """
    return dict(cfg.get(stage) or {})


def _convert_input_paths(input_dir: Path, key: str, include_annotations: bool) -> List[Path]:
    """Return the source files the convert stage reads for *key*.

    Args:
        input_dir: Root of the extracted tier holding `<key>.jsonl`.
        key: Dataset key.
        include_annotations: Whether the convert stage also joins the
            `<key>.annotations.jsonl.gz` sidecar.

    Returns:
        The `<key>.jsonl` path, plus the annotations sidecar path when
        `include_annotations` is True. Paths are returned whether or not
        they exist on disk.
    """
    paths = [input_dir / f"{key}.jsonl"]
    if include_annotations:
        paths.append(input_dir / f"{key}.annotations.jsonl.gz")
    return paths


def _convert_input_files(
    input_dir: Path, key: str, include_annotations: bool, previous: Optional[Dict[str, Any]] = None
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Describe *key*'s convert inputs by size and content hash.

    A file whose size and mtime match *previous* keeps its recorded hash
    instead of being reread: the mtime only decides whether to rehash, never
    whether the input changed, so a touched or copied file is not a change.

    Args:
        input_dir: Root of the extracted tier holding `<key>.jsonl`.
        key: Dataset key.
        include_annotations: Whether convert also joins the annotations sidecar.
        previous: The `input_files` recorded in the unit's last sentinel.

    Returns:
        Per file name `{size, sha256, mtime_ns}`, or `None` for an absent file.
    """
    previous = previous or {}
    files: Dict[str, Optional[Dict[str, Any]]] = {}
    for path in _convert_input_paths(input_dir, key, include_annotations):
        if not path.is_file():
            files[path.name] = None
            continue
        st = path.stat()
        known = previous.get(path.name) or {}
        if known.get("size") == st.st_size and known.get("mtime_ns") == st.st_mtime_ns and known.get("sha256"):
            sha = str(known["sha256"])
        else:
            sha = file_sha256(path)
        files[path.name] = {"size": st.st_size, "sha256": sha, "mtime_ns": st.st_mtime_ns}
    return files


def _input_files_digest(files: Dict[str, Optional[Dict[str, Any]]]) -> str:
    """Return the convert input digest: each file's size and content hash, no mtime.

    Args:
        files: Output of `_convert_input_files`.

    Returns:
        A `sha256:` digest over the files' names, sizes and content hashes.
    """
    return combine_named_digests({name: f"{f['size']}:{f['sha256']}" if f else None for name, f in files.items()})


def _run_info(project_root: Path) -> Dict[str, Any]:
    """Return the informational sentinel fields: git commit and datatrove version.

    Args:
        project_root: Repository root to read the commit from.

    Returns:
        Mapping with `git_commit` and `datatrove_version`, each `None` when
        it cannot be determined.
    """
    try:
        commit: Optional[str] = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        logger.debug("git commit unavailable: %s", exc)
        commit = None
    try:
        datatrove: Optional[str] = package_version("datatrove")
    except PackageNotFoundError:
        datatrove = None
    return {"git_commit": commit, "datatrove_version": datatrove}


def _bucket_keys_by_effective_hash(
    keys: List[str],
    stage: str,
    cfg: Dict[str, Any],
    overrides: Dict[str, Any],
    extra: bytes,
) -> Dict[str, List[str]]:
    """Group *keys* by their effective-config hash for *stage*.

    Datasets that resolve to the same effective stage config share a hash
    and can run in one executor; each distinct override forms its own
    bucket. A dataset with no override hashes identically to the plain
    global slice, so the all-defaults case stays a single bucket.

    Args:
        keys: Dataset keys to bucket, in run order.
        stage: Scoped stage name.
        cfg: The parsed pretrain config.
        overrides: The `overrides:` mapping.
        extra: Stage-level extra bytes folded into the hash (stopwords /
            spam lexicon / roster); identical across keys of a stage.

    Returns:
        Mapping of effective-config hash to the keys sharing it. Bucket
        insertion order follows first appearance.
    """
    buckets: Dict[str, List[str]] = {}
    for key in keys:
        slice_ = effective_stage_config(cfg, overrides, key, stage)
        buckets.setdefault(config_hash(slice_, extra=extra), []).append(key)
    return buckets


def _dataset_keys_payload(dataset_keys: List[str]) -> bytes:
    """Return canonical bytes for the dataset key list (for hashing).

    Args:
        dataset_keys: Dataset keys this run will process. Order is
            normalized via `sorted` so positional `kzb solar` and
            `solar kzb` produce the same hash.

    Returns:
        UTF-8 JSON bytes of the sorted key list.
    """
    import json as _json

    return _json.dumps(sorted(dataset_keys), ensure_ascii=False).encode("utf-8")


def _stage_extra(stage: str, stopwords_bytes: bytes, spam_bytes: bytes, dataset_keys_bytes: bytes) -> bytes:
    """Return extra bytes folded into the hash for a stage.

    Corpus stages (exact_dedup, sentence_dedup, statistics) fold in the
    dataset roster so adding or removing a dataset invalidates them.
    Scoped stages (convert, language, spam, quality, repetition) exclude
    the roster so per-dataset work survives roster changes. Stopword file
    contents are folded for the stages that consume them (quality,
    statistics); the spam lexicon/domain contents are folded for the spam
    stage so editing a list invalidates it.

    Args:
        stage: Stage name.
        stopwords_bytes: Raw bytes of the stopword file.
        spam_bytes: Raw bytes of the spam lexicon and domain lists.
        dataset_keys_bytes: Canonical JSON bytes of the sorted roster.

    Returns:
        Bytes to fold into the stage's sentinel hash.
    """
    roster = b"" if is_scoped(stage) else dataset_keys_bytes
    if stage == "spam":
        # Spam is scoped, so `roster` is empty; the lexicon/domain bytes
        # are what make an edited list invalidate the stage.
        return spam_bytes
    if stage in ("quality", "statistics"):
        return stopwords_bytes + b"\x00" + roster if roster else stopwords_bytes
    return roster


def _resolve_requested_stages(stage: str, run_all: bool) -> Tuple[str, ...]:
    """Resolve which stages a run executes.

    A subset run (`run_all` False) with `--stage all` runs only the
    scoped stages and stops before the corpus stages. With `--all`,
    `all` means every stage. An explicit single stage is returned as-is.

    Args:
        stage: The `--stage` value (a stage name or `"all"`).
        run_all: True when `--all` was passed.

    Returns:
        The stage names to execute, in pipeline order.
    """
    if stage != "all":
        return (stage,)
    return STAGE_NAMES if run_all else SCOPED_STAGES


def _apply_force(output_dir: Path, *, stage: str, run_all: bool, dataset_keys: List[str]) -> None:
    """Apply `--force` per the scoped/corpus force matrix.

    Args:
        output_dir: Curation output root.
        stage: The `--stage` value (`"all"` or a stage name).
        run_all: True when `--all` was passed.
        dataset_keys: Keys in play (the full roster under `--all`, else
            the positional subset).
    """
    paths = CuratePaths(input_folder=output_dir, output_dir=output_dir)

    # Whole-corpus reset only when --all is combined with the default
    # (all) stage. A subset `--stage all` must never wipe other datasets.
    if run_all and stage == "all":
        if output_dir.exists():
            for child in output_dir.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        logger.warning("--force: cleared %s", output_dir)
        return

    # Forcing a corpus stage (only reachable under --all): drop the
    # corpus stage folders (data + sentinel) from that stage downstream.
    if stage in CORPUS_STAGES:
        affected = cascade_from(stage)
        for name in affected:
            shutil.rmtree(paths.stage_dir(name), ignore_errors=True)
        if set(affected) & {"exact_dedup", "sentence_dedup"}:
            shutil.rmtree(paths.dedup_state_dir, ignore_errors=True)
        for name in affected:
            shutil.rmtree(_staging_dir(paths, name), ignore_errors=True)
        logger.warning("--force --stage %s: removed %s", stage, list(affected))
        return

    # Forcing scoped work: drop the requested keys' per-dataset sentinels
    # and shard subfolders for the affected scoped stages, then drop the
    # corpus stages' sentinels (their data is rebuilt on the next --all).
    scoped_affected = SCOPED_STAGES if stage == "all" else tuple(s for s in cascade_from(stage) if is_scoped(s))
    for name in scoped_affected:
        folder = paths.stage_dir(name)
        invalidate_dataset_sentinels(folder, dataset_keys)
        for key in dataset_keys:
            shutil.rmtree(folder / key, ignore_errors=True)
    for name in CORPUS_STAGES:
        (paths.stage_dir(name) / SENTINEL_NAME).unlink(missing_ok=True)
    shutil.rmtree(paths.dedup_state_dir, ignore_errors=True)
    shutil.rmtree(output_dir / "_partial", ignore_errors=True)
    logger.warning(
        "--force: reset scoped stages %s for %s + corpus sentinels",
        list(scoped_affected),
        dataset_keys,
    )


def recount(
    *,
    input_dir: Optional[Path],
    output_dir: Optional[Path],
    pretrain_config: Path,
    extract_config: Optional[Path] = None,
) -> None:
    """Backfill true per-source record counts onto existing scoped sentinels.

    Runs that pre-date the per-source fix stamped a shared bucket total into
    every bucket-mate's sentinel, so `records_in`/`records_out` were identical
    across all datasets that ran under one config. This recomputes each
    surviving sentinel's counts from the on-disk per-key shards (via
    `per_key_stage_counts`) and rewrites only the count fields, leaving the
    config hash and completion time untouched so nothing is treated as a fresh
    run. Stages and datasets without a sentinel are skipped, not fabricated.

    Args:
        input_dir: Override for the pretrain config's input_dir, or None.
        output_dir: Override for the pretrain config's output_dir, or None.
        pretrain_config: Path to the curation config. Required, so an
            experiment's own corpus variant is never assumed away.
        extract_config: Path to extract.yaml, or None for the default.
    """
    project_root = _find_project_root()
    pretrain_path = pretrain_config
    extract_path = extract_config or (project_root / "configs" / "data" / "extract.yaml")
    cfg = _load_yaml(pretrain_path)
    input_dir, output_dir = _resolve_dirs(input_dir, output_dir, cfg)
    paths = CuratePaths(input_folder=input_dir, output_dir=output_dir)
    dataset_keys = _list_datasets(extract_path)

    for stage_name in SCOPED_STAGES:
        stage_folder = paths.stage_dir(stage_name)
        keys = [key for key in dataset_keys if dataset_sentinel_path(stage_folder, key).exists()]
        if not keys:
            continue
        per_key = per_key_stage_counts(stage_name, paths, keys)
        for key in keys:
            key_in, key_out = per_key[key]
            update_dataset_sentinel_counts(stage_folder, key, records_in=key_in, records_out=key_out)
            logger.info(
                "[recount] %s/%s records_in=%d records_out=%d",
                stage_name,
                key,
                key_in,
                key_out,
            )


#: Comment block written above the lock file's entries.
_LOCK_HEADER = (
    "# Generated by `curate_pretraining_corpus.py run` (and `status --adopt`); do not edit.\n"
    "# One entry per unit: the lineage its sentinel recorded when it was built.\n"
)

#: Status reason for a unit whose sentinel disagrees with the committed lock file.
LOCK_DIFFERS = "differs from lock"


@dataclass
class _Setup:
    """Everything a run, a status check or an adoption derives from the config.

    Attributes:
        cfg: The parsed pretrain config.
        overrides: The config's per-dataset `overrides:` mapping.
        paths: Resolved curation paths.
        project_root: Repository root.
        lock_path: The lock file beside the pretrain config.
        roster: Every pretraining dataset key in `extract.yaml`.
        stopwords: Loaded stopword set.
        stopwords_raw: Raw stopword bytes folded into config hashes.
        spam_assets: Loaded spam lexicons and domain blocklist.
    """

    cfg: Dict[str, Any]
    overrides: Dict[str, Any]
    paths: CuratePaths
    project_root: Path
    lock_path: Path
    roster: List[str]
    stopwords: Set[str]
    stopwords_raw: bytes
    spam_assets: SpamAssets

    def extra(self, stage: str) -> bytes:
        """Return the extra bytes folded into *stage*'s config hash.

        Args:
            stage: Stage name.

        Returns:
            See `_stage_extra`; corpus stages fold in the roster.
        """
        return _stage_extra(stage, self.stopwords_raw, self.spam_assets.raw_bytes, _dataset_keys_payload(self.roster))

    def expected_hash(self, stage: str, key: Optional[str] = None) -> str:
        """Return the config hash a unit must carry to be current.

        Args:
            stage: Stage name.
            key: Dataset key for a scoped stage; ignored for corpus stages.

        Returns:
            The effective (override-merged) config hash of a scoped unit, or
            the stage slice's hash for a corpus stage.
        """
        if is_scoped(stage):
            return config_hash(
                effective_stage_config(self.cfg, self.overrides, cast(str, key), stage), self.extra(stage)
            )
        return config_hash(_stage_slice(stage, self.cfg), extra=self.extra(stage))

    def include_annotations(self, key: str) -> bool:
        """Return whether convert joins *key*'s annotations sidecar."""
        return bool(effective_stage_config(self.cfg, self.overrides, key, "convert").get("include_annotations", False))


def lock_path_for(pretrain_config: Path) -> Path:
    """Return the lock file that sits beside a curation config.

    Args:
        pretrain_config: Path to the curation config, e.g. `configs/data/curate.yaml`.

    Returns:
        `<stem>.lock.yaml` in the same folder, e.g. `configs/data/curate.lock.yaml`.
    """
    return pretrain_config.with_name(f"{pretrain_config.stem}.lock.yaml")


def _load_setup(
    input_dir: Optional[Path], output_dir: Optional[Path], pretrain_config: Path, extract_config: Optional[Path]
) -> _Setup:
    """Load the config and everything derived from it.

    Args:
        input_dir: Override for the pretrain config's input_dir, or None.
        output_dir: Override for the pretrain config's output_dir, or None.
        pretrain_config: Path to the curation config.
        extract_config: Path to extract.yaml, or None for the default.

    Returns:
        The loaded `_Setup`.
    """
    project_root = _find_project_root()
    extract_path = extract_config or (project_root / "configs" / "data" / "extract.yaml")
    cfg = _load_yaml(pretrain_config)
    resolved_input, resolved_output = _resolve_dirs(input_dir, output_dir, cfg)
    stopwords, stopwords_raw = _load_stopwords(cfg)
    return _Setup(
        cfg=cfg,
        overrides=cfg.get("overrides") or {},
        paths=CuratePaths(input_folder=resolved_input, output_dir=resolved_output),
        project_root=project_root,
        lock_path=lock_path_for(pretrain_config),
        roster=_list_datasets(extract_path),
        stopwords=stopwords,
        stopwords_raw=stopwords_raw,
        spam_assets=_load_spam_assets(cfg),
    )


def _upstream_digest(paths: CuratePaths, stage: str, key: str) -> Optional[str]:
    """Return the document digest the upstream unit of (*stage*, *key*) recorded.

    Args:
        paths: Resolved curation paths.
        stage: A scoped stage after `convert`.
        key: Dataset key.

    Returns:
        The upstream sentinel's document digest, or `None` when it has none.
    """
    sentinel = read_sentinel(paths.stage_dir(cast(str, upstream_stage(stage))) / key)
    return sentinel.document_digest if sentinel is not None else None


def _scoped_reason(
    setup: _Setup, stage: str, key: str
) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
    """Judge whether the unit (*stage*, *key*) is current.

    Args:
        setup: The loaded setup.
        stage: Scoped stage name.
        key: Dataset key.

    Returns:
        Tuple `(reason, input_digest, input_files)`: the stale reason or
        `None`, the unit's input digest as it stands now, and for `convert`
        the source-file descriptions (else `None`).
    """
    folder = setup.paths.stage_dir(stage) / key
    sentinel = read_sentinel(folder)
    files: Optional[Dict[str, Any]] = None
    if stage == "convert":
        previous = sentinel.input_files if sentinel is not None else None
        files = _convert_input_files(setup.paths.input_folder, key, setup.include_annotations(key), previous)
        digest: Optional[str] = _input_files_digest(files)
    else:
        digest = _upstream_digest(setup.paths, stage, key)
    reason = stale_reason(
        sentinel,
        folder,
        expected_hash=setup.expected_hash(stage, key),
        stage_version=STAGE_VERSIONS[stage],
        input_digest=digest,
    )
    return reason, digest, files


def _corpus_inputs(setup: _Setup, stage: str) -> List[str]:
    """Return the roster keys with upstream output that a corpus stage reads.

    Folders left upstream by keys outside the roster (e.g. benchmarks) are
    never read.

    Args:
        setup: The loaded setup.
        stage: Corpus stage name.

    Returns:
        Roster keys, in roster order.
    """
    up_dir = setup.paths.stage_dir(cast(str, upstream_stage(stage)))
    return [key for key in setup.roster if _has_stage_output(up_dir, key)]


def _corpus_input_digest(paths: CuratePaths, stage: str, input_keys: List[str]) -> Optional[str]:
    """Return a corpus stage's input digest.

    Args:
        paths: Resolved curation paths.
        stage: Corpus stage name.
        input_keys: The keys it reads (see `_corpus_inputs`).

    Returns:
        For the first corpus stage, a combination of each input key's
        upstream document digest; for later ones, the upstream stage's
        document digest (`None` when it has none).
    """
    upstream = cast(str, upstream_stage(stage))
    if is_scoped(upstream):
        return combine_named_digests({key: _upstream_digest(paths, stage, key) for key in input_keys})
    sentinel = read_sentinel(paths.stage_dir(upstream))
    return sentinel.document_digest if sentinel is not None else None


def _corpus_reason(setup: _Setup, stage: str, input_keys: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """Judge whether a corpus stage is current.

    Args:
        setup: The loaded setup.
        stage: Corpus stage name.
        input_keys: The keys it reads.

    Returns:
        Tuple `(reason, input_digest)`: the stale reason or `None`, and the
        stage's input digest as it stands now.
    """
    folder = setup.paths.stage_dir(stage)
    digest = _corpus_input_digest(setup.paths, stage, input_keys)
    reason = stale_reason(
        read_sentinel(folder),
        folder,
        expected_hash=setup.expected_hash(stage),
        stage_version=STAGE_VERSIONS[stage],
        input_digest=digest,
    )
    return reason, digest


def _legacy_units(paths: CuratePaths, keys: List[str], corpus: bool) -> List[str]:
    """List units whose sentinel predates lineage tracking.

    Args:
        paths: Resolved curation paths.
        keys: Dataset keys whose scoped units to check.
        corpus: Also check the corpus stages.

    Returns:
        `<stage folder>/<key>` (or `<stage folder>`) of every legacy sentinel.
    """
    found = []
    for stage in STAGE_NAMES:
        if is_scoped(stage):
            folders = [(f"{STAGE_DIRS[stage]}/{key}", paths.stage_dir(stage) / key) for key in keys]
        elif corpus:
            folders = [(STAGE_DIRS[stage], paths.stage_dir(stage))]
        else:
            continue
        for name, folder in folders:
            sentinel = read_sentinel(folder)
            if sentinel is not None and sentinel.is_legacy:
                found.append(name)
    return found


def _integrity_failure(stage: str, errors: Dict[str, str]) -> RuntimeError:
    """Return the error raised when a stage's output fails its integrity check.

    Args:
        stage: Stage name.
        errors: Failure description per dataset key.

    Returns:
        A `RuntimeError` naming every failing dataset.
    """
    detail = "; ".join(f"{key}: {error}" for key, error in errors.items())
    return RuntimeError(f"[{stage}] integrity check failed, previous output kept: {detail}")


def _run_scoped_bucket(
    setup: _Setup,
    stage: str,
    bucket_keys: List[str],
    effective: Dict[str, Any],
    bucket_hash: str,
    inputs: Dict[str, Tuple[Optional[str], Optional[Dict[str, Any]]]],
    workers: int,
    log_dir: Optional[Path],
    info: Dict[str, Any],
) -> Tuple[int, int]:
    """Build one config bucket of a scoped stage into staging, check it, promote it.

    Every unit is checked before any is promoted, so a failure keeps every
    old output and sentinel in place.

    Args:
        setup: The loaded setup.
        stage: Scoped stage name.
        bucket_keys: Dataset keys sharing one effective config.
        effective: That effective config slice.
        bucket_hash: Its config hash.
        inputs: Per key, the input digest and (convert) source-file descriptions.
        workers: Worker count, for the stage and for scanning.
        log_dir: Per-task log folder (convert only).
        info: Informational sentinel fields.

    Returns:
        The bucket's aggregate `(records_in, records_out)` from the stage runner.

    Raises:
        RuntimeError: If any unit fails its integrity check.
    """
    paths = setup.paths
    staging = _staging_dir(paths, stage)
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    upstream = upstream_stage(stage)
    up_dir = paths.stage_dir(upstream) if upstream is not None else None
    view = _filter_stage_subset(up_dir, bucket_keys) if up_dir is not None else None
    # _stage_runner reads cfg[stage], so hand it the bucket's effective slice.
    bucket_cfg = {**setup.cfg, stage: effective}
    try:
        runner = _stage_runner(
            stage,
            paths,
            bucket_cfg,
            workers,
            setup.stopwords,
            setup.spam_assets,
            dataset_keys=bucket_keys,
            input_view=view,
            log_dir=log_dir,
            output_folder=staging,
        )
        counts = runner()
    finally:
        if view is not None:
            shutil.rmtree(view, ignore_errors=True)

    errors: Dict[str, str] = {}
    for key in bucket_keys:
        if up_dir is None:
            in_scan = scan_files([paths.input_folder / f"{key}.jsonl"], id_key="uid", digest=False)
        else:
            in_scan = scan_files(shard_files(up_dir / key), digest=False, workers=workers)
        out_scan = scan_files(shard_files(staging / key), workers=workers)
        error = check_integrity(in_scan, out_scan)
        if error:
            errors[key] = error
            continue
        input_digest, input_files = inputs[key]
        write_sentinel(
            staging / key,
            config_slice=effective,
            config_hash_value=bucket_hash,
            records_in=in_scan.records,
            records_out=out_scan.records,
            stage_version=STAGE_VERSIONS[stage],
            input_digest=input_digest,
            document_digest=out_scan.document_digest,
            input_files=input_files,
            info=info,
        )
    if errors:
        shutil.rmtree(staging, ignore_errors=True)
        raise _integrity_failure(stage, errors)
    for key in bucket_keys:
        _swap_into_place(staging / key, paths.stage_dir(stage) / key)
    shutil.rmtree(staging, ignore_errors=True)
    return counts


def _curate_scoped(
    setup: _Setup, stage: str, dataset_keys: List[str], workers: int, log_dir: Path, info: Dict[str, Any]
) -> None:
    """Rebuild every stale unit of one scoped stage.

    Args:
        setup: The loaded setup.
        stage: Scoped stage name.
        dataset_keys: Keys this run covers.
        workers: Worker count.
        log_dir: Per-task log folder for the convert stage.
        info: Informational sentinel fields.
    """
    paths = setup.paths
    upstream = upstream_stage(stage)
    up_dir = paths.stage_dir(upstream) if upstream is not None else None
    todo: List[str] = []
    inputs: Dict[str, Tuple[Optional[str], Optional[Dict[str, Any]]]] = {}
    no_input: List[str] = []
    for key in dataset_keys:
        # Keys with no input (never extracted, or filtered out upstream) keep
        # whatever they have: an unmounted extracted tier must not wipe output.
        has_input = (
            (paths.input_folder / f"{key}.jsonl").is_file() if up_dir is None else _has_stage_output(up_dir, key)
        )
        if not has_input:
            no_input.append(key)
            continue
        reason, digest, files = _scoped_reason(setup, stage, key)
        if reason:
            logger.info("[%s] %s: %s", stage, key, reason)
            todo.append(key)
            inputs[key] = (digest, files)
    if no_input:
        logger.info("[%s] skipping %d dataset(s) with no input: %s", stage, len(no_input), ", ".join(no_input))
    if not todo:
        logger.info("[%s] all requested datasets current; skipping.", stage)
        return

    # Datasets sharing an effective config run together in one executor.
    extra = setup.extra(stage)
    buckets = _bucket_keys_by_effective_hash(todo, stage, setup.cfg, setup.overrides, extra)
    logger.info("[%s] %d dataset(s) in %d config group(s)", stage, len(todo), len(buckets))
    for bucket_hash, bucket_keys in buckets.items():
        effective = effective_stage_config(setup.cfg, setup.overrides, bucket_keys[0], stage)
        overridden = [k for k in bucket_keys if (setup.overrides.get(k) or {}).get(stage)]
        if overridden:
            logger.info("[%s] override group %s <- %s", stage, overridden, effective)
        if stage == "convert":
            n_datasets, input_bytes = _extracted_input_summary(paths.input_folder, bucket_keys)
            logger.info("[convert] starting (%d dataset(s), %s)", n_datasets, _human_bytes(input_bytes))
        else:
            logger.info("[%s] starting%s", stage, _starting_input_hint(paths, stage))
        records_in, records_out = _run_scoped_bucket(
            setup,
            stage,
            bucket_keys,
            effective,
            bucket_hash,
            inputs,
            workers,
            log_dir if stage == "convert" else None,
            info,
        )
        logger.info(
            "[%s] done for %d dataset(s) (bucket records_in=%d, records_out=%d)",
            stage,
            len(bucket_keys),
            records_in,
            records_out,
        )


def _curate_corpus(setup: _Setup, stage: str, workers: int, info: Dict[str, Any]) -> None:
    """Rebuild a corpus stage when it is stale, resuming an unfinished build.

    Args:
        setup: The loaded setup.
        stage: Corpus stage name.
        workers: Worker count.
        info: Informational sentinel fields.

    Raises:
        RuntimeError: If a dataset's output fails its integrity check.
    """
    paths = setup.paths
    input_keys = _corpus_inputs(setup, stage)
    if not input_keys:
        logger.info("[%s] no datasets with upstream output; skipping.", stage)
        return
    reason, input_digest = _corpus_reason(setup, stage, input_keys)
    if reason is None:
        logger.info("[%s] sentinel current; skipping.", stage)
        return
    logger.info("[%s] %s", stage, reason)
    up_dir = paths.stage_dir(cast(str, upstream_stage(stage)))
    view = _filter_stage_subset(up_dir, input_keys, holder=paths.output_dir / "_inputs" / stage)
    # One task per input shard caps a task's memory at one shard,
    # and fixes the task count across reruns so a crash can resume.
    tasks, shards = _input_fingerprint(view)
    expected_hash = setup.expected_hash(stage)
    resumed = _prepare_corpus_stage(paths, stage, expected_hash, tasks, f"{input_digest}|{shards}")
    logger.info(
        "[%s] %s%s (tasks=%d, workers=%d)",
        stage,
        "resuming" if resumed else "starting",
        _starting_input_hint(paths, stage),
        tasks,
        workers,
    )
    staging = _staging_dir(paths, stage)
    runner = _stage_runner(
        stage,
        paths,
        setup.cfg,
        workers,
        setup.stopwords,
        setup.spam_assets,
        dataset_keys=input_keys,
        input_view=view,
        tasks=tasks,
        output_folder=staging,
    )
    records_in, records_out = runner()

    document_digest: Optional[str] = None
    if stage != "statistics":
        errors: Dict[str, str] = {}
        digests: List[str] = []
        for key in input_keys:
            in_scan = scan_files(shard_files(up_dir / key), digest=False, workers=workers)
            out_scan = scan_files(shard_files(staging / key), workers=workers)
            error = check_integrity(in_scan, out_scan)
            if error:
                errors[key] = error
            digests.append(cast(str, out_scan.document_digest))
        if errors:
            # Resuming would only reproduce the failure, so start fresh next time.
            shutil.rmtree(staging, ignore_errors=True)
            shutil.rmtree(paths.logs_dir(stage), ignore_errors=True)
            raise _integrity_failure(stage, errors)
        document_digest = merge_digests(digests)
    (staging / PROGRESS_NAME).unlink(missing_ok=True)
    write_sentinel(
        staging,
        config_slice=_stage_slice(stage, setup.cfg),
        config_hash_value=expected_hash,
        records_in=records_in,
        records_out=records_out,
        stage_version=STAGE_VERSIONS[stage],
        input_digest=input_digest,
        document_digest=document_digest,
        info=info,
    )
    _swap_into_place(staging, paths.stage_dir(stage))
    if stage in ("exact_dedup", "sentence_dedup"):
        _purge_dedup_state(paths, stage)
    shutil.rmtree(view, ignore_errors=True)
    logger.info("[%s] done (records_in=%d, records_out=%d)", stage, records_in, records_out)


def _lock_entry(sentinel: Sentinel) -> Dict[str, Any]:
    """Return the lock-file entry for one unit's sentinel.

    Args:
        sentinel: The unit's sentinel.

    Returns:
        Its lineage and counts, without informational fields.
    """
    return {
        "config_hash": sentinel.config_hash,
        "stage_version": sentinel.stage_version,
        "input_digest": sentinel.input_digest,
        "document_digest": sentinel.document_digest,
        "records_in": sentinel.records_in,
        "records_out": sentinel.records_out,
    }


def _write_lock(setup: _Setup) -> None:
    """Rewrite the lock file from every roster unit's sentinel on disk.

    Args:
        setup: The loaded setup.
    """
    entries: Dict[str, Any] = {}
    for stage in STAGE_NAMES:
        folder = setup.paths.stage_dir(stage)
        if is_scoped(stage):
            units = {key: read_sentinel(folder / key) for key in setup.roster}
            entries[stage] = {key: _lock_entry(s) for key, s in units.items() if s is not None}
        else:
            sentinel = read_sentinel(folder)
            if sentinel is not None:
                entries[stage] = _lock_entry(sentinel)
    write_lock(setup.lock_path, entries, header=_LOCK_HEADER)
    logger.info("wrote %s", setup.lock_path)


def curate(
    *,
    datasets: List[str],
    run_all: bool,
    stage: str,
    input_dir: Optional[Path],
    output_dir: Optional[Path],
    force: bool,
    workers: int,
    pretrain_config: Path,
    extract_config: Optional[Path] = None,
    mlflow_enabled: Optional[bool] = None,
) -> None:
    """Run the curation pipeline (argv-free entry point).

    Each requested stage rebuilds only its stale units (see `stale_reason`),
    and the lock file beside the config is rewritten after a successful run.

    Args:
        datasets: Positional dataset keys. Must be empty when run_all is True.
        run_all: Process every dataset from the extract config.
        stage: `--stage` value (`"all"` or a stage name).
        input_dir: Override for the pretrain config's input_dir, or None.
        output_dir: Override for the pretrain config's output_dir, or None.
        force: Apply the `--force` reset matrix.
        workers: Worker count (0 = auto).
        pretrain_config: Path to the curation config. Required, so an
            experiment's own corpus variant is never assumed away.
        extract_config: Path to extract.yaml, or None for the default.
        mlflow_enabled: Tri-state override for MLflow pretrain tracking. None
            defers to the pretrain config's `mlflow.enabled`; True/False force it
            on/off. Tracking only runs after a full `--all` build.

    Raises:
        ValueError: If `datasets` is non-empty while `run_all` is True.
        RuntimeError: If a unit still carries a legacy sentinel, or a stage's
            output fails its integrity check.
    """
    if run_all and datasets:
        raise ValueError("datasets must be empty when run_all is True")
    setup = _load_setup(input_dir, output_dir, pretrain_config, extract_config)
    cfg = setup.cfg
    paths = setup.paths
    output_dir = paths.output_dir
    # Validate overrides against the full roster up front so a typo or
    # out-of-bounds section fails before any stage runs.
    validate_overrides(setup.overrides, setup.roster)

    dataset_keys = list(setup.roster) if run_all else list(datasets)
    # `workers` is a CPU budget, not an item count. The convert stage caps
    # it at the dataset count itself (run_convert_stage); the datatrove
    # stages use it as their `tasks` rank count, where shards -- far more
    # numerous than datasets -- are the unit of work. So an explicit
    # --max-workers is honored as-is; auto (0) resolves to cpu_count // 2.
    workers = workers if workers > 0 else max(1, (os.cpu_count() or 2) // 2)
    if run_all:
        logger.info("Running on all %d datasets (workers=%d)", len(dataset_keys), workers)
    else:
        logger.info(
            "Running on %d dataset(s): %s (workers=%d)",
            len(dataset_keys),
            ", ".join(dataset_keys),
            workers,
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    if force:
        _apply_force(output_dir, stage=stage, run_all=run_all, dataset_keys=dataset_keys)

    legacy = _legacy_units(paths, dataset_keys, corpus=run_all)
    if legacy:
        raise RuntimeError(
            f"{len(legacy)} unit(s) carry a sentinel from before lineage tracking (e.g. {legacy[0]}). "
            "Run `curate_pretraining_corpus.py status --adopt` first to adopt them without a rebuild."
        )

    info = _run_info(setup.project_root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    convert_log_dir = setup.project_root / "logs" / Path(__file__).stem / stamp / "convert"
    for stage_name in _resolve_requested_stages(stage, run_all):
        if is_scoped(stage_name):
            _curate_scoped(setup, stage_name, dataset_keys, workers, convert_log_dir, info)
        elif run_all:
            _curate_corpus(setup, stage_name, workers, info)
        else:
            logger.warning("[%s] corpus stage requires --all; skipping.", stage_name)
    _write_lock(setup)

    # Post-hoc MLflow tracking: only after a full build, reflecting the corpus
    # as it now exists on disk (decoupled from which stages ran this time).
    if run_all:
        mlflow_cfg = cfg.get("mlflow") or {}
        enabled = bool(mlflow_cfg.get("enabled", False)) if mlflow_enabled is None else mlflow_enabled
        if enabled:
            from slm4ie.data.curate.tracking import DEFAULT_EXPERIMENT, log_pretrain_run

            log_pretrain_run(
                output_dir,
                cfg,
                enabled=True,
                experiment=mlflow_cfg.get("experiment", DEFAULT_EXPERIMENT),
                tracking_uri=mlflow_cfg.get("tracking_uri"),
                force=force,
            )


@dataclass(frozen=True)
class UnitStatus:
    """Whether one unit is current, as `status` reports it.

    Attributes:
        stage: Stage name.
        dataset: Dataset key, or `None` for a corpus stage.
        state: `current`, `stale` (a run would rebuild it) or `missing`
            (nothing to build it from).
        reason: Why the unit is stale, else `None`.
    """

    stage: str
    dataset: Optional[str]
    state: str
    reason: Optional[str] = None


def _unit_status(
    stage: str,
    dataset: Optional[str],
    sentinel: Optional[Sentinel],
    reason: Optional[str],
    has_input: bool,
    locked: Optional[Dict[str, Any]],
    lock_exists: bool,
) -> UnitStatus:
    """Classify one unit from its sentinel, stale reason and lock entry.

    Args:
        stage: Stage name.
        dataset: Dataset key, or `None` for a corpus stage.
        sentinel: The unit's sentinel, or `None`.
        reason: The unit's stale reason, or `None` when current.
        has_input: Whether there is input to build the unit from.
        locked: The unit's lock-file entry, or `None`.
        lock_exists: Whether a lock file exists at all.

    Returns:
        The unit's `UnitStatus`.
    """
    if sentinel is None:
        return UnitStatus(stage, dataset, "stale", NOT_BUILT) if has_input else UnitStatus(stage, dataset, "missing")
    if reason is None and lock_exists and locked != _lock_entry(sentinel):
        reason = LOCK_DIFFERS
    return UnitStatus(stage, dataset, "stale" if reason else "current", reason)


def status(
    *,
    input_dir: Optional[Path],
    output_dir: Optional[Path],
    pretrain_config: Path,
    extract_config: Optional[Path] = None,
    adopt: bool = False,
    workers: int = 1,
) -> List[UnitStatus]:
    """Report every unit as current, stale (with the reason) or missing.

    Read-only unless *adopt* is set: no stage runs and no data is written.
    Only `convert` may read data, to rehash an extracted file whose size or
    mtime moved since its sentinel was written.

    Args:
        input_dir: Override for the pretrain config's input_dir, or None.
        output_dir: Override for the pretrain config's output_dir, or None.
        pretrain_config: Path to the curation config.
        extract_config: Path to extract.yaml, or None for the default.
        adopt: First adopt legacy sentinels (see `adopt_legacy`) and rewrite
            the lock file.
        workers: Processes used to read shards when adopting.

    Returns:
        One `UnitStatus` per roster unit, in pipeline order.
    """
    setup = _load_setup(input_dir, output_dir, pretrain_config, extract_config)
    if adopt:
        adopt_legacy(setup, workers)
        _write_lock(setup)
    lock = read_lock(setup.lock_path)
    lock_exists = setup.lock_path.is_file()
    paths = setup.paths
    results: List[UnitStatus] = []
    for stage in STAGE_NAMES:
        if is_scoped(stage):
            upstream = upstream_stage(stage)
            for key in setup.roster:
                sentinel = read_sentinel(paths.stage_dir(stage) / key)
                if upstream is None:
                    has_input = (paths.input_folder / f"{key}.jsonl").is_file()
                else:
                    has_input = _has_stage_output(paths.stage_dir(upstream), key)
                reason = _scoped_reason(setup, stage, key)[0] if sentinel is not None else None
                locked = (lock.get(stage) or {}).get(key)
                results.append(_unit_status(stage, key, sentinel, reason, has_input, locked, lock_exists))
        else:
            input_keys = _corpus_inputs(setup, stage)
            sentinel = read_sentinel(paths.stage_dir(stage))
            reason = _corpus_reason(setup, stage, input_keys)[0] if sentinel is not None else None
            results.append(_unit_status(stage, None, sentinel, reason, bool(input_keys), lock.get(stage), lock_exists))
    return results


def _scan_all(
    requests: Dict[Tuple[str, str], Tuple[List[Path], str, bool]], workers: int
) -> Dict[Tuple[str, str], UnitScan]:
    """Scan many units at once, one file per worker task.

    Args:
        requests: Per unit id, its files, id field and whether to digest.
        workers: Worker processes; 1 scans in-process.

    Returns:
        The merged `UnitScan` per unit id.
    """
    total = sum(len(files) for files, _, _ in requests.values())
    logger.info("[adopt] scanning %d file(s) across %d unit(s) (workers=%d)", total, len(requests), workers)
    if workers <= 1:
        return {
            unit: scan_documents(files, id_key=id_key, digest=digest)
            for unit, (files, id_key, digest) in requests.items()
        }
    parts: Dict[Tuple[str, str], List[Future]] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for unit, (files, id_key, digest) in requests.items():
            parts[unit] = [pool.submit(scan_documents, [f], id_key=id_key, digest=digest) for f in files]
        done = 0
        scans: Dict[Tuple[str, str], UnitScan] = {}
        for unit, futures in parts.items():
            scans[unit] = merge_scans([f.result() for f in futures])
            done += len(futures)
            logger.info("[adopt] scanned %s/%s (%d/%d files)", unit[0], unit[1], done, total)
    return scans


def adopt_legacy(setup: _Setup, workers: int = 1) -> None:
    """Give legacy sentinels full lineage by reading their outputs once, without a rebuild.

    A legacy unit whose config hash still matches is read once: its document
    digest is computed and its output is checked against its input. Its
    sentinel is then rewritten with lineage, keeping its completion time. A
    unit that fails the check is recorded with the failure, so the next run
    rebuilds it (and, if its documents change, whatever sits downstream).
    Legacy units whose config no longer matches are left alone: they are
    stale either way. No stage runs and no shard is written.

    Args:
        setup: The loaded setup.
        workers: Worker processes for reading shards.
    """
    paths = setup.paths

    def adoptable(sentinel: Optional[Sentinel], expected: str) -> bool:
        return sentinel is not None and sentinel.is_legacy and sentinel.config_hash == expected

    scoped = [
        (stage, key)
        for stage in SCOPED_STAGES
        for key in setup.roster
        if adoptable(read_sentinel(paths.stage_dir(stage) / key), setup.expected_hash(stage, key))
    ]
    corpus = [s for s in CORPUS_STAGES if adoptable(read_sentinel(paths.stage_dir(s)), setup.expected_hash(s))]
    if not scoped and not corpus:
        logger.info("[adopt] no legacy sentinels to adopt")
        return

    requests: Dict[Tuple[str, str], Tuple[List[Path], str, bool]] = {}

    def need(stage: str, key: str, digest: bool) -> None:
        if stage == "extracted":
            source = paths.input_folder / f"{key}.jsonl"
            files, id_key = ([source] if source.is_file() else []), "uid"
        else:
            files, id_key = shard_files(paths.stage_dir(stage) / key), "id"
        known = requests.get((stage, key))
        requests[(stage, key)] = (files, id_key, digest or (known is not None and known[2]))

    for stage, key in scoped:
        need(stage, key, True)
        need(upstream_stage(stage) or "extracted", key, False)
    for stage in corpus:
        if stage != "statistics":
            for key in _corpus_inputs(setup, stage):
                need(stage, key, True)
                need(cast(str, upstream_stage(stage)), key, False)
    scans = _scan_all(requests, workers)
    info = {**_run_info(setup.project_root), "adopted_at": datetime.now(timezone.utc).isoformat()}

    for stage, key in scoped:
        folder = paths.stage_dir(stage) / key
        legacy = cast(Sentinel, read_sentinel(folder))
        out_scan = scans[(stage, key)]
        upstream = upstream_stage(stage)
        in_scan = scans[(upstream or "extracted", key)]
        files: Optional[Dict[str, Any]] = None
        if upstream is None:
            files = _convert_input_files(paths.input_folder, key, setup.include_annotations(key))
            input_digest: Optional[str] = _input_files_digest(files)
            # Without its source file there is nothing to check the output against.
            error = check_integrity(in_scan, out_scan) if requests[("extracted", key)][0] else None
        else:
            input_digest = _upstream_digest(paths, stage, key)
            error = check_integrity(in_scan, out_scan)
        write_sentinel(
            folder,
            config_slice=legacy.config_slice,
            config_hash_value=legacy.config_hash,
            records_in=in_scan.records,
            records_out=out_scan.records,
            stage_version=STAGE_VERSIONS[stage],
            input_digest=input_digest,
            document_digest=out_scan.document_digest,
            input_files=files,
            integrity_error=error,
            info=info,
            completed_at=legacy.completed_at,
        )
        logger.info("[adopt] %s/%s: %s", STAGE_DIRS[stage], key, error or "ok")

    for stage in corpus:
        folder = paths.stage_dir(stage)
        legacy = cast(Sentinel, read_sentinel(folder))
        input_keys = _corpus_inputs(setup, stage)
        document_digest: Optional[str] = None
        errors: Dict[str, str] = {}
        if stage != "statistics":
            upstream = cast(str, upstream_stage(stage))
            for key in input_keys:
                error = check_integrity(scans[(upstream, key)], scans[(stage, key)])
                if error:
                    errors[key] = error
            document_digest = merge_digests(cast(str, scans[(stage, key)].document_digest) for key in input_keys)
        failure = "; ".join(f"{key}: {error}" for key, error in errors.items()) or None
        write_sentinel(
            folder,
            config_slice=legacy.config_slice,
            config_hash_value=legacy.config_hash,
            records_in=legacy.records_in,
            records_out=legacy.records_out,
            stage_version=STAGE_VERSIONS[stage],
            input_digest=_corpus_input_digest(paths, stage, input_keys),
            document_digest=document_digest,
            integrity_error=failure,
            info=info,
            completed_at=legacy.completed_at,
        )
        logger.info("[adopt] %s: %s", STAGE_DIRS[stage], failure or "ok")
