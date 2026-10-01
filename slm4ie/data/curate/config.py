"""Loading the curation config: paths, roster, per-dataset overrides, config hashes.

The pipeline is driven by a single curation config (`configs/data/curate.yaml`
or an experiment's own variant). `load_setup` reads it once into a `Setup`
holding everything a run, a status check or an adoption derives from it: the
resolved paths, the dataset roster from `extract.yaml`, the stopword and spam
assets, the lock file's location, and each unit's expected config hash.

Each scoped stage (convert, language, spam, quality, repetition) consumes the
global section named after it. An optional top-level `overrides:` block lets an
individual dataset deep-merge changes onto those defaults, so a corpus can
tweak one knob without forking the whole config or disturbing the other
datasets. Only scoped stages are overridable: corpus stages (exact_dedup,
sentence_dedup, statistics) and global keys (input_dir, output_dir, stopwords)
operate over the whole corpus and reject per-dataset overrides. The block is
validated at load time against each stage's known knob set, so a typo fails
fast instead of silently doing nothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, List, Optional, Set, Tuple, cast

import yaml

from slm4ie.utils.config import deep_merge
from slm4ie.data.curate.paths import CuratePaths
from slm4ie.data.curate.stages import SCOPED_STAGES, is_scoped
from slm4ie.utils.io import find_project_root as _find_project_root, resolve_project_path
from slm4ie.data.curate.resources.stopwords import load_stopwords
from slm4ie.utils.versioning import config_hash

if TYPE_CHECKING:
    from slm4ie.data.curate.stages.spam import SpamAssets

logger = logging.getLogger(__name__)

#: Knobs each scoped stage accepts as an override. The quality and spam
#: sets mirror `QualityConfig` / `SpamConfig`; `test_curate_overrides.py`
#: asserts they stay in lockstep. `repetition` exposes no knobs today, so
#: it is effectively non-overridable until some are surfaced.
STAGE_KNOBS: Dict[str, FrozenSet[str]] = {
    "convert": frozenset(
        {
            "text_field",
            "id_field",
            "metadata_fields",
            "include_annotations",
            "max_shard_bytes",
        }
    ),
    "language": frozenset(
        {
            "targets",
            "candidates",
            "mode",
            "minimum_relative_distance",
            "low_accuracy",
            "max_chars",
        }
    ),
    "spam": frozenset(
        {
            "min_adult_hits",
            "min_spam_hits",
            "keep_fraction",
            "default_language",
            "url_blocklist",
            "use_ldnoobw",
            "model",
            "model_threshold",
        }
    ),
    "quality": frozenset(
        {
            "min_doc_words",
            "max_doc_words",
            "min_avg_word_length",
            "max_avg_word_length",
            "max_symbol_word_ratio",
            "max_bullet_lines_ratio",
            "max_ellipsis_lines_ratio",
            "max_non_alpha_words_ratio",
            "min_stop_words",
        }
    ),
    "repetition": frozenset(),
}


class OverrideConfigError(ValueError):
    """Raised when the `overrides:` block is malformed or out of bounds."""


def validate_overrides(overrides: Optional[Dict[str, Any]], roster: List[str]) -> None:
    """Validate the `overrides:` block against the dataset roster.

    Args:
        overrides: The parsed `overrides:` mapping (dataset to stage to
            knobs), or None/empty when absent.
        roster: Every dataset key declared in extract.yaml.

    Raises:
        OverrideConfigError: If a dataset key is not in `roster`, a
            section is not a scoped stage, a knob is unknown for its
            stage, or a section/knob value has the wrong shape.
    """
    if not overrides:
        return
    roster_set = set(roster)
    for dataset, sections in overrides.items():
        if dataset not in roster_set:
            raise OverrideConfigError(f"overrides: unknown dataset '{dataset}' (not declared in extract.yaml)")
        if not isinstance(sections, dict):
            raise OverrideConfigError(f"overrides.{dataset}: expected a mapping of stage -> knobs")
        for stage, knobs in sections.items():
            if stage not in SCOPED_STAGES:
                raise OverrideConfigError(
                    f"overrides.{dataset}.{stage}: only scoped stages {sorted(SCOPED_STAGES)} may be overridden"
                )
            if not isinstance(knobs, dict):
                raise OverrideConfigError(f"overrides.{dataset}.{stage}: expected a mapping of knob -> value")
            unknown = set(knobs) - STAGE_KNOBS[stage]
            if unknown:
                raise OverrideConfigError(
                    f"overrides.{dataset}.{stage}: unknown knob(s) "
                    f"{sorted(unknown)}; allowed: {sorted(STAGE_KNOBS[stage])}"
                )


def effective_stage_config(
    cfg: Dict[str, Any],
    overrides: Optional[Dict[str, Any]],
    dataset: str,
    stage: str,
) -> Dict[str, Any]:
    """Return *dataset*'s effective config for *stage*.

    Deep-merges the dataset's stage override (if any) over the global
    stage section. A dataset with no override yields a fresh copy of the
    global slice, byte-identical in content to the pre-overrides
    behavior.

    Args:
        cfg: The parsed pretrain config.
        overrides: The `overrides:` mapping, or None/empty.
        dataset: Dataset key.
        stage: Scoped stage name.

    Returns:
        A new mapping of the effective knobs for the dataset and stage.
    """
    base = dict(cfg.get(stage) or {})
    override = ((overrides or {}).get(dataset) or {}).get(stage) or {}
    return deep_merge(base, override)


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

    Thin wrapper over `slm4ie.data.curate.resources.stopwords.load_stopwords`. Reads the
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
            bundled list under `slm4ie/data/curate/resources/stopwords/`.
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
            `slm4ie/data/curate/resources/spam/`.
    """
    from slm4ie.data.curate.stages.spam import load_spam_assets

    scfg = cfg.get("spam") or {}
    languages = scfg.get("languages") or []
    url_blocklist = bool(scfg.get("url_blocklist", True))
    return load_spam_assets(languages, url_blocklist=url_blocklist)


def stage_slice(stage: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return the config slice that drives *stage*'s sentinel hash.

    Args:
        stage: Stage name.
        cfg: The parsed pretrain config.

    Returns:
        The mapping under the stage's top-level YAML key, or `{}` if
        absent.
    """
    return dict(cfg.get(stage) or {})


def bucket_keys_by_effective_hash(
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


@dataclass
class Setup:
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
        return config_hash(stage_slice(stage, self.cfg), extra=self.extra(stage))

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


def load_setup(
    input_dir: Optional[Path], output_dir: Optional[Path], pretrain_config: Path, extract_config: Optional[Path]
) -> Setup:
    """Load the config and everything derived from it.

    Args:
        input_dir: Override for the pretrain config's input_dir, or None.
        output_dir: Override for the pretrain config's output_dir, or None.
        pretrain_config: Path to the curation config.
        extract_config: Path to extract.yaml, or None for the default.

    Returns:
        The loaded `Setup`.
    """
    project_root = _find_project_root()
    extract_path = extract_config or (project_root / "configs" / "data" / "extract.yaml")
    cfg = _load_yaml(pretrain_config)
    resolved_input, resolved_output = _resolve_dirs(input_dir, output_dir, cfg)
    stopwords, stopwords_raw = _load_stopwords(cfg)
    return Setup(
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
