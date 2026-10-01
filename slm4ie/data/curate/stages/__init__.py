"""The stage registry: names, folders, code versions and how to run each stage.

The curation pipeline runs eight sequential stages, each producing a durable
on-disk artifact under `<output_dir>/<folder>/`. Each stage's code lives in one
module in this package (`convert.py`, `language.py`, ..., with both dedup
stages in `dedup.py`), which exposes a runner taking a `StageRun`. This module
ties together the user-facing CLI name of each stage (`--stage <name>`), the
folder it writes to, the config section that drives it, the module that runs
it, and its version — a hash of that module's source. It imports no stage
module, so it stays cheap for consumers that only need names and folders.
"""

from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from slm4ie.data.curate.paths import CuratePaths
    from slm4ie.data.curate.stages.spam import SpamAssets


#: Stage names in pipeline execution order. `convert` is stage 0: it
#: turns extracted `<key>.jsonl` files into datatrove-shaped shards that
#: the remaining stages consume.
STAGE_NAMES: Tuple[str, ...] = (
    "convert",
    "language",
    "spam",
    "quality",
    "repetition",
    "exact_dedup",
    "sentence_dedup",
    "statistics",
)


#: Stages that process one dataset at a time. They honor a dataset
#: subset, write to canonical `<stage_dir>/<dataset>/`, and are tracked
#: by per-dataset sentinels.
SCOPED_STAGES: Tuple[str, ...] = ("convert", "language", "spam", "quality", "repetition")

#: Stages that operate over the whole corpus at once. They read every
#: dataset under their input folder, are tracked by a corpus-level
#: sentinel whose hash includes the dataset roster, and run only under
#: `--all`.
CORPUS_STAGES: Tuple[str, ...] = ("exact_dedup", "sentence_dedup", "statistics")

#: Stage names plus the `"all"` sentinel that the CLI uses for the default
#: "run everything that needs running" mode.
ALL_STAGE_NAMES: Tuple[str, ...] = STAGE_NAMES + ("all",)


#: Mapping from stage name to the folder under `<output_dir>/` it writes.
STAGE_DIRS: Dict[str, str] = {
    "convert": "00_convert",
    "language": "01_language",
    "spam": "02_spam",
    "quality": "03_quality",
    "repetition": "04_repetition",
    "exact_dedup": "05_exact_dedup",
    "sentence_dedup": "06_sentence_dedup",
    "statistics": "07_statistics",
}


#: The module and function that run each stage. A stage's version is the hash
#: of its module's source; imports are not followed.
STAGE_RUNNERS: Dict[str, Tuple[str, str]] = {
    "convert": ("convert", "run"),
    "language": ("language", "run"),
    "spam": ("spam", "run"),
    "quality": ("quality", "run"),
    "repetition": ("repetition", "run"),
    "exact_dedup": ("dedup", "run_exact_dedup"),
    "sentence_dedup": ("dedup", "run_sentence_dedup"),
    "statistics": ("statistics", "run"),
}

#: Folder holding the stage modules.
_PACKAGE_DIR = Path(__file__).parent


def code_version(stage: str, package_dir: Path = _PACKAGE_DIR) -> str:
    """Return a stage's version: a hash of the module that runs it.

    Any edit to that module — a fix, a refactor, a comment — changes the
    version and reruns the stage; a rerun that reproduces the same documents
    leaves downstream stages current.

    Args:
        stage: One of the values in `STAGE_NAMES`.
        package_dir: Folder holding the stage modules.

    Returns:
        `sha256:` hex digest of the stage module's source.
    """
    module, _ = STAGE_RUNNERS[stage]
    return "sha256:" + hashlib.sha256((package_dir / f"{module}.py").read_bytes()).hexdigest()


#: Current version of every stage, computed from its code when imported.
STAGE_VERSIONS: Dict[str, str] = {name: code_version(name) for name in STAGE_NAMES}


@dataclass
class StageRun:
    """Everything one run of a stage needs.

    Attributes:
        paths: Resolved curation paths.
        config: The stage's effective config section.
        workers: Worker count.
        dataset_keys: Datasets to process (convert reads their `<key>.jsonl`).
        output_folder: Folder to write into (the stage's staging folder).
        input_view: Symlink view of the upstream output restricted to the
            datasets being run; `None` for convert.
        tasks: Task count for the corpus stages; defaults to `workers`.
        log_dir: Per-task log folder (convert only).
        stopwords: Stopword set (quality and statistics).
        spam_assets: Spam lexicons and domain blocklist (spam only).
    """

    paths: CuratePaths
    config: Dict[str, Any]
    workers: int
    dataset_keys: List[str]
    output_folder: Path
    input_view: Optional[Path] = None
    tasks: Optional[int] = None
    log_dir: Optional[Path] = None
    stopwords: Set[str] = field(default_factory=set)
    spam_assets: Optional[SpamAssets] = None


def run_stage(stage: str, job: StageRun) -> Tuple[int, int]:
    """Run *stage* on *job*, importing its module only now.

    Args:
        stage: One of the values in `STAGE_NAMES`.
        job: What to run it on.

    Returns:
        The stage's `(records_in, records_out)` document counts.
    """
    module, function = STAGE_RUNNERS[stage]
    runner: Callable[[StageRun], Tuple[int, int]] = getattr(
        importlib.import_module(f"slm4ie.data.curate.stages.{module}"), function
    )
    return runner(job)


#: Per-stage top-level YAML keys that go into the sentinel config hash.
_CONFIG_SLICE_KEYS: Dict[str, Tuple[str, ...]] = {
    "convert": ("convert",),
    "language": ("language",),
    "spam": ("spam",),
    "quality": ("quality",),
    "repetition": ("repetition",),
    "exact_dedup": ("exact_dedup",),
    "sentence_dedup": ("sentence_dedup",),
    "statistics": ("statistics",),
}


def final_corpus_dir() -> str:
    """Return the folder name (under `<output_dir>/`) holding the final corpus.

    Returns:
        The folder name of the final, fully-deduplicated training
        corpus (the output of the `sentence_dedup` stage).
    """
    return STAGE_DIRS["sentence_dedup"]


def statistics_dir() -> str:
    """Return the folder name (under `<output_dir>/`) holding statistics output.

    Returns:
        The folder name of the corpus statistics output.
    """
    return STAGE_DIRS["statistics"]


def config_slice_keys(stage: str) -> Tuple[str, ...]:
    """Return the top-level YAML keys whose contents drive *stage*'s sentinel hash.

    Args:
        stage: One of the values in `STAGE_NAMES`.

    Returns:
        Tuple of `curate.yaml` top-level keys whose values are included
        in the stage's config hash slice. Stopword *file contents* are
        also included for `quality` and `statistics`, but that is handled
        in `config.py` — those files live outside `curate.yaml`.

    Raises:
        KeyError: If *stage* is not a known stage name.
    """
    return _CONFIG_SLICE_KEYS[stage]


def cascade_from(stage: str) -> Tuple[str, ...]:
    """Return *stage* followed by every downstream stage, in execution order.

    Used by `--force` to reset a stage and every stage downstream of it.

    Args:
        stage: One of the values in `STAGE_NAMES`.

    Returns:
        Tuple starting with *stage* and ending with the last pipeline
        stage in execution order.

    Raises:
        KeyError: If *stage* is not a known stage name.
    """
    if stage not in STAGE_NAMES:
        raise KeyError(stage)
    idx = STAGE_NAMES.index(stage)
    return STAGE_NAMES[idx:]


def is_scoped(stage: str) -> bool:
    """Return True if *stage* is a per-dataset scoped stage.

    Args:
        stage: One of the values in `STAGE_NAMES`.

    Returns:
        True for convert/language/quality/repetition; False for the
        corpus-wide dedup/stats stages.

    Raises:
        KeyError: If *stage* is not a known stage name.
    """
    if stage not in STAGE_NAMES:
        raise KeyError(stage)
    return stage in SCOPED_STAGES


def upstream_stage(stage: str) -> Optional[str]:
    """Return the stage that produces *stage*'s input, or None for the first stage.

    Args:
        stage: One of the values in `STAGE_NAMES`.

    Returns:
        The preceding stage's name, or `None` when *stage* is the first
        stage (`convert`) and therefore has no upstream stage.

    Raises:
        KeyError: If *stage* is not a known stage name.
    """
    if stage not in STAGE_NAMES:
        raise KeyError(stage)
    idx = STAGE_NAMES.index(stage)
    return STAGE_NAMES[idx - 1] if idx > 0 else None
