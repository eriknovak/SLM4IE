"""Stage names, folder mapping, and config-slice helpers for the curate pipeline.

The curation pipeline runs eight sequential stages, each producing a
durable on-disk artifact under `<output_dir>/<folder>/`. This module is
the single source of truth that ties together the user-facing CLI name
of each stage (`--stage <name>`), the folder it writes to, and the
top-level pretrain-config key(s) whose contents determine its sentinel
hash. Consumers should import from here rather than hard-coding stage
names or folder paths.
"""

import ast
import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple


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


#: The code that runs each stage, as `<module>.py` (the whole file) or
#: `<module>.py:<name>` (one top-level function or class in a shared module).
#: Imports are not followed: a stage's version covers only these sources.
STAGE_SOURCES: Dict[str, Tuple[str, ...]] = {
    "convert": ("convert.py", "runner.py:_build_convert_params"),
    "language": ("language.py", "pipeline.py:build_language_executors", "runner.py:_build_language_params"),
    "spam": ("spam.py", "pipeline.py:build_spam_executors", "runner.py:_build_spam_config"),
    "quality": ("pipeline.py:QualityConfig", "pipeline.py:build_quality_executors", "runner.py:_build_quality_config"),
    "repetition": ("pipeline.py:build_repetition_executors",),
    "exact_dedup": ("dedup.py", "pipeline.py:build_exact_dedup_executors"),
    "sentence_dedup": ("dedup.py", "pipeline.py:build_sentence_dedup_executors"),
    "statistics": ("stats.py", "pipeline.py:build_statistics_executors"),
}

#: Folder holding the modules `STAGE_SOURCES` names.
_PACKAGE_DIR = Path(__file__).parent


def _source_text(package_dir: Path, source: str) -> str:
    """Return the text of one `STAGE_SOURCES` entry.

    Args:
        package_dir: Folder holding the curate modules.
        source: `<module>.py` or `<module>.py:<name>`.

    Returns:
        The whole file, or the named top-level definition's source.

    Raises:
        KeyError: If the named definition is not in the module.
    """
    filename, _, name = source.partition(":")
    text = (package_dir / filename).read_text(encoding="utf-8")
    if not name:
        return text
    for node in ast.parse(text).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            return ast.get_source_segment(text, node) or ""
    raise KeyError(f"{name} not found in {filename}")


def code_version(stage: str, package_dir: Path = _PACKAGE_DIR) -> str:
    """Return a stage's version: a hash of the code that runs it.

    Any edit to that code — a fix, a refactor, a comment — changes the version
    and reruns the stage; a rerun that reproduces the same documents leaves
    downstream stages current.

    Args:
        stage: One of the values in `STAGE_NAMES`.
        package_dir: Folder holding the curate modules.

    Returns:
        `sha256:` hex digest over the stage's sources, in `STAGE_SOURCES` order.
    """
    h = hashlib.sha256()
    for source in STAGE_SOURCES[stage]:
        h.update(source.encode("utf-8") + b"\x00" + _source_text(package_dir, source).encode("utf-8") + b"\x00")
    return "sha256:" + h.hexdigest()


#: Current version of every stage, computed from its code when imported.
STAGE_VERSIONS: Dict[str, str] = {name: code_version(name) for name in STAGE_NAMES}


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
        also included for `quality` and `stats`, but that's handled by
        the sentinel module — those keys live outside `curate.yaml`.

    Raises:
        KeyError: If *stage* is not a known stage name.
    """
    return _CONFIG_SLICE_KEYS[stage]


def cascade_from(stage: str) -> Tuple[str, ...]:
    """Return *stage* followed by every downstream stage, in execution order.

    Used by the sentinel runner to cascade-invalidate downstream stages
    when *stage*'s config has changed.

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
