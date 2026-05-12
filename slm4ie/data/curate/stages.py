"""Stage names, folder mapping, and config-slice helpers for the curate pipeline.

The curation pipeline runs six sequential stages, each producing a
durable on-disk artifact under `<output_dir>/<folder>/`. This module is
the single source of truth that ties together the user-facing CLI name
of each stage (`--stage <name>`), the folder it writes to, and the
top-level `curate.yaml` key(s) whose contents determine its sentinel
hash. Consumers should import from here rather than hard-coding stage
names or folder paths.
"""

from typing import Tuple


#: Stage names in pipeline execution order.
STAGE_NAMES: Tuple[str, ...] = (
    "language",
    "quality",
    "repetition",
    "exact_dedup",
    "sentence_dedup",
    "stats",
)


#: Stage names plus the `"all"` sentinel that the CLI uses for the default
#: "run everything that needs running" mode.
ALL_STAGE_NAMES: Tuple[str, ...] = STAGE_NAMES + ("all",)


#: Mapping from stage name to the folder under `<output_dir>/` it writes.
STAGE_DIRS = {
    "language": "01_language",
    "quality": "02_quality",
    "repetition": "03_repetition",
    "exact_dedup": "04_1_dedup",
    "sentence_dedup": "04_2_dedup",
    "stats": "05_statistics",
}


#: Per-stage top-level YAML keys that go into the sentinel config hash.
_CONFIG_SLICE_KEYS = {
    "language": ("language",),
    "quality": ("quality",),
    "repetition": ("repetition",),
    "exact_dedup": ("exact_dedup",),
    "sentence_dedup": ("sentence_dedup",),
    "stats": ("stats",),
}


def final_corpus_dir() -> str:
    """Return the folder name (under `<output_dir>/`) holding the final corpus.

    Returns:
        The folder name of the final, fully-deduplicated training
        corpus (the output of the `sentence_dedup` stage).
    """
    return STAGE_DIRS["sentence_dedup"]


def stats_dir() -> str:
    """Return the folder name (under `<output_dir>/`) holding statistics output.

    Returns:
        The folder name of the corpus statistics output.
    """
    return STAGE_DIRS["stats"]


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
