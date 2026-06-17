"""Per-dataset config overrides for the pretrain curation pipeline.

The pretrain pipeline is driven by a single `pretrain.yaml`. Each scoped
stage (convert, language, spam, quality, repetition) consumes the global
section named after it. An optional top-level `overrides:` block lets an
individual dataset deep-merge changes onto those defaults, so a corpus
can tweak one knob without forking the whole config or disturbing the
other datasets.

Only scoped stages are overridable. Corpus stages (exact_dedup,
sentence_dedup, stats) and global keys (input_dir, output_dir,
stopwords) operate over the whole corpus and reject per-dataset
overrides. The block is validated at load time against each stage's
known knob set so a typo fails fast instead of silently no-opping.
"""

from typing import Any, Dict, FrozenSet, List, Optional

from slm4ie.data.catalog import _deep_merge
from slm4ie.data.curate.stages import SCOPED_STAGES

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


def validate_overrides(
    overrides: Optional[Dict[str, Any]], roster: List[str]
) -> None:
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
            raise OverrideConfigError(
                f"overrides: unknown dataset '{dataset}' "
                "(not declared in extract.yaml)"
            )
        if not isinstance(sections, dict):
            raise OverrideConfigError(
                f"overrides.{dataset}: expected a mapping of stage -> knobs"
            )
        for stage, knobs in sections.items():
            if stage not in SCOPED_STAGES:
                raise OverrideConfigError(
                    f"overrides.{dataset}.{stage}: only scoped stages "
                    f"{sorted(SCOPED_STAGES)} may be overridden"
                )
            if not isinstance(knobs, dict):
                raise OverrideConfigError(
                    f"overrides.{dataset}.{stage}: expected a mapping of knob -> value"
                )
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
        cfg: Parsed pretrain.yaml.
        overrides: The `overrides:` mapping, or None/empty.
        dataset: Dataset key.
        stage: Scoped stage name.

    Returns:
        A new mapping of the effective knobs for the dataset and stage.
    """
    base = dict(cfg.get(stage) or {})
    override = ((overrides or {}).get(dataset) or {}).get(stage) or {}
    return _deep_merge(base, override)
