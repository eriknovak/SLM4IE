"""Load and validate the tokenizer-sweep configuration.

Parses the sweep config passed to it (with the same sibling
`*.local.yaml` deep-merge overlay used elsewhere in the project) into a single
`SweepConfig` consumed by both the training and evaluation code.
Owning the config object here keeps `train.py` and `evaluate.py` free of a
shared import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from slm4ie.tokenizers.corpus import SampleBudget
from slm4ie.utils.config import load_yaml

# ===============================
# Sweep config
# ===============================


@dataclass
class SweepConfig:
    """Resolved settings for the tokenizer training + evaluation sweep.

    Attributes:
        corpus_root (Path): Root of the deduplicated pretraining corpus.
        corpus_datasets (list[str]): Dataset subdirs to sample, empty for all.
        train_budget (SampleBudget): Sampling budget for the training corpus.
        tokenizers (list[str]): Registry keys of the tokenizers to train.
        vocab_sizes (list[int]): Vocabulary sizes to sweep.
        special_tokens (list[str]): Reserved tokens for every tokenizer.
        character_coverage (float): Share of the training sample's character
            occurrences the character-level backends' base alphabet covers;
            1.0 keeps every character.
        sloleks_path (Path): Path to the converted Sloleks JSONL(.gz).
        min_stem_len (int): Minimum stem length for morpheme derivation.
        sloleks_relations_path (Path | None): Path to the converted Sloleks
            word-relations JSONL(.gz) for the derivational gold, or None to skip
            the derivational track.
        output_root (Path): Directory holding per-run artifact subdirs.
        report_dir (Path): Directory for the comparison report.
        eval_budget (SampleBudget): Sampling budget for the held-out eval set.
        renyi_alpha (float): Alpha for Renyi efficiency.
        mlflow_experiment (str): MLflow experiment name.
        mlflow_enabled (bool): Whether to log to MLflow.
        mlflow_tracking_uri (str | None): MLflow tracking URI override.
        mlflow_tags (dict[str, str]): Tags set on every train and eval run,
            such as the experiment's `branch`, `type` and `language`.
        stats_n_resamples (int): Bootstrap resamples (B) for CIs and tests.
        stats_ci_level (float): Confidence level for the bootstrap CIs.
        stats_seed (int): Seed for the shared bootstrap resample indices.
        stats_morph_form_sample (int): Forms drawn for the morph CIs and tests.
        corpus_train_share (float | None): Share of each source's documents on
            the training side of the corpus split, or None to sample the whole
            corpus for both training and evaluation.
        inflection_path (Path | None): Path to the converted Sloleks
            stem-ending split, whose lexemes are split for held-out scoring.
        heldout_lexeme_share (float | None): Share of those lexemes held out
            of the morpheme table, or None to hold none out.
    """

    corpus_root: Path
    corpus_datasets: list[str]
    train_budget: SampleBudget
    tokenizers: list[str]
    vocab_sizes: list[int]
    special_tokens: list[str]
    sloleks_path: Path
    min_stem_len: int
    output_root: Path
    report_dir: Path
    eval_budget: SampleBudget
    renyi_alpha: float
    mlflow_experiment: str
    mlflow_enabled: bool
    sloleks_relations_path: Path | None = None
    mlflow_tracking_uri: str | None = None
    stats_n_resamples: int = 2000
    stats_ci_level: float = 0.95
    stats_seed: int = 12345
    stats_morph_form_sample: int = 200000
    corpus_train_share: float | None = None
    inflection_path: Path | None = None
    heldout_lexeme_share: float | None = None
    mlflow_tags: dict[str, str] = field(default_factory=dict)
    character_coverage: float = 1.0

    @property
    def corpus_split_path(self) -> Path:
        """Path to the persisted corpus split.

        Returns:
            `<output_root>/corpus_split.json`.
        """
        return self.output_root / "corpus_split.json"

    @property
    def corpus_sample_path(self) -> Path:
        """Path to the shared cached training sample.

        Returns:
            `<output_root>/corpus_sample.txt.gz`.
        """
        return self.output_root / "corpus_sample.txt.gz"

    @property
    def eval_sample_path(self) -> Path:
        """Path to the shared cached held-out evaluation sample.

        Returns:
            `<output_root>/eval_sample.txt.gz`.
        """
        return self.output_root / "eval_sample.txt.gz"

    @property
    def lexicon_path(self) -> Path:
        """Path to the morpheme table consumed by the morphological backends.

        Holds the inflectional lexicon, or its union with the derivational
        lexicon when the derivational track is configured.

        Returns:
            `<output_root>/morph_lexicon.jsonl.gz`.
        """
        return self.output_root / "morph_lexicon.jsonl.gz"

    @property
    def infl_lexicon_path(self) -> Path:
        """Path to the inflectional-only gold lexicon for the morph metrics.

        Returns:
            `<output_root>/morph_lexicon_infl.jsonl.gz`.
        """
        return self.output_root / "morph_lexicon_infl.jsonl.gz"

    @property
    def deriv_lexicon_path(self) -> Path:
        """Path to the derivational gold lexicon for the derivational metrics.

        Returns:
            `<output_root>/morph_lexicon_deriv.jsonl.gz`.
        """
        return self.output_root / "morph_lexicon_deriv.jsonl.gz"

    def needs_morphology(self) -> bool:
        """Return True when any selected tokenizer needs the morpheme lexicon.

        Returns:
            True if a morphological backend is in the sweep.
        """
        return any(name.startswith("morph") for name in self.tokenizers)

    def needs_derivational(self) -> bool:
        """Return True when a derivational gold source is configured.

        The derivational metrics apply to every tokenizer, so this is
        independent of whether a morphological backend is in the sweep.

        Returns:
            True if `sloleks_relations_path` is set.
        """
        return self.sloleks_relations_path is not None


# ===============================
# Loading
# ===============================


def _budget_from_dict(raw: dict[str, Any]) -> SampleBudget:
    """Build a SampleBudget from a config `budget` mapping.

    Args:
        raw: The `budget` sub-mapping.

    Returns:
        Populated budget with defaults for missing keys.
    """
    return SampleBudget(
        max_bytes=raw.get("max_bytes"),
        max_docs=raw.get("max_docs"),
        seed=raw.get("seed", 0),
        weight_key=raw.get("weight_key", "dataset"),
        source_weights=dict(raw.get("source_weights", {})),
    )


def load_sweep_config(config_path: Path) -> SweepConfig:
    """Load the tokenizer-sweep config, applying any local overlay.

    Args:
        config_path: Path to a sweep config YAML.

    Returns:
        The resolved sweep configuration.

    Raises:
        FileNotFoundError: If `config_path` does not exist.
        ValueError: If required fields are missing or malformed.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Tokenizer config not found: {config_path}")

    # load the config sections
    raw = load_yaml(config_path)

    corpus = raw.get("corpus") or {}
    morphology = raw.get("morphology") or {}
    output = raw.get("output") or {}
    evaluation = raw.get("eval") or {}
    mlflow_cfg = raw.get("mlflow") or {}
    stats_cfg = raw.get("stats") or {}

    tokenizers = raw.get("tokenizers") or []
    vocab_sizes = raw.get("vocab_sizes") or []

    # validate the required fields
    missing: list[str] = []
    if not corpus.get("root"):
        missing.append("corpus.root")
    if not tokenizers:
        missing.append("tokenizers")
    if not vocab_sizes:
        missing.append("vocab_sizes")
    if not morphology.get("sloleks_path"):
        missing.append("morphology.sloleks_path")
    if not output.get("root"):
        missing.append("output.root")
    if morphology.get("heldout_lexeme_share") is not None and not morphology.get("inflection_path"):
        missing.append("morphology.inflection_path (needed by morphology.heldout_lexeme_share)")
    if missing:
        raise ValueError(f"Tokenizer config {config_path} missing fields: {', '.join(missing)}")
    character_coverage = raw.get("character_coverage", 1.0)
    if not isinstance(character_coverage, (int, float)) or not 0.0 < character_coverage <= 1.0:
        raise ValueError(f"Tokenizer config {config_path}: character_coverage {character_coverage!r} outside (0, 1]")

    # resolve the output paths
    output_root = Path(output["root"])
    report_dir = Path(output.get("report_dir") or output_root / "_reports")

    # build the sweep config
    return SweepConfig(
        corpus_root=Path(corpus["root"]),
        corpus_datasets=list(corpus.get("datasets") or []),
        train_budget=_budget_from_dict(corpus.get("budget") or {}),
        tokenizers=list(tokenizers),
        vocab_sizes=[int(v) for v in vocab_sizes],
        special_tokens=list(raw.get("special_tokens") or []),
        sloleks_path=Path(morphology["sloleks_path"]),
        min_stem_len=int(morphology.get("min_stem_len", 2)),
        sloleks_relations_path=(
            Path(morphology["sloleks_relations_path"]) if morphology.get("sloleks_relations_path") else None
        ),
        output_root=output_root,
        report_dir=report_dir,
        eval_budget=_budget_from_dict(evaluation.get("corpus_budget") or {}),
        renyi_alpha=float(evaluation.get("renyi_alpha", 2.5)),
        mlflow_experiment=mlflow_cfg.get("experiment", "slm4ie/tokenizers/sweep"),
        mlflow_enabled=bool(mlflow_cfg.get("enabled", False)),
        mlflow_tracking_uri=mlflow_cfg.get("tracking_uri"),
        stats_n_resamples=int(stats_cfg.get("n_resamples", 2000)),
        stats_ci_level=float(stats_cfg.get("ci_level", 0.95)),
        stats_seed=int(stats_cfg.get("seed", 12345)),
        stats_morph_form_sample=int(stats_cfg.get("morph_form_sample", 200000)),
        corpus_train_share=_optional_float((corpus.get("split") or {}).get("train_share")),
        inflection_path=Path(morphology["inflection_path"]) if morphology.get("inflection_path") else None,
        heldout_lexeme_share=_optional_float(morphology.get("heldout_lexeme_share")),
        mlflow_tags={str(k): str(v) for k, v in (mlflow_cfg.get("tags") or {}).items()},
        character_coverage=float(character_coverage),
    )


def _optional_float(value: Any) -> float | None:
    """Return `value` as a float, or None when it is unset.

    Args:
        value: A config value, possibly None.

    Returns:
        The float value, or None.
    """
    return None if value is None else float(value)
