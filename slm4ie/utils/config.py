"""Load and validate the tokenizer-sweep configuration.

Parses `configs/tokenizers/tokenizers.yaml` (with the same sibling
`*.local.yaml` deep-merge overlay used elsewhere in the project) into a single
`TokenizerSweepConfig` consumed by both the training and analysis scripts.
Owning the config object here keeps `train.py` and `analysis.py` free of a
shared import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from slm4ie.data.catalog import _deep_merge
from slm4ie.tokenizers.corpus import SampleBudget


@dataclass
class TokenizerSweepConfig:
    """Resolved settings for the tokenizer training + evaluation sweep.

    Attributes:
        corpus_root (Path): Root of the deduplicated pretraining corpus.
        corpus_datasets (List[str]): Dataset subdirs to sample, empty for all.
        train_budget (SampleBudget): Sampling budget for the training corpus.
        tokenizers (List[str]): Registry keys of the tokenizers to train.
        vocab_sizes (List[int]): Vocabulary sizes to sweep.
        special_tokens (List[str]): Reserved tokens for every tokenizer.
        sloleks_path (Path): Path to the converted Sloleks JSONL(.gz).
        min_stem_len (int): Minimum stem length for morpheme derivation.
        sloleks_relations_path (Optional[Path]): Path to the converted Sloleks
            word-relations JSONL(.gz) for the derivational gold, or None to skip
            the derivational track.
        output_root (Path): Directory holding per-run artifact subdirs.
        report_dir (Path): Directory for the comparison report.
        eval_budget (SampleBudget): Sampling budget for the held-out eval set.
        renyi_alpha (float): Alpha for Renyi efficiency.
        mlflow_experiment (str): MLflow experiment name.
        mlflow_enabled (bool): Whether to log to MLflow.
        mlflow_tracking_uri (Optional[str]): MLflow tracking URI override.
        stats_n_resamples (int): Bootstrap resamples (B) for CIs and tests.
        stats_ci_level (float): Confidence level for the bootstrap CIs.
        stats_seed (int): Seed for the shared bootstrap resample indices.
        stats_morph_form_sample (int): Forms drawn for the morph CIs and tests.
    """

    corpus_root: Path
    corpus_datasets: List[str]
    train_budget: SampleBudget
    tokenizers: List[str]
    vocab_sizes: List[int]
    special_tokens: List[str]
    sloleks_path: Path
    min_stem_len: int
    output_root: Path
    report_dir: Path
    eval_budget: SampleBudget
    renyi_alpha: float
    mlflow_experiment: str
    mlflow_enabled: bool
    sloleks_relations_path: Optional[Path] = None
    mlflow_tracking_uri: Optional[str] = None
    stats_n_resamples: int = 2000
    stats_ci_level: float = 0.95
    stats_seed: int = 12345
    stats_morph_form_sample: int = 200000

    @property
    def corpus_sample_path(self) -> Path:
        """Path to the shared cached training sample.

        Returns:
            Path: `<output_root>/corpus_sample.txt.gz`.
        """
        return self.output_root / "corpus_sample.txt.gz"

    @property
    def eval_sample_path(self) -> Path:
        """Path to the shared cached held-out evaluation sample.

        Returns:
            Path: `<output_root>/eval_sample.txt.gz`.
        """
        return self.output_root / "eval_sample.txt.gz"

    @property
    def lexicon_path(self) -> Path:
        """Path to the morpheme table consumed by the morphological backends.

        Holds the inflectional lexicon, or its union with the derivational
        lexicon when the derivational track is configured.

        Returns:
            Path: `<output_root>/morph_lexicon.jsonl.gz`.
        """
        return self.output_root / "morph_lexicon.jsonl.gz"

    @property
    def infl_lexicon_path(self) -> Path:
        """Path to the inflectional-only gold lexicon for the morph metrics.

        Returns:
            Path: `<output_root>/morph_lexicon_infl.jsonl.gz`.
        """
        return self.output_root / "morph_lexicon_infl.jsonl.gz"

    @property
    def deriv_lexicon_path(self) -> Path:
        """Path to the derivational gold lexicon for the derivational metrics.

        Returns:
            Path: `<output_root>/morph_lexicon_deriv.jsonl.gz`.
        """
        return self.output_root / "morph_lexicon_deriv.jsonl.gz"

    def needs_morphology(self) -> bool:
        """Return True when any selected tokenizer needs the morpheme lexicon.

        Returns:
            bool: True if a morphological backend is in the sweep.
        """
        return any(name.startswith("morph") for name in self.tokenizers)

    def needs_derivational(self) -> bool:
        """Return True when a derivational gold source is configured.

        The derivational metrics apply to every tokenizer, so this is
        independent of whether a morphological backend is in the sweep.

        Returns:
            bool: True if `sloleks_relations_path` is set.
        """
        return self.sloleks_relations_path is not None


def _budget_from_dict(raw: Dict[str, Any]) -> SampleBudget:
    """Build a SampleBudget from a config `budget` mapping.

    Args:
        raw (Dict[str, Any]): The `budget` sub-mapping.

    Returns:
        SampleBudget: Populated budget with defaults for missing keys.
    """
    return SampleBudget(
        max_bytes=raw.get("max_bytes"),
        max_docs=raw.get("max_docs"),
        seed=raw.get("seed", 0),
        weight_key=raw.get("weight_key", "dataset"),
        source_weights=dict(raw.get("source_weights", {})),
    )


def load_tokenizer_config(config_path: Path) -> TokenizerSweepConfig:
    """Load the tokenizer-sweep config, applying any local overlay.

    Args:
        config_path (Path): Path to `configs/tokenizers/tokenizers.yaml`.

    Returns:
        TokenizerSweepConfig: The resolved sweep configuration.

    Raises:
        FileNotFoundError: If `config_path` does not exist.
        ValueError: If required fields are missing or malformed.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Tokenizer config not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    local_path = config_path.with_suffix(".local.yaml")
    if local_path.exists():
        local_raw = yaml.safe_load(local_path.read_text(encoding="utf-8")) or {}
        raw = _deep_merge(raw, local_raw)

    corpus = raw.get("corpus") or {}
    morphology = raw.get("morphology") or {}
    output = raw.get("output") or {}
    evaluation = raw.get("eval") or {}
    mlflow_cfg = raw.get("mlflow") or {}
    stats_cfg = raw.get("stats") or {}

    tokenizers = raw.get("tokenizers") or []
    vocab_sizes = raw.get("vocab_sizes") or []
    missing: List[str] = []
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
    if missing:
        raise ValueError(f"Tokenizer config {config_path} missing fields: {', '.join(missing)}")

    output_root = Path(output["root"])
    report_dir = Path(output.get("report_dir") or output_root / "_reports")

    return TokenizerSweepConfig(
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
        mlflow_experiment=mlflow_cfg.get("experiment", "slm4ie/tokenization/slovenian"),
        mlflow_enabled=bool(mlflow_cfg.get("enabled", False)),
        mlflow_tracking_uri=mlflow_cfg.get("tracking_uri"),
        stats_n_resamples=int(stats_cfg.get("n_resamples", 2000)),
        stats_ci_level=float(stats_cfg.get("ci_level", 0.95)),
        stats_seed=int(stats_cfg.get("seed", 12345)),
        stats_morph_form_sample=int(stats_cfg.get("morph_form_sample", 200000)),
    )
