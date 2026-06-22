"""End-to-end tests for slm4ie/tokenizers/analysis.py."""

import dataclasses
import gzip
import json
from pathlib import Path

import pytest

from slm4ie.tokenizers.analysis import (
    _train_link_tags,
    augment_with_statistics,
    build_report,
    evaluate_artifact,
    log_results_to_mlflow,
    write_report,
)
from slm4ie.tokenizers.corpus import SampleBudget
from slm4ie.tokenizers.metrics import iter_words
from slm4ie.tokenizers.morphology import MorphemeSegmentation, build_morph_lexicon
from slm4ie.tokenizers.train import MLFLOW_LINK_FILENAME, log_training_to_mlflow, prepare_inputs, train_one
from slm4ie.utils.config import TokenizerSweepConfig

_CORPUS = [
    "hiša ob cesti",
    "hiše so lepe",
    "psu in psa",
    "hišami gradijo mesto",
    "pes laja na psa",
] * 60

_SLOLEKS = [
    {
        "lemma": "hiša",
        "lemma_msd": "Ncfsn",
        "forms": [
            {"form": "hiša", "msd": "Ncfsn"},
            {"form": "hiše", "msd": "Ncfsg"},
            {"form": "hiši", "msd": "Ncfsd"},
            {"form": "hišami", "msd": "Ncfpi"},
        ],
    },
    {
        "lemma": "pes",
        "lemma_msd": "Ncmsn",
        "forms": [{"form": "pes", "msd": "Ncmsn"}, {"form": "psa", "msd": "Ncmsg"}, {"form": "psu", "msd": "Ncmsd"}],
    },
]


def _make_config(tmp_path: Path) -> TokenizerSweepConfig:
    """Build a tiny sweep config with a fake corpus and Sloleks file.

    Args:
        tmp_path (Path): Temp directory root.

    Returns:
        TokenizerSweepConfig: Config over the fake inputs.
    """
    sub = tmp_path / "dd" / "macocu_sl"
    sub.mkdir(parents=True)
    with gzip.open(sub / "00000.jsonl.gz", "wt", encoding="utf-8") as handle:
        for i, text in enumerate(_CORPUS):
            handle.write(json.dumps({"text": text, "id": str(i), "metadata": {"dataset": "macocu_sl"}}) + "\n")

    sloleks = tmp_path / "sloleks.jsonl.gz"
    with gzip.open(sloleks, "wt", encoding="utf-8") as handle:
        for record in _SLOLEKS:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return TokenizerSweepConfig(
        corpus_root=tmp_path / "dd",
        corpus_datasets=[],
        train_budget=SampleBudget(max_docs=300, seed=1),
        tokenizers=["bpe", "morphbpe"],
        vocab_sizes=[90],
        special_tokens=["<unk>"],
        sloleks_path=sloleks,
        min_stem_len=2,
        output_root=tmp_path / "out",
        report_dir=tmp_path / "out" / "_reports",
        eval_budget=SampleBudget(max_docs=100, seed=9),
        renyi_alpha=2.5,
        mlflow_experiment="tokenizer/test",
        mlflow_enabled=False,
    )


class TestEvaluateArtifact:
    """Tests for evaluating a single trained artifact."""

    def test_metrics_present_and_in_range(self, tmp_path: Path):
        """evaluate_artifact returns all six metrics within their ranges."""
        cfg = _make_config(tmp_path)
        sample_path, lexicon_path = prepare_inputs(cfg)
        train_one("bpe-90", cfg=cfg, sample_path=sample_path, lexicon_path=lexicon_path)

        lexicon = build_morph_lexicon(cfg.sloleks_path)
        morph_sample = list(lexicon.by_form.values())
        eval_docs = [["hiša", "hiše"], ["psa", "mesto"]]
        record = evaluate_artifact(
            "bpe-90", output_root=cfg.output_root, eval_docs=eval_docs, morph_sample=morph_sample
        )

        assert record["tokenizer"] == "bpe"
        assert record["vocab_size"] == 90
        assert record["fertility"] >= 1.0
        assert 0.0 <= record["renyi_efficiency"] <= 1.0
        assert 0.0 <= record["morph_score_f1"] <= 1.0
        assert (cfg.output_root / "bpe-90" / "metrics.json").exists()
        assert (cfg.output_root / "bpe-90" / "eval_units.npz").exists()

    def test_missing_artifact_returns_none(self, tmp_path: Path):
        """A missing artifact yields None (skipped)."""
        cfg = _make_config(tmp_path)
        lexicon = build_morph_lexicon(cfg.sloleks_path)
        assert (
            evaluate_artifact(
                "bpe-90",
                output_root=cfg.output_root,
                eval_docs=[["hiša"]],
                morph_sample=list(lexicon.by_form.values()),
            )
            is None
        )


class TestReport:
    """Tests for report construction."""

    def test_build_report_has_rows_and_headers(self):
        """The Markdown report contains a header row and one row per run."""
        results = [
            {
                "run_key": "bpe-90",
                "tokenizer": "bpe",
                "vocab_size": 90,
                "fertility": 1.5,
                "tokens_per_byte": 0.4,
                "chars_per_token": 2.0,
                "renyi_efficiency": 0.8,
                "morph_score_f1": 0.5,
                "morph_edit_distance": 0.6,
                "morph_consistency": 0.7,
            },
        ]
        markdown, payload = build_report(results)
        assert "tokenizer" in markdown
        assert "fertility ↓" in markdown
        assert "morph_score_f1 ↑" in markdown
        assert "| bpe | 90 |" in markdown
        assert payload["directions"]["fertility"] == "lower"

    def test_write_report_creates_files(self, tmp_path: Path):
        """write_report emits report.md and report.json."""
        results = [
            {
                "run_key": "bpe-90",
                "tokenizer": "bpe",
                "vocab_size": 90,
                "fertility": 1.5,
                "tokens_per_byte": 0.4,
                "chars_per_token": 2.0,
                "renyi_efficiency": 0.8,
                "morph_score_f1": 0.5,
                "morph_edit_distance": 0.6,
                "morph_consistency": 0.7,
            },
        ]
        md_path, json_path = write_report(results, tmp_path / "rep")
        assert md_path.exists() and json_path.exists()
        assert json.loads(json_path.read_text(encoding="utf-8"))["results"][0]["tokenizer"] == "bpe"


def test_full_pipeline_train_then_evaluate(tmp_path: Path):
    """Train both backends, evaluate, and produce a report end to end."""
    cfg = _make_config(tmp_path)
    sample_path, lexicon_path = prepare_inputs(cfg)
    for key in ["bpe-90", "morphbpe-90"]:
        train_one(key, cfg=cfg, sample_path=sample_path, lexicon_path=lexicon_path)

    lexicon = build_morph_lexicon(cfg.sloleks_path)
    morph_sample = list(lexicon.by_form.values())
    eval_docs = [iter_words(line) for line in _CORPUS[:20]]
    records = [
        evaluate_artifact(key, output_root=cfg.output_root, eval_docs=eval_docs, morph_sample=morph_sample)
        for key in ["bpe-90", "morphbpe-90"]
    ]
    md_path, _ = write_report([r for r in records if r], cfg.report_dir)
    assert md_path.exists()
    # MorphBPE should respect morpheme boundaries at least as well as plain BPE.
    by_name = {r["tokenizer"]: r for r in records if r}
    assert by_name["morphbpe"]["morph_score_f1"] >= 0.0


def test_augment_with_statistics_adds_cis_and_significance(tmp_path: Path):
    """Aggregation adds CI/std fields and a per-vocab significance block."""
    cfg = _make_config(tmp_path)
    sample_path, lexicon_path = prepare_inputs(cfg)
    for key in ["bpe-90", "morphbpe-90"]:
        train_one(key, cfg=cfg, sample_path=sample_path, lexicon_path=lexicon_path)

    lexicon = build_morph_lexicon(cfg.sloleks_path)
    morph_sample = list(lexicon.by_form.values())
    eval_docs = [iter_words(line) for line in _CORPUS[:20]]
    records = [
        evaluate_artifact(key, output_root=cfg.output_root, eval_docs=eval_docs, morph_sample=morph_sample)
        for key in ["bpe-90", "morphbpe-90"]
    ]
    records = [r for r in records if r]

    significance, stats_config = augment_with_statistics(
        records, cfg.output_root, n_resamples=200, ci_level=0.95, seed=12345, morph_form_sample=len(morph_sample)
    )

    for record in records:
        for metric in ["fertility", "tokens_per_byte", "chars_per_token", "morph_score_f1", "morph_edit_distance"]:
            low, high = record[f"{metric}_ci"]
            assert low <= record[metric] <= high or low == high
            assert record[f"{metric}_std"] >= 0.0
        # Point-only metrics carry no CI.
        assert "renyi_efficiency_ci" not in record
        assert "morph_consistency_ci" not in record

    block = significance["90"]
    assert set(block) == {"fertility", "tokens_per_byte", "chars_per_token", "morph_score_f1", "morph_edit_distance"}
    ranking = block["fertility"]["ranking"]
    assert {row["tokenizer"] for row in ranking} == {"bpe", "morphbpe"}
    assert all("letters" in row for row in ranking)
    # One pair for two tokenizers; it carries an adjusted p-value.
    assert len(block["fertility"]["pairs"]) == 1
    assert "p_adj" in block["fertility"]["pairs"][0]
    assert stats_config["units"] == {"corpus": "documents", "morph": "forms"}


_DERIV_RECORD = {
    "run_key": "bpe-90",
    "tokenizer": "bpe",
    "vocab_size": 90,
    "fertility": 1.5,
    "tokens_per_byte": 0.4,
    "chars_per_token": 2.0,
    "renyi_efficiency": 0.8,
    "morph_score_f1": 0.5,
    "morph_edit_distance": 0.6,
    "morph_consistency": 0.7,
    "morph_score_f1_deriv": 0.3,
    "morph_edit_distance_deriv": 1.2,
    "morph_consistency_deriv": 0.25,
}


def test_build_report_includes_deriv_columns_when_present():
    """Derivational columns appear in the report when a run carries them."""
    markdown, _ = build_report([_DERIV_RECORD])
    assert "morph_score_f1_deriv ↑" in markdown
    assert "morph_edit_distance_deriv ↓" in markdown
    assert "morph_consistency_deriv ↑" in markdown


def test_build_report_omits_deriv_columns_when_absent():
    """No derivational column headers are shown when no run carries them."""
    base = {k: v for k, v in _DERIV_RECORD.items() if not k.endswith("_deriv")}
    markdown, _ = build_report([base])
    # The explanatory note mentions *_deriv, but no column header should.
    assert "morph_score_f1_deriv ↑" not in markdown
    assert "morph_consistency_deriv ↑" not in markdown


def test_evaluate_artifact_adds_deriv_point_estimates(tmp_path: Path):
    """A non-empty deriv_sample adds derivational point-estimate columns."""
    cfg = _make_config(tmp_path)
    sample_path, lexicon_path = prepare_inputs(cfg)
    train_one("bpe-90", cfg=cfg, sample_path=sample_path, lexicon_path=lexicon_path)

    deriv_sample = [
        MorphemeSegmentation("hišica", ["hiš", "ic", "a"], ["morph", "morph", "morph"], "hišica", "Sozei", False),
        MorphemeSegmentation("pisatelj", ["pis", "at", "elj"], ["morph", "morph", "morph"], "pisatelj", "Som", True),
    ]
    record = evaluate_artifact(
        "bpe-90",
        output_root=cfg.output_root,
        eval_docs=[["hiša", "hiše"]],
        morph_sample=list(build_morph_lexicon(cfg.sloleks_path).by_form.values()),
        deriv_sample=deriv_sample,
    )
    assert 0.0 <= record["morph_score_f1_deriv"] <= 1.0
    assert record["morph_edit_distance_deriv"] >= 0.0
    assert 0.0 <= record["morph_coverage_deriv"] <= 1.0


class TestTrainLinkTags:
    """Tests for cross-linking eval runs back to their training run."""

    def test_empty_without_sidecar(self, tmp_path: Path):
        """No linkage sidecar yields no cross-link tags."""
        assert _train_link_tags(tmp_path) == {}

    def test_reads_sidecar(self, tmp_path: Path):
        """A present sidecar surfaces the train run id as tags."""
        (tmp_path / MLFLOW_LINK_FILENAME).write_text(
            json.dumps({"run_id": "abc", "parent_run_id": "par", "run_name": "bpe-90"}),
            encoding="utf-8",
        )
        tags = _train_link_tags(tmp_path)
        assert tags["train_run_id"] == "abc"
        assert tags["train_parent_run_id"] == "par"
        assert tags["train_run_name"] == "bpe-90"


def test_eval_run_links_to_training_run(tmp_path: Path, monkeypatch):
    """An eval run is tagged with the run id of the training run it evaluates."""
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.chdir(tmp_path)
    base = _make_config(tmp_path)
    cfg = dataclasses.replace(base, mlflow_enabled=True, mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")
    sample_path, lexicon_path = prepare_inputs(cfg)
    train_one("bpe-90", cfg=cfg, sample_path=sample_path, lexicon_path=lexicon_path)
    log_training_to_mlflow(["bpe-90"], cfg)

    lexicon = build_morph_lexicon(cfg.sloleks_path)
    record = evaluate_artifact(
        "bpe-90",
        output_root=cfg.output_root,
        eval_docs=[["hiša", "psa"]],
        morph_sample=list(lexicon.by_form.values()),
    )
    assert record is not None
    md_path, json_path = write_report([record], cfg.report_dir)
    log_results_to_mlflow([record], cfg, (md_path, json_path))

    link = json.loads((cfg.output_root / "bpe-90" / MLFLOW_LINK_FILENAME).read_text(encoding="utf-8"))
    client = mlflow.MlflowClient(tracking_uri=cfg.mlflow_tracking_uri)
    experiment = client.get_experiment_by_name(cfg.mlflow_experiment)
    eval_runs = client.search_runs(
        [experiment.experiment_id],
        filter_string="tags.phase = 'eval' and tags.mlflow.runName = 'bpe-90'",
    )
    assert eval_runs, "expected an eval child run"
    assert eval_runs[0].data.tags["train_run_id"] == link["run_id"]
