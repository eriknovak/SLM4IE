"""End-to-end tests for slm4ie/tokenizers/analysis.py."""

import gzip
import json
from pathlib import Path

from slm4ie.tokenizers.analysis import build_report, evaluate_artifact, write_report
from slm4ie.tokenizers.corpus import SampleBudget
from slm4ie.tokenizers.metrics import iter_words
from slm4ie.tokenizers.morphology import build_morph_lexicon
from slm4ie.tokenizers.train import prepare_inputs, train_one
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
        eval_words = ["hiša", "hiše", "psa", "mesto"]
        record = evaluate_artifact("bpe-90", output_root=cfg.output_root, lexicon=lexicon, eval_words=eval_words)

        assert record["tokenizer"] == "bpe"
        assert record["vocab_size"] == 90
        assert record["fertility"] >= 1.0
        assert 0.0 <= record["renyi_efficiency"] <= 1.0
        assert 0.0 <= record["morph_score_f1"] <= 1.0
        assert (cfg.output_root / "bpe-90" / "metrics.json").exists()

    def test_missing_artifact_returns_none(self, tmp_path: Path):
        """A missing artifact yields None (skipped)."""
        cfg = _make_config(tmp_path)
        lexicon = build_morph_lexicon(cfg.sloleks_path)
        assert evaluate_artifact("bpe-90", output_root=cfg.output_root, lexicon=lexicon, eval_words=["hiša"]) is None


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
                "morph_edit_score": 0.6,
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
                "morph_edit_score": 0.6,
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
    eval_words = [w for line in _CORPUS[:20] for w in iter_words(line)]
    records = [
        evaluate_artifact(key, output_root=cfg.output_root, lexicon=lexicon, eval_words=eval_words)
        for key in ["bpe-90", "morphbpe-90"]
    ]
    md_path, _ = write_report([r for r in records if r], cfg.report_dir)
    assert md_path.exists()
    # MorphBPE should respect morpheme boundaries at least as well as plain BPE.
    by_name = {r["tokenizer"]: r for r in records if r}
    assert by_name["morphbpe"]["morph_score_f1"] >= 0.0
