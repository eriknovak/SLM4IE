# Running the tokenizer sweep

`scripts/sweep_tokenizers.py` trains six tokenizers across a vocabulary sweep,
scores them with six metrics, and exports each as a HuggingFace tokenizer for
LM pretraining. Library code lives in `slm4ie/tokenizers/`.

The sweep is shared machinery, so its settings are a shared registry and this
page only covers how to run it; an experiment that varies the sweep keeps its
own dials under its slug's `configs/` and its findings in its record. Adding a
tokenizer means a new `@register_tokenizer` backend under
`slm4ie/tokenizers/backends/`, not a new script.

## What it trains

Six tokenizers, each faithful to its original work: byte-level **BPE**
(GPT-2/RoBERTa style), character-level **charBPE** (the clean MorphBPE
ablation), BERT-style **WordPiece**, SentencePiece **Unigram**, and the
from-scratch **MorphBPE** (morpheme-constrained training, standard inference —
arXiv 2502.00894) and **MorphPiece** (byte-level BPE + a Sloleks-derived
MorphTable — arXiv 2307.07262). Each runs at three vocabulary sizes
(16k/32k/64k). Driven by
[`configs/tokenizers/sweep.yaml`](../configs/tokenizers/sweep.yaml).

## Prerequisites

```bash
uv sync --group tokenizers
```

The sweep consumes the deduplicated corpus (`pretrain/06_sentence_dedup/`) for training
and two Sloleks-derived golds produced by `prepare_datasets.py tokenization`:
the inflectional gold (`tokenization/sloleks.jsonl.gz`) and the derivational
gold (`tokenization/sloleks_relations.jsonl.gz`). The morph metrics report
against the inflectional gold (with bootstrap confidence intervals); the
derivational gold adds point-estimate `*_deriv` columns and also enriches the
morphological backends' morpheme table. Build both before running the sweep —
see [data-pipeline.md](data-pipeline.md).

## Commands

```bash
SWEEP=configs/tokenizers/sweep.yaml

uv run python scripts/prepare_datasets.py tokenization sloleks sloleks_relations  # prerequisite: morph golds
uv run python scripts/sweep_tokenizers.py sample   --config "$SWEEP"         # persistent sample + lexicon
uv run python scripts/sweep_tokenizers.py train    --config "$SWEEP" --all   # train the 6x3 sweep
uv run python scripts/sweep_tokenizers.py evaluate --config "$SWEEP" --all   # 6 metrics + report.md/json
uv run python scripts/sweep_tokenizers.py export   --config "$SWEEP" --all   # HuggingFace tokenizer dirs

# Or one tokenizer (optionally one vocab size) instead of --all:
uv run python scripts/sweep_tokenizers.py train --config "$SWEEP" --tokenizer bpe
uv run python scripts/sweep_tokenizers.py train --config "$SWEEP" --tokenizer bpe --vocab-size 16000
```

The `sample` step materializes the shared, seeded training sample (and the
morpheme lexicon) once into `tokenizers/corpus_sample.txt.gz`, so every `train`
run reuses the identical sample instead of re-drawing it. It is optional —
`train` builds the sample on first use — but running it up front keeps the
sample persistent and reproducible across reruns, and across a `--max-workers`
change after a crash. Pass `--force` to rebuild it.

`train` / `evaluate` / `export` share a one-or-all selection: `--all`, or one
`--tokenizer <name>` optionally narrowed by `--vocab-size <n>` (the two modes
are mutually exclusive). `train` and `evaluate` also take `--force` and
`--max-workers`.

Artifacts land under `/vault/data/SLM4IE/tokenizers/<name>-<vocab>/`; the
comparison report is written to `tokenizers/_reports/report.md`. The sweep logs
to MLflow by default under experiment
`slm4ie/tokenizers/sweep`; the tracking URI is read from
`MLFLOW_TRACKING_URI` (falling back to a local SQLite store), overridable via
`mlflow.tracking_uri` in a `sweep.local.yaml` overlay. Disable
with `mlflow.enabled: false`.

## The six metrics

**Fertility** (tokens/word, ↓), **CTC** compression (tokens-per-byte ↓ /
chars-per-token ↑), **Rényi efficiency** (↑), **MorphScore** boundary F1 (↑),
**Morph-Edit-Distance** (raw, ↓), and **Morph-Consistency** F1 (↑) — the last
two following the MorphBPE paper (arXiv 2502.00894), MorphScore following
Arnett et al. The morph metrics are offset-based and score against a
Sloleks-derived *inflectional* silver gold, so read them as relative
comparators, not absolute morphological accuracy.

## Using a tokenizer downstream

The `export` step writes a HuggingFace tokenizer directory into each artifact.
The five fast tokenizers load with `AutoTokenizer.from_pretrained(<dir>)` and
expose `decode` and `return_offsets_mapping` natively; MorphPiece is a custom
slow tokenizer loaded with
`slm4ie.tokenizers.hf_export.load_pretrained(<dir>)`, exposing
`encode_with_offsets`. Offset mapping (token → source-character span) is the
mechanism for aligning encoder predictions back to the original text.

## SLURM

Batch scripts for cluster execution live under [`slurm/`](../slurm/):

```bash
sbatch slurm/tokenizer_train.sbatch
sbatch slurm/tokenizer_evaluate.sbatch
sbatch slurm/tokenizer_export.sbatch
```
