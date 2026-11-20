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
`--max-workers`; `export` takes `--target` and `--force` (see below).

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

Model training runs in other projects, and each model expects its own special
tokens and input wrapping. The `export` section of the sweep config holds one
**target** per training project: the HuggingFace role of each special token
(for example `cls_token: <s>`), the templates that wrap a single text and a
text pair, and the model's maximum input length. The `modernbert` target names
`<s>` as classifier and start token and `</s>` as separator and end token, and
wraps a text as `<s> … </s>` and a pair as `<s> A </s> B </s>`. A new training
project gets a new target there, not new code.

```bash
uv run python scripts/sweep_tokenizers.py export --config "$SWEEP" --all                      # every target
uv run python scripts/sweep_tokenizers.py export --config "$SWEEP" --all --target modernbert  # one target
```

Each export lands in its own directory, so one trained tokenizer serves several
training projects:

```text
tokenizers/<name>-<vocab>/hf-<target>/
  tokenizer.json            vocabulary, normalizer, pre-tokenizer, template post-processor
  tokenizer_config.json     special-token roles and model_max_length
  special_tokens_map.json
  export_lineage.json       digests of the trained artifact and the target settings, files written
tokenizers/_reports/export-summary.json   one record per <run>/<target>
```

A MorphPiece export has no `tokenizer.json`. In its place it holds the
tokenizer's code, `hf_morph_piece.py`, and the model files that code reads:
`morphpiece.json` (the vocabulary), `morphpiece_bpe.json` (the byte-level BPE
path) and `morpheme_table.jsonl.gz`.

A fast export loads with `AutoTokenizer.from_pretrained(<dir>)` and no code from
this repository. MorphPiece looks up each word in its morpheme table, a step the
fast `tokenizers` library does not have, so its export is a slow tokenizer whose
code travels with it. It loads with
`AutoTokenizer.from_pretrained(<dir>, trust_remote_code=True)`, which runs the
copied `hf_morph_piece.py`. A training project whose code calls
`AutoTokenizer.from_pretrained(<dir>)` without that flag, as ModernBERT's export
and check scripts do, must add it before it can use a MorphPiece export. Before an export is moved into place, it is reloaded and must
encode the first `export.parity_lines` lines of the evaluation sample to the
same ids as the trained tokenizer, or that run fails and the command exits
non-zero. For Unigram, two splits whose total scores differ by at most two
float32 steps at that score's size also pass. SentencePiece sums a split's
score in float32, so it cannot tell such splits apart, and the `tokenizers`
library may pick the other one; on very long documents such splits can differ
by a token.

An export is skipped while its lineage digests match the trained artifact and
the target; retraining a run or editing a target exports it again, and
`--force` exports regardless. When a re-export fails, the earlier directory is
kept and its summary record says `stale`. Its `export_digest` (the artifact it
was built from) and `artifact_digest` (the artifact now) differ when the run was
retrained since that export. Every `exported` or `stale` record also gives the
path, the vocabulary size and the id of each special token, which a training
project's own config may need; a `failed` record gives the reason instead. An export run rewrites the records of the runs it selects and
keeps the others, so `export --all` brings the whole summary up to date.

The special-token ids differ by backend, because SentencePiece reserves its own
ids first: the Unigram runs give `<unk>` 0, `<s>` 1, `</s>` 2, `<pad>` 3 and
`<mask>` 4, while the other backends give `<pad>` 0, `<unk>` 1, `<s>` 2, `</s>` 3
and `<mask>` 4. Read the ids of each export from its summary record, for example
for the special-token flags of ModernBERT's conversion scripts.

The fast tokenizers expose `decode` and `return_offsets_mapping` natively. A slow
tokenizer has no `return_offsets_mapping`, so MorphPiece exposes
`encode_with_offsets` instead. Offset mapping (token → source-character span) is
the mechanism for aligning encoder predictions back to the original text.

## SLURM

Batch scripts for cluster execution live under [`slurm/`](../slurm/):

```bash
sbatch slurm/tokenizer_train.sbatch
sbatch slurm/tokenizer_evaluate.sbatch
sbatch slurm/tokenizer_export.sbatch
```
