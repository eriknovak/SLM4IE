# Data pipeline

How to get from nothing to the on-disk data every experiment reads. Install
first — see [setup.md](setup.md) — then `uv sync --group data`.

All scripts are thin CLI wrappers around `slm4ie/` modules. They read shared
registries from `configs/` and, where a script belongs to one experiment, that
experiment's dials from `experiments/<category>/<slug>/configs/`. Run them via
`uv run` (recommended) or after activating `.venv/`.

## What it produces

The pipeline materializes a shared on-disk tree under `/vault/data/SLM4IE/`
(`data/` in the repo is a gitignored symlink to it). Data an experiment derives
for itself sits under that root's own `experiments/<category>/<slug>/{raw,interim,final}`
tier instead:

```text
raw/<key>/...                                  # original downloads
extracted/                                     # canonical unified form
  <key>.jsonl
  <key>.annotations.jsonl.gz
pretrain/                                      # corpus-wide curation output
  00_convert/<key>/*.jsonl.gz                    # datatrove `Document` shape
  01_language/<key>/*.jsonl.gz
  02_spam/<key>/*.jsonl.gz                       # adult/SEO-spam removal
  03_quality/<key>/*.jsonl.gz
  04_repetition/<key>/*.jsonl.gz
  05_exact_dedup/<key>/*.jsonl.gz                # exact dedup
  06_sentence_dedup/<key>/*.jsonl.gz             # sentence dedup — final corpus
  07_statistics/                                 # corpus-wide stats
tasks/<task>/<dataset>/{train,val,test}.jsonl.gz  # SFT + eval
tokenization/<dataset>.jsonl.gz                # tokenizer-quality data
```

## Configs and scripts

Three scripts read four shared registries plus one experiment-owned config:

| Config                                                                                                                                                           | Subcommand                          | Purpose                                                                                          |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------ |
| [`configs/data/download.yaml`](../configs/data/download.yaml)                                                                                                    | `prepare_datasets.py download`      | Raw corpus + benchmark download catalog.                                                          |
| [`configs/data/extract.yaml`](../configs/data/extract.yaml)                                                                                                      | `prepare_datasets.py extract`       | Sources to normalize into `extracted/`.                                                           |
| [`configs/data/tokenization.yaml`](../configs/data/tokenization.yaml)                                                                                            | `prepare_datasets.py tokenization`  | Tokenizer-quality datasets (lexicon-derived).                                                     |
| [`configs/data/tasks.yaml`](../configs/data/tasks.yaml)                                                                                                          | `prepare_datasets.py tasks`         | Registry of `<task>/<dataset>` entries with roles, sources, splits, labels.                       |
| [`configs/data/curate.yaml`](../configs/data/curate.yaml)                                                                                                         | `curate_pretraining_corpus.py run`  | Eight-stage curation pipeline (stage 0 = datatrove convert; stages 1–7 = filter/spam/dedup/statistics). |

End-to-end command flow:

```bash
uv run python scripts/prepare_datasets.py download --config configs/data/download.yaml --all
uv run python scripts/prepare_datasets.py extract  --config configs/data/extract.yaml  --all
uv run python scripts/curate_pretraining_corpus.py run --config configs/data/curate.yaml --all
uv run python scripts/prepare_datasets.py tokenization --all
uv run python scripts/prepare_datasets.py tasks        --all      # every family in tasks.yaml
```

Curation is documented separately in
[pretraining-corpus.md](pretraining-corpus.md).

## Parallelism and per-dataset logs

Every `prepare_datasets.py` subcommand accepts a `--max-workers` flag and
processes multiple datasets concurrently:

- `--max-workers 0` (default) — auto: `min(cpu_count // 2, n_datasets)` for CPU-bound steps, capped at 4 for `download` to stay polite to remote servers.
- `--max-workers 1` — serial path; tracebacks are unwrapped, console keeps the verbose output, and the inner per-dataset progress bar is shown.
- `--max-workers N` — that many workers, capped at the number of selected datasets.

Per-dataset logs are always written to
`logs/<step>/<UTC-timestamp>/<key>.log`, regardless of worker count. The log
directory is printed to stderr at startup. In parallel mode (`> 1`) the console
only prints a periodic summary line (`running=R done=D skipped=S failed=F
waiting=W`) every 30 seconds — the per-dataset INFO lines and inner tqdm bars
are routed to the log files instead, so concurrent workers don't garble each
other on stderr.

`curate_pretraining_corpus.py run` accepts the same flag but is
**whole-pipeline**, not per-dataset, so the log routing above does not apply.
See [pretraining-corpus.md](pretraining-corpus.md).

## Download

Download raw corpora declared in
[`configs/data/download.yaml`](../configs/data/download.yaml). Selection is
explicit: pass one or more dataset keys as positional arguments, or pass
`--all`. Bare invocation errors out.

```bash
# Download every enabled dataset in the config
uv run python scripts/prepare_datasets.py download --all

# Download specific datasets (positional, mutually exclusive with --all)
uv run python scripts/prepare_datasets.py download fineweb2 cc100

# Force re-download with custom output directory
uv run python scripts/prepare_datasets.py download --all --output-dir /path/to/data --force

# Download only non-pretraining datasets (`role: benchmark` or `role: lexicon`)
uv run python scripts/prepare_datasets.py download --all --only-benchmarks

# Download only pretraining corpora (`role: pretrain`, the default role)
uv run python scripts/prepare_datasets.py download --all --exclude-benchmarks

# Use a different YAML in configs/data/ — `--config-name benchmarks` reads
# configs/data/benchmarks.yaml instead of the default `download.yaml`.
uv run python scripts/prepare_datasets.py download --all --config-name benchmarks

# Download four datasets in parallel (thread pool; default cap is 4)
uv run python scripts/prepare_datasets.py download fineweb2 cc100 mc4 hplt --max-workers 4
```

## Extract

Extract and convert raw downloads to unified JSONL using
[`configs/data/extract.yaml`](../configs/data/extract.yaml). Selection is
explicit: pass dataset keys as positional arguments, or pass `--all`.

```bash
# Extract every dataset declared in extract.yaml
uv run python scripts/prepare_datasets.py extract --all

# Extract specific datasets (positional, mutually exclusive with --all)
uv run python scripts/prepare_datasets.py extract macocu_sl

# Re-extract a dataset whose output already exists
uv run python scripts/prepare_datasets.py extract macocu_sl --force

# Extract several datasets in parallel (process pool)
uv run python scripts/prepare_datasets.py extract macocu_sl classla_web_sl kzb --max-workers 3

# Use a different YAML in configs/data/ (without the .yaml suffix)
uv run python scripts/prepare_datasets.py extract --all --config-name extract_dev

# Override the configured input/output directories from the CLI
uv run python scripts/prepare_datasets.py extract --all \
    --input-dir /vault/data/SLM4IE/raw \
    --output-dir /vault/data/SLM4IE/extracted
```

For annotated corpora (CoNLL-U, TEI with `<w>`, CLASSLA-web JSONL, COLESLAW),
extraction writes two files per dataset under `extracted/`:

- `<key>.jsonl` — text + `source` / `domain` / `doc_id` / `metadata`, consumed both by curation's stage 0 (which lifts it into datatrove's `Document` shape) and by the task converters.
- `<key>.annotations.jsonl.gz` — gzipped per-document annotations as parallel arrays (`forms`, `lemmas`, `upos`, `feats`, `sentences`, plus `spans` when present), kept separate to avoid loading them during text-only training.

The downstream task converters (`spans`, `sentiment`, `superglue`) join these
two files on the fly via `slm4ie.data.io_utils.iter_joined_records`, so no
intermediate merged file is materialized.

## Tokenizer-quality data

The `tokenization` step materializes lexicon-derived datasets used only for
tokenizer / morphology evaluation — they never enter the pretraining corpus.
This covers Sloleks 3.1 (Slovenian inflectional lexicon → inflectional gold) and
the Sloleks 2.0 word relations (CLARIN 11356/1986 → derivational silver gold;
~66k lemmas decomposed with underscores, ~5k linguist-verified). Both are CC
BY-SA 4.0. Configuration lives in
[`configs/data/tokenization.yaml`](../configs/data/tokenization.yaml); the
script also reads [`configs/data/download.yaml`](../configs/data/download.yaml)
to resolve per-dataset raw subdirectories.

```bash
# Convert every dataset declared in tokenization.yaml
uv run python scripts/prepare_datasets.py tokenization --all

# Convert one dataset, overwriting if the output already exists
uv run python scripts/prepare_datasets.py tokenization sloleks --force

# Run in parallel
uv run python scripts/prepare_datasets.py tokenization --all --max-workers 4
```

Output goes to `tokenization/<dataset>.jsonl.gz`. Existing outputs are skipped
unless `--force` is passed.

## Task datasets

One step covers every task family. It reads
[`configs/data/tasks.yaml`](../configs/data/tasks.yaml), a flat registry keyed
`<task>/<dataset>`, routes each entry to the converter that entry names, and
writes to `tasks/<task>/<dataset>/<split>.jsonl.gz` using a task-family schema
(TypedDicts in [`slm4ie/data/schema.py`](../slm4ie/data/schema.py)). Each entry
declares:

- `role` — `finetune_and_eval` or `held_out`; the registry, not directory placement, enforces train/test isolation across families. A `held_out` entry never writes a `train` split (records are re-bucketed, not dropped).
- `source` — `{kind: extracted, keys: […]}` for document-shaped sources joined via `extracted/`, or `{kind: raw, keys: […]}` for task-native bundles (SuperGLUE-SL) read straight from `raw/`.
- `splits`, `labels`, `suite`, `language`, `license`.

Adding a new task dataset is a one-entry edit to `tasks.yaml`; the appropriate
converter (defaulted by the `converters:` map at the top of the file) will pick
it up.

```bash
# Every entry in the registry, whatever its family
uv run python scripts/prepare_datasets.py tasks --all

# A subset, named by entry key
uv run python scripts/prepare_datasets.py tasks ner/ssj500k ner/suk
uv run python scripts/prepare_datasets.py tasks sentiment/sentinews
uv run python scripts/prepare_datasets.py tasks nli/cb nli/rte

# SuperGLUE-SL reads the HumanT variant by default
uv run python scripts/prepare_datasets.py tasks --all --variant googlemt
```

Existing outputs are skipped unless `--force` is passed, and `--max-workers`
gives per-entry parallelism. Only the GLiNER-compatible schema is produced for
span tasks.
