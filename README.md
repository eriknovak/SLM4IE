![Image alt](./docs/assets/imgs/banner/slm4ie_banner_dark_bg.png#gh-dark-mode-only)
![Image alt](./docs/assets/imgs/banner/slm4ie_banner_light_bg.png#gh-light-mode-only)

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache 2.0"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.13%2B-blue.svg" alt="Python 3.13+"></a>
  <a href="https://github.com/eriknovak/SLM4IE"><img src="https://img.shields.io/badge/Open%20Source-%E2%9D%A4-brightgreen.svg" alt="Open Source"></a>
</p>

SLM4IE develops small language models (SLMs) for zero-shot information extraction across European languages, with emphasis on Slovenian. The project targets three limitations of current LLMs:

- **Compute cost:** LLMs require infrastructure beyond reach of smaller organizations for local deployment
- **Low-resource gaps:** Limited training data for sensitive domains and underrepresented languages
- **Output inconsistency:** Unreliable structured extraction from generative models

We build computationally efficient models optimized for commodity hardware, create multilingual benchmark datasets for sensitive domains, and evaluate against existing SLMs and LLMs. All artifacts (models, datasets, code) will be released publicly where possible.

## Requirements

- **Python** ≥ 3.13 (declared in [`pyproject.toml`](pyproject.toml) and [`.python-version`](.python-version))
- **[uv](https://docs.astral.sh/uv/)** — recommended package and environment manager
- **Git** — for cloning the repository and CLARIN.SI dataset access
- **HuggingFace account** — required for gated datasets (e.g., `FineWeb-2`)
- **Disk space** — pretraining corpora total tens of GB; plan accordingly
- **GPU (optional)** — required for tokenizer/model training; CPU sufficient for data preparation

## Setup

### Install dependencies

Clone the repository and create the virtual environment via `uv`:

```bash
git clone https://github.com/eriknovak/SLM4IE.git
cd SLM4IE
uv sync
```

This creates `.venv/` and installs both runtime and dev dependencies pinned in `uv.lock`. Activate the environment for ad-hoc commands:

```bash
source .venv/bin/activate
```

Or prefix individual commands with `uv run` to skip activation.

### Enable the git hooks

The repository ships a pre-commit hook under `.githooks/` that blocks commits
containing presigned-URL credentials (`X-Amz-Signature` / `X-Amz-Credential`).
Activate it once per clone:

```bash
git config core.hooksPath .githooks
```

Secrets and ephemeral values (such as presigned download URLs) belong in a
gitignored `configs/**/*.local.yaml` overlay, which `load_config` deep-merges
over the matching base config — never in the committed YAML.

### HuggingFace authentication

Some datasets (e.g., `FineWeb-2`, gated corpora) require a HuggingFace access token. Authenticate once via the new unified `hf` CLI (shipped with `huggingface_hub` ≥ 0.34, which replaces the deprecated `huggingface-cli`):

```bash
hf auth login
```

Paste a token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) when prompted. The token is stored under `~/.cache/huggingface/` and picked up automatically by `huggingface_hub` and `datasets` — no `HF_TOKEN` environment variable or `.env` file needed.

For non-interactive use (e.g., CI, SLURM), pass the token directly:

```bash
hf auth login --token "$HF_TOKEN"
```

To verify:

```bash
hf auth whoami
```

## Project Structure

```text
SLM4IE/
├── configs/                # YAML configuration files
│   ├── data/                 # download, extract, pretrain, tasks, tokenization, synthetic
│   ├── models/               # model architecture configs
│   ├── tokenizers/           # tokenizer training configs
│   ├── training/             # pretrain.yaml, finetune_ner.yaml
│   └── experiments/          # end-to-end experiment configs
├── slm4ie/                 # Library source
│   ├── data/                 # download, extract, processing, schema, synthetic
│   ├── models/               # model components and registry
│   ├── tokenizers/           # tokenizer training and analysis
│   ├── training/             # trainer, callbacks, evaluation
│   └── utils/                # config, I/O, MLflow helpers
├── scripts/                # CLI entry points (thin wrappers around slm4ie/)
│   ├── data/                 # download.py, extract.py, to_pretrain.py, to_tokenization.py,
│   │                         #   to_spans.py, to_sentiment.py, to_superglue.py, generate_synthetic.py
│   ├── tokenizers/           # train.py, analyze.py
│   ├── train.py              # model pretraining/fine-tuning
│   └── evaluate.py           # benchmark evaluation
├── slurm/                  # SLURM batch scripts for HPC training
├── notebooks/              # exploratory Jupyter notebooks
├── tests/                  # pytest test suite
├── docs/                   # documentation and assets
├── pyproject.toml          # project metadata and dependencies
└── uv.lock                 # locked dependency versions
```

## Running Scripts

All scripts are CLI wrappers around `slm4ie/` modules and read from YAML configs in `configs/`. Run them via `uv run` (recommended) or after activating `.venv/`.

### Data pipeline

The data pipeline materializes a five-tier on-disk tree under `/vault/data/SLM4IE/`:

```text
raw/<key>/...                                  # original downloads (download.py)
extracted/                                     # canonical unified form (extract.py)
  <key>.jsonl
  <key>.annotations.jsonl.gz
pretrain/                                      # corpus-wide curate output (to_pretrain.py)
  00_convert/<key>/*.jsonl.gz                    # datatrove `Document` shape
  01_language/<key>/*.jsonl.gz
  02_quality/<key>/*.jsonl.gz
  03_repetition/<key>/*.jsonl.gz
  04_1_dedup/<key>/*.jsonl.gz                    # exact dedup
  04_2_dedup/<key>/*.jsonl.gz                    # sentence dedup — final corpus
  05_statistics/                                 # corpus-wide stats
tasks/<task>/<dataset>/{train,val,test}.jsonl.gz  # SFT + eval (to_spans/sentiment/superglue)
tokenization/<dataset>.jsonl.gz                # tokenizer-quality data (to_tokenization.py)
```

Five YAML configs drive the seven data scripts:

| Config                                                             | Script(s)                                           | Purpose                                                                                     |
| ------------------------------------------------------------------ | --------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| [`configs/data/download.yaml`](configs/data/download.yaml)         | `download.py`                                       | Raw corpus + benchmark download catalog.                                                    |
| [`configs/data/extract.yaml`](configs/data/extract.yaml)           | `extract.py`                                        | Sources to normalize into `extracted/`.                                                     |
| [`configs/data/pretrain.yaml`](configs/data/pretrain.yaml)         | `to_pretrain.py`                                    | Seven-stage curate pipeline (stage 0 = datatrove convert; stages 1–6 = filter/dedup/stats). |
| [`configs/data/tokenization.yaml`](configs/data/tokenization.yaml) | `to_tokenization.py`                                | Tokenizer-quality datasets (lexicon-derived).                                               |
| [`configs/data/tasks.yaml`](configs/data/tasks.yaml)               | `to_spans.py`, `to_sentiment.py`, `to_superglue.py` | Registry of `<task>/<dataset>` entries with roles, sources, splits, labels.                 |

End-to-end command flow:

```bash
uv run python scripts/data/download.py       --config configs/data/download.yaml      --all
uv run python scripts/data/extract.py        --config configs/data/extract.yaml       --all
uv run python scripts/data/to_pretrain.py    --pretrain-config configs/data/pretrain.yaml --all
uv run python scripts/data/to_tokenization.py --all
uv run python scripts/data/to_spans.py       --all       # reads tasks.yaml automatically
uv run python scripts/data/to_sentiment.py   --all
uv run python scripts/data/to_superglue.py   --all
```

#### Parallelism and per-dataset logs

`download.py`, `extract.py`, `to_tokenization.py`, `to_spans.py`, `to_sentiment.py`, and `to_superglue.py` all accept a `--max-workers` flag and process multiple datasets concurrently:

- `--max-workers 0` (default) — auto: `min(cpu_count // 2, n_datasets)` for CPU-bound steps, capped at 4 for `download.py` to stay polite to remote servers.
- `--max-workers 1` — serial path; tracebacks are unwrapped, console keeps today's verbose output, and the inner per-dataset progress bar is shown.
- `--max-workers N` — that many workers, capped at the number of selected datasets.

Per-dataset logs are always written to `logs/<script>/<UTC-timestamp>/<key>.log`, regardless of worker count. The log directory is printed to stderr at startup. In parallel mode (`> 1`) the console only prints a periodic summary line (`running=R done=D skipped=S failed=F waiting=W`) every 30 seconds — the per-dataset INFO lines and inner tqdm bars are routed to the log files instead, so concurrent workers don't garble each other on stderr.

`to_pretrain.py` accepts the same `--max-workers` flag but is **whole-pipeline**, not per-dataset — every parallel datatrove executor inside one stage uses the same worker count, so the per-dataset log routing above does not apply. Its default is `--max-workers 1` (serial) so a casual `--all` invocation does not silently saturate the box; `--max-workers 0` falls back to `cpu_count // 2` and `--tasks` is accepted as a back-compat alias.

#### Download

Download raw corpora declared in [`configs/data/download.yaml`](configs/data/download.yaml):

Selection is explicit: pass one or more dataset keys as positional arguments, or pass `--all`. Bare invocation errors out.

```bash
# Download every enabled dataset in the config
uv run python scripts/data/download.py --all

# Download specific datasets (positional, mutually exclusive with --all)
uv run python scripts/data/download.py fineweb2 cc100

# Force re-download with custom output directory
uv run python scripts/data/download.py --all --output-dir /path/to/data --force

# Download only evaluation benchmarks (datasets marked `benchmark: true`)
uv run python scripts/data/download.py --all --only-benchmarks

# Download only pretraining corpora (skip benchmarks)
uv run python scripts/data/download.py --all --exclude-benchmarks

# Use a different YAML in configs/data/ — `--config-name benchmarks` reads
# configs/data/benchmarks.yaml instead of the default `download.yaml`.
uv run python scripts/data/download.py --all --config-name benchmarks

# Download four datasets in parallel (thread pool; default cap is 4)
uv run python scripts/data/download.py fineweb2 cc100 mc4 hplt --max-workers 4
```

#### Extract

Extract and convert raw downloads to unified JSONL using [`configs/data/extract.yaml`](configs/data/extract.yaml). Selection is explicit: pass dataset keys as positional arguments, or pass `--all`.

```bash
# Extract every dataset declared in extract.yaml
uv run python scripts/data/extract.py --all

# Extract specific datasets (positional, mutually exclusive with --all)
uv run python scripts/data/extract.py macocu_sl

# Re-extract a dataset whose output already exists
uv run python scripts/data/extract.py macocu_sl --force

# Extract several datasets in parallel (process pool)
uv run python scripts/data/extract.py macocu_sl classla_web_sl kzb --max-workers 3

# Use a different YAML in configs/data/ (without the .yaml suffix)
uv run python scripts/data/extract.py --all --config-name extract_dev

# Override the configured input/output directories from the CLI
uv run python scripts/data/extract.py --all \
    --input-dir /vault/data/SLM4IE/raw \
    --output-dir /vault/data/SLM4IE/extracted
```

For annotated corpora (CoNLL-U, TEI with `<w>`, CLASSLA-web JSONL, COLESLAW), extraction writes two files per dataset under `extracted/`:

- `<key>.jsonl` — text + `source` / `domain` / `doc_id` / `metadata`, consumed both by `to_pretrain.py`'s stage 0 (which lifts it into datatrove's `Document` shape) and by the task converters.
- `<key>.annotations.jsonl.gz` — gzipped per-document annotations as parallel arrays (`forms`, `lemmas`, `upos`, `feats`, `sentences`, plus `spans` when present), kept separate to avoid loading them during text-only training.

The downstream task converters (`to_spans`, `to_sentiment`, `to_superglue`) join these two files on the fly via `slm4ie.data.io_utils.iter_joined_records`, so no intermediate merged file is materialized.

#### Pretraining corpus (`to_pretrain.py`)

`to_pretrain.py` builds the **final pretraining corpus** as a sequence of seven independent, sentinel-skippable stages on top of [datatrove](https://github.com/huggingface/datatrove). Stage 0 lifts `extracted/*.jsonl` into datatrove's `Document` shape; stages 1–6 cover language filtering, Gopher within-document quality and repetition heuristics, cross-corpus exact and sentence deduplication, and corpus statistics. There is no separate datatrove-conversion step — the old `to_datatrove.py` lives inside stage 0 now. Each stage writes a durable on-disk artifact and a `.complete` sentinel under `output_dir`; on rerun, a stage whose config slice hash is unchanged is skipped, and editing one section of [`configs/data/pretrain.yaml`](configs/data/pretrain.yaml) cascade-invalidates that stage plus every downstream stage. `input_dir` is the folder of `<key>.jsonl` files from `extract.py`; `output_dir` is the pretrain-owned tree. The dataset key list still comes from `configs/data/extract.yaml`.

##### Install the extra

```bash
uv sync --extra curate
```

The `curate` extra pulls in `datatrove`, `lingua-language-detector`, `spacy` (Slovenian word/sentence tokenization), `classla` (Slovenian lemmatizer for the keyword TF-IDF pass), `orjson`, `tokenizers`, `xxhash`, `nltk`, and a few smaller helpers. We deliberately skip datatrove's own `processing` and `multilingual` extras because those transitively pull in `fasttext-numpy2-wheel`, which has no Python 3.13 wheel and would require a C++17 toolchain to build.

##### Canonical command

```bash
uv run python scripts/data/to_pretrain.py --all
```

This iterates all seven stages in order, skipping any whose sentinel hash matches the current config. The final corpus lands at `<output_dir>/04_2_dedup/<dataset>/<rank>.jsonl.gz`; statistics at `<output_dir>/05_statistics/`.

##### Seven user-facing stages

Each stage reads its predecessor's output and writes a numbered folder. The two dedup sub-stages are independent: `04_1_dedup` cleans whole-document duplicates across the corpus; `04_2_dedup` runs sentence-level dedup over that result.

| CLI name         | Folder           | Operates on | What it does                                                                                                                                            |
| ---------------- | ---------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `convert`        | `00_convert/`    | per-doc     | lift `extracted/<key>.jsonl` into datatrove `Document` shards (`text` / `id` / `metadata`); carries `dataset` and `domain` for source-weighted sampling |
| `language`       | `01_language/`   | per-doc     | lingua-py language detection (tag or filter)                                                                                                            |
| `quality`        | `02_quality/`    | per-doc     | Gopher within-document quality heuristics (length, word lengths, symbol/bullet/ellipsis ratios, stopword floor)                                         |
| `repetition`     | `03_repetition/` | per-doc     | Gopher within-document repetition heuristics (duplicate paragraphs/lines, top-n-gram saturation, dup-n-gram fractions)                                  |
| `exact_dedup`    | `04_1_dedup/`    | corpus-wide | whole-document exact dedup (xxhash64 of `doc.text`)                                                                                                     |
| `sentence_dedup` | `04_2_dedup/`    | corpus-wide | N-sentence sliding-window dedup (final corpus)                                                                                                          |
| `stats`          | `05_statistics/` | corpus-wide | word/n-gram tables and (optional) classla TF-IDF keywords (single-process)                                                                              |

Each stage's sentinel hash covers its own top-level `pretrain.yaml` section. The `quality` and `stats` hashes additionally fold in the contents of the stopword file, and every stage's hash folds in the sorted list of dataset keys this run will process — so editing `stopwords_sl.txt`, switching between `--all` and a positional subset, or adding a dataset to `extract.yaml` all correctly trigger rebuilds.

Internally each dedup stage chains three datatrove executors via `depends=`: signature → find (single-worker reducer over signatures) → filter + write. The sig/find scratch lives at `<output_dir>/_dedup_state/` and is purged when the stage's sentinel lands. The stats stage is single-process because `CorpusStats` keeps global counters on its instance. The sentence-dedup blocks use `Languages.slovenian` so datatrove dispatches its bundled Slovenian `SpaCyTokenizer` for sentence boundaries.

##### Output layout

```text
<input_dir>/                                upstream input (extract.py)
├── <key>.jsonl
└── <key>.annotations.jsonl.gz

<output_dir>/                               to_pretrain.py owns this entire tree
├── 00_convert/
│   ├── <key>/<rank>.jsonl.gz               ← datatrove `Document` shards
│   └── .complete                           sentinel: stage hash + counts
├── 01_language/
│   ├── <key>/<rank>.jsonl.gz               ← post-language-filter shards
│   └── .complete
├── 02_quality/
│   ├── <key>/<rank>.jsonl.gz
│   └── .complete
├── 03_repetition/
│   ├── <key>/<rank>.jsonl.gz
│   └── .complete
├── 04_1_dedup/
│   ├── <key>/<rank>.jsonl.gz               ← post-exact-dedup shards
│   └── .complete
├── 04_2_dedup/
│   ├── <key>/<rank>.jsonl.gz               ← final pretraining corpus
│   └── .complete
├── 05_statistics/
│   ├── aggregate.json                      corpus-wide totals + tables
│   ├── per_dataset/<key>.json              per-dataset doc/word breakdowns
│   └── .complete
├── _dedup_state/                           sig/find scratch (auto-purged
│                                           when each dedup sentinel lands)
└── _logs/<stage>/                          datatrove per-executor logs
```

##### Useful invocations

```bash
# Run all seven stages, skipping any whose config slice hash is unchanged.
uv run python scripts/data/to_pretrain.py --all

# Run only one stage. If its hash diverges from the recorded sentinel,
# downstream sentinels are dropped so the next --all picks them up.
uv run python scripts/data/to_pretrain.py --all --stage quality

# Force-rebuild a stage and every downstream stage. Removes their data
# folders AND sentinels; --force without --stage clears <output_dir>.
uv run python scripts/data/to_pretrain.py --all --force --stage quality

# Single dataset, or a subset. The dataset key list folds into every
# stage's hash, so a subset rerun will not silently reuse a previous
# full-corpus output. (Switching between subsets / --all triggers rebuilds.)
uv run python scripts/data/to_pretrain.py kzb solar

# Parallelism. Default is 1 (serial). 0 = cpu_count // 2. --tasks is an alias.
uv run python scripts/data/to_pretrain.py --all --max-workers 8
uv run python scripts/data/to_pretrain.py --all --max-workers 0

# Override pretrain.yaml paths from the CLI.
uv run python scripts/data/to_pretrain.py --all \
    --input-dir /tmp/in --output-dir /tmp/out
```

##### Configuration

`configs/data/pretrain.yaml` has one top-level section per stage (`convert:`, `language:`, `quality:`, `repetition:`, `exact_dedup:`, `sentence_dedup:`, `stats:`) plus shared `input_dir`, `output_dir`, and a `stopwords:` path used by both `quality` and `stats`. Each section is the **exclusive input** to that stage's sentinel hash slice, so edits propagate as far downstream as needed and no further. Defaults match the Gopher paper for the heuristic filters, 64-bit xxhash for exact dedup, 3-sentence windows for sentence dedup, and top-5000 word / bigram / trigram + top-200 TF-IDF keyword tables for stats. To skip the (slow) classla-lemmatized keyword pass, set `stats.compute_keywords: false` in the YAML — there is no longer a `--no-keywords` CLI flag.

The first run of the keyword stage downloads the Slovenian classla model (~200 MB) under `~/.classla_resources/`.

#### Tokenizer-quality data (`to_tokenization.py`)

`to_tokenization.py` materializes lexicon-derived datasets used only for tokenizer / morphology evaluation — they never enter the pretraining corpus. Currently this covers Sloleks 3.1 (Slovenian inflectional lexicon). Configuration lives in [`configs/data/tokenization.yaml`](configs/data/tokenization.yaml); the script also reads [`configs/data/download.yaml`](configs/data/download.yaml) to resolve per-dataset raw subdirectories.

```bash
# Convert every dataset declared in tokenization.yaml
uv run python scripts/data/to_tokenization.py --all

# Convert one dataset, overwriting if the output already exists
uv run python scripts/data/to_tokenization.py sloleks --force

# Run in parallel
uv run python scripts/data/to_tokenization.py --all --max-workers 4
```

Output goes to `tokenization/<dataset>.jsonl.gz`. Existing outputs are skipped unless `--force` is passed.

#### Task datasets (`to_spans`, `to_sentiment`, `to_superglue`)

The three task converters all read [`configs/data/tasks.yaml`](configs/data/tasks.yaml), a flat registry keyed `<task>/<dataset>`. They write to `tasks/<task>/<dataset>/<split>.jsonl.gz` using a task-family schema (TypedDicts in [`slm4ie/data/schema.py`](slm4ie/data/schema.py)). Each entry declares:

- `role` — `finetune_and_eval` or `held_out`; the registry, not directory placement, enforces train/test isolation across families.
- `source` — `{kind: extracted, keys: […]}` for document-shaped sources joined via `extracted/`, or `{kind: raw, keys: […]}` for task-native bundles (SuperGLUE-SL) read straight from `raw/`.
- `splits`, `labels`, `suite`, `language`, `license`.

Adding a new task dataset is a one-entry edit to `tasks.yaml`; the appropriate converter (defaulted by the `converters:` map at the top of the file) will pick it up.

```bash
# NER (GLiNER-style output)
uv run python scripts/data/to_spans.py --all                        # every ner/* entry
uv run python scripts/data/to_spans.py ner/ssj500k ner/suk          # subset

# Sentiment
uv run python scripts/data/to_sentiment.py --all                    # every sentiment/* entry
uv run python scripts/data/to_sentiment.py sentiment/sentinews

# SuperGLUE-SL families (nli, qa, coref, wsd, commonsense)
uv run python scripts/data/to_superglue.py --all                    # HumanT variant by default
uv run python scripts/data/to_superglue.py --variant googlemt --all
uv run python scripts/data/to_superglue.py nli/cb nli/rte
```

All three skip existing outputs unless `--force` is passed and accept `--max-workers` for per-entry parallelism. The legacy `--schema {gliner|conll|generic}` flag on `to_spans.py` is gone — only the GLiNER-compatible schema is produced now.

#### Synthetic data

```bash
uv run python scripts/data/generate_synthetic.py # synthetic IE data via LLM APIs
```

### Tokenizer

```bash
uv run python scripts/tokenizers/train.py     # train tokenizer from config
uv run python scripts/tokenizers/analyze.py   # compare tokenizers
```

### Model training and evaluation

```bash
uv run python scripts/train.py     # pretrain or fine-tune from configs/training/*.yaml
uv run python scripts/evaluate.py  # evaluate on benchmarks/*.yaml
```

### SLURM (HPC)

Batch scripts for cluster execution live under [`slurm/`](slurm/):

```bash
sbatch slurm/tokenizer_train.sbatch
sbatch slurm/train.sbatch
sbatch slurm/evaluate.sbatch
sbatch slurm/generate.sbatch
```

### Tests

```bash
uv run pytest                # full test suite
uv run pytest tests/data     # subset
```

## Pretraining Corpora

Slovenian text corpora used for language model pretraining (configured in [`configs/data/download.yaml`](configs/data/download.yaml)).

### CLARIN.SI sources

| Dataset                                                                          | Domain        | Description                                                                                                                   |
| -------------------------------------------------------------------------------- | ------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| [CLASSLA-web.sl 2.0](https://www.clarin.si/repository/xmlui/handle/11356/2079)   | web           | Annotated Slovenian web corpus from the CLASSLA project.                                                                      |
| [CLASSLAWiki-sl](https://www.clarin.si/repository/xmlui/handle/11356/1427)       | wiki          | Slovenian Wikipedia with linguistic annotations (CoNLL-U).                                                                    |
| [MaCoCu-sl 2.0](https://www.clarin.si/repository/xmlui/handle/11356/1795)        | web           | Slovenian web corpus from the MaCoCu project (XML/TEI).                                                                       |
| [ParlaMint-SI 5.0](https://www.clarin.si/repository/xmlui/handle/11356/2004)     | parliamentary | Slovenian parliamentary minutes, annotated TEI.                                                                               |
| [COLESLAW 1.0](https://www.clarin.si/repository/xmlui/handle/11356/2095)         | legal         | Corpus of Slovenian legal texts.                                                                                              |
| [PoVeJMo-VeMo-Med 1.0](https://www.clarin.si/repository/xmlui/handle/11356/1983) | medical       | Slovenian medical texts from the PoVeJMo project.                                                                             |
| [OSS 1.0](https://www.clarin.si/repository/xmlui/handle/11356/1774)              | scientific    | 2.59B words / 3.26B tokens from 151K scientific texts (monographs, articles, theses) from Slovenian universities (2000–2022). |
| [siParl 4.0](https://www.clarin.si/repository/xmlui/handle/11356/1936)           | parliamentary | 239M words from parliamentary minutes (1990–2022), TEI XML. May overlap with ParlaMint-SI.                                    |
| [Janes-News 1.0](https://www.clarin.si/repository/xmlui/handle/11356/1140)       | news          | 14.8M tokens from news article comments (2007–2015). Informal register.                                                       |
| [KZB 1.0](https://www.clarin.si/repository/xmlui/handle/11356/1872)              | scientific    | 25M words / 33.6M tokens of curated scientific monographs and papers (2000–2023).                                             |

### HuggingFace sources

| Dataset                                                                  | Domain | Description                                                                                                       |
| ------------------------------------------------------------------------ | ------ | ----------------------------------------------------------------------------------------------------------------- |
| [FinePDF](https://huggingface.co/datasets/HuggingFaceFW/finepdfs)        | web    | Slovenian (`slv_Latn`) PDF-derived text.                                                                          |
| [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)     | web    | Slovenian (`slv_Latn`) high-quality web corpus.                                                                   |
| [mC4](https://huggingface.co/datasets/allenai/c4)                        | web    | Cleaned multilingual Common Crawl, ~5 GB+ for Slovenian.                                                          |
| [HPLT 2.0 Cleaned](https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned) | web    | HPLT project web crawl (CommonCrawl + Internet Archive), cleaned tier; Slovenian config `slv_Latn` (~10.3M rows). |

### Direct HTTP sources

| Dataset                                                            | Domain | Description                                                                                                                                                                                                              |
| ------------------------------------------------------------------ | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [CC100](https://data.statmt.org/cc-100/)                           | web    | Monolingual CommonCrawl filtered with fastText (Facebook AI, XLM-R), ~1.4 GB compressed for Slovenian. Fetched directly from `statmt.org`; the HuggingFace mirror is script-based and no longer supported by `datasets`. |
| [Legal-mC4](https://huggingface.co/datasets/joelniklaus/legal-mc4) | legal  | Legal-domain text filtered from mC4, ~32.5K documents / ~107M words for Slovenian. Fetched directly from the HuggingFace LFS endpoint; the repo's loading script is no longer supported by `datasets`.                   |

### Disabled by default

Optional sources requiring extra access (gated datasets, manual login, copyright restrictions): `KAS 2.0` (CLARIN academic login), `Janes-Forum/Blog`, `Solar 3.0`, `CulturaX` (HF gated). Not bulk-downloadable: `Gigafida 2.x`, `Metafida 1.0`, `Trendi`.

## Benchmarks

Slovenian evaluation datasets used for downstream IE tasks. Benchmarks are declared in [`configs/data/download.yaml`](configs/data/download.yaml) with `benchmark: true` and a `tasks:` list, so they share the download pipeline with pretraining corpora. Use `--only-benchmarks` to fetch just the evaluation datasets.

| Dataset                                                                       | Source    | Tasks                                     | Description                                                                                                                                                                                                                                                                                                      |
| ----------------------------------------------------------------------------- | --------- | ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [SUK 1.1](https://www.clarin.si/repository/xmlui/handle/11356/1959)           | CLARIN.SI | POS, LEMMA, DEP, NER, SRL, COREF, WSD, SA | ~1M tokens / 881K words / 2,913 texts manually annotated with MULTEXT-East V6, JOS, and Universal Dependencies. Integrates ssj500k 2.3, Ambiga, ElexisWSD, and SentiCoref subcorpora. License: CC BY-SA 4.0.                                                                                                     |
| [ssj500k 2.3](https://www.clarin.si/repository/xmlui/handle/11356/1434)       | CLARIN.SI | POS, LEMMA, DEP, NER, SRL                 | ~500K tokens manually annotated with MSD tags, lemmas, UD syntax (UD 2.8), named entities, and semantic role labels. Foundation corpus for SUK 1.1. License: CC BY-NC-SA 4.0.                                                                                                                                    |
| [Slovene SuperGLUE](https://www.clarin.si/repository/xmlui/handle/11356/1380) | CLARIN.SI | QA, NLI, WSD, COREF, MRC                  | Slovene translation of SuperGLUE (BoolQ, CB, COPA, MultiRC, ReCoRD, RTE, WiC, WSC). Mix of human and Google MT translation. License: CC BY 4.0. Convert to per-task evaluation files with `scripts/data/to_superglue.py`.                                                                                        |
| [SentiNews 1.0](https://www.clarin.si/repository/xmlui/handle/11356/1110)     | CLARIN.SI | SA                                        | Slovene news sentiment with three-level annotations (sentence, paragraph, document) and 3-class labels. Directly downloadable. License: CC BY-SA 4.0. Convert to evaluation JSONL with `scripts/data/to_sentiment.py`.                                                                                           |
| [Sloleks 3.1](https://www.clarin.si/repository/xmlui/handle/11356/2080)       | CLARIN.SI | TOKENIZER                                 | Slovenian inflectional lexicon (lemmas + word forms with MULTEXT-East V6 / JOS MSDs). **Tokenizer / morphology evaluation only** — intentionally absent from `extract.yaml`, never enters the pretraining corpus. Distributed as TEI XML. License: CC BY-SA 4.0. Convert with `scripts/data/to_tokenization.py`. |

### Task abbreviations

- **POS** — part-of-speech tagging
- **LEMMA** — lemmatization
- **DEP** — dependency parsing
- **NER** — named entity recognition
- **SRL** — semantic role labeling
- **COREF** — coreference resolution
- **WSD** — word sense disambiguation
- **SA** — sentiment analysis
- **NLI** — natural language inference
- **QA** — question answering
- **MRC** — machine reading comprehension
- **TOKENIZER** — tokenizer / morphology evaluation (lexicon-based, not a downstream IE task)

## Acknowledgments

The project is funded by ARIS (Slovenian Research and Innovation Agency) under the project number [Z2-70067](https://cris.cobiss.net/ecris/si/sl/project/24346).

<figure>
  <img src="https://github.com/eriknovak/SLM4IE/blob/main/docs/assets/imgs/aris.png?raw=true" alt="ARIS Logo" width="460" />
</figure>
