<p align="center">
  <img src="https://github.com/eriknovak/SLM4IE/blob/main/docs/assets/imgs/logo.png?raw=true" alt="logo" style="width: 60%;">
</p>

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

```
SLM4IE/
├── configs/                # YAML configuration files
│   ├── data/                 # download, extract, processing, benchmarks, synthetic
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
│   ├── data/                 # download.py, extract.py, process.py, analyze.py, generate_synthetic.py
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

#### Parallelism and per-dataset logs

`download.py`, `extract.py`, `to_datatrove.py`, `to_spans.py`, `to_sentiment.py`, and `to_tokenizer_eval.py` all accept a `--max-workers` flag and process multiple datasets concurrently:

- `--max-workers 0` (default) — auto: `min(cpu_count // 2, n_datasets)` for CPU-bound steps, capped at 4 for `download.py` to stay polite to remote servers.
- `--max-workers 1` — serial path; tracebacks are unwrapped, console keeps today's verbose output, and the inner per-dataset progress bar is shown.
- `--max-workers N` — that many workers, capped at the number of selected datasets.

Per-dataset logs are always written to `logs/<script>/<UTC-timestamp>/<key>.log`, regardless of worker count. In parallel mode (`> 1`) the console only prints a periodic summary line (`running=R done=D skipped=S failed=F waiting=W`) every 30 seconds — the per-dataset INFO lines and inner tqdm bars are routed to the log files instead, so concurrent workers don't garble each other on stderr.

`curate.py` is the exception: it parallelizes internally via datatrove's `LocalPipelineExecutor` (see `--workers` below) and is not wrapped by `--max-workers`.

#### Download

Download raw corpora declared in [`configs/data/download.yaml`](configs/data/download.yaml):

```bash
# Download all enabled datasets
uv run python scripts/data/download.py

# Download specific datasets
uv run python scripts/data/download.py --datasets fineweb2 cc100

# Force re-download with custom output directory
uv run python scripts/data/download.py --output-dir /path/to/data --force

# Download only evaluation benchmarks (datasets marked `benchmark: true`)
uv run python scripts/data/download.py --only-benchmarks

# Download only pretraining corpora (skip benchmarks)
uv run python scripts/data/download.py --exclude-benchmarks

# Download four datasets in parallel (thread pool; default cap is 4)
uv run python scripts/data/download.py --datasets fineweb2 cc100 mc4 hplt --max-workers 4
```

#### Extract

Extract and convert raw downloads to unified JSONL using [`configs/data/extract.yaml`](configs/data/extract.yaml):

```bash
uv run python scripts/data/extract.py
uv run python scripts/data/extract.py --datasets macocu_sl

# re-extract a dataset whose output already exists
uv run python scripts/data/extract.py --datasets macocu_sl --force

# extract several datasets in parallel (process pool)
uv run python scripts/data/extract.py --datasets macocu_sl classla_web_sl kzb --max-workers 3
```

For annotated corpora (CoNLL-U, TEI with `<w>`, CLASSLA-web JSONL, COLESLAW), extraction writes two files per dataset:

- `<key>.jsonl` — text + `source` / `domain` / `doc_id` / `metadata`, used directly for pretraining.
- `<key>.annotations.jsonl.gz` — gzipped per-document annotations as parallel arrays (`forms`, `lemmas`, `upos`, `feats`, `sentences`), kept separate to avoid loading them during text-only training.

The downstream converters (`to_datatrove`, `to_spans`) join these two files on the fly via `slm4ie.data.io_utils.iter_joined_records`, so no intermediate merged file is materialized.

#### Datatrove format

Convert the per-dataset JSONL into the [datatrove](https://github.com/huggingface/datatrove) `Document` shape (`text` / `id` / `metadata`) so the corpus can be filtered, deduped, and sharded with datatrove pipelines:

```bash
# Convert one dataset
uv run python scripts/data/to_datatrove.py kzb

# Convert every dataset in extract.yaml
uv run python scripts/data/to_datatrove.py --all

# Re-convert every dataset in extract.yaml
uv run python scripts/data/to_datatrove.py --all --force

# Convert every dataset in parallel
uv run python scripts/data/to_datatrove.py --all --max-workers 4
```

Output goes to `<output_dir>/datatrove/<key>.jsonl.gz` (override with `--output-dir`). Existing outputs are skipped unless `--force` is passed. Every record carries `dataset` and `domain` at the top level so downstream filters and source-weighted sampling can use them via `document.metadata`. `JsonlReader("…/datatrove/*.jsonl.gz")` ingests the whole corpus in one go.

#### Spans format

Convert the per-dataset JSONL into span-level IE training files (GLiNER / CoNLL / generic) for fine-tuning encoder models on entity-style tasks:

```bash
# GLiNER training shape
uv run python scripts/data/to_spans.py kzb --schema gliner

# every dataset, lossless generic shape
uv run python scripts/data/to_spans.py --all --schema generic

# parallel
uv run python scripts/data/to_spans.py --all --schema gliner --max-workers 4
```

Output goes to `<output_dir>/spans/<schema>/<key>.jsonl.gz`. Existing outputs are skipped unless `--force` is passed. The converter expects each annotations payload to carry a `spans` field (`[start, end, label]` triples or `{start, end, label}` dicts with end-exclusive token indices); records without spans are skipped with a warning.

#### Sentiment format

Convert raw SA benchmark downloads (e.g. `sentinews`) into evaluation-ready JSONL with normalized `{negative, neutral, positive}` labels. The converter reads directly from the raw download tree (it bypasses the `extract.py` step) and emits one record per item with `id`, `text`, `label`, `label_id`, `level`, and `metadata`:

```bash
# Convert SentiNews (all annotation levels present in the download)
uv run python scripts/data/to_sentiment.py sentinews

# Restrict to document-level sentiment
uv run python scripts/data/to_sentiment.py sentinews --levels document

# Convert every SA-tagged benchmark dataset declared in download.yaml
uv run python scripts/data/to_sentiment.py --all

# parallel
uv run python scripts/data/to_sentiment.py --all --max-workers 4
```

Output goes to `<raw-dir>/eval/sentiment/<key>.jsonl.gz` (override with `--output-dir`), accompanied by a `label_map.json` so the integer `label_id` encoding is traceable. Existing outputs are skipped unless `--force` is passed.

#### SuperGLUE format

Convert the extracted Slovene SuperGLUE distribution into per-task per-split JSONL files for fine-tuning and SloBENCH-style evaluation. Each task is materialized in its native SuperGLUE schema (BoolQ, CB, COPA, RTE, ReCoRD, WiC, WSC pass through unchanged); MultiRC is flattened to one row per answer by default for classification convenience:

```bash
# All 8 tasks, all splits, HumanT variant
uv run python scripts/data/to_superglue.py

# Only CB and RTE, val split, GoogleMT variant
uv run python scripts/data/to_superglue.py --tasks CB RTE --splits val --variant googlemt

# Keep MultiRC in its native nested shape
uv run python scripts/data/to_superglue.py --tasks MultiRC --no-flatten-multirc
```

Output goes to `<raw-dir>/eval/superglue_sl/<variant>/<task>/<split>.jsonl.gz` (override with `--output-dir`). The converter expects the raw download to contain a `SuperGLUE-HumanT/` or `SuperGLUE-GoogleMT/` directory with one subdirectory per task and `train.jsonl` / `val.jsonl` / `test.jsonl` inside. Existing outputs are skipped unless `--force` is passed.

#### Curation (datatrove)

`curate.py` produces the **final pretraining corpus** in a single invocation: language verification, cross-dataset deduplication, and corpus statistics, all fused into one [datatrove](https://github.com/huggingface/datatrove) pipeline. Inputs and outputs are read from `configs/data/curate.yaml` (`input_dir`, `output_dir`) — `input_dir` is the folder of `<key>.jsonl.gz` datatrove shards from the previous stage and `output_dir` is where the deduplicated training corpus and statistics are written. The dataset key list is still pulled from `configs/data/extract.yaml`.

##### Install the extra

```bash
uv sync --extra curate
```

The `curate` extra pulls in `datatrove`, `lingua-language-detector`, `spacy` (Slovenian word/sentence tokenization), `classla` (Slovenian lemmatizer for the keyword TF-IDF pass), `orjson`, `tokenizers`, `xxhash`, `nltk`, and a few smaller helpers. We deliberately skip datatrove's own `processing` and `multilingual` extras because those transitively pull in `fasttext-numpy2-wheel`, which has no Python 3.13 wheel and would require a C++17 toolchain to build.

##### Canonical command

```bash
uv run python scripts/data/curate.py --all
```

This runs the full pipeline end-to-end on every dataset declared in `configs/data/extract.yaml`. The output is the `output_dir` from `configs/data/curate.yaml` and nothing else — every intermediate artifact lives in a `tempfile.TemporaryDirectory` that is removed at the end of the run.

##### How the pipeline runs (6 datatrove executors)

`build_curate_executors()` chains six `LocalPipelineExecutor`s with `depends=`. Calling `.run()` on the last executor walks the chain backwards and runs the rest in order. Find stages can't be merged with the rest because they consume all signatures from disk in a single pass; the corpus-stats stage stays single-process because `CorpusStats` keeps global counters on its instance.

```text
TMP = tempfile.TemporaryDirectory()      (or <output_dir>/_dedup/ with --debug)

executor 1   (per-shard, parallel via --workers)
    JsonlReader(<input_dir>)
        → LinguaLanguageFilter            # tag metadata.language + score
        → JsonlWriter(TMP/lang_tagged)    # tagged shards for executor 3
        → ExactDedupSignature             # writes hashes to TMP/exact_sigs

executor 2   (single worker — find stage)
    ExactFindDedups
        reads  TMP/exact_sigs
        writes TMP/exact_dups

executor 3   (per-shard, parallel via --workers)
    JsonlReader(TMP/lang_tagged)
        → ExactDedupFilter                # drops whole-doc duplicates
        → JsonlWriter(TMP/after_exact)    # filtered shards for executor 5
        → SentenceDedupSignature          # writes 3-sent hashes to TMP/sent_sigs

executor 4   (single worker — find stage)
    SentenceFindDedups
        reads  TMP/sent_sigs
        writes TMP/sent_dups

executor 5   (per-shard, parallel via --workers)
    JsonlReader(TMP/after_exact)
        → SentenceDedupFilter             # drops duplicate sentence spans
        → JsonlWriter(<output_dir>)       # the training corpus

executor 6   (single worker — global stats)
    JsonlReader(<output_dir>)
        → CorpusStats                     # accumulates global counters
                                          # writes <output_dir>/statistics/
```

Both signature steps use `Languages.slovenian`, so datatrove dispatches to its bundled Slovenian `SpaCyTokenizer` for word and sentence boundaries. Running them with the English default would corrupt the dedup signature.

##### Output layout

```text
<input_dir>/                                upstream input (unchanged)
└── <key>.jsonl.gz

<output_dir>/                               curate.py owns this entire tree
├── <key>.jsonl.gz                          ← deduplicated training corpus
├── statistics/
│   ├── aggregate.json                      corpus-wide totals + tables
│   └── per_dataset/<key>.json              per-dataset doc/word breakdowns
└── _dedup/                                 only with --debug
    ├── lang_tagged/<key>.jsonl.gz          (executor 1 output)
    ├── after_exact/<key>.jsonl.gz          (executor 3 output)
    ├── exact_sigs/, exact_dups/            (datatrove dedup state)
    ├── sent_sigs/, sent_dups/              (datatrove dedup state)
    ├── exact_dropped/<key>.jsonl.gz        (whole-doc duplicates dropped)
    └── sentence_dropped/<key>.jsonl.gz     (docs dropped post sentence-dedup)
```

##### Useful invocations

```bash
# Single dataset (still all three concerns; dedup operates within that
# one shard only — for cross-dataset dedup, use --all).
uv run python scripts/data/curate.py kzb

# Re-run only the stats stage against the existing curated corpus —
# useful after editing top_k / keyword_top_k in configs/data/curate.yaml.
uv run python scripts/data/curate.py --all --stage stats

# Skip the (slow) classla-lemmatized TF-IDF keyword pass.
uv run python scripts/data/curate.py --all --no-keywords

# Debug: keep dedup state and dropped-duplicate JSONL shards under
# <output_dir>/_dedup/ for inspection. Or --debug-dir <path> to put
# them somewhere else.
uv run python scripts/data/curate.py --all --debug

# Parallelism: per-shard executors (1, 3, 5) run with this many workers.
# Find stages and the final stats stage stay single-worker by design.
# Pass --workers 0 to use every available core. --tasks is a back-compat alias.
uv run python scripts/data/curate.py --all --workers 4

# Rebuild from scratch (the existing <output_dir> is preserved by default).
uv run python scripts/data/curate.py --all --force

# Override curate.yaml paths from the CLI (e.g. for ad-hoc runs).
uv run python scripts/data/curate.py --all \
    --input-dir /tmp/in --output-dir /tmp/out
```

`--stage lang` and `--stage dedup` exist for debugging only — their outputs live in the auto-cleaned tempdir, so they vanish when the script exits. Combine with `--debug` to make those intermediates inspectable. `--stage stats` is the only single-stage mode that produces a real persistent artifact (it re-reads `<output_dir>` and refreshes `<output_dir>/statistics/`).

##### Configuration

`configs/data/curate.yaml` controls the input/output folders, language candidate set, dedup thresholds, n-gram orders, and stopword path. `input_dir` and `output_dir` are required; both can be overridden per run via `--input-dir` / `--output-dir`. The default `<output_dir>/statistics/aggregate.json` includes top-5000 word / bigram / trigram tables and top-200 TF-IDF keywords per (domain, dataset) bucket — adjust those numbers in the config rather than via CLI flags.

The first run of the keyword stage downloads the Slovenian classla model (~200 MB) under `~/.classla_resources/`. Pass `--no-keywords` to skip it — the rest of the stats are populated from datatrove's bundled spaCy tokenizer and don't need any model download.

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

| Dataset                                                                       | Source    | Tasks                                     | Description                                                                                                                                                                                                               |
| ----------------------------------------------------------------------------- | --------- | ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [SUK 1.1](https://www.clarin.si/repository/xmlui/handle/11356/1959)           | CLARIN.SI | POS, LEMMA, DEP, NER, SRL, COREF, WSD, SA | ~1M tokens / 881K words / 2,913 texts manually annotated with MULTEXT-East V6, JOS, and Universal Dependencies. Integrates ssj500k 2.3, Ambiga, ElexisWSD, and SentiCoref subcorpora. License: CC BY-SA 4.0.              |
| [ssj500k 2.3](https://www.clarin.si/repository/xmlui/handle/11356/1434)       | CLARIN.SI | POS, LEMMA, DEP, NER, SRL                 | ~500K tokens manually annotated with MSD tags, lemmas, UD syntax (UD 2.8), named entities, and semantic role labels. Foundation corpus for SUK 1.1. License: CC BY-NC-SA 4.0.                                             |
| [Slovene SuperGLUE](https://www.clarin.si/repository/xmlui/handle/11356/1380) | CLARIN.SI | QA, NLI, WSD, COREF, MRC                  | Slovene translation of SuperGLUE (BoolQ, CB, COPA, MultiRC, ReCoRD, RTE, WiC, WSC). Mix of human and Google MT translation. License: CC BY 4.0. Convert to per-task evaluation files with `scripts/data/to_superglue.py`. |
| [SentiNews 1.0](https://www.clarin.si/repository/xmlui/handle/11356/1110)     | CLARIN.SI | SA                                        | Slovene news sentiment with three-level annotations (sentence, paragraph, document) and 3-class labels. Directly downloadable. License: CC BY-SA 4.0. Convert to evaluation JSONL with `scripts/data/to_sentiment.py`.    |
| [Sloleks 3.1](https://www.clarin.si/repository/xmlui/handle/11356/2080)       | CLARIN.SI | TOKENIZER                                 | Slovenian inflectional lexicon (lemmas + word forms with MULTEXT-East V6 / JOS MSDs). **Tokenizer / morphology evaluation only** — intentionally absent from `extract.yaml`, never enters the pretraining corpus. Distributed as TEI XML. License: CC BY-SA 4.0. Convert with `scripts/data/to_tokenizer_eval.py`. |

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
