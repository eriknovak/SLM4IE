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
  02_spam/<key>/removed/*.jsonl.gz               # dropped docs + spam_reason
  03_quality/<key>/*.jsonl.gz
  04_repetition/<key>/*.jsonl.gz
  05_exact_dedup/<key>/*.jsonl.gz                # exact dedup
  06_sentence_dedup/<key>/*.jsonl.gz             # sentence dedup — final corpus
  07_statistics/                                 # corpus-wide stats
tasks/<task>/<dataset>/{train,val,test}.jsonl.gz  # SFT + eval
tokenization/<dataset>.jsonl.gz                # tokenizer-quality data
pretokenized/<corpus>/<tokenizer>-<length>/    # pretokenized outputs for other projects
  <format>/{train,validation}/  manifest.json
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
[pretraining-corpus.md](pretraining-corpus.md). Tokenizing the curated corpus
into pretokenized outputs for other projects (`scripts/pretokenize_corpus.py`,
registry [`configs/data/pretokenize.yaml`](../configs/data/pretokenize.yaml))
is documented in [pretokenized-corpus.md](pretokenized-corpus.md).

## Parallelism and per-dataset logs

Every `prepare_datasets.py` subcommand accepts a `--max-workers` flag and
processes multiple datasets concurrently:

- `--max-workers 0` (default) — auto: the memory-aware default below for CPU-bound steps, capped at 4 for `download` to stay polite to remote servers.
- `--max-workers 1` — serial path; tracebacks are unwrapped, console keeps the verbose output, and the inner per-dataset progress bar is shown.
- `--max-workers N` — that many workers, capped at the number of selected datasets. Memory is not checked.

### Worker count by memory

The automatic worker count is the smallest of three limits, and at least 1:

- half the cores the process may use (CPU affinity, so SLURM and `taskset` limits count);
- the number of items to process;
- 80% of the machine's available memory (`MemAvailable`) divided by the step's memory per worker.

Tokenizer evaluation also caps the count at 8. Every run logs the count and the limit that set it, for example
`tasks: 11 workers, limited by memory (cores=40 items=30 available=111.8 GB per worker=8.0 GB)`.
The same default sizes the pools of extraction, tokenizer-quality data, task
datasets, the curation convert stage, tokenizer training and evaluation, and
the pretokenized corpus.

Each step declares its memory per worker in the `workers:` mapping of its
registry. A step without one assumes 8 GB, and the log says so:

```yaml
workers:
  memory_per_worker_gb: 4        # peak memory of one worker, in GB
  max_items_per_worker: 50       # optional: replace a worker after 50 items
```

The mapping sits at the top level of `extract.yaml`, `tokenization.yaml`,
`tasks.yaml` and `curate.yaml` (no curation stage hashes it, so changing it
rebuilds nothing). `pretokenize.yaml` keeps it in `defaults`, and an entry's
own `workers:` replaces it. `configs/tokenizers/sweep.yaml` has one mapping per
pool, `workers: {train: {...}, evaluate: {...}}`.

`max_items_per_worker` replaces each pool worker after that many items, so
memory that grows across items is returned to the system. It is off by
default. Replaced workers start fresh instead of forking, so they re-import
their modules. Curation and tokenizer evaluation reject it at load, since
their pools cannot replace workers.

Inside each worker of the shared process pool, the HuggingFace tokenizers'
thread count (`RAYON_NUM_THREADS`) is set to the cores divided by the workers, so 20
workers do not each start one thread per core. Extraction runs its own shard
pool and builds no tokenizer, so it does not set it.

### Measuring memory per worker

Run the step on one worker over its largest item, and read the peak resident
memory:

```bash
/usr/bin/time -v uv run python scripts/prepare_datasets.py tasks <largest-entry> --max-workers 1 --force \
  2>&1 | grep "Maximum resident"
```

Declare about 1.2 to 2 times the peak, rounded up to a whole GB, with the date
and the item it was measured on in a comment. One worker's peak can vary
threefold between datasets, so measure on the heaviest one.

### Running a long job under a memory cap

A job that overruns memory can bring down the whole machine: the kernel's
out-of-memory killer may pick the systemd user manager first, which closes
every tmux pane. Run long jobs in a systemd scope with a memory cap, so an
overrun stops only that job:

```bash
systemd-run --user --scope --unit=<job-name> -p MemoryMax=90G -p MemorySwapMax=0 \
  -p CollectMode=inactive-or-failed \
  uv run python scripts/prepare_datasets.py tasks --all
```

The automatic count reads the machine's available memory, not the cap. Under a
cap well below the free memory, pass `--max-workers` sized from the cap.

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
`--all`. Bare invocation errors out. A `manual: true` entry is never fetched:
`--all` only checks its folder holds files and logs its note otherwise.

Two optional entry fields record how the catalog's corpora relate, declared
once here and read by every consumer (the extract registry does not repeat
them):

- `contains` — keys whose documents the entry fully includes (`suk` contains
  `ssj500k`). Transitive. Keys must exist, an entry may not list itself, and a
  cycle is a config error.
- `overlaps` — keys sharing part of the entry's documents, usually the same
  upstream crawl. Symmetric and never transitive; declared on one side.

`role` says what a dataset is for; `contains` and `overlaps` say what it is made
of. Curation's `--all` keeps one representative per containment group, and the
task registry refuses train/held-out pairs that share documents. The audit
behind every relation is the containment map in [datasets.md](datasets.md#containment-map).

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

- `<key>.jsonl` — text + `source` / `domain` / `doc_id` / `uid` / `native_id` / `metadata`, consumed both by curation's stage 0 (which lifts it into datatrove's `Document` shape) and by the task converters. `doc_id` is unique within the dataset (extraction drops a repeat with identical text and fails on one with different text), `uid` is `<key>:<doc_id>`, and `native_id` is the raw source's own identifier when it has one.
- `<key>.annotations.jsonl.gz` — gzipped per-document annotations as parallel arrays (`forms`, `lemmas`, `upos`, `feats`, `space_after`, `sentences`), kept separate to avoid loading them during text-only training. A CoNLL-U source whose tokens carry `NER=` IOB tags also gets `spans`: one `[start, end, label]` triple per entity, where `start`/`end` are character offsets into `text` (`end` exclusive, so `text[start:end]` is the entity surface) and `label` is upper-case (`PER`, `LOC`, `ORG`, `MISC`, `DERIV-PER`). An annotated document without entities carries `spans: []`; a source without the tag has no `spans` key.

The downstream task converters (`spans`, `sentiment`, `pos`, `superglue`) join these
two files on the fly via `slm4ie.data.extract.records.iter_joined_records`, so no
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

- `role` — `finetune_and_eval`, `held_out` or `finetune_only`; the registry, not directory placement, enforces train/test isolation across families. A `held_out` entry never writes a `train` split (records are re-bucketed, not dropped). A `finetune_only` entry keeps train, val and test for per-entry fine-tuning, yet counts as `held_out` in the isolation check, so no zero-shot training mix reads it.
- `source` — `{kind: extracted, keys: […]}` for document-shaped sources joined via `extracted/`, or `{kind: raw, keys: […]}` for task-native bundles (SuperGLUE-SL) read straight from `raw/`. An extracted source may add `exclude: […]`: catalog keys whose documents are dropped by `doc_id` (`ner/suk` excludes `ssj500k`).
- `splits`, `labels`, `suite`, `language`, `license`.
- `split_on` (optional) — assigns splits from a document metadata field for sources with a published split (`spans` and `pos` converters). Its `otherwise` may be one split or a list; documents that reach a list are hash-split across it by document id. `ner/slavicner` keeps US Election 2020 as test and hash-splits the other topics into train and val.

Source isolation is checked when the registry loads, before anything is written:
a `finetune_and_eval` entry and a `held_out` or `finetune_only` entry of the
same task may not read the same key, and neither side's key may contain the
other's (the `contains` relation in `download.yaml`, followed transitively)
unless that side excludes the contained key. A violation fails with both entry
names and the containing key.

The `pos` family (`PosExample`: `{id, words, tags}`, id `<doc>:s<n>`) is built by
the `pos` converter for `pos/ssj500k`, `pos/suk` and `pos/janes_tag`. Each takes
the role of the NER entry from the same source, and the hash split keys on the
document id, so one document's sentences share a split. A sentence whose word
and tag counts differ is refused and counted.

`ner/suk`, `ner/janes_tag`, `ner/slavicner` and `sentiment/twitter_15eu_sl` are
`finetune_only` and take the standard 70/15/15 hash split (`slavicner` as above).
`sentiment/twitter_15eu_sl` has no download source yet, so it stays unbuilt. The
SuperGLUE entries switch roles after #53.

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

## Per-entry fine-tuning

`scripts/finetune_encoders.py` trains a pretrained encoder with a task head on
one entry's train split, picks the checkpoint on its val split and scores it on
its test split (library code in `slm4ie/finetune/`; needs the `evaluation`
group). The shared fine-tuning config,
[`configs/evaluation/finetune.yaml`](../configs/evaluation/finetune.yaml), sets
the learning rates, seeds, epochs and per-family overrides; the
encoders come from `--encoder` or an experiment dial (`--dial`) merged over the
shared file. The `run` subcommand does the work. Entries come as positional keys or `--all`; `--all` skips entries whose split
files are missing, with a warning and a count, while a named entry with missing
files fails. Each encoder, entry, learning rate and seed is one MLflow run
named `<encoder>-<task>-<dataset>-lr<lr>-seed<seed>`.

```bash
# Print the planned runs without training
uv run --group evaluation python scripts/finetune_encoders.py run pos/ssj500k \
    --encoder EMBEDDIA/sloberta --dry-run

# Every built entry
uv run --group evaluation python scripts/finetune_encoders.py run --all \
    --encoder <name> --experiment slm4ie/<slug> --tag tier=local
```

The `search` subcommand runs a hyperparameter search instead, from a `search`
block that a dial supplies. Each encoder and entry gets one Optuna study: a
`grid` sampler runs every combination of the listed values, for an ablation
table, and a `tpe` sampler draws a fixed number of trials with Optuna's
Tree-structured Parzen Estimator, optionally pruning trials whose per-epoch val
score falls below the median. Trials train with the family's first seed and are
picked on val; test is never scored inside a study. The best setting is then
fine-tuned once per seed and scored on test. The study lives in `--storage` (or
`$OPTUNA_STORAGE`), so a stopped search resumes. With `--experiment`, each
study is one MLflow parent run with a child run per trial and per final seed;
the parent logs the best setting and the final seeds' mean and standard
deviation of each test score.

```bash
# Print the planned studies and trials without training
uv run --group evaluation python scripts/finetune_encoders.py search pos/ssj500k \
    --encoder EMBEDDIA/sloberta --dial <dial>.yaml --dry-run

# Run them, stored for resumption
uv run --group evaluation python scripts/finetune_encoders.py search pos/ssj500k \
    --encoder EMBEDDIA/sloberta --dial <dial>.yaml --storage sqlite:////abs/path/optuna.db \
    --experiment slm4ie/<slug> --tag tier=local
```
