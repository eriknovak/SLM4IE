---
title: Curation
---

# Curation (datatrove)

`scripts/data/curate.py` produces the **final pretraining corpus** in a
single invocation: language verification, cross-dataset deduplication,
and corpus statistics, all fused into one
[datatrove](https://github.com/huggingface/datatrove) pipeline.

Inputs and outputs are read from
[`configs/data/curate.yaml`](https://github.com/eriknovak/SLM4IE/blob/main/configs/data/curate.yaml)
(`input_dir`, `output_dir`):

- `input_dir` is the folder of `<key>.jsonl.gz` datatrove shards from the
  previous stage.
- `output_dir` is where the deduplicated training corpus and statistics
  are written.

The dataset key list is still pulled from
`configs/data/extract.yaml`.

## Install the curate extra

```bash
uv sync --extra curate
```

The `curate` extra pulls in `datatrove`, `lingua-language-detector`,
`spacy` (Slovenian word/sentence tokenization), `classla` (Slovenian
lemmatizer for the keyword TF-IDF pass), `orjson`, `tokenizers`,
`xxhash`, `nltk`, and a few smaller helpers.

We deliberately skip datatrove's own `processing` and `multilingual`
extras because those transitively pull in `fasttext-numpy2-wheel`, which
has no Python 3.13 wheel and would require a C++17 toolchain to build.

## Canonical command

```bash
uv run python scripts/data/curate.py --all
```

This runs the full pipeline end-to-end on every dataset declared in
`configs/data/extract.yaml`. The output is the `output_dir` from
`configs/data/curate.yaml` and nothing else — every intermediate artifact
lives in a `tempfile.TemporaryDirectory` that is removed at the end of
the run.

## How the pipeline runs (5 datatrove executors)

`build_curate_executors()` chains five `LocalPipelineExecutor`s with
`depends=`. Calling `.run()` on the last executor walks the chain
backwards and runs the rest in order. Find stages can't be merged with
the rest because they consume all signatures from disk in a single pass;
everything else fuses cleanly.

```text
TMP = tempfile.TemporaryDirectory()      (or <output_dir>/_dedup/ with --debug)

executor 1   (per-shard, parallel via --tasks)
    JsonlReader(<input_dir>)
        → LinguaLanguageFilter            # tag metadata.language + score
        → JsonlWriter(TMP/lang_tagged)    # tagged shards for executor 3
        → ExactDedupSignature             # writes hashes to TMP/exact_sigs

executor 2   (single worker — find stage)
    ExactFindDedups
        reads  TMP/exact_sigs
        writes TMP/exact_dups

executor 3   (per-shard, parallel via --tasks)
    JsonlReader(TMP/lang_tagged)
        → ExactDedupFilter                # drops whole-doc duplicates
        → JsonlWriter(TMP/after_exact)    # filtered shards for executor 5
        → SentenceDedupSignature          # writes 3-sent hashes to TMP/sent_sigs

executor 4   (single worker — find stage)
    SentenceFindDedups
        reads  TMP/sent_sigs
        writes TMP/sent_dups

executor 5   (single worker — global stats)
    JsonlReader(TMP/after_exact)
        → SentenceDedupFilter             # drops duplicate sentence spans
        → CorpusStats                     # accumulates global counters
        → JsonlWriter(<output_dir>)       # the training corpus
```

Both signature steps use `Languages.slovenian`, so datatrove dispatches
to its bundled Slovenian `SpaCyTokenizer` for word and sentence
boundaries. **Running them with the English default would corrupt the
dedup signature.**

## Output layout

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

## Useful invocations

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

# Parallelism: per-shard executors (1, 3) run with this many tasks.
# Find and stats stages stay single-worker by design.
uv run python scripts/data/curate.py --all --tasks 4

# Rebuild from scratch (the existing <output_dir> is preserved by default).
uv run python scripts/data/curate.py --all --force

# Override curate.yaml paths from the CLI (e.g. for ad-hoc runs).
uv run python scripts/data/curate.py --all \
    --input-dir /tmp/in --output-dir /tmp/out
```

`--stage lang` and `--stage dedup` exist for **debugging only** — their
outputs live in the auto-cleaned tempdir, so they vanish when the script
exits. Combine with `--debug` to make those intermediates inspectable.

`--stage stats` is the only single-stage mode that produces a real
persistent artifact (it re-reads `<output_dir>` and refreshes
`<output_dir>/statistics/`).

## Configuration

[`configs/data/curate.yaml`](https://github.com/eriknovak/SLM4IE/blob/main/configs/data/curate.yaml)
controls the input/output folders, language candidate set, dedup
thresholds, n-gram orders, and stopword path. `input_dir` and
`output_dir` are required; both can be overridden per run via
`--input-dir` / `--output-dir`.

The default `<output_dir>/statistics/aggregate.json` includes top-5000
word / bigram / trigram tables and top-200 TF-IDF keywords per
(domain, dataset) bucket — adjust those numbers in the config rather
than via CLI flags.

!!! note "First-run model download"
    The first run of the keyword stage downloads the Slovenian classla
    model (~200 MB) under `~/.classla_resources/`. Pass `--no-keywords`
    to skip it — the rest of the stats are populated from datatrove's
    bundled spaCy tokenizer and don't need any model download.
