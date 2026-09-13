# Building the pretraining corpus

`scripts/curate_pretraining_corpus.py run` builds the final pretraining corpus
as a sequence of eight independent, sentinel-skippable stages on top of
[datatrove](https://github.com/huggingface/datatrove). Stage 0 lifts
`extracted/*.jsonl` into datatrove's `Document` shape; stages 1–7 cover language
filtering, adult/SEO-spam removal, Gopher within-document quality and repetition
heuristics, cross-corpus exact and sentence deduplication, and corpus
statistics.

Each stage writes a durable on-disk artifact and a `.complete` sentinel under
`output_dir`. On rerun, a stage whose config slice hash is unchanged is skipped,
and editing one section of the config cascade-invalidates that stage plus every
downstream stage. `input_dir` is the folder of `<key>.jsonl` files from the
extract step; `output_dir` is the pretrain-owned tree. The dataset key list
comes from [`configs/data/extract.yaml`](../configs/data/extract.yaml).

The settings are a shared registry, so the corpus is built once and reused by
every experiment. The config is still passed explicitly, since an experiment may
curate a variant of its own from its slug's `configs/`.

## Install the dependency group

```bash
uv sync --group corpus
```

The `corpus` group pulls in `datatrove`, `lingua-language-detector`, `spacy`
(Slovenian word/sentence tokenization), `classla` (Slovenian lemmatizer for the
keyword TF-IDF pass), `orjson`, `tokenizers`, `xxhash`, `nltk`, and a few
smaller helpers. It deliberately skips datatrove's own `processing` and
`multilingual` extras, because those transitively pull in
`fasttext-numpy2-wheel`, which has no Python 3.13 wheel and would require a
C++17 toolchain to build.

## Canonical command

```bash
CURATION=configs/data/curate.yaml

uv run python scripts/curate_pretraining_corpus.py run --config "$CURATION" --all
```

This iterates all eight stages in order, skipping any whose sentinel hash
matches the current config. The final corpus lands at
`<output_dir>/06_sentence_dedup/<dataset>/<rank>.jsonl.gz`; statistics at
`<output_dir>/07_statistics/`.

## The eight stages

Each stage reads its predecessor's output and writes a numbered folder. The two
dedup stages are independent: `05_exact_dedup` cleans whole-document duplicates
across the corpus; `06_sentence_dedup` runs sentence-level dedup over that
result.

| CLI name         | Folder               | Operates on | What it does                                                                                                                                            |
| ---------------- | -------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `convert`        | `00_convert/`        | per-doc     | lift `extracted/<key>.jsonl` into datatrove `Document` shards (`text` / `id` / `metadata`); carries `dataset` and `domain` for source-weighted sampling |
| `language`       | `01_language/`       | per-doc     | lingua-py language detection (tag or filter)                                                                                                            |
| `spam`           | `02_spam/`           | per-doc     | adult/SEO-spam removal via per-language lexicons + URL/domain blocklist + optional model hook                                                           |
| `quality`        | `03_quality/`        | per-doc     | Gopher within-document quality heuristics (length, word lengths, symbol/bullet/ellipsis ratios, stopword floor)                                         |
| `repetition`     | `04_repetition/`     | per-doc     | Gopher within-document repetition heuristics (duplicate paragraphs/lines, top-n-gram saturation, dup-n-gram fractions)                                  |
| `exact_dedup`    | `05_exact_dedup/`    | corpus-wide | whole-document exact dedup (xxhash64 of `doc.text`)                                                                                                     |
| `sentence_dedup` | `06_sentence_dedup/` | corpus-wide | N-sentence sliding-window dedup (final corpus)                                                                                                          |
| `statistics`     | `07_statistics/`     | corpus-wide | word/n-gram tables and (optional) classla TF-IDF keywords (single-process)                                                                              |

Internally each dedup stage chains three datatrove executors via `depends=`:
signature → find (single-worker reducer over signatures) → filter + write. The
sig/find scratch lives at `<output_dir>/_dedup_state/` and is purged when the
stage's sentinel lands. The statistics stage is single-process because
`CorpusStats` keeps global counters on its instance. The sentence-dedup blocks
use `Languages.slovenian` so datatrove dispatches its bundled Slovenian
`SpaCyTokenizer` for sentence boundaries.

## Sentinels: what triggers a rebuild

Each stage's sentinel hash covers its own top-level config section. The
`quality` and `statistics` hashes additionally fold in the contents of the
stopword file, the `spam` hash folds in the spam lexicon and domain-list
contents, and every stage's hash folds in the sorted list of dataset keys this
run will process — so editing `stopwords/sl.txt`, editing a `spam/<code>/*.txt`
list, switching between `--all` and a positional subset, or adding a dataset to
`extract.yaml` all correctly trigger rebuilds.

> **Note — refreshed inputs auto-rebuild.** `convert` is the only stage that
> reads the `extracted/` tier, so it also tracks a **size + modification-time
> fingerprint** of each source `<key>.jsonl` (and its `.annotations.jsonl.gz`
> sidecar when `include_annotations` is on). Re-extracting a dataset — e.g.
> folding new weekly windows into a living corpus with `prepare_datasets.py
> extract <key> --force` — changes that fingerprint, which marks `convert` stale
> for that key and cascades through every downstream scoped and corpus stage. So
> a plain `curate_pretraining_corpus.py run --all` after re-extraction re-folds
> the updated data with no `--force` needed. The fingerprint is **size and time
> only, never the file contents** (hashing the whole multi-hundred-GB tier on
> every run would be prohibitive): an in-place edit that somehow preserved both
> byte size and mtime would go undetected, and a content-preserving copy/`touch`
> that bumps mtime triggers a harmless rebuild. A sentinel carrying no
> fingerprint is grandfathered — its key is rebuilt only if the source file is
> newer than the recorded completion time.

## Per-dataset overrides

An optional top-level `overrides:` block lets a single dataset patch any
**scoped** stage's config without forking the file. It is keyed by dataset, then
by stage, and deep-merges over the global section (unspecified knobs inherit the
default):

```yaml
overrides:
  slovenian_news:
    quality:
      max_ellipsis_lines_ratio: 0.9   # news prose uses "…" mid-article
```

Only scoped stages (`convert`, `language`, `spam`, `quality`, `repetition`) are
overridable — naming a corpus stage (`exact_dedup` / `sentence_dedup` /
`statistics`) or a global key (`input_dir` / `output_dir` / `stopwords`), an
unknown dataset, or an unknown knob is a hard error at load. Datasets are
bucketed by their effective config: those sharing one run together in a single
executor, so only overridden datasets pay isolation cost. Each dataset's
sentinel hashes its effective (merged) config, so adding or editing an override
re-runs only that dataset's stage plus its downstream and the corpus
dedup/statistics; a dataset with no override is byte-identical to before and
never re-runs spuriously. `repetition` exposes no knobs today, so it is
effectively non-overridable until some are surfaced.

## Output layout

```text
<input_dir>/                                upstream input (extract step)
├── <key>.jsonl
└── <key>.annotations.jsonl.gz

<output_dir>/                               curate_pretraining_corpus.py owns this entire tree
├── 00_convert/
│   ├── <key>/<rank>.jsonl.gz               ← datatrove `Document` shards
│   └── .complete                           sentinel: stage hash + counts
├── 01_language/
│   ├── <key>/<rank>.jsonl.gz               ← post-language-filter shards
│   └── .complete
├── 02_spam/
│   ├── <key>/<rank>.jsonl.gz               ← post-spam-filter shards
│   └── .complete
├── 03_quality/
│   ├── <key>/<rank>.jsonl.gz
│   └── .complete
├── 04_repetition/
│   ├── <key>/<rank>.jsonl.gz
│   └── .complete
├── 05_exact_dedup/
│   ├── <key>/<rank>.jsonl.gz               ← post-exact-dedup shards
│   └── .complete
├── 06_sentence_dedup/
│   ├── <key>/<rank>.jsonl.gz               ← final pretraining corpus
│   └── .complete
├── 07_statistics/
│   ├── aggregate.json                      corpus-wide totals + tables
│   ├── per_dataset/<key>.json              per-dataset doc/word breakdowns
│   └── .complete
├── _dedup_state/                           sig/find scratch (auto-purged
│                                           when each dedup sentinel lands)
└── _logs/<stage>/                          datatrove per-executor logs
```

## Useful invocations

```bash
CURATION=configs/data/curate.yaml

# Run all eight stages, skipping any whose config slice hash is unchanged.
uv run python scripts/curate_pretraining_corpus.py run --config "$CURATION" --all

# Run only one stage. If its hash diverges from the recorded sentinel,
# downstream sentinels are dropped so the next --all picks them up.
uv run python scripts/curate_pretraining_corpus.py run --config "$CURATION" --all --stage quality

# Force-rebuild a stage and every downstream stage. Removes their data
# folders AND sentinels; --force without --stage clears <output_dir>.
uv run python scripts/curate_pretraining_corpus.py run --config "$CURATION" --all --force --stage quality

# Single dataset, or a subset. The dataset key list folds into every
# stage's hash, so a subset rerun will not silently reuse a previous
# full-corpus output. (Switching between subsets / --all triggers rebuilds.)
uv run python scripts/curate_pretraining_corpus.py run --config "$CURATION" kzb solar

# Parallelism. Default is 1 (serial). 0 = cpu_count // 2. --tasks is an alias.
uv run python scripts/curate_pretraining_corpus.py run --config "$CURATION" --all --max-workers 8
uv run python scripts/curate_pretraining_corpus.py run --config "$CURATION" --all --max-workers 0

# Override the configured paths from the CLI.
uv run python scripts/curate_pretraining_corpus.py run --config "$CURATION" --all \
    --input-dir /tmp/in --output-dir /tmp/out
```

`--max-workers` is **whole-pipeline**, not per-dataset: every parallel datatrove
executor inside one stage uses the same worker count, so the per-dataset log
routing of `prepare_datasets.py` does not apply here. The default is 1 (serial)
so a casual `--all` invocation does not silently saturate the box.

## Configuration

[`configs/data/curate.yaml`](../configs/data/curate.yaml)
has one top-level section per stage (`convert:`, `language:`, `spam:`,
`quality:`, `repetition:`, `exact_dedup:`, `sentence_dedup:`, `statistics:`)
plus shared `input_dir`, `output_dir`, and a `stopwords:` path used by both
`quality` and `statistics`. Each section is the **exclusive input** to that
stage's sentinel hash slice, so edits propagate as far downstream as needed and
no further. Defaults match the Gopher paper for the heuristic filters, 64-bit
xxhash for exact dedup, 3-sentence windows for sentence dedup, and top-5000
word / bigram / trigram + top-200 TF-IDF keyword tables for statistics. The slow
classla-lemmatized keyword pass is toggled in the YAML with
`statistics.compute_keywords`.

The first run of the keyword stage downloads the Slovenian classla model
(~200 MB) under `~/.classla_resources/`.

## Diagnosing foreign-language leakage

The word tables surface English function words at well under 1% of mass. The
`diagnose` subcommand says where that residue lives: whole foreign documents
that slipped past the language stage, which a config change can tighten, or
English passages embedded in Slovenian documents, which the project accepts.

```bash
uv run python scripts/curate_pretraining_corpus.py diagnose --config "$CURATION" \
    --candidates sl,en,de,hr,sr,it,fr
```

It samples the finished corpus and writes nothing. Narrowing `--candidates` to
the likely confounders is much faster than the config's full European set and
still separates Slovenian from English. `--base-dir` points it at a stage other
than the final corpus; `--per-dataset` and `--max-shards-per-dataset` size the
sample.
