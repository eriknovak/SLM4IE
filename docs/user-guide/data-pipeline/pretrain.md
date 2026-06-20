---
title: Pretraining corpus
---

# Pretraining corpus (`to_pretrain.py`)

!!! success "Verified end-to-end"
    This route has been run end-to-end. The corpus it produced is profiled on
    the [Corpus Statistics](../../datasets/corpus-statistics.md) page
    (28.3M documents, 7.23B words).

`scripts/data/to_pretrain.py` builds the final pretraining corpus as a sequence
of **eight independent, sentinel-skippable stages** on top of
[datatrove](https://github.com/huggingface/datatrove). Stage 0 lifts
`extracted/*.jsonl` into datatrove's `Document` shape; stages 1–7 cover language
filtering, adult/SEO-spam removal, Gopher within-document quality and repetition
heuristics, cross-corpus exact and sentence deduplication, and corpus
statistics. Driven by
[`configs/data/pretrain.yaml`](https://github.com/eriknovak/SLM4IE/blob/main/configs/data/pretrain.yaml).

## Install the extra

```bash
uv sync --extra curate
```

The `curate` extra pulls in `datatrove`, `lingua-language-detector`, `spacy`
(Slovenian word/sentence tokenization), `classla` (lemmatizer for the keyword
TF-IDF pass), and a few smaller helpers. datatrove's own `processing` and
`multilingual` extras are deliberately skipped — they pull in
`fasttext-numpy2-wheel`, which has no Python 3.13 wheel.

## Canonical command

```bash
uv run python scripts/data/to_pretrain.py --all
```

This iterates all eight stages in order, skipping any whose config-slice hash
matches the recorded sentinel. The final corpus lands at
`pretrain/05_2_dedup/<dataset>/<rank>.jsonl.gz`; statistics at
`pretrain/06_statistics/`.

## The eight stages

Each stage reads its predecessor's output and writes a numbered folder plus a
`.complete` sentinel. The two dedup sub-stages are independent: `05_1_dedup`
removes whole-document duplicates across the corpus; `05_2_dedup` runs
sentence-level dedup over that result.

| CLI name | Folder | Scope | What it does |
|----------|--------|-------|--------------|
| `convert` | `00_convert/` | per-doc | lift `extracted/<key>.jsonl` into datatrove `Document` shards; carries `dataset` and `domain` for source-weighted sampling |
| `language` | `01_language/` | per-doc | lingua-py language detection (tag or filter) |
| `spam` | `02_spam/` | per-doc | adult/SEO-spam removal via per-language lexicons + URL/domain blocklist |
| `quality` | `03_quality/` | per-doc | Gopher within-document quality heuristics (length, word lengths, symbol/bullet/ellipsis ratios, stopword floor) |
| `repetition` | `04_repetition/` | per-doc | Gopher within-document repetition heuristics (duplicate paragraphs/lines, top-n-gram saturation) |
| `exact_dedup` | `05_1_dedup/` | corpus-wide | whole-document exact dedup (xxhash64 of `doc.text`) |
| `sentence_dedup` | `05_2_dedup/` | corpus-wide | N-sentence sliding-window dedup (final corpus) |
| `stats` | `06_statistics/` | corpus-wide | word/n-gram tables and optional classla TF-IDF keywords |

## Sentinels and incremental reruns

Each stage's sentinel hash covers its own top-level `pretrain.yaml` section, so
editing one section cascade-invalidates that stage plus every downstream stage.
The hash also folds in the sorted list of dataset keys this run will process —
so switching between `--all` and a positional subset, or adding a dataset to
`extract.yaml`, correctly triggers rebuilds.

!!! note "Refreshed inputs auto-rebuild"
    `convert` also tracks a size + modification-time fingerprint of each source
    `<key>.jsonl`. Re-extracting a dataset (`extract.py <key> --force`) marks
    `convert` stale for that key and cascades downstream, so a plain
    `to_pretrain.py --all` re-folds the updated data with no `--force` needed.
    The fingerprint is size and time only — never the file contents.

!!! note "Per-dataset overrides"
    An optional top-level `overrides:` block lets a single dataset patch any
    **scoped** stage's config without forking the file. It is keyed by dataset,
    then by stage, and deep-merges over the global section:

    ```yaml
    overrides:
      slovenian_news:
        quality:
          max_ellipsis_lines_ratio: 0.9   # news prose uses "…" mid-article
    ```

    Only scoped stages (`convert`, `language`, `spam`, `quality`, `repetition`)
    are overridable; naming a corpus stage or an unknown knob is a hard error at
    load.

## Useful invocations

```bash
# Run all eight stages, skipping any whose config-slice hash is unchanged.
uv run python scripts/data/to_pretrain.py --all

# Run only one stage. If its hash diverges, downstream sentinels are dropped
# so the next --all picks them up.
uv run python scripts/data/to_pretrain.py --all --stage quality

# Force-rebuild a stage and everything downstream (removes data + sentinels).
uv run python scripts/data/to_pretrain.py --all --force --stage quality

# A single dataset, or a subset.
uv run python scripts/data/to_pretrain.py kzb solar

# Parallelism. Default is 1 (serial). 0 = cpu_count // 2.
uv run python scripts/data/to_pretrain.py --all --max-workers 8
```

!!! warning "Whole-pipeline workers"
    Unlike the other converters, `--max-workers` here is a **whole-pipeline**
    count — every parallel datatrove executor inside a stage uses it. The
    default is `1` (serial) so a casual `--all` does not saturate the box.

## Configuration

`configs/data/pretrain.yaml` has one top-level section per stage plus shared
`input_dir`, `output_dir`, and a `stopwords:` path used by both `quality` and
`stats`. Defaults match the Gopher paper for the heuristic filters, 64-bit
xxhash for exact dedup, 3-sentence windows for sentence dedup, and
top-5000 word/bigram/trigram + top-200 TF-IDF keyword tables for stats. To skip
the slow classla-lemmatized keyword pass, set `stats.compute_keywords: false`.
The first keyword run downloads the Slovenian classla model (~200 MB) under
`~/.classla_resources/`.

## Output layout

```text
pretrain/
├── 00_convert/<key>/<rank>.jsonl.gz      ← datatrove Document shards
├── 01_language/<key>/<rank>.jsonl.gz
├── 02_spam/<key>/<rank>.jsonl.gz
├── 03_quality/<key>/<rank>.jsonl.gz
├── 04_repetition/<key>/<rank>.jsonl.gz
├── 05_1_dedup/<key>/<rank>.jsonl.gz      ← post-exact-dedup
├── 05_2_dedup/<key>/<rank>.jsonl.gz      ← final pretraining corpus
├── 06_statistics/
│   ├── aggregate.json                    corpus-wide totals + tables
│   └── per_dataset/<key>.json            per-dataset breakdowns
├── _dedup_state/                         sig/find scratch (auto-purged)
└── _logs/<stage>/                        datatrove per-executor logs
```

See [Corpus Statistics](../../datasets/corpus-statistics.md) for what
`06_statistics/aggregate.json` contains for the current corpus.
