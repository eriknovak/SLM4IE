# Building the pretraining corpus

`scripts/curate_pretraining_corpus.py run` builds the final pretraining corpus
as a sequence of eight independent, sentinel-skippable stages on top of
[datatrove](https://github.com/huggingface/datatrove). Stage 0 lifts
`extracted/*.jsonl` into datatrove's `Document` shape; stages 1–7 cover language
filtering, adult/SEO-spam removal, Gopher within-document quality and repetition
heuristics, cross-corpus exact and sentence deduplication, and corpus
statistics.

Each stage writes a durable on-disk artifact and a `.complete` sentinel under
`output_dir`, one per dataset for the scoped stages. A rerun rebuilds only what
something output-affecting changed for — config, code, or input documents — and
stops as soon as a rebuilt stage reproduces the same documents (see
[Versioning](#versioning-what-triggers-a-rebuild)). `input_dir` is the folder of `<key>.jsonl` files from the
extract step; `output_dir` is the pretrain-owned tree. The dataset key list
comes from [`configs/data/extract.yaml`](../configs/data/extract.yaml); entries
marked `role: benchmark` (evaluation gold such as SUK) are skipped by `--all`
so they never enter the corpus, and `access: gated` marks licence-bound sources
whose totals are reported separately from the open ones.

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

This iterates all eight stages in order, skipping every unit that is current,
and rewrites the lock file `configs/data/curate.lock.yaml`. The final corpus lands at
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

The `spam` stage matches its lexicon by stem, so inflected forms count: an
entry's last token drops one trailing vowel when it has five or more letters
and then accepts up to three more letters (`joške` matches `joškah`,
`joškami`). `min_adult_hits` / `min_spam_hits` count **distinct stems**, so
one word repeated or inflected counts once. Every dropped document is written
with its `metadata.spam_reason` and `metadata.spam_terms` (the matched stems)
to `02_spam/<key>/removed/<rank>.jsonl.gz`. That folder travels with the unit
but is outside its integrity check and document digest, and the next stage
never reads it.

Internally each dedup stage chains three datatrove executors via `depends=`:
signature → find (single-worker reducer over signatures) → filter + write. The
sig/find scratch lives at `<output_dir>/_partial/<stage folder>.scratch/`, beside
the stage's staging folder, and is purged when the stage is promoted. The
statistics stage maps `CorpusStats` over the corpus
into per-task partials and reduces them in one process. The sentence-dedup blocks
use `Languages.slovenian` so datatrove dispatches its bundled Slovenian
`SpaCyTokenizer` for sentence boundaries; its signature step packs hashes into
numpy chunks rather than datatrove's per-sentence Python tuples, which cuts its
memory about tenfold and writes identical files.

The corpus-wide stages read only the roster's datasets (through a symlink view
at `<output_dir>/_inputs/<stage>/`), so folders left upstream by keys dropped
from `extract.yaml` or marked `role: benchmark` never reach them. They run one
datatrove task per input shard, so a task holds at most one shard in memory and
`--max-workers` only sets how many tasks run at once.

**Resuming a crashed corpus stage.** A corpus stage builds in
`<output_dir>/_partial/<stage folder>/` and records its config hash, stage
version, task count and inputs in `.in_progress.json` there when it starts.
Rerunning the same command after a crash resumes: datatrove skips the tasks it
marked complete under `_logs/<stage>/`, and the dedup scratch is kept. If any of
those values changed, the stage starts fresh and first clears its staging
folder, logs and scratch. The previously promoted output stays in place until
the new one is finished and checked.

## Versioning: what triggers a rebuild

A **unit** is one scoped stage for one dataset (`03_quality/kzb/`) or one corpus
stage (`05_exact_dedup/`). Each unit's sentinel records its lineage:

| Field             | What it is                                                                                         |
| ----------------- | -------------------------------------------------------------------------------------------------- |
| `config_hash`     | hash of the stage's (override-merged) config section, plus the files it names — see below          |
| `stage_version`   | a hash of the stage's module in `slm4ie/data/curate/stages/` (both dedup stages share `dedup.py`); imports are not followed |
| `input_digest`    | the upstream unit's recorded document digest; for `convert`, the size and SHA-256 of the extracted source file(s) |
| `document_digest` | order-independent hash of the documents the unit wrote                                             |
| `shards`          | every file the unit wrote, by relative path and byte size                                          |
| `info`            | git commit, datatrove version, completion time — for reference only, never compared               |

A unit is rebuilt when its config hash, stage version or input digest differs,
when its files on disk no longer match `shards` ("output changed on disk"), or
when it failed its integrity check. Nothing else triggers a rebuild:

- **Worker count, shard layout, compression and mtimes never count.** The
  document digest sums a SHA-256 per document over its canonical JSON, ignoring
  the reader-stamped `metadata.file_path`, so the same documents give the same
  digest however they are split. Copying, `rsync`-ing or `touch`-ing shards or
  extracted files changes nothing; a touched extracted file is rehashed once.
- **Early cutoff.** A stage that reruns — say after a refactor or a comment
  edit changed its code hash — and reproduces the same documents leaves every
  downstream unit current.

The config hash covers the stage's own top-level section. `quality` and
`statistics` also fold in the stopword file, `spam` folds in its lexicon and
domain lists, and the corpus stages fold in the sorted roster, so editing
`stopwords/sl.txt` or a `spam/<code>/*.txt` list, or adding a dataset to
`extract.yaml`, rebuilds what it should. Scoped stages exclude the roster, so a
new dataset leaves the others' scoped work alone.

**Atomic swap and integrity check.** A unit is written to
`<output_dir>/_partial/<stage folder>/`, then checked: every document id may
appear in the output at most as often as in the input, and the output may hold
no more records than the input. Only then does it replace its old folder by
rename, sentinel included. A failed check raises and keeps the old output and
sentinel; a crash mid-run leaves them intact too. Stray shards from an earlier,
wider run can therefore never survive into a rebuilt unit.

**Re-extracted inputs rebuild automatically.** Re-extracting a dataset (e.g.
`prepare_datasets.py extract <key> --force`) changes its content hash, so a
plain `run --all` refolds it — and whatever downstream it actually changes. A
dataset whose extracted file is absent is skipped, never emptied.

**Lock file.** After every successful run — a dataset subset, a single stage
or `--all` — `configs/data/curate.lock.yaml` (beside whichever config was
passed; the first run creates it) is rebuilt from the sentinels on disk: one
entry per unit with its lineage and counts. It is rewritten only when an entry
changed, so a run that rebuilt nothing leaves it untouched. Committed, it names
the corpus each commit expects.

**Status.** `status` reports every unit as `current`, `stale` with the reason,
or `missing` (nothing to build it from), compares the lock file too, and exits
1 when any unit is stale. It reads no shards and rebuilds nothing.

**Adopting sentinels from before lineage tracking.** `run` refuses to start
while legacy sentinels remain. `status --adopt` reads each legacy unit once,
checks it, computes its document digest and rewrites its sentinel, so a corpus
built earlier becomes current without a rebuild. Units failing the check are
recorded as such and are the only ones the next run rebuilds. Adoption reads
the whole corpus and every extracted file once, with one sequential reader
feeding `--max-workers` parsing processes, so a spinning disk streams instead
of seeking between files. Units that already carry lineage but a stage version
from before code hashes only get the current code version recorded — no read —
on the claim that the code as it stands built them.

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
never re-runs spuriously. The `repetition` knobs are datatrove's Gopher
repetition parameters (`dup_line_frac`, `dup_para_frac`, `dup_line_char_frac`,
`dup_para_char_frac`, and `top_n_grams` / `dup_n_grams` as lists of
`[n, fraction]` pairs).

### Pass-through units

Every content stage (`language`, `spam`, `quality`, `repetition`) takes an
`enabled` knob, default `true`. A dataset with `enabled: false` on a stage is a
**pass-through unit**: the stage's executor never sees it, and its output
folder holds symlinks to the upstream unit's shards plus a sentinel of its own.
That sentinel hashes the effective config (with `enabled: false` in it) and
carries the upstream's document digest, so every downstream unit stays current.
`status` and the run log name such a unit `pass-through`. Setting `enabled`
back to `true` changes the config hash, so the stage rebuilds that dataset.
`convert` cannot be skipped.

```yaml
overrides:
  kas:
    spam:
      enabled: false
```

### Profiles

Once several datasets share a knob set, name it once under `profiles:` and
reference it with `profile:`. The profile applies first, then the dataset's own
knobs deep-merge over it. Profiles are resolved at load time into plain
overrides, so a dataset's effective config — and its hash — is the same whether
its knobs came through a profile or were written inline. An unknown profile, a
corpus stage or an unknown knob inside a profile fails at load.

```yaml
profiles:
  curated:
    spam: {enabled: false}

overrides:
  coleslaw:
    profile: curated
  kas:
    profile: curated
    quality: {min_doc_words: 20}
```

## Output layout

```text
<input_dir>/                                upstream input (extract step)
├── <key>.jsonl
└── <key>.annotations.jsonl.gz

<output_dir>/                               curate_pretraining_corpus.py owns this entire tree
├── 00_convert/
│   └── <key>/
│       ├── <rank>.jsonl.gz                 ← datatrove `Document` shards
│       └── .complete                       sentinel: lineage + counts
├── 01_language/
│   └── <key>/{<rank>.jsonl.gz, .complete}  ← post-language-filter shards
├── 02_spam/
│   └── <key>/{<rank>.jsonl.gz, .complete}  ← post-spam-filter shards
│       └── removed/<rank>.jsonl.gz         ← dropped docs + spam_reason
├── 03_quality/
│   └── <key>/{<rank>.jsonl.gz, .complete}
├── 04_repetition/
│   └── <key>/{<rank>.jsonl.gz, .complete}
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
├── _partial/<stage>/                       staging: a unit builds here, then
│                                           replaces its old folder by rename
├── _partial/<stage>.scratch/               dedup sig/find scratch (purged
│                                           when its stage is promoted)
├── _inputs/<stage>/                        roster view a corpus stage reads
└── _logs/<stage>/                          datatrove per-executor logs and
                                            per-task completion markers
```

## Useful invocations

```bash
CURATION=configs/data/curate.yaml

# Run all eight stages, rebuilding only stale units.
uv run python scripts/curate_pretraining_corpus.py run --config "$CURATION" --all

# What would the next run rebuild, and why? Exits 1 when anything is stale.
uv run python scripts/curate_pretraining_corpus.py status --config "$CURATION"

# One-off: adopt sentinels written before lineage tracking, without a rebuild.
uv run python scripts/curate_pretraining_corpus.py status --config "$CURATION" --adopt --max-workers 16

# Run only one stage. Downstream units see its new document digest and
# rebuild on the next --all, unless its documents came out the same.
uv run python scripts/curate_pretraining_corpus.py run --config "$CURATION" --all --stage quality

# Force-rebuild a stage and every downstream stage. Removes their data
# folders AND sentinels; --force without --stage clears <output_dir>.
uv run python scripts/curate_pretraining_corpus.py run --config "$CURATION" --all --force --stage quality

# Single dataset, or a subset: runs the scoped stages for those keys only.
uv run python scripts/curate_pretraining_corpus.py run --config "$CURATION" kzb solar

# Parallelism. Default is 1 (serial). 0 = cpu_count // 2. --tasks is an alias.
uv run python scripts/curate_pretraining_corpus.py run --config "$CURATION" --all --max-workers 8
uv run python scripts/curate_pretraining_corpus.py run --config "$CURATION" --all --max-workers 0

# Override the configured paths from the CLI.
uv run python scripts/curate_pretraining_corpus.py run --config "$CURATION" --all \
    --input-dir /tmp/in --output-dir /tmp/out
```

`--max-workers` is **whole-pipeline**, not per-dataset: every parallel datatrove
executor inside one stage runs that many tasks at once, so the per-dataset log
routing of `prepare_datasets.py` does not apply here. The default is 1 (serial)
so a casual `--all` invocation does not silently saturate the box.

## Configuration

[`configs/data/curate.yaml`](../configs/data/curate.yaml)
has one top-level section per stage (`convert:`, `language:`, `spam:`,
`quality:`, `repetition:`, `exact_dedup:`, `sentence_dedup:`, `statistics:`)
plus shared `input_dir`, `output_dir`, and a `stopwords:` path used by both
`quality` and `statistics`. Each section is the **exclusive input** to that
stage's config hash, so edits propagate as far downstream as the documents
actually change and no further. Defaults match the Gopher paper for the heuristic filters, 64-bit
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

## Sampling what each stage kept and dropped

Every stage writes only the documents it keeps, so its drops exist on disk only
as the difference between its output and its input. The `sample` subcommand
reconstructs that difference and draws a balanced sample from it, so each
stage's decisions can be judged rather than assumed.

```bash
uv run python scripts/curate_pretraining_corpus.py sample --config "$CURATION" \
    --out data/experiments/data/<category>/<slug>/interim/sample.jsonl \
    --all --max-workers 12
```

The strata are `dataset x stage x decision`, where the decision is `kept` (the
document is in the stage's output) or `dropped` (it is in the stage's input but
not its output). `--per-cell` sets how many documents each cell holds (40),
`--max-chars` how much of each document is written (2,000), and `--seed` makes
the draw reproducible. One row is written per document, listing every cell it
was drawn into, so a document kept by several stages is judged once.

A stage does not keep its documents in the shard they arrived in, so no output
shard can be set against an input shard opposite it, and position pairs them
wrongly: `culturax` alone redistributes every document between `01_language`
and `02_spam`. Deciding what a stage dropped therefore needs the stage's whole
output. Each cell reads all of it once — that gives both the kept sample and an
index of surviving ids, held as one 64-bit hash per document — and then reads
input shards, where any document the index does not know was dropped.
`--shards-per-cell` (3 by default, 0 for all) bounds only that second read, so
the cost is one pass over each stage plus a few input shards.

Two caveats. A source whose document ids are not unique reports fewer drops than
it made, because a dropped document sharing an id with a surviving one reads as
a survivor — `coleslaw` is such a source. And only datasets present in the final
corpus are sampled: sources excluded from the build, or dropped entirely by a
stage, are not part of the roster.
