# Design: Adult / SEO-Spam Filter Stage

Date: 2026-06-08
Status: Approved (pending spec review)
Addresses: `docs/pretrain-corpus-todos.md` Item 1 (🔴 Adult / SEO spam contamination)

## Problem

The completed pretraining corpus (27.18M docs / ~7.87B words) is contaminated
with adult and SEO-spam web content. The global top-200 word table surfaces
escort/porn/dating vocabulary at extreme ranks — `prostitutke` is the ~10th most
frequent content word (10.9M occurrences), alongside `porno`, `seks`,
`kurba`/`kurbe`, and SEO patterns. Neither the lingua language filter nor the
Gopher quality/repetition heuristics remove this, because it is grammatical
Slovenian. A general/professional information-extraction model should not be
pretrained on it.

## Constraints and findings

- No robust universal model exists for "adult + SEO spam" across languages.
- datatrove ships `C4BadWordsFilter` using LDNOOBW word lists for ~28 languages,
  but Slovenian (`sl`), Croatian (`hr`), and Serbian (`sr`) are **not** covered.
- URLs survive into the corpus for most web sources (e.g. a sampled c4 spam doc
  carries `metadata.url` pointing at an SEO weight-loss-tea domain), so a
  URL/domain blocklist is viable — but `cc100` carries no URL metadata, so URL
  cannot be the only signal.
- The pipeline is sentinel-tracked; a new stage is a clean addition to
  `stages.py` + `pipeline.py` + `pretrain.yaml` + the `to_pretrain.py` runner.

## Decisions (from brainstorming)

1. Approach: **lexicon + URL blocklist, model-pluggable** (not a single
   multilingual model).
2. Action: **drop, with a configurable `keep_fraction`**; write the match reason
   to metadata for retained-flagged docs and tally drop reasons in stage stats.
3. Placement: **new scoped stage `spam`, after `language`, before `quality`**.
4. Language configs: **ship curated `sl` + `en`; auto-wire LDNOOBW** for other
   languages on demand; URL blocklist is language-agnostic.
5. Renumber downstream stage folders (approved).
6. Ship both the `adult` and `spam` (SEO/scam) word-list categories.

## Architecture

New per-dataset scoped stage `spam` inserted between `language` and `quality`.
Stages reference folders by name (`STAGE_DIRS`), so renumbering is a dict change.

New folder numbering:

```text
00_convert  01_language  02_spam (NEW)  03_quality  04_repetition
05_1_dedup  05_2_dedup  06_statistics
```

The filter runs before quality/repetition/dedup, so those stages rebuild into the
new folder names regardless; renumbering adds no extra compute. The old
`02_quality`, `03_repetition`, `04_1_dedup`, `04_2_dedup`, `05_statistics`
directories become orphans to delete after a clean re-run. Re-run is from `spam`
onward (~1–2 days wall-clock).

Data flow:

```text
01_language/<ds>/*.jsonl.gz
  -> JsonlReader
  -> SpamFilter (URL blocklist + lexicon + optional model; keep_fraction)
  -> JsonlWriter
  -> 02_spam/<ds>/*.jsonl.gz
  -> (downstream) 03_quality ...
```

## Components

### `slm4ie/data/curate/spam.py`

- `SpamConfig` dataclass mirroring the YAML knobs (see below), following the
  `QualityConfig` pattern.
- `SpamFilter(BaseFilter)` (datatrove `BaseFilter` subclass):
  - Lazy-loads, per requested/encountered language, two word sets: `adult` and
    `spam` (SEO/scam). Curated repo lists take precedence; LDNOOBW lists are
    fetched via `cached_asset_path_or_download` when `use_ldnoobw` is true and no
    curated file exists for that language.
  - Lazy-loads the URL/domain blocklist: custom `domains.txt` (repo) plus the
    UT1 adult domain list (auto-fetched/cached) when `url_blocklist` is true.
  - `filter(doc)` flags the doc when **any** signal triggers:
    - `metadata.url` registered domain or fqdn is on the blocklist;
    - adult-term hits `>= min_adult_hits`;
    - spam-term hits `>= min_spam_hits`;
    - (optional) model score `>= model_threshold` when a model is configured.
  - Word matching uses whole-word boundaries (flank by non-word chars), mirroring
    `C4BadWordsFilter`, to avoid substring false positives.
  - Language selection: `doc.metadata.get("language", default_language)`.
  - `keep_fraction` (seeded `np.random` uniform, like `C4BadWordsFilter`):
    flagged docs are retained when `uniform() < keep_fraction`; retained docs get
    `metadata.spam_reason` recording why they were flagged.
  - Drops return `(False, reason)` so datatrove records per-reason drop counts in
    `_logs/spam/stats`.

### Assets `slm4ie/data/spam/`

Mirrors the `slm4ie/data/stopwords/` convention (plain-text, one term per line):

```text
slm4ie/data/spam/
  sl/adult.txt    # curated, unambiguous Slovenian adult terms
  sl/spam.txt     # curated Slovenian SEO/scam terms
  en/adult.txt
  en/spam.txt
  domains.txt     # custom adult/spam domains (language-agnostic)
  __init__.py
```

LDNOOBW lists for non-shipped languages are fetched and cached at runtime (not
vendored). The UT1 adult domain list is likewise auto-fetched/cached.

#### Lexicon precision (load-bearing)

Word lists contain **only unambiguous** adult/spam terms (e.g. `prostitutke`,
`porno`, `seks`, `kurba`, `kurbe`, explicit terms; SEO: casino/replica/loan-scam
terms). Ambiguous common words seen in the top-200 (`ženske`, `telo`, `masaža`,
`zmenke`) are deliberately **excluded** — the URL blocklist plus the `>= 2`-hit
threshold catch those contexts without discarding legitimate health/dating/
massage-therapy text. Precision is prioritized over recall for the lexicon.

### `configs/data/pretrain.yaml` — new `spam:` section

```yaml
spam:
  languages: [sl, en]      # curated lists loaded eagerly; others via LDNOOBW on demand
  use_ldnoobw: true        # auto-load LDNOOBW for languages lacking a curated file
  url_blocklist: true      # enable URL/domain blocklist (UT1 adult + custom domains.txt)
  min_adult_hits: 2        # flag when adult-term hits reach this count
  min_spam_hits: 2         # flag when SEO/scam-term hits reach this count
  keep_fraction: 0.0       # fraction of flagged docs to retain (seeded sampling)
  default_language: sl     # language assumed for docs without metadata.language
  model: null              # optional classifier spec; null = lexicon + URL only
```

### `slm4ie/data/curate/pipeline.py`

- `SpamConfig` import + `build_spam_executors(paths, *, tasks, spam_config,
  spam_assets_dir, input_override)` returning one `LocalPipelineExecutor`:
  read `01_language/` → `SpamFilter` → write `02_spam/`. Mirrors
  `build_quality_executors` (including `input_override` for subset symlinks and
  `skip_completed=False`).

### `slm4ie/data/curate/stages.py`

- Add `"spam"` to `STAGE_NAMES` (between `language` and `quality`) and to
  `SCOPED_STAGES`.
- `STAGE_DIRS` renumbered as above; `spam -> "02_spam"`.
- `_CONFIG_SLICE_KEYS["spam"] = ("spam",)`.
- `final_corpus_dir()` / `stats_dir()` keep returning by name (now `05_2_dedup` /
  `06_statistics` via the dict) — no signature change.

### `slm4ie/data/curate/sentinel.py`

Extend the existing "include extra asset-file contents in the sentinel hash"
mechanism (already used for the stopword file in `quality`/`stats`) so the
`spam` stage's hash also covers the loaded spam word-list and domain files.
Editing any list invalidates `spam` and cascades downstream.

### `scripts/data/to_pretrain.py`

- Load the `spam` config slice into a `SpamConfig`.
- Dispatch `spam` in the scoped-stage execution path (it is a `SCOPED_STAGE`,
  so the existing scoped machinery — subset symlinks, per-dataset sentinels —
  applies).

### `tests/data/test_spam.py`

Small in-tree fixtures (tiny lists or the real `sl` list). Cover:

- whole-word boundary matching (no substring false positives);
- adult/spam hit thresholds (`>=` boundary behavior);
- URL domain + fqdn blocklist matching; missing-URL docs are skipped;
- `keep_fraction` determinism under a fixed seed;
- language fallback to `default_language`;
- the curated `sl` list flags the known offenders (`prostitutke`, `porno`, ...).

## Error handling

- A requested curated language in `languages` with no asset file → raise at
  startup (mirrors the stopwords contract).
- On-demand LDNOOBW for a doc's language, uncached and offline → log a warning
  once and treat that language as having an empty list (no crash).
- Missing `metadata.url` → skip the URL check for that doc.
- `model` set but its dependencies/asset are unavailable → raise at startup with
  a clear message.

## Testing and verification

- Unit tests above run under `uv run pytest tests/data/test_spam.py`.
- `uv run ruff check --select D slm4ie/ scripts/` clean on changed files.
- Dry-run the stage on a handful of `c4`/`cc100` shards; inspect the drop rate
  and a sample of dropped docs for precision.
- After the full re-run from `spam`, re-check `06_statistics/aggregate.json`
  `word_freq_top_200` to confirm the offenders are gone and no obvious
  legitimate vocabulary was decimated.

## Out of scope (YAGNI)

- Shipping a default classifier model (only the pluggable hook is built).
- Vendoring the full LDNOOBW set or the UT1 list into the repo (fetched/cached).
- Training-time tagging workflow (action is drop + `keep_fraction`).
- Source-weighting and language-leakage work (separate TODO items 2 and 3).
