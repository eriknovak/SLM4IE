# CLAUDE.md

Project-specific guidance for working in SLM4IE. Global Python and writing rules
in `~/.claude/` still apply; this file only records what is specific to this
repository.

## What this project is

Small language models for zero-shot information extraction across European
languages, with emphasis on Slovenian. It is a **collection of experiments**,
not a service: each one tests a hypothesis, they may depend on each other or
stand alone, and the shared machinery exists to run them.

Three kinds of machinery every experiment draws on:

- Pretraining-corpus preparation (download, extract, curate).
- Tokenizer + model training (SLURM-friendly, MLflow-tracked).
- Evaluation on Slovenian benchmarks (NER, SA, SuperGLUE, etc.) driven by
  the `tasks.yaml` registry.

See `experiments/README.md` for what has been tried and what it showed, and
`docs/` for setup, the data pipeline, corpus curation, the tokenizer sweep and
the dataset catalog. `CONTEXT.md` fixes the vocabulary — use its terms rather
than synonyms when writing code, configs or records.

## Environment

- **Python 3.13+** (pinned in `pyproject.toml` and `.python-version`).
- **uv** is the canonical package manager. Use `uv sync` to install and
  `uv run <cmd>` to execute scripts. Do not invoke `pip` directly.
- Dependencies are **named groups**, not extras: `uv sync` gives base + `dev`,
  and each workstream adds its own (`data`, `corpus`, `tokenizers`,
  `analysis`, all of which pull in `tracking` for MLflow; `docs` builds the
  showcase site).
- **Ruff** is configured with `line-length = 120`, `target-version = "py313"`,
  double quotes, `docstring-code-format = true`. Match this when generating
  code; do not raise the line length.
- **pytest** lives under `tests/`. Run `uv run pytest` (or a path subset);
  end-to-end pipeline runs carry the `slow` marker, so `-m "not slow"` skips
  them.

## Repository layout

```text
experiments/ The experiments themselves — one folder per hypothesis, plus the
             findings book, glossary, and generated HTML report.
slm4ie/      Library code — importable modules only, no CLI logic.
scripts/     Thin CLI wrappers around slm4ie/. Argument parsing + config load.
             One script per pipeline, named verb + object, with one subcommand
             per step: prepare_datasets, curate_pretraining_corpus,
             sweep_tokenizers.
configs/     Shared registries only: the dataset catalogs, the task registry,
             the curation settings and the tokenizer sweep, each named for the
             subcommand that reads it.
slurm/       SLURM batch scripts for HPC training.
tests/       pytest suite mirroring slm4ie/, scripts/ and website/.
docs/        Operator documentation and agent-skill config (agents/).
website/     The showcase site: MkDocs sources, theme overrides, image assets.
data/        Gitignored symlink to /vault/data/SLM4IE/ (see Data layout).
logs/        Gitignored per-run logs written by the scripts.
```

When adding a new pipeline step, put the logic in `slm4ie/` and add a small
wrapper in `scripts/` that loads a YAML from `configs/`.

## Experiment layout

```text
experiments/
  README.md                        the findings book — index of every experiment
  GLOSSARY.md                      metrics, abbreviations, project terms
  build_report.py                  the HTML report builder
  report.html                      generated; never hand-edited
  <category>/<slug>/
    README.md                      the record: hypothesis → decisions → findings → verdict
    configs/<type>-<factor>.yaml   this experiment's dials
    analysis.py                    pulls MLflow runs → tables/ + figures/
    tables/*.csv  figures/*.svg    static, committed, linked from the record
```

Categories are chosen by what the hypothesis is *about*: `data/` (corpora,
splits, filtering), `methods/` (tokenizers, architectures, training recipes),
`validation/` (metrics, benchmarks, evaluation protocol).

No experiment has a record yet. `data/pretraining-corpus-slovenian/` and
`methods/tokenizer-sweep-slovenian/` are reserved slugs awaiting a hypothesis:
their former configs are now shared registries under `configs/`, the prior
tokenizer-sweep runs are parked in MLflow under `slm4ie/archive/`, and the
write-ups of those runs live in `docs/notes/`.

Experiments may depend on each other. The dependency is explicit — the record's
frontmatter carries `builds_on: [../../data/<slug>/]` and its Hypothesis states
what the parent left open. An independent experiment leaves the field empty.

Naming derives from one choice, the slug, so it stays consistent everywhere:

| Thing  | Convention                                       | Example                                    |
| ------ | ------------------------------------------------ | ------------------------------------------ |
| Slug   | 2–4 kebab words naming what is under test        | `tokenizer-sweep-slovenian`                |
| Branch | `exp/<slug>` main-line, `study/<slug>` secondary | `exp/tokenizer-sweep-slovenian`            |
| MLflow | `slm4ie/<category>/<slug>`                       | `slm4ie/methods/tokenizer-sweep-slovenian` |
| Config | `<type>-<factor>.yaml`, or `<type>-<setting>`    | `sweep-vocab-size.yaml`                    |
| Figure | `<config-stem>-<quantity>-by-<dimension>`        | `sweep-fertility-by-vocab.svg`             |

A dial's name says what the experiment varies (`sweep-vocab-size.yaml`); when
nothing varies it names the setting it fixes (`curation-south-slavic.yaml`).

Shared pipeline stages track lineage under `slm4ie/data/<stage>` instead
(`slm4ie/data/extract`, `slm4ie/data/tasks`); those runs belong to no
experiment.

The project's mission is language-specific, so the slug names the language when
it is a **fixed setting** (`tokenizer-sweep-slovenian`) and names a family when
language is the **factor under test** (`tokenizer-sweep-south-slavic`); a
genuinely language-agnostic experiment omits it (`dedup-threshold`). Every run
also carries a `language` tag (`sl`, or `sl,hr,sr`) so MLflow stays exactly
filterable.

## Data layout

Datasets live **outside the repo** at `/vault/data/SLM4IE/`. Never commit
data files; respect the `.gitignore`. `data/` in the repo is a gitignored
symlink to that root, so the paths below are reachable as `data/<tier>/…`. The
tree has seven tiers:

```text
raw/<key>/...                                  # prepare_datasets.py download
extracted/                                     # prepare_datasets.py extract
  <key>.jsonl
  <key>.annotations.jsonl.gz
pretrain/                                      # curate_pretraining_corpus.py run
  00_convert/<key>/*.jsonl.gz                    # datatrove `Document` shape
  01_language/ … 06_sentence_dedup/              # filter/spam/dedup stages
  07_statistics/
tasks/<task>/<dataset>/{train,val,test}.jsonl.gz  # prepare_datasets.py tasks
tokenization/<dataset>.jsonl.gz                # prepare_datasets.py tokenization
tokenizers/<sweep run>/                        # sweep_tokenizers.py artifacts
  _reports/                                      # the sweep comparison report
experiments/<category>/<slug>/                 # per-experiment derived data
  raw/ interim/ final/                           # as downloaded / partial / consumed
```

Everything above the `experiments/` tier is **shared data**: many experiments
read the same corpus, so it is never rebuilt per experiment. Only data an
experiment derives for itself goes under the `experiments/` tier.

Extraction produces two artifacts per dataset under `extracted/`, joined on the
fly downstream — never materialize a merged file:

- `<key>.jsonl` — text + `source` / `domain` / `doc_id` / `metadata`.
- `<key>.annotations.jsonl.gz` — gzipped per-document annotations (parallel
  arrays: `forms`, `lemmas`, `upos`, `feats`, `sentences`, plus `spans` when
  present).

Use `slm4ie.data.io_utils.iter_joined_records` to consume both together.

## Conversion routes — keep them separate

Downstream consumers fork after extraction. There are three routes, and
they own disjoint output trees:

1. **Pretraining (`curate_pretraining_corpus.py run`):** a thin CLI over
   `slm4ie/data/curate/runner.py`, which runs eight sentinel-skippable stages
   (0–7) on top of [datatrove](https://github.com/huggingface/datatrove).
   Stage 0 (`convert`) lifts `extracted/<key>.jsonl` into the `Document` shape
   (`text` / `id` / `metadata`, with `dataset` and `domain` for
   source-weighted sampling); stages 1–7 do language filtering, adult/SEO-spam
   removal, Gopher quality + repetition heuristics, exact + sentence dedup,
   and corpus statistics. Output:
   `pretrain/00_convert/ … pretrain/07_statistics/`.
   Driven by `configs/data/curate.yaml`.
   The annotations sidecar is **not** read here — it would desync after any
   datatrove step that rewrites the text.

2. **Tasks (`prepare_datasets.py tasks`):** reads `configs/data/tasks.yaml`,
   a flat registry keyed `<task>/<dataset>`, and routes each entry to the
   converter backend that entry names (`slm4ie/data/tasks/converters/`), so one
   invocation spans every family. Outputs go to
   `tasks/<task>/<dataset>/<split>.jsonl.gz` using task-family schemas defined
   as TypedDicts in `slm4ie/data/schema.py`.
   - `spans` handles every `ner/*` entry and emits GLiNER-style output.
     Requires a `spans` field in the annotations payload.
   - `sentiment` handles `sentiment/*` entries with normalized
     `{negative, neutral, positive}` labels.
   - `superglue` handles every SuperGLUE-SL subtask, dissolved into
     `nli/`, `qa/`, `coref/`, `wsd/`, `commonsense/`. The `--variant`
     flag picks `humant` (default) or `googlemt`.

   Train/test isolation is enforced by each entry's `role` field
   (`finetune_and_eval` vs `held_out`), **not** by directory placement.
   Document-shaped sources use `source.kind: extracted`; task-native
   bundles (SuperGLUE-SL) bypass `extracted/` via `source.kind: raw`.

3. **Tokenizer quality (`prepare_datasets.py tokenization`):** reads
   `configs/data/tokenization.yaml`, writes `tokenization/<dataset>.jsonl.gz`.
   Lexicon-derived (Sloleks, etc.); never enters the pretraining corpus.

All routes skip existing outputs unless `--force` is passed and accept either
positional keys (`<task>/<dataset>` entry keys for the task route) or `--all`.

### Tokenizer-comparison sweep (shared machinery, not a conversion route)

Its settings are the shared registry `configs/tokenizers/sweep.yaml` and its
deps sit behind the `tokenizers` group. An experiment that varies the sweep
keeps its own dials under its slug's `configs/` and its findings in its record.
`scripts/sweep_tokenizers.py` (subcommands `sample`, `train`, `evaluate`,
`export`; library code in `slm4ie/tokenizers/`) trains six tokenizers across a
vocab sweep, scores them with six metrics, and exports each as a HuggingFace
tokenizer for LM-pretraining. `sample` is optional: it materializes the shared
seeded training sample + morpheme lexicon up front so reruns reuse an identical
persistent sample.

- It is a **consumer**, not a fourth conversion route: it reads the deduplicated
  corpus (`pretrain/06_sentence_dedup/`) for training and
  `tokenization/sloleks.jsonl.gz` for the morpheme-derived gold, and writes
  artifacts + a report under `/vault/data/SLM4IE/tokenizers/`.
- Each backend is faithful to its original work (byte-level BPE, char-level
  charBPE, BERT WordPiece, SentencePiece Unigram, MorphBPE =
  constrained-training/standard-inference, MorphPiece = byte-level BPE +
  MorphTable).
- The morpheme gold is derived from Sloleks (`slm4ie/tokenizers/morphology.py`)
  and is **inflectional silver gold** — the morph metrics are offset-based
  relative comparators, not absolute morphology.
- Morph-Edit-Distance (raw, lower-better) and Morph-Consistency (F1) follow the
  MorphBPE paper (arXiv 2502.00894); MorphScore follows Arnett et al.
- Adding new tokenizers means new `@register_tokenizer` backends under
  `slm4ie/tokenizers/backends/`, not new scripts.

## Documentation style — Google-style docstrings (REQUIRED)

**Every public module, class, function, and method MUST have a Google-style
docstring.** This is the single most enforced rule in this codebase. No
exceptions for "small" or "obvious" functions, no reST syntax leaking in.

Banned (reST / Sphinx — never use):

- `::` literal-block marker, section underlines (`====`, `----`).
- Field lists: `:param x:`, `:returns:`, `:raises:`, `:rtype:`.
- Double backticks (`` ``code`` ``) for inline code — use single backticks.
- Directives like `.. note::`, `.. code-block::`, `.. deprecated::`.

Required shape:

```python
def extract(dataset: str, force: bool = False) -> Path:
    """Extract a raw dataset to unified JSONL.

    Args:
        dataset: Key from configs/data/extract.yaml.
        force: Re-extract even if the output already exists.

    Returns:
        Path to the produced `<key>.jsonl` file.

    Raises:
        KeyError: If `dataset` is not declared in the config.
    """
```

Type hints required on all public signatures. Use `typing` collection generics
(`List`, `Dict`, `Optional`, etc.) for consistency across the codebase.

### Verify with ruff before claiming Python work is done

Ruff bundles pydocstyle, and `pyproject.toml` already pins
`[tool.ruff.lint.pydocstyle] convention = "google"`. Always run this against the
files you changed (or `slm4ie/ scripts/ experiments/` for a sweep) **before**
reporting a Python change as complete or committing it:

```bash
uv run ruff check --select D <changed-paths>
```

This catches missing docstrings (`D1xx`), formatting drift (`D2xx`), wording
(`D3xx`), section ordering and Google-convention violations (`D4xx`). Fix
every reported issue — do not silence with `# noqa` unless the rule is
genuinely wrong for that file, and then say why in the same line.

If the change touches argument lists or return types, also eyeball that
`Args:` / `Returns:` / `Raises:` still match the new signature; ruff
checks structure but not semantic agreement with the code.

## Conventions worth honoring

- Configs are YAML; never hardcode dataset URLs, paths, or hyperparameters in
  Python. Shared **registries** — anything every experiment reuses — live in
  `configs/`; an experiment's **dials** live in
  `experiments/<category>/<slug>/configs/`. A shared registry is named for the
  subcommand that reads it, so `prepare_datasets.py extract` reads
  `configs/data/extract.yaml` and `curate_pretraining_corpus.py run` reads
  `configs/data/curate.yaml`.
- Secrets and ephemeral values (presigned download URLs, machine-local paths) go
  in a gitignored `*.local.yaml` sibling, which `load_config` deep-merges over
  the base config. The `.githooks/pre-commit` hook blocks presigned-URL
  credentials; enable it with `git config core.hooksPath .githooks`.
- Scripts in `scripts/` should stay thin: parse args, load config, dispatch
  into `slm4ie/`. Don't hide library logic inside a script. A new step joins an
  existing script as a subcommand; a new script needs a pipeline of its own.
  Shared argument shapes (selection, workers, config paths, log dirs) come from
  `slm4ie/utils/cli.py`.
- Annotated extractors should keep text and annotations split (see Data
  layout). Don't add a "merged" output without a strong reason.
- Tests under `tests/data/` use small fixtures committed in-tree; do not
  point tests at `/vault/data/SLM4IE/`.

## Things to avoid

- Committing data, model checkpoints, or `.env` files.
- Using `pip install` instead of `uv add` / `uv sync`.
- Bumping `line-length` above 120 to fit a long line — refactor instead.
- Cross-importing between `scripts/` modules; share via `slm4ie/`.
- Reintroducing the retired data tiers `processed/`, `final/`, `benchmarks/`,
  or the retired `--schema {gliner|conll|generic}` flag; only the GLiNER task
  schema remains.
- Adding a fourth conversion route alongside pretraining curation / task
  conversion / tokenization without discussing it first — the split is
  intentional. New task families belong as new entries in `tasks.yaml`,
  not as new scripts.

## Agent skills

### Issue tracker

Issues are tracked as GitHub issues in `eriknovak/SLM4IE` via the `gh` CLI.
See `docs/agents/issue-tracker.md`.

### Triage labels

Default canonical labels (`needs-triage`, `needs-info`, `ready-for-agent`,
`ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` at the root is the only domain doc. There is no
`docs/adr/`; delivery decisions go in commit messages and PR bodies, experiment
decisions in the record's `## Decisions`.

## labflow

archetype: project-repo
