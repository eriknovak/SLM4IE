# CLAUDE.md

Project-specific guidance for working in SLM4IE. Global Python and writing rules
in `~/.claude/` still apply; this file only records what is specific to this
repository.

## What this project is

Small language models for zero-shot information extraction across European
languages, with emphasis on Slovenian. Three workstreams live here:

- Pretraining-corpus preparation (download, extract, curate via `to_pretrain.py`).
- Tokenizer + model training (SLURM-friendly, MLflow-tracked).
- Evaluation on Slovenian benchmarks (NER, SA, SuperGLUE, etc.) driven by
  the `tasks.yaml` registry.

See `README.md` for the dataset catalog and end-to-end commands.

## Environment

- **Python 3.13+** (pinned in `pyproject.toml` and `.python-version`).
- **uv** is the canonical package manager. Use `uv sync` to install and
  `uv run <cmd>` to execute scripts. Do not invoke `pip` directly.
- **Ruff** is configured with `line-length = 120`, `target-version = "py313"`,
  double quotes, `docstring-code-format = true`. Match this when generating
  code; do not raise the line length.
- **pytest** lives under `tests/`. Run `uv run pytest` (or a path subset).

## Repository layout

```text
slm4ie/      Library code — importable modules only, no CLI logic.
scripts/     Thin CLI wrappers around slm4ie/. Argument parsing + config load.
configs/     YAML configs (data/, models/, tokenizers/, training/, experiments/).
slurm/       SLURM batch scripts for HPC training.
tests/       pytest suite mirroring slm4ie/.
notebooks/   Exploratory only; not part of the pipeline.
docs/        Project docs and assets.
```

When adding a new pipeline step, prefer extending `slm4ie/` with the logic and
adding a small wrapper in `scripts/` that loads a YAML from `configs/`.

## Data layout

Datasets live **outside the repo** at `/vault/data/SLM4IE/`. Never commit
data files; respect the `.gitignore`. The tree has five tiers:

```text
raw/<key>/...                                  # download.py
extracted/                                     # extract.py — canonical L2
  <key>.jsonl
  <key>.annotations.jsonl.gz
pretrain/                                      # to_pretrain.py
  00_convert/<key>/*.jsonl.gz                    # datatrove `Document` shape
  01_language/ … 05_2_dedup/                     # filter/spam/dedup stages
  06_statistics/
tasks/<task>/<dataset>/{train,val,test}.jsonl.gz  # to_spans / to_sentiment / to_superglue
tokenization/<dataset>.jsonl.gz                # to_tokenization.py
```

- `scripts/data/extract.py` produces, per dataset, two artifacts under
  `extracted/` that are joined on the fly downstream — never materialize a
  merged file:
  - `<key>.jsonl` — text + `source` / `domain` / `doc_id` / `metadata`.
  - `<key>.annotations.jsonl.gz` — gzipped per-document annotations
    (parallel arrays: `forms`, `lemmas`, `upos`, `feats`, `sentences`,
    plus `spans` when present).
- Use `slm4ie.data.io_utils.iter_joined_records` to consume both together.
- Old tier names (`processed/`, `final/`, `benchmarks/`) are gone. Don't
  reintroduce them.

## Conversion routes — keep them separate

Downstream consumers fork after extraction. There are three routes, and
they own disjoint output trees:

1. **Pretraining (`to_pretrain.py`):** runs eight sentinel-skippable stages
   on top of [datatrove](https://github.com/huggingface/datatrove). Stage 0
   (`convert`) lifts `extracted/<key>.jsonl` into the `Document` shape
   (`text` / `id` / `metadata`, with `dataset` and `domain` for
   source-weighted sampling); stages 1–7 do language filtering, adult/SEO-spam
   removal, Gopher quality + repetition heuristics, exact + sentence dedup,
   and corpus stats. Output: `pretrain/00_convert/ … pretrain/06_statistics/`.
   Driven by `configs/data/pretrain.yaml`. The annotations sidecar is
   **not** read here — it would desync after any datatrove step that
   rewrites the text.

2. **Tasks (`to_spans`, `to_sentiment`, `to_superglue`):** all three read
   `configs/data/tasks.yaml`, a flat registry keyed `<task>/<dataset>`.
   They write `tasks/<task>/<dataset>/<split>.jsonl.gz` using task-family
   schemas defined as TypedDicts in `slm4ie/data/schema.py`.
   - `to_spans.py` handles every `ner/*` entry and emits GLiNER-style
     output. (The old `--schema {gliner|conll|generic}` flag is gone; only
     the GLiNER schema remains.) Requires a `spans` field in the
     annotations payload.
   - `to_sentiment.py` handles `sentiment/*` entries with normalized
     `{negative, neutral, positive}` labels.
   - `to_superglue.py` handles every SuperGLUE-SL subtask, dissolved into
     `nli/`, `qa/`, `coref/`, `wsd/`, `commonsense/`. The `--variant`
     flag picks `humant` (default) or `googlemt`.

   Train/test isolation is enforced by each entry's `role` field
   (`finetune_and_eval` vs `held_out`), **not** by directory placement.
   Document-shaped sources use `source.kind: extracted`; task-native
   bundles (SuperGLUE-SL) bypass `extracted/` via `source.kind: raw`.

3. **Tokenizer quality (`to_tokenization.py`):** reads
   `configs/data/tokenization.yaml`, writes `tokenization/<dataset>.jsonl.gz`.
   Lexicon-derived (Sloleks, etc.); never enters the pretraining corpus.

All converters skip existing outputs unless `--force` is passed and accept
either a single dataset key (or `<task>/<dataset>` entry key for the task
converters) or `--all`.

### Tokenizer-comparison stage (not a conversion route)

`scripts/tokenizers/{train,analyze}.py` (library code in `slm4ie/tokenizers/`,
config `configs/tokenizers/tokenizers.yaml`, deps behind the `tokenize` extra)
train five tokenizers across a vocab sweep and score them with six metrics. It
is a **consumer**, not a fourth conversion route: it reads the deduplicated
corpus (`pretrain/05_2_dedup/`) for training and `tokenization/sloleks.jsonl.gz`
for the morpheme-derived gold, and writes artifacts + a report under
`/vault/data/SLM4IE/tokenizers/`. The morpheme gold is derived from Sloleks
(`slm4ie/tokenizers/morphology.py`) and is **inflectional silver gold** — the
morph metrics are relative comparators, not absolute morphology. Adding new
tokenizers means new `@register_tokenizer` backends under
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
`[tool.ruff.lint.pydocstyle] convention = "google"`. Always run this against
the files you changed (or `slm4ie/ scripts/` for a sweep) **before** reporting
a Python change as complete or committing it:

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
  Python — read them from `configs/`.
- Scripts in `scripts/` should stay thin: parse args, load config, dispatch
  into `slm4ie/`. Don't hide library logic inside a script.
- Annotated extractors should keep text and annotations split (see Data
  layout). Don't add a "merged" output without a strong reason.
- Tests under `tests/data/` use small fixtures committed in-tree; do not
  point tests at `/vault/data/SLM4IE/`.

## Things to avoid

- Committing data, model checkpoints, or `.env` files.
- Using `pip install` instead of `uv add` / `uv sync`.
- Bumping `line-length` above 120 to fit a long line — refactor instead.
- Cross-importing between `scripts/` modules; share via `slm4ie/`.
- Adding a fourth conversion route alongside `to_pretrain` / task
  converters / `to_tokenization` without discussing it first — the split is
  intentional. New task families belong as new entries in `tasks.yaml`,
  not as new scripts.
