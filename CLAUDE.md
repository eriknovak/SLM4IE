# CLAUDE.md

Project-specific guidance for working in SLM4IE. Global Python and writing rules
in `~/.claude/` still apply; this file only records what is specific to this
repository.

## What this project is

Small language models for zero-shot information extraction across European
languages, with emphasis on Slovenian. Three workstreams live here:

- Pretraining-corpus preparation (download, extract, datatrove conversion).
- Tokenizer + model training (SLURM-friendly, MLflow-tracked).
- Evaluation on Slovenian benchmarks (NER, SA, SuperGLUE, etc.).

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

- Datasets live **outside the repo** at `/vault/data/SLM4IE/`. Never commit
  data files; respect the `.gitignore`.
- `scripts/data/extract.py` produces, per dataset, two artifacts that are
  joined on the fly downstream — never materialize a merged file:
  - `<key>.jsonl` — text + `source` / `domain` / `doc_id` / `metadata`.
  - `<key>.annotations.jsonl.gz` — gzipped per-document annotations
    (parallel arrays: `forms`, `lemmas`, `upos`, `feats`, `sentences`,
    plus `spans` when present).
- Use `slm4ie.data.io_utils.iter_joined_records` to consume both together.

## Two conversion routes — keep them separate

Downstream consumers fork after extraction:

1. **Pretraining (datatrove):** `scripts/data/to_datatrove.py` →
   `<output>/datatrove/<key>.jsonl.gz` in datatrove's `Document` shape
   (`text` / `id` / `metadata`). Carries `dataset` and `domain` at the top
   level for source-weighted sampling.
2. **IE / evaluation (spans + task-specific):**
   - `scripts/data/to_spans.py --schema {gliner|conll|generic}` →
     `<output>/spans/<schema>/<key>.jsonl.gz`. Requires a `spans` field in
     the annotations payload.
   - `scripts/data/to_sentiment.py` → `<raw>/eval/sentiment/<key>.jsonl.gz`
     with normalized `{negative, neutral, positive}` labels and a
     `label_map.json`.
   - `scripts/data/to_superglue.py` →
     `<raw>/eval/superglue_sl/<variant>/<task>/<split>.jsonl.gz`.

All converters skip existing outputs unless `--force` is passed and accept
either a single dataset key or `--all`.

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
- Adding a third conversion route alongside datatrove/spans without
  discussing it first — the split is intentional.
