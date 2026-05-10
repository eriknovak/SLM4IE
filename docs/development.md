---
title: Development
---

# Development

Conventions and tooling for contributors. Global Python style rules from
`~/.claude/` apply on top of these project-specific notes.

## Environment

- **Python 3.13+** is required (pinned in `pyproject.toml` and
  `.python-version`).
- **uv** is the canonical package manager. Use `uv sync` to install and
  `uv run <cmd>` to execute scripts. Do not invoke `pip` directly.

Install the development extra:

```bash
uv sync --extra dev
```

Optional extras:

```bash
uv sync --extra curate   # corpus curation pipeline (datatrove, classla, ...)
uv sync --extra docs     # documentation toolchain (mkdocs-material, ...)
```

## Repository layout

```text
slm4ie/      Library code — importable modules only, no CLI logic.
scripts/     Thin CLI wrappers around slm4ie/. Argument parsing + config load.
configs/     YAML configs (data/, models/, tokenizers/, training/, experiments/).
slurm/       SLURM batch scripts for HPC training.
tests/       pytest suite mirroring slm4ie/.
notebooks/   Exploratory only; not part of the pipeline.
docs/        Project documentation and assets.
```

When adding a new pipeline step, prefer extending `slm4ie/` with the logic
and adding a small wrapper in `scripts/` that loads a YAML from `configs/`.

## Documentation style — Google-style docstrings (REQUIRED)

Every public module, class, function, and method must have a Google-style
docstring. No exceptions for "small" or "obvious" functions, no reST syntax.

Banned (reST / Sphinx):

- `::` literal-block marker, section underlines (`====`, `----`).
- Field lists: `:param x:`, `:returns:`, `:raises:`, `:rtype:`.
- Double backticks for inline code — use single backticks.
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

Type hints are required on all public signatures. Use `typing` collection
generics (`List`, `Dict`, `Optional`, etc.) for consistency across the
codebase.

## Verify with ruff before claiming Python work is done

Ruff bundles pydocstyle, and `pyproject.toml` pins
`[tool.ruff.lint.pydocstyle] convention = "google"`. Run this against
changed files (or `slm4ie/ scripts/` for a sweep) **before** reporting a
Python change as complete:

```bash
uv run ruff check --select D <changed-paths>
```

This catches missing docstrings (`D1xx`), formatting drift (`D2xx`),
wording (`D3xx`), section ordering, and Google-convention violations
(`D4xx`). Fix every reported issue. Do not silence with `# noqa` unless
the rule is genuinely wrong, and then say why on the same line.

## Testing

Run the full pytest suite:

```bash
uv run pytest
```

Run a subset:

```bash
uv run pytest tests/data
```

Tests under `tests/data/` use small fixtures committed in-tree; do not
point tests at `/vault/data/SLM4IE/`.

## Conventions

- Configs are YAML; never hardcode dataset URLs, paths, or hyperparameters
  in Python — read them from `configs/`.
- Scripts in `scripts/` should stay thin: parse args, load config, dispatch
  into `slm4ie/`. Don't hide library logic inside a script.
- Annotated extractors keep text and annotations split (see
  [Extract](user-guide/data-pipeline/extract.md)). Don't add a "merged"
  output without a strong reason.
- `line-length = 120`. Refactor instead of bumping it.
- Don't cross-import between `scripts/` modules; share via `slm4ie/`.

## Things to avoid

- Committing data, model checkpoints, or `.env` files.
- Using `pip install` instead of `uv add` / `uv sync`.
- Adding a third conversion route alongside datatrove/spans without
  discussing it first — the split is intentional.

## Building the documentation locally

```bash
uv sync --extra docs
uv run mkdocs serve
```

The site is served at `http://127.0.0.1:8000`. Strict-build any change
before pushing:

```bash
uv run mkdocs build --strict
```

Documentation is deployed automatically to GitHub Pages on every push to
`main` via `.github/workflows/documentation.yaml`.
