---
title: Tests
---

# Tests

The project uses [pytest](https://docs.pytest.org/) for testing.

## Install the dev extra

```bash
uv sync --extra dev
```

## Run

```bash
uv run pytest                # full test suite
uv run pytest tests/data     # subset
```

## Conventions

- Tests live under `tests/` and mirror the `slm4ie/` layout.
- Fixtures used by `tests/data/` are small and committed in-tree — never
  point tests at the live `/vault/data/SLM4IE/` data store.
- Pytest configuration lives in `pyproject.toml`
  (`[tool.pytest.ini_options].testpaths`).

## Linting

Ruff is configured with Google-style docstring checks:

```bash
uv run ruff check --select D <changed-paths>
```

See [Development](../development.md) for the full conventions.
