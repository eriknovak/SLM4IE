---
title: Installation
---

# Installation

## Requirements

- **Python** ≥ 3.13 (declared in `pyproject.toml` and `.python-version`)
- **[uv](https://docs.astral.sh/uv/)** — recommended package and environment
  manager
- **Git** — for cloning the repository and CLARIN.SI dataset access
- **HuggingFace account** — required for gated datasets (e.g.,
  `FineWeb-2`); see [HuggingFace Authentication](huggingface-auth.md)
- **Disk space** — pretraining corpora total tens of GB; plan accordingly
- **GPU (optional)** — required for tokenizer/model training; CPU
  sufficient for data preparation

## Clone and install

Clone the repository and create the virtual environment via `uv`:

```bash
git clone https://github.com/eriknovak/SLM4IE.git
cd SLM4IE
uv sync
```

This creates `.venv/` and installs both runtime and dev dependencies pinned
in `uv.lock`. Activate the environment for ad-hoc commands:

```bash
source .venv/bin/activate
```

Or prefix individual commands with `uv run` to skip activation.

## Optional extras

The base `uv sync` only pulls runtime essentials. Install extras as needed:

| Extra    | Command                    | Purpose                                                     |
|----------|----------------------------|-------------------------------------------------------------|
| `dev`    | `uv sync --extra dev`      | Test runner (`pytest`) and linter (`ruff`) for contributors |
| `curate` | `uv sync --extra curate`   | datatrove pipeline for corpus curation (see [Curation](../user-guide/data-pipeline/curate.md)) |
| `docs`   | `uv sync --extra docs`     | MkDocs Material toolchain for building this documentation site |

Multiple extras can be combined:

```bash
uv sync --extra dev --extra curate
```

## Next steps

- Authenticate with HuggingFace: [HuggingFace Authentication](huggingface-auth.md)
- Get oriented in the codebase: [Project Structure](project-structure.md)
- Start running the pipeline: [User Guide](../user-guide/index.md)
