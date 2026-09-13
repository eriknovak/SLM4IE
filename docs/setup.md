# Setup

Everything needed before running any pipeline. See
[data-pipeline.md](data-pipeline.md) for what to run once this is done.

## Requirements

- **Python** ≥ 3.13 (declared in [`pyproject.toml`](../pyproject.toml) and [`.python-version`](../.python-version))
- **[uv](https://docs.astral.sh/uv/)** — the canonical package and environment manager
- **Git** — for cloning the repository and CLARIN.SI dataset access
- **HuggingFace account** — required for gated datasets (e.g., `FineWeb-2`)
- **Disk space** — pretraining corpora total tens of GB; plan accordingly
- **GPU (optional)** — required for tokenizer/model training; CPU sufficient for data preparation

## Install dependencies

Clone the repository and create the virtual environment via `uv`:

```bash
git clone https://github.com/eriknovak/SLM4IE.git
cd SLM4IE
uv sync
```

This creates `.venv/` and installs the base plus the `dev` group, pinned in
`uv.lock` — about 10 packages, enough to read the configs and build
`experiments/report.html`. Every workstream beyond that is a named dependency
group, not an extra, so you install only what you are about to run:

| Group        | Pull in with                 | Needed for                                                                        |
| ------------ | ---------------------------- | --------------------------------------------------------------------------------- |
| `dev`        | (default)                    | `pytest`, `ruff`                                                                   |
| `tracking`   | (included by the four below) | MLflow — experiment runs and data-pipeline lineage                                 |
| `data`       | `uv sync --group data`       | download → extract → task conversion (`prepare_datasets.py`)                       |
| `corpus`     | `uv sync --group corpus`     | building the pretraining corpus (`curate_pretraining_corpus.py`, datatrove stack)  |
| `tokenizers` | `uv sync --group tokenizers` | training and scoring tokenizers (`sweep_tokenizers.py`)                            |
| `analysis`   | `uv sync --group analysis`   | an experiment's `analysis.py` — MLflow tables + datachart figures                  |

`uv sync --all-groups` installs everything.

Activate the environment for ad-hoc commands:

```bash
source .venv/bin/activate
```

Or prefix individual commands with `uv run` to skip activation.

## Enable the git hooks

The repository ships a pre-commit hook under `.githooks/` that blocks commits
containing presigned-URL credentials (`X-Amz-Signature` / `X-Amz-Credential`).
Activate it once per clone:

```bash
git config core.hooksPath .githooks
```

Secrets and ephemeral values (such as presigned download URLs) belong in a
gitignored `*.local.yaml` sibling overlay, which `load_config` deep-merges
over the matching base config — never in the committed YAML.

## HuggingFace authentication

Some datasets (e.g., `FineWeb-2`, gated corpora) require a HuggingFace access
token. Authenticate once via the unified `hf` CLI (shipped with
`huggingface_hub` ≥ 0.34, which replaces the deprecated `huggingface-cli`):

```bash
hf auth login
```

Paste a token from
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) when
prompted. The token is stored under `~/.cache/huggingface/` and picked up
automatically by `huggingface_hub` and `datasets` — no `HF_TOKEN` environment
variable or `.env` file needed.

For non-interactive use (e.g., CI, SLURM), pass the token directly:

```bash
hf auth login --token "$HF_TOKEN"
```

To verify:

```bash
hf auth whoami
```

## Tests

```bash
uv run pytest                # full test suite
uv run pytest tests/data     # subset
uv run pytest -m "not slow"  # skip end-to-end pipeline runs
```
