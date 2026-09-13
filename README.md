![Image alt](./website/assets/imgs/banner/slm4ie_banner_dark_bg.png#gh-dark-mode-only)
![Image alt](./website/assets/imgs/banner/slm4ie_banner_light_bg.png#gh-light-mode-only)

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache 2.0"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.13%2B-blue.svg" alt="Python 3.13+"></a>
  <a href="https://github.com/eriknovak/SLM4IE"><img src="https://img.shields.io/badge/Open%20Source-%E2%9D%A4-brightgreen.svg" alt="Open Source"></a>
</p>

SLM4IE develops small language models (SLMs) for zero-shot information extraction across European languages, with emphasis on Slovenian. The project targets three limitations of current LLMs:

- **Compute cost:** LLMs require infrastructure beyond reach of smaller organizations for local deployment
- **Low-resource gaps:** Limited training data for sensitive domains and underrepresented languages
- **Output inconsistency:** Unreliable structured extraction from generative models

We build computationally efficient models optimized for commodity hardware, create multilingual benchmark datasets for sensitive domains, and evaluate against existing SLMs and LLMs. All artifacts (models, datasets, code) will be released publicly where possible.

The repository is a **collection of experiments**, not a service. Each one tests
a hypothesis; the shared machinery exists to run them.
[**What we have tried and what it showed**](experiments/README.md) is the place
to start.

## Quick start

```bash
git clone https://github.com/eriknovak/SLM4IE.git
cd SLM4IE
uv sync --group data

uv run python scripts/prepare_datasets.py download --all   # raw corpora + benchmarks
uv run python scripts/prepare_datasets.py extract  --all   # normalize to unified JSONL
uv run python scripts/prepare_datasets.py tasks    --all   # evaluation task datasets
```

Requires Python ≥ 3.13 and [uv](https://docs.astral.sh/uv/). Datasets are large
and live outside the repo. Full prerequisites, dependency groups, git hooks and
HuggingFace authentication: [`docs/setup.md`](docs/setup.md).

## Documentation

| Page                                                     | Covers                                                              |
| -------------------------------------------------------- | ------------------------------------------------------------------- |
| [`docs/setup.md`](docs/setup.md)                         | Requirements, install, dependency groups, git hooks, HF auth, tests |
| [`docs/data-pipeline.md`](docs/data-pipeline.md)         | Download, extract, task conversion, tokenizer-quality data          |
| [`docs/pretraining-corpus.md`](docs/pretraining-corpus.md) | The eight-stage corpus curation pipeline                          |
| [`docs/datasets.md`](docs/datasets.md)                   | Catalog of every pretraining corpus and benchmark                   |
| [`docs/tokenizer-sweep.md`](docs/tokenizer-sweep.md)     | Training and scoring the six tokenizer backends                     |
| [`experiments/README.md`](experiments/README.md)         | The findings book — one entry per experiment                        |
| [`CONTEXT.md`](CONTEXT.md)                               | The project's vocabulary                                            |

## Repository layout

```text
experiments/  The experiments — one folder per hypothesis, plus the findings
              book, the glossary, and the generated HTML report
configs/      Shared registries: the dataset catalogs and the task registry
slm4ie/       Library source — importable modules only
scripts/      CLI entry points, one per pipeline, thin wrappers around slm4ie/
slurm/        SLURM batch scripts for HPC training
tests/        pytest suite
docs/         This documentation, plus agent-skill config
website/      The project showcase site (Material for MkDocs), deployed to GitHub Pages
```

Three scripts drive everything: `prepare_datasets.py` (download, extract, tasks,
tokenization), `curate_pretraining_corpus.py` (the pretraining corpus), and
`sweep_tokenizers.py` (the tokenizer sweep).

## Data

Datasets live **outside the repository**, under `/vault/data/SLM4IE/`; `data/`
in the repo is a gitignored symlink to it. Never commit data files.

Slovenian pretraining corpora come from CLARIN.SI (CLASSLA-web, MaCoCu,
ParlaMint-SI, COLESLAW, OSS, siParl and more), HuggingFace (FineWeb-2, FinePDF,
mC4, HPLT) and direct HTTP (CC100, Legal-mC4). Evaluation uses SUK 1.1,
ssj500k 2.3, Slovene SuperGLUE and SentiNews, with Sloleks 3.1 as the tokenizer
morphology lexicon. Sources, licenses and per-dataset notes:
[`docs/datasets.md`](docs/datasets.md).

## Acknowledgments

The project is funded by ARIS (Slovenian Research and Innovation Agency) under the project number [Z2-70067](https://cris.cobiss.net/ecris/si/sl/project/24346).

<figure>
  <img src="https://github.com/eriknovak/SLM4IE/blob/main/website/assets/imgs/aris.png?raw=true" alt="ARIS Logo" width="460" />
</figure>
