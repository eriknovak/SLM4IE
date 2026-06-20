---
title: Home
hide:
  - navigation
  - toc
---

<p align="center">
  <img src="assets/imgs/logo/slm4ie_logo_color.svg" alt="SLM4IE logo" style="width: 40%; max-width: 320px;">
</p>

<p align="center">
  <a href="https://github.com/eriknovak/SLM4IE/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache 2.0"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.13%2B-blue.svg" alt="Python 3.13+"></a>
  <a href="https://github.com/eriknovak/SLM4IE"><img src="https://img.shields.io/badge/Open%20Source-%E2%9D%A4-brightgreen.svg" alt="Open Source"></a>
</p>

# SLM4IE

SLM4IE develops small language models (SLMs) for zero-shot information
extraction across European languages, with emphasis on Slovenian.

## Why SLM4IE

The project targets three limitations of current LLMs:

- **Compute cost** — LLMs require infrastructure beyond the reach of smaller
  organizations for local deployment.
- **Low-resource gaps** — limited training data for sensitive domains and
  underrepresented languages.
- **Output inconsistency** — unreliable structured extraction from generative
  models.

We build computationally efficient models optimized for commodity hardware,
create multilingual benchmark datasets for sensitive domains, and evaluate
against existing SLMs and LLMs. All artifacts (models, datasets, code) will be
released publicly where possible.

<div markdown>
[Get Started](getting-started/installation.md){ .md-button .md-button--primary }
[Browse the User Guide](user-guide/index.md){ .md-button }
[GitHub :fontawesome-brands-github:](https://github.com/eriknovak/SLM4IE){ .md-button }
</div>

## Project status

The workstreams are at different stages of maturity, and this site is explicit
about which is which. Pages for workflows that have **not yet been run
end-to-end** carry a 🚧 *In progress* banner.

| Area | Status |
|------|--------|
| Download → Extract → Pretraining corpus | **Verified** — run end-to-end; see [Corpus Statistics](datasets/corpus-statistics.md) |
| Task converters (NER, sentiment, SuperGLUE) | Implemented, not yet run |
| Tokenizer comparison sweep | Implemented, not yet run |
| Model training & evaluation | Implemented, not yet run |

## Workstreams

- **[Data pipeline](user-guide/data-pipeline/index.md)** — download, extract,
  and curate Slovenian corpora into a pretraining corpus and into IE training
  shapes.
- **[Datasets](datasets/index.md)** — catalog of pretraining corpora and
  downstream benchmarks (NER, SA, SuperGLUE, tokenizer evaluation), plus
  [statistics](datasets/corpus-statistics.md) of the curated corpus.
- **[Tokenizers](user-guide/tokenizers.md)** — train and score six tokenizers
  across a vocab sweep for Slovenian morphology.
- **[Training](user-guide/training.md)** and
  **[Evaluation](user-guide/evaluation.md)** — pretrain or fine-tune SLMs from
  YAML configs, with MLflow tracking and SLURM batch scripts.

## Acknowledgments

The project is funded by ARIS (Slovenian Research and Innovation Agency)
under the project number
[Z2-70067](https://cris.cobiss.net/ecris/si/sl/project/24346).

<p align="center">
  <img src="assets/imgs/aris.png" alt="ARIS Logo" width="320">
</p>
