---
title: Home
hide:
  - navigation
  - toc
---

<p align="center">
  <img src="assets/imgs/logo.png" alt="SLM4IE logo" style="width: 50%;">
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

## Workstreams

- **[Data pipeline](user-guide/data-pipeline/index.md)** — download, extract,
  and convert Slovenian corpora to pretraining and IE training shapes.
- **[Datasets](datasets/index.md)** — curated catalog of pretraining corpora
  and downstream benchmarks (NER, SA, SuperGLUE, tokenizer evaluation).
- **[Training](user-guide/training.md)** — pretrain or fine-tune SLMs from
  YAML configs, with MLflow tracking and SLURM batch scripts.
- **[Evaluation](user-guide/evaluation.md)** — run benchmark evaluation
  against the produced models.

## Acknowledgments

The project is funded by ARIS (Slovenian Research and Innovation Agency)
under the project number
[Z2-70067](https://cris.cobiss.net/ecris/si/sl/project/24346).

<p align="center">
  <img src="assets/imgs/aris.png" alt="ARIS Logo" width="320">
</p>
