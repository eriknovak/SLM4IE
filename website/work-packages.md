---
title: Work packages
template: work-packages.html
subtitle: The work programme of the SLM4IE project, and how the work is organised.
description: What the SLM4IE project does, work package by work package, and how the work connects.
hide:
  - navigation
---

The project runs in three work packages over two years. They are not a relay:
WP1 prepares the data, WP2 builds the model and its tokenizer, WP3 trains and
evaluates — and what WP3 measures feeds straight back into the data preparation
and the architecture, so all three stay open until the end.

<figure class="slm-figure" markdown>
<div class="slm-figure__panel" markdown>
![Chart of the project’s work programme](assets/imgs/work-packages.svg)
</div>
<figcaption>Figure 1: Chart of the project’s work programme.</figcaption>
</figure>

## WP1 — Data acquisition and preparation

WP1 turns raw text into data a model can learn from. **T1.1 Data collection**
gathers text in European languages, with an emphasis on Slavic ones. **T1.2 Data
analysis** checks its quality and shows which languages and domains are well
covered and which are thin; what it finds also guides the tokenizer in T2.2.
**T1.3 Data enrichment and generation** fills those gaps, going back and forth
with the analysis until the coverage holds. **T1.4 Data formatting** shapes the
result for pretraining and fine-tuning and hands it to model training in T3.1.
The curated datasets are published openly.

## WP2 — Model development

WP2 designs the model that data goes into. **T2.1 Model architecture**
assembles the components best suited to information extraction, drawing on both
encoder-only and decoder-only designs. **T2.2 Tokenization algorithm** adapts how
text is split into tokens for morphologically rich European languages, informed
by the analysis in T1.2. Both feed model training in T3.1.

## WP3 — Model training and evaluation

WP3 trains the models and tests them. **T3.1 Model training** brings together
the formatted data from T1.4 and the architecture and tokenizer from WP2.
**T3.2 Model evaluation** measures the trained models against established
benchmarks and against both larger models and models of a similar size. Its
results feed back into WP1 and WP2, the loop across the top of Figure 1, so the
data, the architecture and the tokenizer are refined as results come in. The
models that hold up are published openly.

## Deliverables

Four reports mark the project's progress, two in each project year. Between
reports, notable results appear under [News](news/index.md), and every
experiment is recorded
[in the repository](https://github.com/eriknovak/SLM4IE/blob/main/experiments/README.md)
as it runs.
