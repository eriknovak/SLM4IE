---
title: Slovene training data — a dataset for tokenizers, models and information extraction
slug: slovene-training-data
short: Slovene training data
status: open
members:
  - ../data/data-landscape-slovenian/
  - ../data/curation-quality-slovenian/
  - ../data/curation-thresholds-slovenian/
planned:
  - slug: medical-ie-slovenian
    short: Medical extraction examples
    title: how to build native Slovene medical examples for information extraction, since open supply has too few
    builds_on: [data-landscape-slovenian]
  - slug: kpi-coverage-slovenian
    short: KPI coverage check
    title: whether the curated corpus and the task sets together meet every KPI, scientific extraction examples included
    builds_on: [curation-quality-slovenian, medical-ie-slovenian]
---

# Slovene training data — a dataset for tokenizers, models and information extraction

## Question

Can the project assemble, from open sources, a Slovene dataset large and clean enough to train its tokenizers and small language models, and to fine-tune them for information extraction? The project needs this because every later experiment trains or evaluates on it, and the proposal behind the project commits to minimum amounts of text and examples per domain. An answer says which sources the dataset draws on, how much each domain holds once low-quality text is removed, and where examples have to be built rather than found.

## What we now believe

- **Open Slovene text is enough to pretrain on in medicine and science, before curation**: the survey of open datasets found medicine's new native sources far past the token target ([data-landscape-slovenian:F2]) and science past both targets on text written in Slovene ([data-landscape-slovenian:F3]).
- **The curation pipeline cannot yet be trusted with that domain text**: what it keeps is clean ([curation-quality-slovenian:F3]), but every content filter drops mostly good text ([curation-quality-slovenian:F1]), how well a filter decides swings with the source ([curation-quality-slovenian:F2]), and the quality filter removes most of the medical source ([curation-quality-slovenian:F7]).
- **Medical examples have to be built**: almost all of medicine's open documents are translation or machine output ([data-landscape-slovenian:F1]), and the machine-written sets gather in medicine rather than across the catalogue ([data-landscape-slovenian:F6]).

## Threads

- **Pretraining text per domain**: whether open native text is enough to train tokenizers and models in each domain — [data-landscape-slovenian:H1], [data-landscape-slovenian:H3]
- **Information-extraction examples**: whether native examples exist to fine-tune and evaluate extraction in each domain — [data-landscape-slovenian:H2], [data-landscape-slovenian:H4]
- **Curation of the corpus**: whether the pipeline removes bad text while keeping good domain text, per stage and per source — [curation-quality-slovenian:H1], [curation-quality-slovenian:H2], [curation-quality-slovenian:H3], [curation-quality-slovenian:H4]
- **Filter tuning per source**: whether settings chosen per source let each content filter drop what it targets, and mostly bad text — [curation-thresholds-slovenian:H1], [curation-thresholds-slovenian:H2], [curation-thresholds-slovenian:H3], [curation-thresholds-slovenian:H4], [curation-thresholds-slovenian:H5]
- **Provenance of the text**: how much of the supply was written in Slovene rather than translated or generated — [data-landscape-slovenian:F1], [data-landscape-slovenian:F6]

## Open

- how to set the curation filters per source so curated domain text survives while web spam is still removed — answered by curation-thresholds-slovenian
- how native medical extraction examples are built at the scale KPI 2 asks for — answered by medical-ie-slovenian
- whether the curated corpus and the task sets together meet every KPI, including extraction examples for science — answered by kpi-coverage-slovenian
