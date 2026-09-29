---
title: Slovene training data — a dataset for tokenizers, models and information extraction
slug: slovene-training-data
short: Slovene training data
status: open
members:
  - ../data/data-landscape-slovenian/
planned:
  - slug: curation-quality-slovenian
    short: Corpus curation quality
    title: whether the curation pipeline drops low-quality text without discarding valid domain text
    ticket: "#2"
    builds_on: [data-landscape-slovenian]
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

- **Open Slovene text is enough to pretrain on in medicine and science**: the survey of open datasets found medicine's new native sources far past the token target ([data-landscape-slovenian:F2]) and science past both targets on text written in Slovene ([data-landscape-slovenian:F3]).
- **Medical examples have to be built**: almost all of medicine's open documents are translation or machine output ([data-landscape-slovenian:F1]), and the machine-written sets gather in medicine rather than across the catalogue ([data-landscape-slovenian:F6]).

## Threads

- **Pretraining text per domain**: whether open native text is enough to train tokenizers and models in each domain — [data-landscape-slovenian:H1], [data-landscape-slovenian:H3]
- **Information-extraction examples**: whether native examples exist to fine-tune and evaluate extraction in each domain — [data-landscape-slovenian:H2], [data-landscape-slovenian:H4]
- **Provenance of the text**: how much of the supply was written in Slovene rather than translated or generated — [data-landscape-slovenian:F1], [data-landscape-slovenian:F6]

## Open

- whether the curation pipeline keeps valid medical and scientific text while removing low-quality text — answered by curation-quality-slovenian
- how native medical extraction examples are built at the scale KPI 2 asks for — answered by medical-ie-slovenian
- whether the curated corpus and the task sets together meet every KPI, including extraction examples for science — answered by kpi-coverage-slovenian
