---
title: Can a model under one billion parameters extract Slovene information within reach of frontier LLMs?
slug: zero-shot-ie-slovenian
short: Zero-shot Slovene extraction
status: open
members:
  - ../validation/ie-benchmark-slovenian/
planned:
  - slug: gliner-backbones-slovenian
    short: GLiNER backbones
    title: which backbone, a Slovene or multilingual encoder or a small decoder LLM, gives a GLiNER model the best zero-shot Slovene NER
    ticket: "#45"
    builds_on: [ie-benchmark-slovenian]
  - slug: domain-finetune-slovenian
    short: Domain fine-tuning
    title: whether GLiNER models fine-tuned on silver medical and scientific examples come within 10% of the best frontier LLM on the domain gold sets
    ticket: "#51"
    builds_on: [gliner-backbones-slovenian, medical-ie-slovenian, science-ie-slovenian, encoder-pretraining-slovenian]
---

# Can a model under one billion parameters extract Slovene information within reach of frontier LLMs?

## Question

Frontier LLMs such as Claude, GPT and Gemini extract entities from text in many languages without task-specific training, but they are large, paid per call and cannot run on a local GPU. The project commits to models under one billion parameters that fit in 8 GB of GPU memory and come within 5–10% of those LLMs on Slovene information extraction (KPIs 5–7). GLiNER is the candidate design: an encoder scores each text span against label names, so new labels need no retraining. The question is whether such a model, built on the right backbone and fine-tuned on medical and scientific examples, reaches that margin. An answer gives each model's score next to the best LLM's on the same test sets, with its parameters, peak memory, and cost per document.

## What we now believe

_No experiment has concluded yet._

## Threads

- **Reference scores**: where Claude and open LLMs on one local GPU stand on Slovene extraction, and whether a local model can label the silver data — [ie-benchmark-slovenian:C1], [ie-benchmark-slovenian:Q1]

## Open

- where frontier and open LLMs stand on Slovene extraction, which fixes the margin KPI 7 is measured against — answered by ie-benchmark-slovenian
- which backbone a Slovene GLiNER model should use — answered by gliner-backbones-slovenian
- whether domain fine-tuning closes the gap on medical and scientific text — answered by domain-finetune-slovenian
