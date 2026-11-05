---
title: How close an open LLM on one local GPU comes to Claude on Slovene information extraction
slug: ie-benchmark-slovenian
short: Extraction benchmark
category: validation
branch: exp/ie-benchmark-slovenian
base_commit: 909c513
status: draft
ticket: "#42"
pr:
mlflow:
builds_on: []
varies: []
uses: []
tests: []
evidence: {}
types: [baseline]
concluded:
---

# How close an open LLM on one local GPU comes to Claude on Slovene information extraction

## TL;DR

- **Hypothesis**: open — the first round scores Claude and local open LLMs zero-shot on four Slovene NER test sets.
- **Next**: prepare the Janes-Tag and SlavicNER test sets, then build the shared scorer and the model backends.

## Hypothesis

- **Question**: How close does the best open LLM that runs on one 16 GB GPU come to the best Claude model when both extract information from Slovene text zero-shot, under one fixed protocol?
- **Decision it informs**: whether the silver-label experiments (silver-ner-slovenian, medical-ie-slovenian, science-ie-slovenian) annotate their Slovene documents with a local open model or with Claude.
- **Statement**: on Slovene named-entity recognition, the best Claude model leads the best local open LLM by more than 5 points of strict span F1, so silver annotation needs Claude.
- **Rationale**:
  - **The silver-label experiments need an annotator, and the two kinds cost differently.** KPI 2 asks for more than 10,000 medical examples, and the silver experiments label native documents with an LLM. A local model costs only GPU time and keeps the text on project hardware. Claude runs through a subscription whose rate limits bound how many documents a day it can label.
  - **No zero-shot LLM result for Slovene NER was found.** The ticket's draft expected a frontier lead of more than 10 F1 points on NER. That expectation comes from English work, where small open models trail frontier models most on span extraction. GaMS, trained further on Slovene, may narrow the gap for its size.
  - **The test sets avoid open web copies where they can.** Janes-Tag, ssj500k and SUK are distributed through the CLARIN.SI repository behind a download, which lowers the chance that a model saw them in training. SlavicNER has been public on GitHub since 2017, so a lead that appears only there hints at memorisation.
  - **The data can carry the comparison, by the counts on disk.** The SUK test split holds 733 documents with 18,548 gold spans and ssj500k holds 233 documents with 1,031. Janes-Tag adds about 1,500 short social-media texts, and SlavicNER about 250 news documents with about 10,000 mentions. The sentiment and SuperGLUE-SL entries cannot be scored yet, because SentiNews has no extractor and the SuperGLUE-SL test files carry no labels. They wait for a later round. Counts in [test-set sizes per task registry entry](tables/scoping-test-sets.csv).
- **Predictions**: every clause is read on the test splits of four NER entries, `ner/ssj500k`, `ner/suk`, `ner/janes_tag` and `ner/slavicner`, each model prompted zero-shot once per document ([D3], [D9]). The Claude arm and the local arm each send their best model, picked on the validation splits ([D1], [D7]). Intervals are 95% paired bootstraps over test documents ([D5]).
  - **C1 — The best local open LLM against the best Claude model on strict span F1**: read once per test set; better if the interval of local minus Claude lies above +5 points, worse if it lies below −5 points — equivalent inside the margin and undecided when it straddles; equivalent or better lets the silver experiments annotate locally ([D2]; measured by [M3]).
  - **Q1 · diagnostic · measured — Where each model stands on each test set**: read on the table of strict span F1 per model and test set and on the figure of F1 against peak GPU memory; reported as strict micro F1 with its 95% interval — it is the reference the KPI 7 margins are measured against ([D1]; measured by [M3]).
  - **Verdict rule**: adopted, the local annotator, when C1 is equivalent or better on all four test sets. Inconclusive when no set reads worse but at least one is undecided. Rejected when at least one set reads worse and none reads equivalent or better. Partial when some sets read worse and others equivalent or better, the genre reading: the local model annotates only the genres where it holds.
  - **Read beside the clauses, not as clauses**: partial-overlap span F1, macro F1 over labels and F1 per label; the miss rate per model and set ([D6]); tokens in and out; the list-price cost per 1,000 documents for Claude ([D8]); peak GPU memory and documents per second for the local models; strict F1 per SUK subcorpus; the gap on SlavicNER beside the gap on the download-gated sets.
- **Outcome**: open

## Rounds

### R1 — NER on four test sets, Claude against local open models

- **Opened**: 2026-11-06
- **Tests**: whether a local open model is close enough to Claude on Slovene NER to annotate the silver sets, which three later experiments wait on.
- **Clauses**: C1, Q1
- **Budget**: three Claude models and about eight local models, each over the validation and test splits of four entries, about 90 tracked runs. The round ends early when no local model passes the miss-rate gate ([D6]).
- **Pre-registered at**:
- **Closed**:

## Design

- **Inputs**: the validation and test splits of four NER task registry entries: `ner/ssj500k` and `ner/suk` as built, and `ner/janes_tag` and `ner/slavicner`, which `labflow:data` adds as held-out entries, described in `## Datasets`.
- **Method / factors**: one pass from a test document to a score per model and test set.
  - Write a plain-language description for every label of every label set ([D4], [M1]).
  - Prompt every model once per document with the same template and JSON schema, at temperature 0 ([D3], [M2]).
  - Align the returned mentions to character spans, score strict and partial-overlap span F1, and bootstrap over documents ([D5], [M3]).
  - Pick each arm's best model on the validation splits and compare the two on the test splits ([D1], [M4]).
  - Factors: the arm (Claude, local open model) is scientific. Each arm's best is picked from its models ([D7]). The Claude arm holds Opus, Sonnet and Haiku 5.5. The local arm holds Qwen3.5, Gemma 4, Llama and GaMS at two or more sizes that fit on the GPU. The prompt, the schema, the decoding and the label descriptions are fixed for every model.
- **Metrics**: primary strict span F1; secondary partial-overlap span F1, macro F1 and miss rate — each defined in `experiments/GLOSSARY.md`
- **Protocol**: one tracked MLflow run per model, entry and split under `slm4ie/validation/ie-benchmark-slovenian`, tagged `language=sl`. Each run logs every raw answer, so `analysis.py` rescores without calling a model again. Decoding is greedy, so each model runs once with no seeds.
- **Compute**: local GPU — one GeForce 4080 GPU with 16 GB serves the open models through the local vllm-service, or through HuggingFace transformers when vllm cannot serve a model. Models over about 9B parameters run quantised.
- **People**: about 1 hour — the user reviews the label descriptions before the first tracked run.
- **Data access**: open — Janes-Tag 2.1 from CLARIN.SI under CC BY-SA 4.0; SlavicNER from its GitHub release, for research use.
- **External dependency**: Claude subscription — the Claude models run through the user's subscription in Claude Code's headless mode, whose rate limits bound throughput; the open models need the local vllm-service.

## Datasets

<Written by labflow:data at the data-ready gate: one row per test set, in the
order the pipeline meets them, each linking its statistics tables.>

## Methods

<Written by labflow:code as each step lands: one `### M<n>` entry per step,
M1 label descriptions, M2 prompting, M3 alignment and scoring, M4 picking the
best model per arm.>

## Decisions

### D1 — Pick each arm's best model on the validation splits, then compare on the test splits

- **Decision**: the best model of each arm is the one with the highest strict span F1 averaged over the four validation splits. C1 then compares the two picked models on the test splits only.
- **Why**: picking the highest of about eight scores on the test split favours the model that sampling noise lifted most, which overstates the winner.
- **Alternatives**:
  - Best on test: simpler, but the picked score is biased upward.
  - Compare every local model with Claude: about eight comparisons, with a correction that would widen every interval.
- **History**:
  - 2026-11-06 (before runs) picked on the mean validation F1 over four entries.

### D2 — Set the margin for local against Claude at 5 F1 points

- **Decision**: C1 reads a difference within ±5 points of strict span F1 as equivalent.
- **Why**: KPI 7 asks for models within 5–10% of frontier LLMs, and 5 points is about 6% of an F1 near 80. Silver-ner-slovenian uses the same 5 points between models trained on silver and on gold labels.
- **History**:
  - 2026-11-06 (before runs) 5 points, proposed for approval at the start gate.

### D3 — Give every model the same zero-shot prompt and JSON schema, and count an unparseable answer as empty

- **Decision**: every model sees one prompt template per task family, with the label descriptions and the document, and must answer as JSON against one schema listing mention text and label. An answer that does not parse scores as no mentions.
- **Why**: one protocol makes scores comparable across this and later experiments. An annotator that cannot produce valid output is not usable, so the failure counts against it.
- **Alternatives**:
  - Retry until the answer parses: hides a failure an annotation pipeline would meet.
- **History**:
  - 2026-11-06 (before runs) one zero-shot template, one schema, no retries.

### D4 — Pass labels as plain-language descriptions and score each test set on its own labels

- **Decision**: each label (PER, LOC, ORG, MISC, DERIV-PER; SlavicNER's PER, ORG, LOC, EVT, PRO) reaches the model as a short description written from its annotation guidelines. No label set is mapped onto another.
- **Why**: the medical and scientific experiments bring new label sets, and the same prompt must serve them. SlavicNER's event and product labels test that now.
- **History**:
  - 2026-11-06 (before runs) one description per label, reviewed by the user.

### D5 — Align returned mention strings to spans in reading order, and resample whole documents

- **Decision**: each returned mention is matched to the first unused occurrence of its text after the previous match. A mention with no occurrence counts as a false positive. A document is the unit the bootstrap resamples, and both arms are resampled on the same documents.
- **Why**: LLMs copy text more reliably than they count character offsets. Spans in one document are not independent, so resampling documents keeps the interval honest. The paired draw removes the variance the arms share.
- **History**:
  - 2026-11-06 (before runs) reading-order alignment, paired document bootstrap.

### D6 — Admit a model as its arm's best only if it parses on at least 90% of documents

- **Decision**: a model whose miss rate (the share of documents with an unparseable answer) exceeds 10% on any validation split cannot be its arm's best. It is still scored and reported.
- **Why**: a model that fails the format often wins or loses on parsing, not on extraction, and could not run an annotation pipeline unattended.
- **History**:
  - 2026-11-06 (before runs) 10% floor per validation split.

### D7 — Count as local only open models that run on one 16 GB GPU

- **Decision**: the local arm holds open-weight models served on the project's 16 GB GPU, through the vllm-service or HuggingFace transformers, quantised when a model would not fit otherwise.
- **Why**: the annotator must run on project hardware, and KPI 5 judges memory on an 8–16 GB GPU.
- **Alternatives**:
  - Open models through OpenRouter: no GPU limit, but no memory figure, and GaMS is not hosted there.
- **History**:
  - 2026-11-06 (before runs; user) local GPU only, vllm-service first, transformers for the rest.

### D8 — Run Claude through the subscription, and report its cost at list price

- **Decision**: the Claude models run through the user's subscription. Their cost per 1,000 documents is the logged tokens priced at the API list price on the day of the run.
- **Why**: the subscription has no per-call price, but a list-price figure lets later readers compare it with API use.
- **Alternatives**:
  - GPT and Gemini through OpenRouter in this round: deferred to a later round.
- **History**:
  - 2026-11-06 (before runs; user) Claude only, through the subscription.

### D9 — Score four NER test sets, preferring download-gated sources

- **Decision**: the test sets are ssj500k, SUK without its ssj500k documents, Janes-Tag 2.1 and the Slovene part of SlavicNER. SBL-51abbr is left out, being too small for an interval narrower than the margin.
- **Why**: download-gated CLARIN.SI data is less likely to be in a model's training data. Janes-Tag adds non-standard social-media text, and SlavicNER adds size, two new labels and a public-data contrast.
- **History**:
  - 2026-11-06 (before runs; user) Janes-Tag and SlavicNER added to the two built entries.

## Findings

<Written by labflow:report, one living entry per question the runs answer.>

## Verdict

<Written once, at conclusion, by labflow:report after user approval.>

## Reproduce

<Written at conclusion by labflow:report from the working plan's Log.>
