---
title: Curation pipeline quality — Slovenian
slug: curation-quality-slovenian        # MLflow experiment: slm4ie/data/curation-quality-slovenian
category: data
line: main-line
branch: exp/curation-quality-slovenian
base_commit: a0b3306
status: draft
ticket: "#2"
pr:
mlflow:
builds_on: []
types: []
concluded:
---
# Curation pipeline quality — Slovenian

## TL;DR

- **Hypothesis**: open
- **Next**: rerun the corpus without benchmark sources, then draw the judged sample

## Hypothesis

- **Statement**: The eight-stage curation pipeline removes low-quality records without discarding valid domain-specific text; each stage's drop rate is explained by its intended filter, not by domain or source artefacts.
- **Rationale**: The stages are Gopher-style heuristics and cross-corpus dedup tuned on English web crawls. The first build lost 30 M of 54 M documents in sentence dedup alone, and the news ellipsis override showed one default already misfired on Slovene prose. Whether the rest of the drops are correct has only been checked by intuition, never against labels.
- **Predictions**: Confirmed if every stage's precision of drops against the calibrated judge is at least 0.8, no source has a stage precision below 0.6, and the residual bad rate among kept documents is below 0.1 on every judged dimension. Refuted if any stage drops mostly good text (precision below 0.6 overall) or if a curated domain source (medical, scientific, legal) loses more than half its documents to a stage whose drops the judge does not confirm.
- **Outcome**: open

## Design

- **Data**: `data/extracted/*.jsonl` for all 22 extracted sources minus the benchmark sources suk and ssj500k ([D1]); rerun through `configs/data/curate.yaml` into `data/pretrain/`; a stratified judged sample and a 300-document manual calibration set under `data/experiments/data/curation-quality-slovenian/`.
- **Method / factors**: rebuild the corpus [M1]; draw the source × stage × decision sample [M2]; judge it with the command-line-driven language-model judge [M3]; label the calibration set in the marimo labelling notebook [M4]; score judge agreement [M5]; compute per-stage precision and residual rates [M6]; check dedup drops and residual near-duplication [M7]; extract throughput [M8]. Nothing is varied: this is an assessment, not a tuning study ([D6]).
- **Metrics**: primary precision of drops per stage against the calibrated judge; secondary residual bad rate among kept documents, judge–human Cohen's kappa, residual near-duplicate rate (MinHash), sentence-dedup drops by reason, documents per second per stage — each defined in `experiments/GLOSSARY.md`.
- **Protocol**: one tracked corpus build (lineage under `slm4ie/data/curate`); sampling seed fixed; judge run once at the agreed cap, rerun only after a rubric revision that the kappa gate forces; analysis runs tracked under `slm4ie/data/curation-quality-slovenian`.
- **Compute**: local machine (40 cores, 125 GB RAM, one 16 GB GPU) with 6.4 TB free on `/vault`; the Slovenian national compute cluster SLING if a stage cannot fit ([D7]). Judge tokens bounded by batch caps ([D3]).

## Methods

_Written by labflow:code as each step lands._

## Decisions

### D1 — Sources in the rerun

- **Decision**: All extracted sources except the benchmark corpora suk and ssj500k. Gated sources (gigafida, kas, solar, culturax) stay in, flagged by licence, so totals can be reported with and without them.
- **Why**: Benchmarks inside the pretraining corpus contaminate evaluation. Gated sources are real training material for an internal model, but a public release cannot include them, so both totals matter for KPI 3 in kpi-coverage-slovenian.
- **Alternatives**:
  - Rerun `--all` as before: keeps the benchmark leak.
  - Drop gated sources too: loses the internal-model total.
- **History**:
  - 2026-09-14 first version, grill Q15

### D2 — What a judged unit is

- **Decision**: One document truncated to its first 2,000 characters. Strata are source × stage × decision (kept or dropped), 15 documents per cell, sampled with a fixed seed; empty cells are skipped.
- **Why**: The Gopher filters see the whole document but decide on surface statistics, so 2,000 characters carry the evidence that matters and keep judge tokens bounded. Fifteen per cell gives roughly ±25% precision per cell, tight enough at stage × source aggregation.
- **History**:
  - 2026-09-14 first version, grill Q21

### D3 — The judge

- **Decision**: Claude Sonnet 5 invoked from a standalone script through the Claude Code command-line tool (`claude -p`), with configurable model, prompt file, concurrency, batch size and maximum batch count; one JSON row per document; already-judged documents are skipped on rerun.
- **Why**: A separate script makes the run resumable and bounds token use, which prior analyses of this kind showed to be the dominant cost. Subagents inside a session cannot be capped or resumed the same way.
- **Alternatives**:
  - Session subagents: no cap, no resume.
  - Local open model: weaker on Slovene, no calibration evidence.
- **History**:
  - 2026-09-14 first version, grill Q17/Q21

### D4 — Judge rubric

- **Decision**: Per document: language (sl / not), text type (natural prose / boilerplate / list / code / garbage), adult-or-spam (yes/no), coherence (1–5), looks machine-translated (yes/no), contains PII (yes/no), domain from the shared taxonomy (medical, scientific, legal, news, parliamentary, academic, encyclopedic, forum/social, general-web, finance, other).
- **Why**: Each stage targets one of these dimensions, so precision can be scored per stage. The domain label doubles as gold for the classifier in kpi-coverage-slovenian; machine-translation and PII flags cost nothing and matter for a public release.
- **History**:
  - 2026-09-14 first version, grill Q12/Q16

### D5 — Calibration gate

- **Decision**: A 300-document set drawn from the same strata is labelled by the project lead on the same rubric in a marimo notebook. Stage metrics are reported only once Cohen's kappa is at least 0.6 on every dimension; below that the rubric is revised and the judge rerun.
- **Why**: Judge labels are only evidence once their agreement with a human is known. Kappa 0.6 is the conventional floor for substantial agreement.
- **History**:
  - 2026-09-14 first version, grill Q22

### D6 — Measure, do not tune

- **Decision**: No stage threshold is varied. Sentence-dedup drops are logged by reason (duplicate windows removed versus short remainder discarded) so a later tuning experiment has its baseline.
- **Why**: The hypothesis is about whether the current pipeline is trustworthy; varying floors would turn it into a tuning study with a different hypothesis.
- **History**:
  - 2026-09-14 first version, grill Q27

### D7 — Compute placement

- **Decision**: Run locally with workers sized to the machine and resource use monitored; move a stage to the SLING cluster only if it cannot fit in memory or disk.
- **Why**: The first build completed locally; the rerun differs only by excluding two small sources. SLING is available for one year if needed.
- **History**:
  - 2026-09-14 first version, grill Q24

## Findings

_Written by labflow:report._

## Verdict


## Reproduce

_Written at conclusion._
