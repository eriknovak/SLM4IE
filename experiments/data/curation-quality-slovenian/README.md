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
- **Next**: pairwise judge pass on the four content filters, then dedup statistics and the analysis

## Hypothesis

- **Statement**: The eight-stage curation pipeline removes low-quality records without discarding valid domain-specific text; each stage's drop rate is explained by its intended filter, not by domain or source artefacts.
- **Rationale**: The stages are Gopher-style heuristics and cross-corpus dedup tuned on English web crawls. The first build lost 30 M of 54 M documents in sentence dedup alone, and the news ellipsis override showed one default already misfired on Slovene prose. Whether the rest of the drops are correct has only been checked by intuition, never against labels.
- **Predictions**: Confirmed if every stage's precision of drops against the calibrated judge is at least 0.8, no source has a stage precision below 0.6, and the residual bad rate among kept documents is below 0.1 on every judged dimension. Refuted if any stage drops mostly good text (precision below 0.6 overall) or if a curated domain source (medical, scientific, legal) loses more than half its documents to a stage whose drops the judge does not confirm.
- **Outcome**: open

## Design

- **Data**: `data/extracted/*.jsonl` for all 22 extracted sources minus the benchmark sources suk and ssj500k ([D1]); rerun through `configs/data/curate.yaml` into `data/pretrain/`; a stratified judged sample and a 300-document manual calibration set under `data/experiments/data/curation-quality-slovenian/`.
- **Method / factors**: rebuild the corpus [M1]; draw the source × stage × decision sample [M2]; judge it blind on the seven-dimension rubric [M3]; hand-label a calibration subset in the marimo notebook [M4]; score agreement on the collapsed keep-or-drop decision [M5]; compute per-stage drop precision and residual bad rate at both bars [M6, D8]; run a pairwise pass over the four content filters, per source [M7, D2]; assess the two dedup stages by duplication statistics, judging only the drops those statistics cannot match [M8, D9]; compute corpus statistics that need no judge [M9]; extract throughput [M10]. Nothing is varied: this is an assessment, and the repairs it motivates belong to `curation-thresholds-slovenian` ([D6]).
- **Metrics**: primary precision of drops per stage and source against the calibrated judge, at the lenient bar with the strict bar reported beside it ([D8]); secondary the pairwise win rate of kept over dropped text per source and stage, residual bad rate among kept documents, judge–human Cohen's kappa on the collapsed decision, twin-match rate for dedup drops and residual near-duplicate rate among the kept (MinHash), sentence-dedup drops by reason, and the no-judge corpus statistics — source × stage survival, domain mix before and after, length distributions, type–token ratio, out-of-vocabulary rate against Sloleks, language-identification confidence, and totals with and without gated sources. Throughput is documents per second per stage. Each is defined in `experiments/GLOSSARY.md`.
- **Protocol**: one tracked corpus build (lineage under `slm4ie/data/curate`); sampling seed fixed at 20260916; judge run once at the agreed cap, rerun only after a rubric revision that the agreement gate forces; the pairwise pass drawn from the same sample; a second draw under a new seed reserved, unjudged, for validating the tuning experiment ([D10]); analysis runs tracked under `slm4ie/data/curation-quality-slovenian`.
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

- **Decision**: Two units. For the blind rubric pass, one document truncated to its first 2,000 characters; strata are source × stage × decision (kept or dropped), 40 documents per cell, sampled with a fixed seed, empty cells skipped. Six stages make a keep-or-drop decision, so 20 sources fill at most 240 cells and the judged set stays under 10,000 documents. For the pairwise pass, one kept document shown against one dropped document from the same source and the same stage, 20 pairs per source per stage over the four content filters, each pair asked in both orders.
- **Why**: The Gopher filters see the whole document but decide on surface statistics, so 2,000 characters carry the evidence that matters and keep judge tokens bounded. Forty per cell gives roughly ±15% precision per cell, so a single source's drop precision at one stage can be read on its own rather than only in aggregate.
- **History**:
  - 2026-09-14 first version, grill Q21
  - 2026-09-17 raised from 15 to 40 per cell; the funnel from the finished build showed per-source stage drops worth reading individually (medical loses 85% at `quality`), and 10,000 documents is within the judge budget
  - 2026-09-25 pairwise unit added, per source rather than pooled. Absolute scales proved unreliable between annotators — human and judge matched on only 39.7% of the 1-5 coherence scale — while a forced choice between two documents needs no shared scale at all. Per-source draws are what the prediction "no source below 0.6" is actually about; 20 sources × 4 stages × 20 pairs × 2 orders is roughly 3,200 comparisons, about $4 on the Batch API

### D3 — The judge

- **Decision**: A standalone script with two interchangeable backends: the Batch API (default), which costs half of standard rates and enforces the reply shape with a structured-output schema, and the Claude Code command-line tool (`claude -p`), which needs no API key and is used to try a rubric change quickly. Model, prompt file, group size and maximum request count are configurable; one JSON row per document; already-judged documents are skipped on rerun.
- **Why**: A separate script makes the run resumable and bounds token use. Subagents inside a session cannot be capped or resumed the same way. Batch pricing halves the bill for work with no deadline, and grouping documents per request amortises the rubric, which would otherwise cost more than every verdict put together. Prompt caching earns nothing here and is deliberately not used: the rubric is about 1,500 tokens, under Haiku 4.5's 4,096-token minimum cacheable prefix, and batch requests run hours apart so a five-minute cache would miss anyway.
- **Alternatives**:
  - Session subagents: no cap, no resume.
  - Local open model: weaker on Slovene, no calibration evidence.
- **History**:
  - 2026-09-14 first version, grill Q17/Q21
  - 2026-09-18 Batch API added as the default backend and the CLI kept as the second; the whole 8,930-document run costs about $5 on Haiku 4.5 or $10 on Sonnet 5, so the model is chosen by the kappa gate rather than by cost

### D4 — Judge rubric

- **Decision**: Per document: language (sl / other code / mixed), text type (prose / boilerplate / list / code / garbage), adult-or-spam (yes/no), coherence (1–5), looks machine-translated (yes/no), contains PII (yes/no), domain from the catalog's taxonomy (medical, scientific, legal, news, parliamentary, academic, wiki, forum, blog, student, web, finance, other). The judge is blind: it sees the document's id and text only, never the stage or the decision under test. The rubric lives in `configs/judge-rubric.md`.
- **Why**: Each stage targets one of these dimensions, so precision can be scored per stage. The domain label doubles as gold for the classifier in kpi-coverage-slovenian; machine-translation and PII flags cost nothing and matter for a public release. For the stage metrics the seven dimensions collapse to the one decision the pipeline makes — keep or drop — by the bar in [D8]; the dimensions then serve as the reason a document failed, not as seven separate measurements.
- **History**:
  - 2026-09-14 first version, grill Q12/Q16
  - 2026-09-17 domain vocabulary aligned with `configs/data/extract.yaml` so a judged label can be compared against the source's declared domain; blindness written into the rubric

### D5 — Calibration gate

- **Decision**: The project lead labelled 116 documents of the 300-document calibration set on the rubric, blind to the judge, and labelling stops there. The gate is Cohen's kappa of at least 0.6 on the collapsed keep-or-drop decision ([D8]) — not on the seven dimensions separately, which are reported descriptively. The gate is met: kappa 0.66, raw agreement 94.0%. About 30 further adjudications are drawn at the end of the analysis, over cases where the judge and the pipeline flatly conflict, from the twelve sources the 116 do not reach.
- **Why**: The pipeline makes one binary decision, so that is the judgement the judge must be trusted on; demanding agreement on every dimension measured something else and could not be met. Exact agreement on the 1-5 coherence scale was 39.7% and on the 13-way domain 69%, while the collapsed decision agreed 94.0%, coherence fell within one point 94.0% of the time, and the judge showed no systematic offset (-0.04 points). Kappa 0.6 is the conventional floor for substantial agreement. Stopping at 116 widens the interval to roughly ±0.13 but does not weaken validity, and the 116 cover all twelve stage × decision cells.
- **Caveat**: The 116 fall in the calibration file's order, so they reach 8 of 20 sources and all of them web-derived (c4, cc100, culturax, fineweb2, finepdf, coleslaw, classla_web_sl, classlawiki_sl). Agreement on curated prose — medical, parliamentary, legal, academic — is therefore untested, which is what the 30 end-of-analysis adjudications are aimed at.
- **History**:
  - 2026-09-14 first version, grill Q22
  - 2026-09-22 briefly cut to 100 documents for the labeller's time, then restored to 300 the same evening; the smaller set would have carried a kappa interval of about ±0.15 against ±0.09 on 300
  - 2026-09-25 gate rewritten onto the collapsed decision after 116 labels showed the seven-dimension gate was unreachable by construction; labelling stopped at 116 rather than 300 (grill round 1, Q4)
  - 2026-09-18 weighted kappa adopted for coherence after Haiku and Sonnet agreed on only 52% of the 1-5 scale while 98% of their disagreements were a single point, and one pair (Sonnet 5 against Haiku 4) accounted for 87 of 280 documents; the coherence anchors were rewritten as an ordered decision with an explicit 4/5 test

### D6 — Measure, do not tune

- **Decision**: No stage threshold is varied. Sentence-dedup drops are logged by reason (duplicate windows removed versus short remainder discarded) so a later tuning experiment has its baseline.
- **Why**: The hypothesis is about whether the current pipeline is trustworthy; varying floors would turn it into a tuning study with a different hypothesis.
- **History**:
  - 2026-09-14 first version, grill Q27
  - 2026-09-25 reaffirmed once the audit showed the content filters drop mostly usable text: the repairs that finding calls for go to `curation-thresholds-slovenian`, which varies the dials and validates on a fresh draw ([D10])

### D7 — Compute placement

- **Decision**: Run locally with workers sized to the machine and resource use monitored; move a stage to the SLING cluster only if it cannot fit in memory or disk.
- **Why**: The first build completed locally; the rerun differs only by excluding two small sources. SLING is available for one year if needed.
- **History**:
  - 2026-09-14 first version, grill Q24

### D8 — The bar for bad text

- **Decision**: A document counts as bad when the judge scores coherence 2 or less, or calls it garbage or boilerplate, or flags it as adult-or-spam. That is the primary bar for every drop-precision and residual-bad-rate number. The stricter bar — coherence 3 or less on the same conditions — is reported in a column beside it. Both are fixed before the final metrics are computed.
- **Why**: Every precision number moves with this line, so choosing it after seeing the results would make the conclusion an artefact of the threshold. The lenient bar is the one the human labels calibrate: agreement there is kappa 0.66 against 0.47 at the strict line. Reporting both shows that the finding does not rest on where the line sits.
- **History**:
  - 2026-09-25 first version, grill round 2 Q1

### D9 — How the dedup stages are assessed

- **Decision**: Duplication statistics first. Every document the dedup stages dropped is matched against the surviving corpus with MinHash; the record reports the share with a genuine surviving twin, the residual near-duplicate rate among kept documents, and sentence-dedup drops by reason. The judge is asked only about drops that no match above threshold explains.
- **Why**: Quality judging is provably blind here. Among judged documents, `exact_dedup` kept and dropped text was usable at 96.1% against 96.6%, and `sentence_dedup` at 94.2% against 98.0% — the dropped side scoring *higher*, because dedup removes copies of good text. A stage that removes duplicates can only be judged on whether what it removed was duplicated.
- **History**:
  - 2026-09-25 first version, grill round 2 Q3

### D10 — What is held back from tuning

- **Decision**: The tuning experiment may use the 8,930 judged documents and the 116 human labels to choose thresholds, but no claim of improvement rests on them. It draws a fresh sample under a new seed, judged only after the thresholds are frozen, and confirms the result with a judge from a different model family.
- **Why**: Thresholds tuned against one judge's verdicts stop being neutral evidence of anything — the pipeline would be fitted to that judge's taste on those documents. A fresh draw controls for the documents, a second judge controls for the judge, and the sampler is cached per cell so the extra draw costs unattended time rather than attention.
- **History**:
  - 2026-09-25 first version, grill round 2 Q4

## Findings

_Written by labflow:report._

## Verdict


## Reproduce

_Written at conclusion._
