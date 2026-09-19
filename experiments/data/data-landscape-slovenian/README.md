---
title: Open Slovene data landscape — KPI coverage
slug: data-landscape-slovenian             # MLflow experiment: slm4ie/data/data-landscape-slovenian
category: data
line: main-line
branch: exp/data-landscape-slovenian
base_commit: a0b3306
status: draft
ticket: "#1"
pr:
mlflow:
builds_on: []
types: []
concluded:
---
# Open Slovene data landscape — KPI coverage

## TL;DR

- **Hypothesis**: open
- **Next**: finish the catalogue, verify the sizes that carry the verdict

## Hypothesis

- **Statement**: Open-access Slovene text sources, catalogued by domain, size, annotation and document length, contain enough medical and scientific material to meet KPI 2 and KPI 4 without synthetic data.
- **Rationale**: The project already downloads 22 Slovene sources, but nobody has written down what else exists openly, under what licence, in what domain, and whether it is annotated. Earlier survey work left an intuition — medicine is the gap — and no notes. The corpus build makes the intuition concrete: medicine finished at 352 documents and 808,176 words, the smallest domain by two orders of magnitude. Whether that is a fact about Slovene or a fact about which sources were picked can only be settled by looking outside the repo.
- **Predictions**: Confirmed if openly downloadable Slovene text reaches more than 500k tokens in both medicine and science (KPI 4), with more than 10k examples in each (KPI 2), from sources that are not already in the download registry. Refuted if medicine falls short under every access filter, which makes the multilingual fallback or synthetic data a necessity rather than a choice.
- **Outcome**: open

## Design

- **Data**: No corpus is built here. The deliverable is a catalogue: one row per dataset found in any searched source, committed as CSV under `tables/`.
- **Method / factors**: Search eight source families [D2]; record every dataset found, whatever its access [D3]; verify each row against its landing page; aggregate per domain under three access filters [D8]. Nothing is varied — this is a survey.
- **Metrics**: primary openly downloadable words per domain; secondary annotated examples per domain. Both compared against KPI 2 (at least two domains, medicine and science, over 10k examples each) and KPI 4 (over 500k tokens per domain).
- **Protocol**: Every size is unverified until read from official statistics or a downloaded sample [D6]. The search date and the families covered are recorded, so a later refresh knows what to re-check.
- **Compute**: None beyond web search and one MLflow lineage run logging the catalogue as an artifact.

## Methods

The search ran on 2026-09-17: four agents in parallel, one per source family of
D2, each required to return the URL it read for every row and to leave a field
empty rather than guess. Their 304 rows are kept verbatim as the experiment's
evidence, one JSON file per family, under
`data/experiments/data/data-landscape-slovenian/raw/`.

`analysis.py` is the whole pipeline and reads only those files plus
`configs/data/download.yaml`. It merges the families into one row per dataset
(D10), marks each row that the project already downloads — matching on CLARIN
and LINDAT handles, Hugging Face repos, and the names of the four corpora the
registry fetches by hand — converts word counts to tokens at D6's factor,
scores each row against the two KPIs, and totals every domain under D8's three
access filters. It writes `tables/` and `figures/` and is safe to rerun: every
output is rebuilt from the evidence files.

## Decisions

### D1 — Language scope

- **Decision**: Slovene across all domains, with a medical-only side table for other European languages.
- **Why**: The project is Slovene-first, but if open Slovene medicine falls short, the fallback is medical text in a related or larger language. Sizing that fallback costs little while the search is already running.
- **History**:
  - 2026-09-14 first version, grill Q6

### D2 — Sources searched

- **Decision**: CLARIN.SI, HuggingFace Hub, Zenodo, ELG, LINDAT, OPUS (including parallel medical such as EMEA), Common Crawl derivatives, and Slovene medical publishers and institutions (Zdravniški vestnik, NIJZ, ZZZS, JAZMP, Wikipedia medical categories), plus synthetic and translated medical sets. Every hit is verified against its landing page.
- **Why**: These are where Slovene text is actually published. Naming them fixes the survey's scope, so the record can say what was covered and a refresh knows what to re-check.
- **History**:
  - 2026-09-14 first version, grill Q20
  - 2026-09-17 searched by four parallel agents, one per family, each required to return the URL it read for every row and to leave a field empty rather than guess

### D3 — Inclusion

- **Decision**: Every dataset found is a row, regardless of access. An `access` column (open / login / gated / not downloadable) decides what counts toward the verdict.
- **Why**: A source that exists but is not packaged — a journal archive of individual PDFs, say — is not supply today but is the answer to "what would close the gap". Excluding it would lose the most actionable finding.
- **History**:
  - 2026-09-14 first version, grill Q20

### D4 — Catalogue schema

- **Decision**: One row per dataset with: name, source handle or URL, licence, access, languages, domains, reported document count, reported word or token count, size verified, annotation type, document-length class, format, already in the download registry, KPI fit.
- **Why**: Each column answers a question the KPIs ask. Annotation type separates pretraining supply from IE fine-tuning supply; document-length class separates a sentence bank from a document corpus, which matters for pretraining.
- **History**:
  - 2026-09-14 first version, grill Q19
  - 2026-09-19 the single `KPI fit` column became two, `kpi2_fit` and `kpi4_fit`, because a row can be decided for one KPI and undecidable for the other: most rows report a document count but not a word count. Three columns were added in the same pass — `tokens_estimated` (the word count at D6's factor, so the KPI comparison is not recomputed by every reader), `family` (which searched source returned the row) and `mirrors` (the addresses of the copies merged into it under D10)

### D5 — Domain taxonomy

- **Decision**: medical, scientific, legal, news, parliamentary, academic, encyclopedic, forum/social, general-web, finance, other. A dataset may carry several. The headline science figure is scientific plus academic.
- **Why**: Shared with the sibling experiments so the three records can be read against each other.
- **History**:
  - 2026-09-14 first version, grill Q12

### D6 — Token unit and verification

- **Decision**: Sizes are recorded as the words the source reports, with about 2 subword tokens per word applied for KPI comparisons. A size counts as verified only when read from a downloaded sample or an official statistics page; a publisher's prose claim is unverified.
- **Why**: Published corpus sizes are frequently rounded, stale, or measured differently from what the project would ingest. Marking the provenance of each number keeps an unverified claim from silently becoming a KPI verdict. The token factor stays an estimate until the project tokenizer exists.
- **History**:
  - 2026-09-14 first version, grill Q3/Q14

### D7 — Two readings of KPI 2

- **Decision**: KPI 2 is reported twice: once as pretraining documents, once as annotated IE examples.
- **Why**: "10k examples" means different things for pretraining and for information extraction, and the proposal commits to both. Reporting one number would hide which commitment is met.
- **History**:
  - 2026-09-14 first version, grill Q10

### D8 — Outputs

- **Decision**: The catalogue CSV and per-domain summary tables under `tables/`, figures via datachart, produced by `analysis.py`. Per-domain totals are reported under three access filters: open only, open plus login, and all. No MLflow runs beyond a lineage run logging the catalogue as an artifact.
- **Why**: The three filters are the honest way to state supply: what anyone can download, what a registered researcher can, and what exists at all. A single total would conflate them.
- **History**:
  - 2026-09-14 first version
  - 2026-09-19 a third figure, `reported-sizes-by-domain`, was added. The token figure cannot draw a domain whose datasets nobody sized, and medicine is exactly that domain, so without the third figure the most important result would have been an empty slot on the axis

### D9 — Reading a missing size

- **Decision**: A domain total that falls short of a KPI threshold while the domain still holds datasets nobody sized is reported as `unknown`, not as a miss. A total that clears the threshold is `yes` whatever is missing, since the missing datasets could only raise it. Every table carries the count of datasets behind each verdict that reported no size.
- **Why**: D6 says an unreported size is not a verified zero. Medicine is the case that forces the rule: 34 catalogued datasets, not one of which publishes a word count, which is a fact about Slovene medical publishing rather than about how much Slovene medical text exists. Reporting that as "KPI 4 not met" would have turned a gap in the evidence into a finding.
- **History**:
  - 2026-09-19 first version

### D10 — Merging a dataset catalogued twice

- **Decision**: Rows are merged on the dataset name, normalised. The row from the repository that publishes the dataset wins over the row from an aggregator mirroring it; the duplicates only fill fields the winner left empty, and their addresses are kept in a `mirrors` column.
- **Why**: ELG and LINDAT both re-list CLARIN.SI holdings, so 13 datasets arrived twice and would have been double-counted in every per-domain total. Keeping the publisher's record keeps the licence and size as that publisher states them; keeping the mirrors means a reader can still reach the copy the other catalogue offered.
- **History**:
  - 2026-09-19 first version

## Findings

_Written by labflow:report._

## Verdict


## Reproduce

_Written at conclusion._
