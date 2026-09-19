---
title: Open Slovene data landscape — KPI coverage
slug: data-landscape-slovenian             # MLflow experiment: slm4ie/data/data-landscape-slovenian
category: data
line: main-line
branch: exp/data-landscape-slovenian
base_commit: a0b3306
status: running
ticket: "#1"
pr:
mlflow:
builds_on: []
types: []
concluded:
---
# Open Slovene data landscape — KPI coverage

## TL;DR

- **Hypothesis**: leaning refuted for medicine, confirmed for science
- Medicine holds 3.04M open documents but only 22,435 of them were written in Slovene; the rest is translation and machine output [F1]
- No open Slovene medical dataset publishes a size, so KPI 4 for medicine cannot be read at all [F2]
- Science clears both KPIs on native prose alone, at 299,961 documents and 5.28B estimated tokens [F3]
- **Next**: request access to the three gated `texdata` medical sets, and size the native medical sources by sampling

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

### M1 — The searched rows, one file per source family

- **Input**: The eight source families of [D1] and [D2]. Four agents searched them in parallel on 2026-09-17, one per family, each required to return the URL it read for every row and to leave a field empty rather than guess.
- **Output**: One JSON file per family under `data/experiments/data/data-landscape-slovenian/raw/`, each a list of rows on [D4]'s schema. 304 rows in the first pass, 15 more from the second pass on Hugging Face ([M2]).
- **How**: Nothing is computed here. The agents' output is kept verbatim as the experiment's evidence, so every later number can be traced to the row and the URL it came from, and a refresh can replace one family without disturbing the others. The full Hugging Face harvest the fourth agent worked from — 657 Slovene-tagged repositories — is kept beside the curated rows as `huggingface-harvest.json`.
- **Code**: `analysis.py::load_rows`
- **Settings**: `FAMILY_FILES` in `analysis.py`

### M2 — The second pass on Hugging Face

- **Input**: The 657-repository harvest from [M1] and the catalogue as it stood after the first pass.
- **Output**: `huggingface-recheck.json`, 15 rows the first pass missed, on the same schema.
- **How**: Two datasets the project already downloads were absent from the catalogue, which meant the first pass had curated the harvest too tightly rather than searched it too narrowly. The harvest was re-read for repositories not already catalogued, filtered to those naming medicine or science in the repository id or the card, and each candidate's Slovene configuration was sized through the Hugging Face dataset-viewer size API on 2026-09-19. A repository whose size the API refuses — it answers 501 for a repository with no viewer and 401 for a gated one — is recorded unsized rather than estimated.
- **Code**: `analysis.py::load_rows`
- **Settings**: none

### M3 — One row per dataset

- **Input**: Every row from [M1] and [M2], in family order.
- **Output**: One row per dataset, carrying a `mirrors` field with the addresses of the copies folded into it.
- **How**: Rows are keyed on the dataset name with every non-alphanumeric character dropped. The first row under a key wins, and the family order puts the repository that publishes a dataset ahead of the aggregator that mirrors it ([D10]); later rows under the same key only fill fields the winner left empty, and their URLs are appended to `mirrors`. 13 datasets arrived twice this way, all of them CLARIN.SI holdings that ELG or LINDAT re-list.
- **Code**: `analysis.py::merge_duplicates`
- **Settings**: `FAMILY_FILES` in `analysis.py`

### M4 — The derived columns

- **Input**: The merged rows from [M3], and `configs/data/download.yaml`.
- **Output**: Each row gains `tokens_estimated`, `provenance`, `in_registry`, `kpi2_fit` and `kpi4_fit`.
- **How**: `tokens_estimated` is the reported word count at [D6]'s two tokens per word, left empty when no word count was reported. `in_registry` compares the row's own address and the addresses in `mirrors` against the registry, matching CLARIN and LINDAT handles and Hugging Face repositories; the four corpora the registry fetches by hand carry neither, so they are matched on a fragment of the catalogue's name for the same corpus. `provenance` reads the name and notes for the terms of [D11]. `kpi2_fit` is set only for rows carrying medicine or science and asks whether the row alone reaches 10,000 documents; `kpi4_fit` asks whether it reaches 500,000 estimated tokens. Either is `unknown` when the count it needs was never reported ([D9]).
- **Code**: `analysis.py::annotate`
- **Settings**: `TOKENS_PER_WORD`, `KPI2_EXAMPLES`, `KPI4_TOKENS`, `GENERATED_TERMS`, `TRANSLATED_TERMS` in `analysis.py`

### M5 — Per-domain supply under each access filter

- **Input**: The annotated Slovene rows from [M4]; the rows marked as the other-language medical fallback are excluded and written to their own table.
- **Output**: `tables/supply-by-domain-and-access.csv`, one line per domain and access filter, and the four figures.
- **How**: A row counts toward every domain it carries, so a dataset tagged medical and scientific is in both totals and the columns do not sum to the catalogue. For each domain and each of [D8]'s three access filters the script totals words, estimated tokens, documents, documents from datasets written in Slovene, and documents from datasets carrying information-extraction annotation, and counts the datasets behind each total that reported no size. Each KPI verdict then reads its total against its threshold under [D9]: cleared is `yes` whatever is missing, short with nothing missing is `no`, and short with sizes missing is `unknown`.
- **Code**: `analysis.py::summarise`
- **Settings**: `DOMAINS`, `ACCESS_FILTERS`, `IE_ANNOTATIONS` in `analysis.py`

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
  - 2026-09-19 the Hugging Face family was searched a second time [M2]. Two corpora the project already downloads, FinePDFs and Legal-mC4, were missing from the catalogue although the harvest held both, so the first pass had curated the harvest too tightly rather than searched too narrowly. The harvest is now kept as evidence beside the curated rows, so a third pass re-reads it instead of re-searching the Hub

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
- **Why**: An unreported size is not a verified zero [D6]. Medicine forces the rule: 43 catalogued datasets, none of them publishing a word count [F2]. Reporting that as KPI 4 missed would turn a gap in the evidence into a finding about Slovene.
- **History**:
  - 2026-09-19 first version

### D10 — Merging a dataset catalogued twice

- **Decision**: Rows are merged on the dataset name, normalised. The row from the repository that publishes the dataset wins over the row from an aggregator mirroring it; the duplicates only fill fields the winner left empty, and their addresses are kept in a `mirrors` column.
- **Why**: ELG and LINDAT both re-list CLARIN.SI holdings, so 13 datasets arrived twice and would have been double-counted in every per-domain total. Keeping the publisher's record keeps the licence and size as that publisher states them; keeping the mirrors means a reader can still reach the copy the other catalogue offered.
- **History**:
  - 2026-09-19 first version

### D11 — Provenance of the Slovene text

- **Decision**: Every row is read as `native`, `translated` or `generated` from the terms its own name and notes use. `generated` is tested first, so a set that was translated and then automatically annotated counts as machine output. Per-domain totals carry the native document count beside the full one.
- **Why**: The Hypothesis asks whether open sources cover the KPIs *without* synthetic data, and a document count cannot answer that. Medicine forces it: the domain clears KPI 2 by two orders of magnitude, almost all of it instruction sets and machine-translated material [F1].
- **Alternatives**: Judge each row by hand — rejected: a judgement living outside the script cannot be rerun as the catalogue grows.
- **History**:
  - 2026-09-19 first version

## Findings

### F1 — Open Slovene medicine is mostly not written in Slovene · key

- **Summary**: Medicine holds 3.04M open documents, of which 22,435 were written in Slovene; the rest is translation and machine output.
- **Runs**: none — this experiment is a survey, and its evidence is the committed catalogue rather than MLflow runs
- **Result**: ![Documents per domain, split into text written in Slovene and text translated or machine-written, against the KPI 2 threshold of 10,000. Medicine is the only domain where the machine-written and translated share is the larger of the two, by two orders of magnitude.](figures/documents-by-domain-and-provenance.svg)
- **Reading**: Medicine is the only domain in the taxonomy whose supply inverts: everywhere else the documents written in Slovene outnumber the translated and machine-written ones, and in medicine they are outnumbered roughly 135 to 1. The total is carried by five GaMS-Instruct medical instruction sets and the HUMADEX NER set, which was built by translating English medical question-answer pairs and annotating them automatically. What remains after [D11]'s reading is 22,435 documents across eight sources, and 17,701 of those are PoVeJMo-VeMo-Med, which the project already downloads. This refutes the Prediction as it was meant rather than as it was worded: openly downloadable medicine does pass 10k examples, but not "without synthetic data", and not from sources outside the download registry. It would be overturned by sources that are native Slovene medical prose at scale — the three gated `texdata` sets are the nearest candidates, and none of them can be read without access.
- **Implication**: The medical corpus cannot be grown from open data alone at the volume KPI 4 implies. Either the gated and not-downloadable sources are pursued, or the multilingual fallback becomes the plan rather than the reserve.
- **History**:
  - 2026-09-19 first result, from the catalogue at the second Hugging Face pass [M2]

### F2 — No open Slovene medical dataset publishes a size · key

- **Summary**: All 43 catalogued medical datasets report no word count, so KPI 4 for medicine cannot be read either way.
- **Runs**: none — survey
- **Result**: ![Datasets per domain, stacked into those whose source reports a size and those that do not. Medicine is the only domain with no sized dataset at all.](figures/reported-sizes-by-domain.svg)
- **Reading**: Every other domain in the taxonomy has at least a few datasets whose publisher states a word or token count; medicine has none, at any access level. The domain's KPI 4 verdict is therefore `unknown` under [D9] rather than a miss, and the token figure cannot draw medicine at all. This neither confirms nor refutes the Prediction — it says the Prediction is unreadable on present evidence, which is a different result from the one the experiment set out to get. The gap is closable: the sources are mostly small enough to download and count, and PoVeJMo-VeMo-Med is already in the corpus, where it measured 352 documents and 808,176 words after curation. What cannot be closed by counting is the gated material, where the size is behind the same approval as the text.
- **Implication**: Sizing the eight native medical sources by sampling is the cheapest way to turn this `unknown` into a number, and should happen before the verdict.
- **History**:
  - 2026-09-19 first result, from the catalogue at the second Hugging Face pass [M2]

### F3 — Science clears both KPIs on native prose alone · key

- **Summary**: Science reaches 299,961 openly downloadable documents written in Slovene and 5.28B estimated tokens, both far past their thresholds.
- **Runs**: none — survey
- **Result**: [Per-domain supply under each access filter](tables/supply-by-domain-and-access.csv)
- **Reading**: Reading science as scientific plus academic ([D5]), the open filter alone gives 433,671 documents, of which 299,961 are native, and 5.28B estimated tokens against KPI 4's 500,000. The margin is large enough that neither the two-tokens-per-word estimate ([D6]) nor the eight scientific and ten academic datasets that report no size can change the verdict. This confirms the Prediction for science on every reading: the threshold is cleared on native text, from sources outside the download registry, without the access filter having to be loosened. The one reading it does not satisfy is [D7]'s second: no scientific dataset carries information-extraction annotation at all, so science supplies pretraining text and evaluation material for the academic sub-domain only.
- **Implication**: Science needs no further sourcing for pretraining. Information-extraction evaluation in the scientific domain has to be built, not found.
- **History**:
  - 2026-09-19 first result, from the catalogue at the second Hugging Face pass [M2]

### F4 — The catalogue is mostly supply the project does not use · supporting F3

- **Summary**: 266 of the 295 catalogued Slovene datasets are outside the download registry.
- **Runs**: none — survey
- **Result**: [The catalogue](tables/catalogue.csv)
- **Reading**: The registry holds 31 entries and the catalogue matches 29 of them, missing only the Sloleks relations lexicon and the private Slovenian News living corpus, neither of which is openly published. The other 266 rows are datasets nobody on the project has drawn on. Most of that surplus is in `other` and `general-web`, where the corpus is already far past every threshold, so the headline number overstates how much of it matters; the surplus that bears on [F3] is the ten scientific and twelve academic datasets that are new. This qualifies [F3] rather than extending it: science clears its thresholds on new supply, not only on what is already downloaded.
- **Implication**: The scientific and academic rows flagged new are the shortlist for the next corpus build; the general-web surplus is not worth ingesting.
- **History**:
  - 2026-09-19 first result, from the catalogue at the second Hugging Face pass [M2]

### F5 — Two aggregators re-list a twentieth of the catalogue · minor

- **Summary**: 13 CLARIN.SI datasets arrive a second time through ELG or LINDAT and would otherwise be counted twice.
- **Runs**: none — survey
- **Result**: [The catalogue](tables/catalogue.csv), `mirrors` column
- **History**:
  - 2026-09-19 first result, from the catalogue at the second Hugging Face pass [M2]


_Written by labflow:report._

## Verdict


## Reproduce

_Written at conclusion._
