---
title: Open Slovene data landscape — KPI coverage
slug: data-landscape-slovenian             # MLflow experiment: slm4ie/data/data-landscape-slovenian
category: data
line: main-line
branch: exp/data-landscape-slovenian
base_commit: a0b3306
status: concluded
ticket: "#1"
pr: "#7"
mlflow: http://localhost:5555/#/experiments/12
builds_on: []
types: [lineage]
concluded: 2026-09-28
---
# Open Slovene data landscape — KPI coverage

## TL;DR

- **Hypothesis**: Open native Slovene text covers science on both KPIs and medicine on tokens, but not medicine on examples. Refuted (concluded 2026-09-28): medical KPI 2 ([H2]) falls short, while [H1], [H3] and [H4] hold.
- **F1 — Open Slovene medicine is mostly not written in Slovene.** Medicine holds 3.04M open documents, of which 24,757 were written in Slovene; the rest is translation and machine output.
- **F2 — Medicine's new native supply is large in words and small in documents.** The four new native medical sources counted by this experiment hold about 7.9M words, thirty times KPI 4, in under four thousand documents.
- **F3 — Science clears both KPIs on native prose alone.** Native science outside the download registry reaches 147,291 open documents and 33.6M estimated tokens, both far past their thresholds.
- **Next**: medical information-extraction examples have to be built rather than found; the new native medical and scientific sources are the shortlist for the next corpus build.

## Hypothesis

- **Statement**: Open-access Slovene text sources, catalogued by domain, size, annotation and document length, contain enough medical and scientific material to meet KPI 2 and KPI 4 without synthetic data.
- **Rationale**: Nobody had written down what Slovene text exists openly beyond what the project already downloads.
  - **The project downloads 22 Slovene sources and knows little else.** Nobody has recorded what else exists openly, under what licence, in what domain, and whether it is annotated.
  - **Earlier survey work left an intuition that medicine is the gap, and no notes.** The project's pretraining corpus build, the pipeline that downloads, filters and deduplicates its sources and counts words per source, makes it concrete. Medicine finished at 352 documents and 808,176 words, a hundred times smaller than the next smallest domain.
  - **Only a search outside the repo can tell the two readings apart.** The gap is either a fact about Slovene or a fact about which sources were picked.
- **Predictions**: Each clause counts openly downloadable Slovene datasets outside the download registry, written in Slovene rather than translated or machine-generated ([D11]), with tokens estimated at two per word ([D6]). An example is one document.
  - **H1 — Medicine has enough native text to pretrain on**: confirmed if it reaches 500,000 estimated tokens (KPI 4); refuted if it falls short under every access filter — medical pretraining then needs the multilingual fallback or synthetic data.
  - **H2 — Medicine has enough native examples**: confirmed if it reaches 10,000 documents (KPI 2); refuted if it falls short under every access filter — medical examples then have to be built or synthesised.
  - **H3 — Science has enough native text to pretrain on**: confirmed if scientific plus academic datasets reach 500,000 estimated tokens; refuted if they fall short under every access filter.
  - **H4 — Science has enough native examples**: confirmed if scientific plus academic datasets reach 10,000 documents; refuted if they fall short under every access filter.
  - **Verdict rule**: confirmed when H1 to H4 all hold; refuted when any of them falls short under every access filter; inconclusive otherwise.
  - **Read beside the clauses, not as clauses**: annotated information-extraction examples per domain, the second reading of KPI 2 ([D7]), are reported but decide no clause.
- **Outcome**: refuted

## Design

- **Inputs**: The rows eight searched source families returned for Slovene datasets, and the text this experiment fetched from four native medical sources to size them, described in `## Datasets`.
- **Method / factors**: One pass from searched rows to per-domain supply. Nothing is varied; this is a survey.
  - Search the source families and keep every row with the URL it came from ([M1]), then re-read the Hugging Face harvest for rows the first pass missed ([M2]).
  - Merge datasets catalogued twice into one row ([M3]).
  - Size the reachable native medical sources by fetching and counting them ([M6]).
  - Check how the text of every row the keyword reading marks non-native came to exist, and which translation system made it ([M7]).
  - Add token estimates, provenance, registry overlap and KPI fit to every row ([M4]).
  - Total each domain under three access filters and read each total against its KPI ([M5]).
- **Metrics**: primary estimated tokens per domain against KPI 4; secondary documents per domain, native documents per domain, and annotated examples per domain against KPI 2.
- **Protocol**: A size counts only when read from official statistics or counted over downloaded text ([D6]). A total short of a threshold while datasets report no size is `unknown`, not a miss ([D9]). Search dates and families are recorded so a refresh knows what to re-check.
- **Compute**: Web search, polite per-item fetching for the four medical sources, and one MLflow lineage run logging the tables and figures.

## Datasets

- **Survey rows**: One row per dataset a searched source family returned, on the catalogue schema ([D4]), kept as JSON per family under `data/experiments/data/data-landscape-slovenian/raw/`, beside the provenance check of each non-native row ([M7]). [Rows each family returned, kept after merging, and already downloaded](tables/dataset-survey-rows-statistics.csv)
- **Fetched native medical text**: One item per case report, Wikipedia article, journal article or guideline PDF, fetched under `data/experiments/data/data-landscape-slovenian/interim/<source>/` for counting only. [Items and words per fetched source](tables/native-medical-sizes.csv)

## Methods

### M1 — The searched rows, one file per source family

- **Input**: The eight source families of the language scope ([D1]) and the source list ([D2]). Four agents searched them in parallel on 2026-09-17, one per family.
- **Output**: One JSON file per family under `data/experiments/data/data-landscape-slovenian/raw/`, each a list of rows on the catalogue schema ([D4]).
- **How**: Nothing is computed here.
  1. Each agent returns, for every row, the URL it read, and leaves a field empty rather than guess.
  2. The rows are kept verbatim as the experiment's evidence, so every later number traces to a row and a URL.
  3. One file per family lets a refresh replace one family without disturbing the others.
  4. The full Hugging Face harvest the agent worked from is kept beside the rows as `huggingface-harvest.json`.
- **Code**: `analysis.py::load_rows`
- **Settings**: `FAMILY_FILES` (which family files are read, in merge order) in `analysis.py`

### M2 — The second pass on Hugging Face

- **Input**: The Hugging Face harvest kept in the evidence of the searched rows ([M1]) and the catalogue as it stood after the first pass.
- **Output**: `huggingface-recheck.json`, the rows the first pass missed, on the same schema.
- **How**: Two corpora the project already downloads were missing, so the first pass had curated the harvest too tightly.
  1. Re-read the harvest for repositories not already catalogued.
  2. Keep those naming medicine or science in the repository id or its card.
  3. Size each candidate's Slovene configuration through the Hugging Face dataset-viewer size API on 2026-09-19.
  4. Record a repository the API refuses to size as unsized rather than estimated.
- **Code**: `analysis.py::load_rows`
- **Settings**: none

### M3 — One row per dataset

- **Input**: Every row from the searched rows ([M1]) and the second pass ([M2]), in family order.
- **Output**: One row per dataset, carrying a `mirrors` field with the addresses of the copies folded into it.
- **How**: The family order puts the publishing repository ahead of any aggregator that mirrors it ([D10]).
  1. Key each row on its dataset name with every non-alphanumeric character dropped.
  2. Keep the first row under a key.
  3. Let a later row under the same key fill only fields the kept row left empty.
  4. Append the later row's URL to `mirrors`.
- **Code**: `analysis.py::merge_duplicates`
- **Settings**: `FAMILY_FILES` (the merge order) in `analysis.py`

### M4 — The derived columns

- **Input**: The merged rows ([M3]), the download registry `configs/data/download.yaml`, the corpus build's per-source counts in `pretrain/07_statistics/aggregate.json`, the counted medical sizes ([M6]) and the provenance checks ([M7]).
- **Output**: Each row gains `words_basis`, `tokens_estimated`, `provenance`, `provenance_class`, `translation_system`, `in_registry`, `registry_key`, `kpi2_fit` and `kpi4_fit`.
- **How**: Each column answers one question the KPIs ask.
  1. Match the row, and every address in `mirrors`, against the registry's CLARIN handles and Hugging Face repositories. The four corpora fetched by hand match on a fragment of their name.
  2. Take the word count the source reported. Where there is none, take this experiment's own count ([M6]), then the corpus build's count; `words_basis` says which ([D6]).
  3. Set `tokens_estimated` to the word count times two, empty when no count exists.
  4. Read `provenance` from the terms the row's name and notes use ([D11]). Where the row was checked ([M7]), its checked class decides instead, and its translation system is carried over.
  5. Where this experiment counted a source's items itself ([M6]), take that as the row's `documents`, since it counts in the catalogue's unit.
  6. Set `kpi2_fit` for medical and scientific rows on 10,000 documents, and `kpi4_fit` on 500,000 estimated tokens. Either is `unknown` when its count was never reported ([D9]).
- **Code**: `analysis.py::annotate`
- **Settings**: `TOKENS_PER_WORD` (the tokens-per-word estimate), `KPI2_EXAMPLES` (the KPI 2 threshold), `KPI4_TOKENS` (the KPI 4 threshold), `GENERATED_TERMS` (words marking machine output), `TRANSLATED_TERMS` (words marking translation) in `analysis.py`

### M5 — Per-domain supply under each access filter

- **Input**: The annotated Slovene rows ([M4]). The rows sizing the other-language medical fallback are excluded and written to their own table.
- **Output**: `tables/supply-by-domain-and-access.csv`, one line per domain and access filter; `tables/supply-by-clause.csv`, the totals the Predictions are decided on; and the four figures.
- **How**: A row counts toward every domain it carries, so the domain columns do not sum to the catalogue.
  1. For each domain and each access filter ([D8]), total words, estimated tokens, documents, native documents and annotated examples.
  2. Count the datasets behind each total that reported no size.
  3. Read each total against its threshold: cleared is `yes`, short with nothing missing is `no`, short with sizes missing is `unknown` ([D9]).
  4. For medicine and for science, each row counted once, total again over native rows outside the registry only; these decide H1 to H4.
- **Code**: `analysis.py::summarise`, `analysis.py::clause_supply`
- **Settings**: `DOMAINS` (the domain taxonomy), `ACCESS_FILTERS` (which access classes each filter admits), `IE_ANNOTATIONS` (annotation types that make a row IE supply) in `analysis.py`

### M6 — Sizing the new native medical sources

- **Input**: The four native medical rows outside the registry that can be fetched without scraping a repository that forbids bulk access ([D12]). They are the clinical case reports on Hugging Face, the Slovene Wikipedia's medicine category, the Zdravniški vestnik journal and the faculty's clinical guidelines list.
- **Output**: `tables/native-medical-sizes.csv`, one line per source with items, words, an estimate and its basis. The derived columns ([M4]) read it back as `words_basis` `counted`, `partial` or `sampled`.
- **How**: Words are whitespace tokens throughout, as in the catalogue.
  1. Case reports: read every row through the Hub rows API. A row is one sentence, so the unit reported is the case report.
  2. Wikipedia: fetch plain-text extracts of every article within two subcategory levels of `Kategorija:Medicina`.
  3. Zdravniški vestnik: harvest the journal's whole record list through OAI-PMH and keep records whose language is Slovene.
  4. Draw 40 of those with a fixed seed, fetch each PDF at one request per two seconds, and extract its text with pypdf.
  5. Estimate the journal as mean words per counted article times the Slovene record count.
  6. Guidelines: keep the count of guidelines the faculty page states as the item count, and count words over every link that resolves to a PDF. Dead links are left out, so the word count is a lower bound.
  7. On a rerun, reuse the fetched files and recount.
- **Code**: `size_medical_sources.py::main`
- **Settings**: `SAMPLE_SIZE` (journal articles drawn), `SEED` (the draw's seed), `WIKI_DEPTH` (category levels walked), `POLITE_DELAY` (seconds between requests to a site) in `size_medical_sources.py`

### M7 — Checking how non-native text came to exist

- **Input**: Every catalogue row the keyword reading of provenance ([D11]) did not mark native, with its landing page and mirrors.
- **Output**: `provenance-check.json` beside the searched rows, one entry per row with its class, source language, translation system, generating model, evidence URL and quote, and a confidence. `tables/provenance-by-class.csv` and `tables/translation-systems.csv` total it.
- **How**: Three agents read the sources in parallel on 2026-09-28, each row's landing page first, then the paper or card it cites.
  1. Assign one class: native, human-translated, machine-translated, synthetic (written by a language model), bilingual resource, or mixed.
  2. A bilingual resource is a dictionary, term list or parallel corpus whose translation direction the source does not establish.
  3. A translation post-edited only in part counts as machine-translated.
  4. Record the translation system or generating model only where the source names it, with a verbatim quote; otherwise record it as unknown.
  5. Mark confidence high where the source states it, medium where it is strongly implied, and low where the page could not be read.
  6. Count a mixed row as machine-translated in the systems table only when machine translation is part of its mix.
- **Code**: `analysis.py::provenance_summary`
- **Settings**: `PROVENANCE_OF_CLASS` (how each checked class folds into native, translated or generated) in `analysis.py`

## Decisions

### D1 — Language scope

- **Decision**: Slovene across all domains, with a medical-only side table for other European languages.
- **Why**: The project is Slovene-first, but if open Slovene medicine falls short, the fallback is medical text in a related or larger language. Sizing that fallback costs little while the search is already running.
- **History**:
  - 2026-09-14 first version, design interview, question 6

### D2 — Sources searched

- **Decision**: CLARIN.SI, HuggingFace Hub, Zenodo, ELG, LINDAT, OPUS (with parallel medical sets such as EMEA), and Common Crawl derivatives. Also Slovene medical publishers and institutions (Zdravniški vestnik, NIJZ, ZZZS, JAZMP, Wikipedia medical categories), plus synthetic and translated medical sets. Every hit is verified against its landing page.
- **Why**: These are where Slovene text is actually published. Naming them fixes the survey's scope, so the record can say what was covered and a refresh knows what to re-check.
- **History**:
  - 2026-09-14 first version, design interview, question 20
  - 2026-09-17 searched by four parallel agents, one per family, each required to return the URL it read for every row and to leave a field empty rather than guess
  - 2026-09-19 the Hugging Face family was searched a second time [M2]. Two corpora the project already downloads, FinePDFs and Legal-mC4, were missing although the harvest held both. The harvest is now kept as evidence, so a third pass re-reads it instead of re-searching the Hub

### D3 — Inclusion

- **Decision**: Every dataset found is a row, regardless of access. An `access` column (open / login / gated / not downloadable) decides what counts toward the verdict.
- **Why**: A source that exists but is not packaged — a journal archive of individual PDFs, say — is not supply today but is the answer to "what would close the gap". Excluding it would lose the most actionable finding.
- **History**:
  - 2026-09-14 first version, design interview, question 20

### D4 — Catalogue schema

- **Decision**: One row per dataset. Its columns are name, source handle or URL, licence, access, languages, domains, reported document count, and reported word or token count. The rest are size verified, annotation type, document-length class, format, already in the download registry, and KPI fit.
- **Why**: Each column answers a question the KPIs ask. Annotation type separates pretraining supply from IE fine-tuning supply; document-length class separates a sentence bank from a document corpus, which matters for pretraining.
- **History**:
  - 2026-09-14 first version, design interview, question 19
  - 2026-09-19 the single `KPI fit` column became `kpi2_fit` and `kpi4_fit`, because most rows report a document count but not a word count. `tokens_estimated`, `family` and `mirrors` added in the same pass
  - 2026-09-21 `registry_key`, `words_corpus`, `documents_corpus` and `words_basis` added, so a size taken from the project's own build is distinguishable from one the source reported [D6]

### D5 — Domain taxonomy

- **Decision**: medical, scientific, legal, news, parliamentary, academic, encyclopedic, forum/social, general-web, finance, other. A dataset may carry several. The headline science figure is scientific plus academic.
- **Why**: Shared with the project's other data experiments, so their records can be read against each other.
- **History**:
  - 2026-09-14 first version, design interview, question 12

### D6 — Token unit and verification

- **Decision**: Sizes are recorded as the words the source reports, with about 2 subword tokens per word applied for KPI comparisons. A size counts as verified only when read from a downloaded sample or an official statistics page; a publisher's prose claim is unverified.
- **Why**: Published corpus sizes are frequently rounded, stale, or measured differently from what the project would ingest. Marking the provenance of each number keeps an unverified claim from silently becoming a KPI verdict. The token factor stays an estimate until the project tokenizer exists.
- **History**:
  - 2026-09-14 first version, design interview, question 3/Q14
  - 2026-09-21 a registry source whose publisher reports no size takes the word count the corpus build measured, marked `words_basis: corpus`; that is a count over downloaded text, the stronger verification this decision admits
  - 2026-09-22 a source this experiment fetched and counted itself takes that count ahead of the corpus fallback, marked `counted`, `partial` (a lower bound) or `sampled` (extrapolated from a fixed-seed sample) [M6]

### D7 — Two readings of KPI 2

- **Decision**: KPI 2 is reported twice: once as pretraining documents, once as annotated IE examples.
- **Why**: "10k examples" means different things for pretraining and for information extraction, and the proposal commits to both. Reporting one number would hide which commitment is met.
- **History**:
  - 2026-09-14 first version, design interview, question 10

### D8 — Outputs

- **Decision**: The catalogue CSV and per-domain summary tables under `tables/`, figures via datachart, produced by `analysis.py`. Per-domain totals are reported under three access filters: open only, open plus login, and all. No MLflow runs beyond a lineage run logging the catalogue as an artifact.
- **Why**: The three filters are the honest way to state supply: what anyone can download, what a registered researcher can, and what exists at all. A single total would conflate them.
- **History**:
  - 2026-09-14 first version
  - 2026-09-19 a third figure, `reported-sizes-by-domain`, added, because the token figure cannot draw a domain whose datasets nobody sized, and medicine was that domain

### D9 — Reading a missing size

- **Decision**: A domain total that falls short of a KPI threshold while the domain still holds datasets nobody sized is reported as `unknown`, not as a miss. A total that clears the threshold is `yes` whatever is missing, since the missing datasets could only raise it. Every table carries the count of datasets behind each verdict that reported no size.
- **Why**: An unreported size is not a verified zero, as the token unit and verification rule holds ([D6]). Reporting a medical domain nobody sized as a KPI 4 miss would turn a gap in the evidence into a finding about Slovene.
- **History**:
  - 2026-09-19 first version

### D10 — Merging a dataset catalogued twice

- **Decision**: Rows are merged on the dataset name, normalised. The publisher's row wins over an aggregator's row mirroring it. The duplicates only fill fields the winner left empty, and their addresses are kept in a `mirrors` column.
- **Why**: ELG re-lists CLARIN.SI holdings, so 13 datasets arrived twice and would have been double-counted in every per-domain total. Keeping the publisher's record keeps the licence and size as the publisher states them; the mirrors stay reachable.
- **History**:
  - 2026-09-19 first version
  - 2026-09-28 Why corrected: every merged duplicate came through ELG; LINDAT's rows matched no CLARIN.SI name

### D11 — Provenance of the Slovene text

- **Decision**: Every row is first read as `native`, `translated` or `generated` from the terms its name and notes use. Every row not read as native is then checked against its card or paper into six classes ([M7]), and the checked class decides. Per-domain totals carry the native document count beside the full one.
- **Why**: The Hypothesis asks whether open sources cover the KPIs *without* synthetic data, and a document count alone cannot answer that. Medicine forces it: the domain clears KPI 2 only on instruction sets and machine-translated material [F1].
- **Alternatives**:
  - Check every row by hand, native ones included — rejected: the non-native rows are where a misreading changes what the Predictions count.
  - Keep the keyword reading alone — rejected: it counted dictionaries and human translation memories as translated text and named no translation system.
- **History**:
  - 2026-09-19 first version
  - 2026-09-28 non-native rows checked against their sources into six classes, stored as evidence so the reading still reruns ([M7]); five rows moved to native

### D12 — Which new medical sources to fetch

- **Decision**: Sources reachable through an API or a standard harvesting endpoint are fetched whole or sampled. Sources whose repository states there is no bulk export are not fetched. Of the native medical rows, four were fetched and the University of Ljubljana repository was left out.
- **Why**: A word count over downloaded text is the only verification the token unit rule accepts ([D6]), and none of these sources publishes one. Polite per-item fetching is ordinary research use where the publisher offers an endpoint. The one source saying "no bulk export" is left alone.
- **Alternatives**:
  - Estimate from document counts alone — rejected: half the rows have none, and the units differ (a case-report row is a sentence).
- **History**:
  - 2026-09-21 first version, approved by the project lead before anything was fetched

## Findings

### F1 — Open Slovene medicine is mostly not written in Slovene · key

- **Summary**: Medicine holds 3.04M open documents, of which 24,757 were written in Slovene; the rest is translation and machine output.
- **Runs**: 607de533f6a04e7bbc60368b56ce8cb2
- **Result**: ![Share of each domain's openly downloadable documents that were written in Slovene, with the domain's document total beside its name and medicine's row bolded. Most domains sit at or near full native share; other, legal and academic sit near half, weighed down by translation memories and instruction sets; medicine sits near zero.](figures/documents-by-domain-and-provenance.svg)
- **Reading**:
  - **Medicine is the only domain where translated and machine-written documents far outnumber native ones.** Everywhere else most documents were written in Slovene. Medicine's total is carried by instruction sets, prompt-and-answer pairs written to train chat models, and by one medical question set translated from English and annotated automatically.
  - **It refutes the medical examples clause [H2] as the Statement meant it.** Open medical documents pass KPI 2's threshold only when machine output counts, and the Statement excludes it. Most of the native remainder is PoVeJMo-VeMo-Med, a corpus of Slovene medical texts the project already downloads, which the clause also excludes.
  - **A native medical corpus at scale would overturn it.** None turned up in the eight searched source families. Three Hugging Face datasets from the `texdata` account, gated so they open only on request, are no such candidate. Two are machine-translated from English, and the third is general Slovene web and Wikipedia text ([M7]).
- **Implication**: Medical examples for KPI 2 cannot be collected from open native data. They have to be built, or found beyond the source families this survey searched.
- **History**:
  - 2026-09-19 first result, from the catalogue at the second Hugging Face pass [M2]
  - 2026-09-28 access to the three gated `texdata` sets, requested 2026-09-22, still ungranted; the reading no longer names them as candidates
  - 2026-09-28 the `texdata` sets described from their checked provenance ([M7]): two machine-translated, one general native text. Lineage rerun at e32f209, replaces run 9877583f146a4ba486c42382e64087c3
  - 2026-09-28 Summary corrected to the native count after the sizing pass ([M6]); lineage rerun at b453686, replaces run e953aa30cf0948f9be2e6fcb3658bdde

### F2 — Medicine's new native supply is large in words and small in documents · key

- **Summary**: The four new native medical sources counted by this experiment hold about 7.9M words, thirty times KPI 4, in under four thousand documents.
- **Runs**: 607de533f6a04e7bbc60368b56ce8cb2
- **Result**: [Items and words counted for each new native medical source](tables/native-medical-sizes.csv)
- **Reading**:
  - **The national medical journal, Zdravniški vestnik, carries almost all of the new words.** Its Slovene articles, sized from a fixed-seed sample ([M6]), hold most of the total, and the Slovene Wikipedia's medicine category most of the rest. The case reports and the guidelines list are small: the reports are few once sentences are grouped, and most guideline links are dead.
  - **It confirms the medical text clause [H1] and refutes the medical examples clause [H2].** New native text clears KPI 4 many times over, even at the rough two-tokens-per-word estimate ([D6]). Native medical documents outside the registry stay under ten thousand under every access filter, and even that count holds the journal three times, once per catalogue listing it.
  - **Only a different unit of example would overturn it.** The one open native medical source left unsized, the CURLICAT corpus, is a collection of sentences with no documents to count. Counting sentences or paragraphs as examples would clear KPI 2 easily; counting documents, nothing in reach adds more than a few thousand.
- **Implication**: Medical pretraining text can be grown from open native sources, so the multilingual fallback is a choice rather than a necessity. Medical information-extraction examples have to be built.
- **History**:
  - 2026-09-19 first result, from the catalogue at the second Hugging Face pass [M2]: the domain read `unknown`, with no sized dataset at all
  - 2026-09-21 revised: registry rows take the corpus build's word count when the source reports none ([D6]); the verdict moved from `unknown` to `yes` on held data
  - 2026-09-22 revised: the four reachable new sources fetched and counted ([M6]); KPI 4 now met on new supply, title and Summary changed with it

### F3 — Science clears both KPIs on native prose alone · key

- **Summary**: Native science outside the download registry reaches 147,291 open documents and 33.6M estimated tokens, both far past their thresholds.
- **Runs**: 607de533f6a04e7bbc60368b56ce8cb2
- **Result**: [Native supply outside the download registry, per Prediction domain and access filter](tables/supply-by-clause.csv)
- **Reading**:
  - **Science clears both thresholds under the open filter alone.** Science here is scientific plus academic datasets, each counted once ([D5]). Loosening access only adds to totals already past both thresholds.
  - **It confirms the science text clause [H3] and the science examples clause [H4].** The totals count only native text outside the download registry, as the clauses require. The margin is wide enough that the rough token estimate ([D6]) cannot change it, and the datasets reporting no size could only raise it.
  - **Only the annotated reading of KPI 2 falls short, and it decides no clause.** No scientific dataset carries information-extraction annotation, so under that second reading ([D7]) science supplies pretraining text but no ready-made extraction examples.
- **Implication**: Science needs no further sourcing for pretraining. Information-extraction evaluation in the scientific domain has to be built, not found.
- **History**:
  - 2026-09-19 first result, from the catalogue at the second Hugging Face pass [M2]
  - 2026-09-28 revised: decided on native supply outside the registry, each row counted once, as the clauses require; the earlier total included registry rows. Lineage rerun at b453686

### F4 — The catalogue is mostly supply the project does not use · supporting F3

- **Summary**: 266 of the 295 catalogued Slovene datasets are outside the download registry.
- **Runs**: 607de533f6a04e7bbc60368b56ce8cb2
- **Result**: [The catalogue, one row per dataset](tables/catalogue.csv)
- **Reading**:
  - **Nine in ten catalogued datasets are ones the project does not download.** The catalogue finds nearly every registry entry; the two it misses are not openly published.
  - **This qualifies science clearing both KPIs on native prose ([F3]).** Science clears its thresholds on new supply, not only on what is already downloaded, through its new scientific and academic datasets.
  - **The headline overstates what matters.** Most of the surplus is general-web and miscellaneous text, where every threshold is long cleared. A registry that already held the scientific datasets would undo the qualification.
- **Implication**: The new scientific and academic rows are the shortlist for the next corpus build; the general-web surplus is not worth ingesting.
- **History**:
  - 2026-09-19 first result, from the catalogue at the second Hugging Face pass [M2]

### F5 — An aggregator re-lists a twentieth of the catalogue · minor

- **Summary**: 13 CLARIN.SI datasets arrive a second time through ELG and would otherwise be counted twice.
- **Runs**: 607de533f6a04e7bbc60368b56ce8cb2
- **Result**: [The catalogue, one row per dataset](tables/catalogue.csv), `mirrors` column
- **Reading**: Merging on the normalised dataset name ([D10]) keeps every per-domain total free of these double counts.
- **History**:
  - 2026-09-19 first result, from the catalogue at the second Hugging Face pass [M2]
  - 2026-09-28 corrected to ELG alone: no LINDAT row merged into a CLARIN.SI one

### F6 — Machine-made Slovene is rare in the catalogue and gathers in medicine · supporting F1

- **Summary**: Machine translation touches 15 catalogued datasets and model-written text 9; most wholly synthetic sets are medical.
- **Runs**: 607de533f6a04e7bbc60368b56ce8cb2
- **Result**: [Datasets, documents and words per checked provenance class](tables/provenance-by-class.csv)
- **Reading**:
  - **Most non-native datasets are human translations or bilingual resources, not machine output.** Of the rows checked ([M7]), professional translation memories and parallel corpora outnumber machine-translated and synthetic sets together. Five rows the keyword reading flagged turned out to be native Slovene.
  - **This qualifies medicine's machine-written majority ([F1]).** Machine output is uncommon across the catalogue, but five of the six wholly synthetic sets are medical, and they hold about half of the domain's documents. Medicine's shortfall is a property of that domain's open supply, not of the catalogue as a whole.
  - **Native rows misread as native would overturn the proportion.** Only rows the keyword reading flagged were checked, so a machine-translated set described without the usual words would still count as native.
- **History**:
  - 2026-09-28 first result, from the provenance check ([M7])

### F7 — Google Translate is the translation system named most often · minor

- **Summary**: Of 15 datasets involving machine translation, 10 name the system; Google Translate is the most common.
- **Runs**: 607de533f6a04e7bbc60368b56ce8cb2
- **Result**: [Machine-translated and mixed datasets per translation system](tables/translation-systems.csv)
- **Reading**: The named systems range from Google Translate and DeepL to language models used as translators, so no single system's errors dominate the machine-translated Slovene.
- **History**:
  - 2026-09-28 first result, from the provenance check ([M7])

## Verdict

- **Outcome**: refuted — the medical examples clause [H2] falls short under every access filter; the medical text clause [H1] and both science clauses, [H3] and [H4], hold.
- **Evidence**: Medicine's open documents are machine output ([F1]), and its new native sources hold many words in few documents ([F2]). Science clears both KPIs on native prose the project does not yet download ([F3]), from a catalogue that is mostly such new supply ([F4]).
- **Discussion**:
  - **Open native Slovene text covers science on both KPIs and medicine on tokens, but not medicine on examples.** Medicine has plenty of native words and too few native documents to count as examples.
  - **The medical gap was one of documents, not of words.** The project's corpus build suggested medicine lacked text. Once the national medical journal and Wikipedia were counted, text was plentiful; what is scarce is many separate native medical documents.
  - **The survey measures what is catalogued, not what is written.** Sources outside the eight families, and sources whose repositories forbid bulk access, were not counted. A follow-up that builds medical examples inherits the question of which unit counts as an example.
- **Adoption candidate**: Add Zdravniški vestnik and the Wikipedia medicine category to the download registry, with the new scientific and academic rows ([F4]). The pieces to lift are the journal harvester and the Wikipedia category walk in `size_medical_sources.py`.
- **Next**:
  - Build medical information-extraction examples; open supply cannot provide ten thousand native medical documents.
  - Settle why PoVeJMo-VeMo-Med reports far more documents than the corpus build kept, in the curation-quality experiment (`curation-quality-slovenian`), which tests how much good text the corpus build's filters discard.
  - Decide in a KPI-coverage follow-up whether scientific extraction examples are built or found; none are annotated today.
  - Retry the three gated `texdata` sets if access is ever granted; the request of 2026-09-22 is unanswered.
  - Size Zdravniški vestnik from a larger sample before ingesting it.
- **Limitations**:
  - Under open access, medicine's document total reads `unknown` by the missing-size rule ([D9]) only because of one sentence collection with no document unit. The verdict reads it as a miss.
  - Tokens are estimated at two per word until the project tokenizer exists.
  - The journal's word count is extrapolated from a sample of about forty articles.
  - Only rows the keyword reading marked non-native were checked against their sources; a native row it misread stays native.
  - The three gated `texdata` sets are catalogued but unsized; access was never granted.
  - Source families beyond the eight searched were not covered.

## Reproduce

The searched rows under `data/experiments/data/data-landscape-slovenian/raw/` are the survey's evidence and are not regenerated. The corpus build's statistics must be in place under `data/pretrain/07_statistics/`.

```bash
export MLFLOW_TRACKING_URI=http://localhost:5555
cd "$(git rev-parse --show-toplevel)"

# F2 — fetch and count the new native medical sources (commit ff303ac)
uv run --group analysis python experiments/data/data-landscape-slovenian/size_medical_sources.py

# F1, F2, F3, F4, F5 — catalogue, tables, figures and the lineage run (commit e32f209)
uv run --group analysis python experiments/data/data-landscape-slovenian/analysis.py --mlflow
```
