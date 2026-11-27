---
title: Which hosted LLMs can make Slovene medical and scientific NER examples worth paying for
slug: silver-llm-screen-slovenian
short: LLM screen for silver data
category: data
branch: exp/silver-llm-screen-slovenian
base_commit: db44227
status: draft
kind:
ticket: "#64"
pr:
mlflow:
builds_on: [kpi-coverage-slovenian, ie-benchmark-slovenian]
varies: [model-families/decoder-only]
uses: []
tests: []
evidence: {}
types: [baseline]
concluded:
---

# Which hosted LLMs can make Slovene medical and scientific NER examples worth paying for

## Hypothesis

- **Question**: Which of the LLMs offered through one hosted service write and label Slovene medical and scientific text well enough, at a price the project can pay, to make its named-entity training examples?
- **Decision it informs**: The one to three models that go on to make the full medical and scientific example sets, and whether those sets are built mainly by labelling real text or by writing new text.
- **Builds on**:
  - `kpi-coverage-slovenian`: that experiment found about a million medical and a million scientific documents in the curated corpus, none with entity labels. It left open who labels them.
  - `ie-benchmark-slovenian`: that experiment scored LLMs on general Slovene entity labels against human gold and built the span scorer and the chat layer. It left medical and scientific labels untested.
- **Statement**: No bar is tested. The experiment measures, per model, how correct its Slovene and its labels are in two jobs and two domains, and what each example costs.
- **Rationale**: The models differ in price far more than any earlier result says they differ in quality, so a small screen should already separate them.
  - **Price spans two orders of magnitude.** Among the ten screened models the list price of output runs from 0.50 to 20 US dollars per million tokens ([projected cost per model](tables/scoping-cost-by-model.csv)).
  - **General Slovene labelling is already strong in the costlier models.** The extraction benchmark measured three Claude models between 77 and 94 points of strict span F1 on four general-domain sets. Nothing comparable exists for medical or scientific labels.
  - **A judge model alone has not been trustworthy here.** The curation experiments compared one judge model with one person and reached the agreement bar of 0.6 (Cohen's kappa) in one of four draws. This screen therefore pairs judge ratings with rule checks and a person's blind ratings.
  - **The data can carry the design, by the counts on disk.** The two natively written sources hold far more text than the 10 passages per domain the screen draws. The screen needs 400 model calls and 800 judge calls, projected at about 21 US dollars at list price ([projected cost per model](tables/scoping-cost-by-model.csv)).
- **Predictions**: Each clause is read per model, domain and job on 10 examples per cell. Every value is reported with a 95% bootstrap interval that resamples passages, since spans and words cluster inside a passage. A failed or unparsed output counts as wrong in every share. At this size only wide gaps between models can be told apart.
  - **Q1 · measured — How correct and natural is the Slovene each model writes**: read on `tables/language-rating-by-model.csv`; reported as errors per 100 words and the mean naturalness rating with their intervals, per model over both domains and then per domain, beside the same error count on the real passages as the judges' floor — the answer feeds the shortlist ([D7]; measured by [M4]).
  - **Q2 · measured — How correct are the entity labels each model gives**: read on `tables/label-correctness-by-model.csv`; reported as the share of labelled spans rated correct, per model, domain and job, and, in the labelling job only, the share of pooled correct spans a model found — the answer feeds the shortlist ([D7]; measured by [M4]).
  - **Q3 · diagnostic · measured — Whether written text reads less natural than real text**: read on `tables/naturalness-real-against-written.csv`; reported as the mean naturalness rating of the real passages and of each model's written passages, rated blind in one pool — the answer feeds the choice between labelling real text and writing new text ([D4]; measured by [M4]).
  - **Verdict rule**: Q1 and Q2 both answered is answered. One of them answered is partial, and the reading names which side of quality stays unknown. Neither answered is unanswered.
  - **Read beside the clauses, not as clauses**: the share of outputs passing the rule checks, refusals and failed calls, the cost per example and per 10,000 examples, and the agreement between each judge and the person.
- **Outcome**: open

## Rounds

### R1 — a small screen of ten models on both jobs and both domains

- **Opened**: 2026-11-20
- **Tests**: Which of the ten models are clearly unsuitable on language, labels or cost, so that a second round spends its larger sample only on the best four.
- **Clauses**: Q1, Q2, Q3
- **Budget**: about 25 of the 110 US dollars capped for the whole experiment. The round ends early if its spend reaches 30 US dollars ([D5]).
- **Pre-registered at**:
- **Closed**:

## Design

- **Inputs**: Real passages come from two natively written sources of the curated corpus, openly licensed theses and articles for science and half of medicine, and official product descriptions of medicines for the other half. Writing briefs and label definitions are written for this experiment. All are described in `## Datasets`.
- **Method / factors**: One pass makes the examples, checks them and rates them.
  - Draw 10 real passages per domain by a written rule, and derive one writing brief from each, the same for every model ([M1]).
  - Ask each model to label each real passage and to write a labelled passage for each brief ([M2]).
  - Check every output by rules that need no judge: it parses, its spans match the text, its labels are allowed ([M3]).
  - Have two judge models and one person rate the outputs blind to the model that made them ([M4]).
  - Price each output from the usage the service reports ([M5]).
  - Factors: scientific, the model; cuts, the domain (medical, scientific) and the job (label real text, write new text); fixed, the prompts, the label sets, the passages and briefs, and the sampling settings.
- **Metrics**: primary errors per 100 words and share of spans rated correct; secondary naturalness rating, share of pooled correct spans found, rule-check pass share, cost per 10,000 examples, judge agreement with a person — each defined in `experiments/GLOSSARY.md`
- **Protocol**: One tracked run per model holds its 40 outputs, and one per judge holds its ratings. Each output is made once, with one fixed seed where the service accepts one. A dry run projects the spend before any paid call.
- **Compute**: hosted calls only — 400 model calls and 840 judge calls in the first round, inside a cap of 110 US dollars for the experiment; no GPU.
- **People**: about three hours — one Slovene speaker rates a blind sample of at least 60 outputs.
- **Data access**: none — both sources are already in the curated corpus.
- **External dependency**: OpenRouter — every model and both judges are called through this one hosted service, at its list prices of 2026-11-20.

## Datasets

- **Real passages** (derived from the curated corpus; none labels): one item is a passage of 100 to 300 words of body prose from one document. The twenty passages sit in the experiment's data folder under `final/passages-round1/`, drawn by a written rule from open theses and articles and from medicine product descriptions. [20 passages, 3 sets](tables/dataset-passages-summary.csv) [fields](tables/dataset-passages-fields.csv) [passages and length per set](tables/dataset-passages-statistics.csv)
- **Writing briefs** (own; none labels): one item is a brief that names a text type, a topic, a target length and the labels a written passage must contain, paired with one real passage. The twenty briefs are the file `configs/briefs-passages.yaml`. [20 briefs, 2 domains](tables/dataset-briefs-summary.csv) [fields](tables/dataset-briefs-fields.csv) [briefs asking for each label](tables/dataset-briefs-statistics.csv)

## Methods

### M1 — Draw real passages and pair each with a writing brief

- **Input**: The curated corpus's documents of the open thesis collection and of the medicine product descriptions. One item is a document with its text and its catalogue fields, such as the document type and the subject class.
- **Output**: Twenty passages of 100 to 300 words, each stored with its set, domain and source document, and twenty briefs, one per passage. They are the input of the labelling and writing step ([M2]).
- **How**:
  1. Keep the documents of the wanted type and subject class.
  2. Split each into paragraphs and keep those that read as body prose: long enough, ending like a sentence, mostly letters, with no broken letters from a faulty extraction.
  3. Take runs of whole paragraphs inside the length range that contain the set's working vocabulary.
  4. Shuffle the documents with a fixed seed and take one run from each, and one per medicine, until the set is full.
  5. Write a brief per passage by hand: the same text type, topic and length, none of its wording, and the labels the text must contain.
- **Code**: `slm4ie/silver/passages.py::draw_passages`
- **Settings**: `seed` (seed of the draw, 13), `count` (passages per set), `udc_prefix` (subject classes kept), `cue_words` and `min_cues` (the working vocabulary a passage must contain) — in `configs/passages-sources.yaml`; the briefs are `configs/briefs-passages.yaml`.

### M2 — Have each model label every passage and write a passage for every brief

- **Input**: The passages and briefs of [M1], the label definitions and the prompts. One call sends one passage or one brief to one model.
- **Output**: One stored record per call: the model, the item, the raw reply, the parsed text and mentions, each mention's position in the text, the token counts and the cost. The records feed the rule checks ([M3]).
- **How**:
  1. Build the prompt from the domain's label list and the passage or the brief.
  2. Call the model once, with the fixed seed and temperature where it accepts them, and low reasoning ([D8]).
  3. Parse the reply against the answer shape. A reply off the shape is stored as unparsed.
  4. Find each mention in the text, in reading order, to get its position.
  5. Ask every model for one item before the next item, so that a stop at the spend limit loses whole items.
- **Code**: `slm4ie/silver/make.py::make_examples`
- **Settings**: `seed` (sampling seed, 13), `temperature` (0), `reasoning_effort` (reasoning strength, low), `max_tokens` (longest reply, 16000), per model `accepts_seed`, `accepts_temperature` and `reasoning_effort` ([D8]), `round_usd` (the round's spend limit, 30) — in `configs/screen-models.yaml`; the prompts are `configs/prompts-jobs.yaml`.

### M3 — Check every output by rules that need no judge

- **Input**: The stored records of [M2], with the passage or brief each answers.
- **Output**: One row of yes-or-no flags per record, and the share of records passing each flag per model and job.
- **How**:
  1. The reply parsed.
  2. Every mention was found in the text.
  3. Every label is one of the domain's labels.
  4. Labelling job: the text is the passage, unchanged.
  5. Writing job: every label the brief asks for appears, and the length is within 30% of the target.
  6. A failed call or an unparsed reply fails every flag.
- **Code**: `slm4ie/silver/checks.py::check_output`

### M4 — Rate the outputs blind, by two judge models and one person

- **Input**: The outputs that pass the rule checks of [M3], and the real passages of [M1] as bare text. One unit is a text with its numbered mentions, under an id that hides who made it.
- **Output**: Per judge and unit, a rating: the listed language errors, naturalness and realism from 1 to 5, a verdict per mention on its label and its span, and the entities the annotation missed. The person fills the same rating on a balanced sample.
- **How**:
  1. Pool the units and shuffle them with a fixed seed.
  2. Each judge rates each unit once. A rating off the scale, or one that does not cover the unit's mentions, is dropped and counted.
  3. Errors per 100 words is the number of listed errors divided by the words of the text, times 100.
  4. Draw the person's sample in whole rounds over job, domain and model family, so each appears equally often.
  5. Per rated quantity, compare each judge with the person: raw agreement, agreement within one point, and weighted Cohen's kappa (agreement beyond chance, with near misses counted as partial agreement).
- **Code**:
  - step 1: `slm4ie/silver/rate.py::rating_pool`
  - step 2: `slm4ie/silver/rate.py::rate_outputs`
  - step 3: `slm4ie/silver/rate.py::errors_per_100_words`
  - step 4: `slm4ie/silver/rate.py::draw_person_sample`
  - step 5: `slm4ie/silver/rate.py::agreement`

### M5 — Price each output from the cost the service reports

- **Input**: The token counts and the cost in US dollars that the hosted service returns with every reply.
- **Output**: The cost per stored record, summed per model and job, and projected to 10,000 examples.
- **How**:
  1. Take the reported cost. Where a reply reports none, price its tokens at the model's list price.
  2. Before each call, refuse it when the spend so far plus the model's largest single call would pass the limit.
  3. The projection to 10,000 examples is the mean cost per example of a model and job, times 10,000.
- **Code**: `slm4ie/silver/make.py::price_usd`
- **Settings**: `usd_per_million_input`, `usd_per_million_output` (list prices per model, for the fallback and the dry run) — in `configs/screen-models.yaml`.

## Decisions

### D1 — Screen ten models at ten examples per cell, then compare the best four at a hundred

- **Decision**: In the first round each model makes 10 examples per domain and job, reported as measured values with intervals and no pass level. A second round is planned for the best four models at 100 examples per cell.
- **Why**: A share on 10 examples carries an interval of roughly 25 points either side, enough to drop weak models but not to rank the rest. On 100 it narrows to about 9 points.
- **Alternatives**:
  - One round of 50 per cell for all ten models: about the same cost, wider intervals for the models that could be chosen.
  - One round of 50 per cell with one judge: inside the first cap, but no check on a judge favouring its own model family.
- **History**:
  - 2026-11-20 (before runs; user) set at 10 per cell after 100 was proposed.
  - 2026-11-20 (before runs; user) second round of 100 per cell for four finalists added, so that close models can be compared.

### D2 — Rate with two judge models from different families and one person

- **Decision**: Two judge models outside the screened list rate every output. One person rates a blind sample of at least 60, balanced over job, domain and model family. Per rated quantity, raw agreement and agreement within one point are reported beside weighted Cohen's kappa. Where no judge reaches kappa 0.6, that quantity is read on the person's ratings alone.
- **Why**: Earlier curation experiments found one judge model below that agreement in three of four draws. A second family shows whether a judge favours models related to it.
- **Alternatives**:
  - One judge model: cheaper, but its bias toward its own family could not be seen.
  - Person only: the most trusted ratings, but 400 outputs exceed the hours available.
- **History**:
  - 2026-11-20 (before runs) first version.
  - 2026-11-20 (before runs; user) raw and within-one-point agreement added, and the person's ratings made the fallback, because kappa on 60 ratings carries a range near 0.19 either side.

### D3 — Take real passages only from the two natively written sources

- **Decision**: All ten scientific passages and five medical ones come from the openly licensed collection of Slovene theses and articles. The other five medical passages come from official product descriptions of medicines, which are probably translated. Web-crawl documents and the access-restricted thesis corpus are not used. A passage is body prose of 100 to 300 words, one per document and one per medicine. Each writing brief is derived from one real passage: the same text type, topic and length, and none of its wording.
- **Why**: The real passages are the reference for natural Slovene. Most medical web-crawl documents were found by a classifier, and no experiment checked whether they were written in Slovene or translated.
- **Alternatives**:
  - Classifier-labelled web documents: far more variety, but unknown origin would blur the comparison of real against written text.
  - The access-restricted thesis corpus: larger, but its terms restrict passing its text on, and every passage is sent to a hosted service.
- **History**:
  - 2026-11-20 (before runs) first version.
  - 2026-11-20 (before runs; user) passage rule added and briefs paired with real passages, so written and real text differ only in who wrote them.
  - 2026-11-20 (before runs; user) medical passages split five and five, after the source first taken for patient leaflets proved to hold product descriptions of medicines; only the thesis passages serve as the reference for natural Slovene.

### D4 — Screen both jobs, labelling real text and writing new text

- **Decision**: Every model does both jobs on both domains.
- **Why**: The project plan made labelling real text the main route and writing a filler for rare labels. Measuring both shows whether written text is needed and how much naturalness it costs.
- **Alternatives**:
  - Writing only: half the calls, but no evidence on the cheaper route of labelling text that already exists.
- **History**:
  - 2026-11-20 (before runs; user) both jobs chosen.

### D5 — Stop the runs before total spend passes 110 US dollars

- **Decision**: The runner sums the cost the service reports per call and refuses a call that would pass the round's limit, 30 US dollars in the first round and 110 over both. Downstream work waits on no other build check.
- **Why**: The project has a fixed budget for hosted models, and reasoning tokens can raise a model's cost well above the projection.
- **History**:
  - 2026-11-20 (before runs; user) cap set at 50 US dollars.
  - 2026-11-20 (before runs; user) cap raised to 110 US dollars with the second round.

### D6 — Label medicine with the OMOP domains a text can mention, and science with ten labels

- **Decision**: Medical labels are the domains of the OMOP health-data standard that name something a text can mention, about 24, among them Condition, Drug, Procedure, Measurement, Unit, Route, Device, Visit and Specimen. Science labels are Task, Method, Dataset, Metric, Material, Software, Parameter, Value, Result and Resource. The config file holds the full lists.
- **Why**: Clinical data users already know the OMOP domains. Domains that only organise a database, and domains that combine two others, name nothing a writer mentions, so they would lengthen every prompt and stay unmeasured.
- **Alternatives**:
  - Every OMOP domain, about 45: complete, but many labels would never be used.
  - The seven clinical domains only: shorter prompts, but no labels for units, routes, visits or patient attributes.
- **History**:
  - 2026-11-20 (before runs; user) text-mentionable domains chosen; Resource added to the science labels.

### D7 — Drop a model only when it rates clearly below the costliest one, then keep the four cheapest

- **Decision**: The reference is the costliest screened model. A model is dropped when the interval of its paired difference to the reference lies wholly on the worse side, on errors per 100 words or on the share of spans rated correct, over the 20 shared items of a job. It is also dropped when more than a fifth of its outputs fail the rule checks. The four cheapest remaining models go to the second round.
- **Why**: Ten examples per cell cannot rank strong models, so most will tie. A rule written before the results keeps the choice of finalists from following the numbers.
- **Alternatives**:
  - An absolute pass level: no measured value exists yet in these domains to set one from.
  - The four highest-rated models: at this sample size the order among strong models is mostly noise.
- **History**:
  - 2026-11-20 (before runs; user) first version, after the design review.

### D8 — Send each model only the settings it accepts, and run Mistral Large 4 at high reasoning

- **Decision**: The fixed seed and temperature 0 go only to models that accept them. The four Claude models get neither, and the two GPT models get no temperature. Every model is asked for low reasoning, except Mistral Large 4, which supports only high or none and runs at high. A reply may hold 16,000 tokens, reasoning included.
- **Why**: The service sends a call only to providers that accept every setting it carries, so an unaccepted setting failed every Claude and GPT call in the smoke run. Mistral Large 4 and DeepSeek V4.1 Flash spent the earlier 4,000-token limit on reasoning and returned no answer.
- **Alternatives**:
  - Mistral Large 4 with no reasoning: closer in cost to the others' low level, but it would test the model in a weaker mode than the rest.
  - Drop the provider filter: every setting could be sent, but a provider that silently drops the answer schema could then serve the call.
- **History**:
  - 2026-11-27 (before runs) first version, after the first smoke run, from the service's model listing of that day.
