# Glossary

Terms and abbreviations the records use that a reader arriving cold may not
know, and every metric they report. One line each, `- **Term** (alias, alias):
definition` — the report links the first mention in each block to this page
and shows the definition on hover. Words match in any case; short or
symbolic forms (F1, κ, q05, P@k) match as written. Extend the list whenever a
finding introduces a term; never remove one a record still uses.

The repository's own vocabulary — tier, stage, entry, slug, backend and the rest
— lives in [CONTEXT.md](../CONTEXT.md) instead.

## Metrics

- **Precision**: of the items the system returned, the share that are correct. 1 means nothing wrong was returned.
- **Recall**: of the correct items, the share the system returned. 1 means nothing was missed.
- **F1** (F1 score): harmonic mean of precision and recall; 1 is perfect, 0 means nothing right. Macro F1 averages F1 over classes or units with equal weight; micro F1 pools all decisions first.
- **Precision@k** (P@k): precision over the top k ranked items only.
- **Recall@k** (R@k): the share of correct items found within the top k.
- **Accuracy**: the share of all decisions that are correct; misleading when classes are imbalanced.
- **Cohen's κ** (κ, kappa): agreement between two raters beyond what chance would give; 1 is full agreement, 0 chance level, below 0 worse than chance.
- **Quadratic-weighted κ** (weighted kappa): Cohen's κ for an ordered scale such as 1-5, where a disagreement costs the square of its distance, so a one-point miss counts far less than a three-point one.
- **Wilson interval** (Wilson score interval): a confidence interval for a share that, unlike the textbook one, never runs below 0 or above 1, which matters for shares near either end.
- **Drop precision**: of the documents a curation stage dropped, the share judged bad text; high means the stage removes what it should.
- **Residual bad rate**: of the documents a curation stage kept, the share judged bad text; what the stage failed to catch.
- **Kept win rate**: of the pairs of one kept and one dropped document from the same source and stage, the share where the judge prefers the kept one in both presentation orders.
- **Order disagreement**: the share of pairs whose answer changes when the two documents swap places; it measures the judge's position bias.
- **Side-with-judge share**: of the cases where the judge and the pipeline conflict, the share where a person labelling blind agrees with the judge.
- **Twin match rate**: of a dedup stage's dropped documents, the share whose content survives in the finished corpus as another copy — byte-identical for exact dedup, sharing at least half its three-sentence windows for sentence dedup.
- **Window coverage**: the share of a document's three-sentence windows that another document also contains.
- **Retention**: the share of a source's converted documents that reach the finished corpus.
- **Type-token ratio** (TTR): distinct tokens divided by tokens; it falls as a sample grows, so it is only compared at one fixed token budget.
- **Out-of-vocabulary rate** (OOV rate): the share of alphabetic tokens a lexicon does not list; names, foreign words, code and encoding damage all count as out of vocabulary.
- **CPU hours**: processor time a step consumed, summed over its tasks; comparable between steps and machines.
- **Wall hours** (wall clock): elapsed time a step took on the machine that ran it, at the worker count it was given.

## Terms

- **q05, q50, q95** (q05, q50, q95): the 5th, 50th (median) and 95th percentiles of a distribution.
- **Curation pipeline**: the eight stages that turn extracted text into the pretraining corpus, in order convert, language, spam, quality, repetition, exact dedup, sentence dedup and statistics; the six from language to sentence dedup each keep or drop every document.
- **Content filter**: one of the four curation stages that judge a document's content — the language, spam, quality and repetition filters — as opposed to the two dedup stages, which remove copies.
- **Gopher heuristics** (Gopher-style): the rule-based quality and repetition filters published with DeepMind's Gopher model — thresholds on document length, word length, symbol ratios and repeated n-grams, tuned on English web text.
- **Raw web crawl**: a source scraped from the open web with little editing — here c4, cc100, culturax, fineweb2, hplt, macocu_sl, classla_web_sl and finepdf.
- **Curated source**: a source assembled or edited by people for a purpose — here the parliamentary, medical, legal, scientific, news, wiki and reference corpora.
- **Sloleks**: the reference lexicon of Slovene word forms, used here to check whether a token is an ordinary Slovene word.
- **LLM judge** (judge): a language model given a fixed rubric and asked to label documents, standing in for a human annotator once its agreement with one is measured.
- **Bad text** (lenient bar, strict bar): at the lenient bar, a document the judge scores coherence 2 or lower, labels garbage or boilerplate, or flags as adult or spam; the strict bar also counts coherence 3.
- **Coherence**: the rubric's 1-5 score of whether a document reads as connected, sensible text, from unreadable (1) to fluent, well-formed prose (5).
- **Boilerplate**: text furniture rather than written content — menus, footers, cookie notices, navigation, templated disclaimers.
- **Calibration set**: documents labelled by both a person and the judge, so their agreement is known before the judge's labels are used as evidence.
- **Adjudication set**: documents drawn where the judge and the pipeline flatly disagree, labelled by a person to settle which of the two is right.
- **Stratified sample**: a fixed number of documents from every source, stage and decision, so small sources are represented; its rates describe each cell, not the corpus as a whole.
- **Pairwise comparison**: the judge is shown two documents and asked which is better, or whether they tie; each pair is asked twice with the documents swapped.
- **Position bias**: a judge's tendency to prefer whichever document it is shown first (or second), regardless of content.
- **Grill** (grill round): a structured design interview that questions a plan one decision at a time; a history line's `Q15` names its fifteenth question.
- **SLING**: the Slovenian national supercomputing network, whose clusters the project can use for jobs that do not fit on one machine.
- **Gated source**: a dataset whose licence restricts redistribution, so it is counted separately from what a public release could hold.
- **KPI** (KPI 2, KPI 3, KPI 4): a key performance indicator the ARIS proposal (Z2-70067) commits the project to. KPI 2 asks for more than 10,000 examples in each of at least two domains, medicine and science; KPI 3 for more than 5B tokens per language; KPI 4 for more than 500,000 tokens per domain.
- **Access filter** (open, open+login, all): which datasets a total counts. `open` is what anyone can download, `open+login` adds what a registered user can, `all` adds gated and not-downloadable sources — what exists rather than what is reachable.
- **Provenance** (native, translated, generated): how a dataset's Slovene text came to exist. `native` is written in Slovene, `translated` is carried into Slovene from another language, `generated` is machine-written or automatically annotated.
- **Provenance class** (human-translated, machine-translated, synthetic, bilingual resource, mixed): the finer reading of a checked row. Human-translated text was translated by people; machine-translated by a translation system or a language model; synthetic text was written by a language model; a bilingual resource is a dictionary or parallel corpus whose translation direction is unknown; mixed combines several.
- **Machine translation** (MT): translating text automatically, with a dedicated system such as Google Translate or DeepL or with a language model prompted to translate.
- **Estimated tokens** (tokens_estimated): a reported word count multiplied by two, the working subword-tokens-per-word factor until the project tokenizer exists. Empty when the source reported no word count.
- **Unknown** (as a KPI verdict): the total falls short of the threshold but datasets in that domain reported no size, so the shortfall may be an absence of evidence rather than an absence of text.
- **CLARIN.SI** (CLARIN, clarin.si): the Slovenian node of CLARIN, the European research infrastructure for language resources; the main publisher of Slovene corpora, which it addresses by `11356/<n>` handles.
- **LINDAT** (LINDAT/CLARIAH-CZ): the Czech CLARIN node, which re-publishes some Slovene holdings under `11234/<n>` handles.
- **ELG** (European Language Grid): an EU platform cataloguing language resources and services; for Slovene it largely mirrors CLARIN.SI rather than publishing its own.
- **NER** (named-entity recognition): labelling spans of text with the kind of entity they name — a person, a place, a drug, a diagnosis. One of the information-extraction tasks the project targets.
- **IE** (information extraction): pulling structured facts — entities, relations, attributes — out of running text. The project's target task family.
- **PDF** (Portable Document Format): the page-layout file most Slovene journals and institutions publish articles and guidelines in. Its text has to be extracted before it can be counted, and a scanned PDF yields none.
- **Tokenizer**: the component that splits text into the subword units a language model reads; its vocabulary decides how many tokens a word becomes.
- **OAI-PMH** (Open Archives Initiative Protocol for Metadata Harvesting): the standard interface journals and repositories expose for listing every record they hold, so a whole archive can be enumerated without scraping.
- **Instruction set** (instruction sets): prompt-and-answer pairs written, often by another model, to train a chat model to follow requests; machine output rather than native prose.
- **MLflow**: the experiment-tracking server the project logs every run to, with its parameters, metrics and output files.
- **Lineage run** (lineage): an MLflow run that trains or scores nothing and only records which data and code produced a set of outputs, keeping them as attached files.
- **Download registry** (registry): the project's list of datasets it downloads and builds its corpus from, kept in `configs/data/download.yaml`.
