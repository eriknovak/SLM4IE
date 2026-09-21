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

## Terms

- **q05, q50, q95** (q05, q50, q95): the 5th, 50th (median) and 95th percentiles of a distribution.
- **KPI** (KPI 2, KPI 3, KPI 4): a key performance indicator the ARIS proposal (Z2-70067) commits the project to. KPI 2 asks for more than 10,000 examples in each of at least two domains, medicine and science; KPI 3 for more than 5B tokens per language; KPI 4 for more than 500,000 tokens per domain.
- **Access filter** (open, open+login, all): which datasets a total counts. `open` is what anyone can download, `open+login` adds what a registered user can, `all` adds gated and not-downloadable sources — what exists rather than what is reachable.
- **Provenance** (native, translated, generated): how a dataset's Slovene text came to exist. `native` is written in Slovene, `translated` is carried into Slovene from another language, `generated` is machine-written or automatically annotated.
- **Estimated tokens** (tokens_estimated): a reported word count multiplied by two, the working subword-tokens-per-word factor until the project tokenizer exists. Empty when the source reported no word count.
- **Unknown** (as a KPI verdict): the total falls short of the threshold but datasets in that domain reported no size, so the shortfall may be an absence of evidence rather than an absence of text.
- **CLARIN.SI** (CLARIN, clarin.si): the Slovenian node of CLARIN, the European research infrastructure for language resources; the main publisher of Slovene corpora, which it addresses by `11356/<n>` handles.
- **LINDAT** (LINDAT/CLARIAH-CZ): the Czech CLARIN node, which re-publishes some Slovene holdings under `11234/<n>` handles.
- **ELG** (European Language Grid): an EU platform cataloguing language resources and services; for Slovene it largely mirrors CLARIN.SI rather than publishing its own.
- **NER** (named-entity recognition): labelling spans of text with the kind of entity they name — a person, a place, a drug, a diagnosis. One of the information-extraction tasks the project targets.
- **IE** (information extraction): pulling structured facts — entities, relations, attributes — out of running text. The project's target task family.
- **PDF** (Portable Document Format): the page-layout file most Slovene journals and institutions publish articles and guidelines in. Its text has to be extracted before it can be counted, and a scanned PDF yields none.
