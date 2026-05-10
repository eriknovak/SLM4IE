---
title: Datasets
---

# Datasets

Catalog of Slovenian-language datasets used by SLM4IE, split into two
groups:

- **[Pretraining Corpora](pretraining-corpora.md)** — large unlabeled
  text corpora used for language-model pretraining.
- **[Benchmarks](benchmarks.md)** — annotated evaluation datasets used
  for downstream IE tasks.

Both groups are declared in
[`configs/data/download.yaml`](https://github.com/eriknovak/SLM4IE/blob/main/configs/data/download.yaml).
Benchmarks are flagged with `benchmark: true` and a `tasks:` list, so
they share the same download pipeline as pretraining corpora.

Use `--only-benchmarks` on the [download script](../user-guide/data-pipeline/download.md)
to fetch just the evaluation datasets.

## Task abbreviations

The benchmark catalog uses these task tags:

- **POS** — part-of-speech tagging
- **LEMMA** — lemmatization
- **DEP** — dependency parsing
- **NER** — named entity recognition
- **SRL** — semantic role labeling
- **COREF** — coreference resolution
- **WSD** — word sense disambiguation
- **SA** — sentiment analysis
- **NLI** — natural language inference
- **QA** — question answering
- **MRC** — machine reading comprehension
- **TOKENIZER** — tokenizer / morphology evaluation (lexicon-based, not
  a downstream IE task)
