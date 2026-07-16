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
- **[Corpus Statistics](corpus-statistics.md)** — size and composition of
  the curated pretraining corpus actually produced by the pipeline.
- **[Extraction Formats](extraction-formats.md)** — the raw input formats
  each dataset ships in, and how the extractors map them onto the unified
  schema.

Both groups are declared in
[`configs/data/download.yaml`](https://github.com/eriknovak/SLM4IE/blob/main/configs/data/download.yaml).
Each entry carries a `role` field: `pretrain` (the default) for
pretraining corpora, `benchmark` for evaluation datasets, and `lexicon`
for tokenizer/morphology lexicons. Non-pretrain entries also carry a
`tasks:` list, and all roles share the same download pipeline.

Use `--only-benchmarks` on the [download script](../user-guide/data-pipeline/download.md)
to fetch just the non-pretraining datasets (`role: benchmark` or
`role: lexicon`).

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
