---
title: Benchmarks
---

# Benchmarks

Slovenian evaluation datasets used for downstream IE tasks. Benchmarks
are declared in
[`configs/data/download.yaml`](https://github.com/eriknovak/SLM4IE/blob/main/configs/data/download.yaml)
with `benchmark: true` and a `tasks:` list, so they share the download
pipeline with pretraining corpora. Use `--only-benchmarks` to fetch just
the evaluation datasets.

| Dataset                                                                       | Source    | Tasks                                     | Description                                                                                                                                                                                                               |
| ----------------------------------------------------------------------------- | --------- | ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [SUK 1.1](https://www.clarin.si/repository/xmlui/handle/11356/1959)           | CLARIN.SI | POS, LEMMA, DEP, NER, SRL, COREF, WSD, SA | ~1M tokens / 881K words / 2,913 texts manually annotated with MULTEXT-East V6, JOS, and Universal Dependencies. Integrates ssj500k 2.3, Ambiga, ElexisWSD, and SentiCoref subcorpora. License: CC BY-SA 4.0.              |
| [ssj500k 2.3](https://www.clarin.si/repository/xmlui/handle/11356/1434)       | CLARIN.SI | POS, LEMMA, DEP, NER, SRL                 | ~500K tokens manually annotated with MSD tags, lemmas, UD syntax (UD 2.8), named entities, and semantic role labels. Foundation corpus for SUK 1.1. License: CC BY-NC-SA 4.0.                                             |
| [Slovene SuperGLUE](https://www.clarin.si/repository/xmlui/handle/11356/1380) | CLARIN.SI | QA, NLI, WSD, COREF, MRC                  | Slovene translation of SuperGLUE (BoolQ, CB, COPA, MultiRC, ReCoRD, RTE, WiC, WSC). Mix of human and Google MT translation. License: CC BY 4.0. Convert to per-task evaluation files with `scripts/data/to_superglue.py`. |
| [SentiNews 1.0](https://www.clarin.si/repository/xmlui/handle/11356/1110)     | CLARIN.SI | SA                                        | Slovene news sentiment with three-level annotations (sentence, paragraph, document) and 3-class labels. Directly downloadable. License: CC BY-SA 4.0. Convert to evaluation JSONL with `scripts/data/to_sentiment.py`.    |
| [Sloleks 3.1](https://www.clarin.si/repository/xmlui/handle/11356/2080)       | CLARIN.SI | TOKENIZER                                 | Slovenian inflectional lexicon (lemmas + word forms with MULTEXT-East V6 / JOS MSDs). **Tokenizer / morphology evaluation only** — intentionally absent from `extract.yaml`, never enters the pretraining corpus. Distributed as TEI XML. License: CC BY-SA 4.0. Convert with `scripts/data/to_tokenizer_eval.py`. |

## Task abbreviations

See [Datasets](index.md#task-abbreviations).

## Conversion

Each benchmark is converted to its evaluation-ready shape via a dedicated
pipeline script:

| Benchmark           | Conversion script                                                              |
|---------------------|--------------------------------------------------------------------------------|
| SUK / ssj500k       | [`to_spans.py`](../user-guide/data-pipeline/spans.md) (NER)                    |
| Slovene SuperGLUE   | [`to_superglue.py`](../user-guide/data-pipeline/superglue.md)                  |
| SentiNews           | [`to_sentiment.py`](../user-guide/data-pipeline/sentiment.md)                  |
| Sloleks             | [`to_tokenizer_eval.py`](../user-guide/data-pipeline/tokenizer-eval.md)        |

Once converted, run [Evaluation](../user-guide/evaluation.md) to score
trained models.
