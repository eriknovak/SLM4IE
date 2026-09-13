# Dataset catalog

Every dataset the project draws on, pretraining corpora first and evaluation
benchmarks after. All of them are declared in
[`configs/data/download.yaml`](../configs/data/download.yaml) and fetched by
`prepare_datasets.py download` — see [data-pipeline.md](data-pipeline.md).

## Pretraining corpora

Slovenian text corpora used for language model pretraining.

### CLARIN.SI sources

| Dataset                                                                          | Domain        | Description                                                                                                                   |
| -------------------------------------------------------------------------------- | ------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| [CLASSLA-web.sl 2.0](https://www.clarin.si/repository/xmlui/handle/11356/2079)   | web           | Annotated Slovenian web corpus from the CLASSLA project.                                                                      |
| [CLASSLAWiki-sl](https://www.clarin.si/repository/xmlui/handle/11356/1427)       | wiki          | Slovenian Wikipedia with linguistic annotations (CoNLL-U).                                                                    |
| [MaCoCu-sl 2.0](https://www.clarin.si/repository/xmlui/handle/11356/1795)        | web           | Slovenian web corpus from the MaCoCu project (XML/TEI).                                                                       |
| [ParlaMint-SI 5.0](https://www.clarin.si/repository/xmlui/handle/11356/2004)     | parliamentary | Slovenian parliamentary minutes, annotated TEI.                                                                               |
| [COLESLAW 1.0](https://www.clarin.si/repository/xmlui/handle/11356/2095)         | legal         | Corpus of Slovenian legal texts.                                                                                              |
| [PoVeJMo-VeMo-Med 1.0](https://www.clarin.si/repository/xmlui/handle/11356/1983) | medical       | Slovenian medical texts from the PoVeJMo project.                                                                             |
| [OSS 1.0](https://www.clarin.si/repository/xmlui/handle/11356/1774)              | scientific    | 2.59B words / 3.26B tokens from 151K scientific texts (monographs, articles, theses) from Slovenian universities (2000–2022). |
| [siParl 4.0](https://www.clarin.si/repository/xmlui/handle/11356/1936)           | parliamentary | 239M words from parliamentary minutes (1990–2022), TEI XML. May overlap with ParlaMint-SI.                                    |
| [Janes-News 1.0](https://www.clarin.si/repository/xmlui/handle/11356/1140)       | news          | 14.8M tokens from news article comments (2007–2015). Informal register.                                                       |
| [KZB 1.0](https://www.clarin.si/repository/xmlui/handle/11356/1872)              | scientific    | 25M words / 33.6M tokens of curated scientific monographs and papers (2000–2023).                                             |

### HuggingFace sources

| Dataset                                                                  | Domain | Description                                                                                                       |
| ------------------------------------------------------------------------ | ------ | ----------------------------------------------------------------------------------------------------------------- |
| [FinePDF](https://huggingface.co/datasets/HuggingFaceFW/finepdfs)        | web    | Slovenian (`slv_Latn`) PDF-derived text.                                                                          |
| [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)     | web    | Slovenian (`slv_Latn`) high-quality web corpus.                                                                   |
| [mC4](https://huggingface.co/datasets/allenai/c4)                        | web    | Cleaned multilingual Common Crawl, ~5 GB+ for Slovenian.                                                          |
| [HPLT 2.0 Cleaned](https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned) | web    | HPLT project web crawl (CommonCrawl + Internet Archive), cleaned tier; Slovenian config `slv_Latn` (~10.3M rows). |

### Direct HTTP sources

| Dataset                                                            | Domain | Description                                                                                                                                                                                                              |
| ------------------------------------------------------------------ | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [CC100](https://data.statmt.org/cc-100/)                           | web    | Monolingual CommonCrawl filtered with fastText (Facebook AI, XLM-R), ~1.4 GB compressed for Slovenian. Fetched directly from `statmt.org`; the HuggingFace mirror is script-based and no longer supported by `datasets`. |
| [Legal-mC4](https://huggingface.co/datasets/joelniklaus/legal-mc4) | legal  | Legal-domain text filtered from mC4, ~32.5K documents / ~107M words for Slovenian. Fetched directly from the HuggingFace LFS endpoint; the repo's loading script is no longer supported by `datasets`.                   |

### Disabled by default

Optional sources requiring extra access (gated datasets, manual login,
copyright restrictions): `KAS 2.0` (CLARIN academic login), `Janes-Forum/Blog`,
`Solar 3.0`, `CulturaX` (HF gated). Not bulk-downloadable: `Gigafida 2.x`,
`Metafida 1.0`, `Trendi`.

## Benchmarks

Slovenian evaluation datasets used for downstream IE tasks. Benchmarks are
declared with `role: benchmark` (tokenizer lexicons such as Sloleks use
`role: lexicon`) and a `tasks:` list, so they share the download pipeline with
pretraining corpora. Use `--only-benchmarks` to fetch just the non-pretraining
datasets.

| Dataset                                                                       | Source    | Tasks                                     | Description                                                                                                                                                                                                                                                                                                      |
| ----------------------------------------------------------------------------- | --------- | ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [SUK 1.1](https://www.clarin.si/repository/xmlui/handle/11356/1959)           | CLARIN.SI | POS, LEMMA, DEP, NER, SRL, COREF, WSD, SA | ~1M tokens / 881K words / 2,913 texts manually annotated with MULTEXT-East V6, JOS, and Universal Dependencies. Integrates ssj500k 2.3, Ambiga, ElexisWSD, and SentiCoref subcorpora. License: CC BY-SA 4.0.                                                                                                     |
| [ssj500k 2.3](https://www.clarin.si/repository/xmlui/handle/11356/1434)       | CLARIN.SI | POS, LEMMA, DEP, NER, SRL                 | ~500K tokens manually annotated with MSD tags, lemmas, UD syntax (UD 2.8), named entities, and semantic role labels. Foundation corpus for SUK 1.1. License: CC BY-NC-SA 4.0.                                                                                                                                    |
| [Slovene SuperGLUE](https://www.clarin.si/repository/xmlui/handle/11356/1380) | CLARIN.SI | QA, NLI, WSD, COREF, MRC                  | Slovene translation of SuperGLUE (BoolQ, CB, COPA, MultiRC, ReCoRD, RTE, WiC, WSC). Mix of human and Google MT translation. License: CC BY 4.0. Convert to per-task evaluation files with `prepare_datasets.py tasks`.                                                                                           |
| [SentiNews 1.0](https://www.clarin.si/repository/xmlui/handle/11356/1110)     | CLARIN.SI | SA                                        | Slovene news sentiment with three-level annotations (sentence, paragraph, document) and 3-class labels. Directly downloadable. License: CC BY-SA 4.0. Convert to evaluation JSONL with `prepare_datasets.py tasks`.                                                                                              |
| [Sloleks 3.1](https://www.clarin.si/repository/xmlui/handle/11356/2080)       | CLARIN.SI | TOKENIZER                                 | Slovenian inflectional lexicon (lemmas + word forms with MULTEXT-East V6 / JOS MSDs). **Tokenizer / morphology evaluation only** — intentionally absent from `extract.yaml`, never enters the pretraining corpus. Distributed as TEI XML. License: CC BY-SA 4.0. Convert with `prepare_datasets.py tokenization`. |

### Task abbreviations

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
- **TOKENIZER** — tokenizer / morphology evaluation (lexicon-based, not a downstream IE task)
