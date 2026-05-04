# SLM4IE

**Small Language Models for Zero-Shot Information Extraction in European Languages**

SLM4IE develops small language models (SLMs) for zero-shot information extraction across European languages, with emphasis on Slovenian. The project targets three limitations of current LLMs:

- **Compute cost:** LLMs require infrastructure beyond reach of smaller organizations for local deployment
- **Low-resource gaps:** Limited training data for sensitive domains and underrepresented languages
- **Output inconsistency:** Unreliable structured extraction from generative models

We build computationally efficient models optimized for commodity hardware, create multilingual benchmark datasets for sensitive domains, and evaluate against existing SLMs and LLMs. All artifacts (models, datasets, code) will be released publicly where possible.

## Pretraining Corpora

Slovenian text corpora used for language model pretraining (configured in [`configs/data/download.yaml`](configs/data/download.yaml)).

### CLARIN.SI sources

| Dataset | Domain | Description |
|---|---|---|
| [CLASSLA-web.sl 2.0](https://www.clarin.si/repository/xmlui/handle/11356/2079) | web | Annotated Slovenian web corpus from the CLASSLA project. |
| [CLASSLAWiki-sl](https://www.clarin.si/repository/xmlui/handle/11356/1427) | wiki | Slovenian Wikipedia with linguistic annotations (CoNLL-U). |
| [MaCoCu-sl 2.0](https://www.clarin.si/repository/xmlui/handle/11356/1795) | web | Slovenian web corpus from the MaCoCu project (XML/TEI). |
| [ParlaMint-SI 5.0](https://www.clarin.si/repository/xmlui/handle/11356/2004) | parliamentary | Slovenian parliamentary minutes, annotated TEI. |
| [COLESLAW 1.0](https://www.clarin.si/repository/xmlui/handle/11356/2095) | legal | Corpus of Slovenian legal texts. |
| [PoVeJMo-VeMo-Med 1.0](https://www.clarin.si/repository/xmlui/handle/11356/1983) | medical | Slovenian medical texts from the PoVeJMo project. |
| [OSS 1.0](https://www.clarin.si/repository/xmlui/handle/11356/1774) | scientific | 2.59B words / 3.26B tokens from 151K scientific texts (monographs, articles, theses) from Slovenian universities (2000–2022). |
| [siParl 4.0](https://www.clarin.si/repository/xmlui/handle/11356/1936) | parliamentary | 239M words from parliamentary minutes (1990–2022), TEI XML. May overlap with ParlaMint-SI. |
| [Janes-News 1.0](https://www.clarin.si/repository/xmlui/handle/11356/1140) | news | 14.8M tokens from news article comments (2007–2015). Informal register. |
| [KZB 1.0](https://www.clarin.si/repository/xmlui/handle/11356/1872) | scientific | 25M words / 33.6M tokens of curated scientific monographs and papers (2000–2023). |

### HuggingFace sources

| Dataset | Domain | Description |
|---|---|---|
| [FinePDF](https://huggingface.co/datasets/HuggingFaceFW/finepdfs) | web | Slovenian (`slv_Latn`) PDF-derived text. |
| [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | web | Slovenian (`slv_Latn`) high-quality web corpus. |
| [mC4](https://huggingface.co/datasets/allenai/c4) | web | Cleaned multilingual Common Crawl, ~5 GB+ for Slovenian. |
| [CC100](https://huggingface.co/datasets/statmt/cc100) | web | Monolingual CommonCrawl filtered with fastText (Facebook AI, XLM-R), ~1.4 GB compressed for Slovenian. |

### Disabled by default

Optional sources requiring extra access (gated datasets, manual login, copyright restrictions): `KAS 2.0` (CLARIN academic login), `Janes-Forum/Blog`, `Solar 3.0`, `CulturaX` (HF gated), `Legal-mC4`, `HPLT`. Not bulk-downloadable: `Gigafida 2.x`, `Metafida 1.0`, `Trendi`.

## Benchmarks

Slovenian evaluation datasets used for downstream IE tasks (configured in [`configs/data/benchmarks.yaml`](configs/data/benchmarks.yaml)).

| Dataset | Source | Tasks | Description |
|---|---|---|---|
| [SUK 1.1](https://www.clarin.si/repository/xmlui/handle/11356/1959) | CLARIN.SI | POS, LEMMA, DEP, NER, SRL, COREF, WSD, SA | ~1M tokens / 881K words / 2,913 texts manually annotated with MULTEXT-East V6, JOS, and Universal Dependencies. Integrates ssj500k 2.3, Ambiga, ElexisWSD, and SentiCoref subcorpora. License: CC BY-SA 4.0. |
| [ssj500k 2.3](https://www.clarin.si/repository/xmlui/handle/11356/1434) | CLARIN.SI | POS, LEMMA, DEP, NER, SRL | ~500K tokens manually annotated with MSD tags, lemmas, UD syntax (UD 2.8), named entities, and semantic role labels. Foundation corpus for SUK 1.1. License: CC BY-NC-SA 4.0. |
| [Slovene SuperGLUE](https://www.clarin.si/repository/xmlui/handle/11356/1380) | CLARIN.SI | QA, NLI, WSD, COREF, MRC | Slovene translation of SuperGLUE (BoolQ, CB, COPA, MultiRC, ReCoRD, RTE, WiC, WSC). Mix of human and Google MT translation. License: CC BY 4.0. |
| [Twitter Sentiment 15 EU](https://www.clarin.si/repository/xmlui/handle/11356/1054) | CLARIN.SI | SA | Slovenian slice of a 1.6M-tweet 3-class sentiment corpus across 15 European languages. Tweet text requires re-hydration via the X API. License: CC BY-SA 4.0. |

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

## Acknowledgments

The project is funded by ARIS (Slovenian Research and Innovation Agency) under the project number [Z2-70067](https://cris.cobiss.net/ecris/si/sl/project/24346).

<figure>
  <img src="https://github.com/eriknovak/SLM4IE/blob/main/docs/funding/logo.jpg?raw=true" alt="ARIS Logo" width="420" />
</figure>