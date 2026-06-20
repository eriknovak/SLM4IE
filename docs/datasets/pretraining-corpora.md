---
title: Pretraining Corpora
---

# Pretraining Corpora

Slovenian text corpora used for language model pretraining, configured
in
[`configs/data/download.yaml`](https://github.com/eriknovak/SLM4IE/blob/main/configs/data/download.yaml).

## CLARIN.SI sources

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

## HuggingFace sources

| Dataset                                                                  | Domain | Description                                                                                                       |
| ------------------------------------------------------------------------ | ------ | ----------------------------------------------------------------------------------------------------------------- |
| [FinePDF](https://huggingface.co/datasets/HuggingFaceFW/finepdfs)        | web    | Slovenian (`slv_Latn`) PDF-derived text.                                                                          |
| [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)     | web    | Slovenian (`slv_Latn`) high-quality web corpus.                                                                   |
| [mC4](https://huggingface.co/datasets/allenai/c4)                        | web    | Cleaned multilingual Common Crawl, ~5 GB+ for Slovenian.                                                          |
| [HPLT 2.0 Cleaned](https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned) | web    | HPLT project web crawl (CommonCrawl + Internet Archive), cleaned tier; Slovenian config `slv_Latn` (~10.3M rows). |

## Direct HTTP sources

| Dataset                                                            | Domain | Description                                                                                                                                                                                                              |
| ------------------------------------------------------------------ | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [CC100](https://data.statmt.org/cc-100/)                           | web    | Monolingual CommonCrawl filtered with fastText (Facebook AI, XLM-R), ~1.4 GB compressed for Slovenian. Fetched directly from `statmt.org`; the HuggingFace mirror is script-based and no longer supported by `datasets`. |
| [Legal-mC4](https://huggingface.co/datasets/joelniklaus/legal-mc4) | legal  | Legal-domain text filtered from mC4, ~32.5K documents / ~107M words for Slovenian. Fetched directly from the HuggingFace LFS endpoint; the repo's loading script is no longer supported by `datasets`.                   |

## Disabled by default

Optional sources requiring extra access (gated datasets, manual login,
copyright restrictions): `KAS 2.0` (CLARIN academic login),
`Janes-Forum/Blog`, `Solar 3.0`, `CulturaX` (HF gated). Not bulk-downloadable:
`Gigafida 2.x`, `Metafida 1.0`, `Trendi`.

## Next step

Once downloaded and extracted, the corpora are curated into the final
pretraining corpus via
[`to_pretrain.py`](../user-guide/data-pipeline/pretrain.md). See
[Corpus Statistics](corpus-statistics.md) for the size and composition of the
corpus that route produced.
