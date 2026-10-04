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

Off for `--all` (download and curation alike): `Gigafida 2.2` (presigned URLs
on request; enable it in the gitignored `download.local.yaml`), `Metafida 1.0`
and `Trendi` (not bulk-downloadable). `KAS 2.0` (CLARIN academic login) and the
living `slovenian_news` crawl are `manual: true` but enabled: download only
checks their folders, and curation selects them.

The Janes corpora (`janes_forum`, `janes_blog`, `janes_news`) are disabled for
their informal, non-standard register: they are user-generated forum posts,
blogs and comments, and the pretraining selection keeps to edited, standard
prose for now. Training or evaluating on non-standard text may be revisited
later as an experiment of its own.

## Containment map

Which registry entries hold the same documents. The audit covered all 31
entries of `configs/data/download.yaml` (the issue that asked for it counted
33; the registry holds 31) against each other. Each verdict rests on the
publisher's own documentation where it settles the question, and on a
measurement over the extracted tier where it does not ([method](#measured-overlap)).
A pair not listed below is `disjoint`.

The verdicts live in the registry as `contains` and `overlaps` (see
[data-pipeline.md](data-pipeline.md#download)). A measured pair is `contains`
when at least 99% of the smaller side's sampled sentences recur in the larger,
and `overlaps` when either side shares at least 10%; below that it stays
disjoint, its figure kept here.

| Group                | Representative     | Members                                                                             | Relation                                       | Evidence                                                                                                                                                                                            |
| -------------------- | ------------------ | ----------------------------------------------------------------------------------- | ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| SUK                  | `suk` (benchmark)  | `ssj500k`                                                                           | contains                                       | [SUK 1.1](https://www.clarin.si/repository/xmlui/handle/11356/1959)                                                                                                                                 |
| SUK and SentiNews    | —                  | `suk`, `sentinews`                                                                  | overlaps: SentiCoref's 837 documents           | [SentiCoref 1.0](https://www.clarin.si/repository/xmlui/handle/11356/1285)                                                                                                                          |
| Parliament           | `siparl`           | `parlamint_si`                                                                      | contains; measured 100%                        | [siParl 4.0](https://www.clarin.si/repository/xmlui/handle/11356/1936), [ParlaMint paper §2.1.14](https://epubl.ktu.edu/object/elaba:119556500/119556500.pdf)                                       |
| mC4                  | `c4`               | `legal_mc4`                                                                         | contains; measured 100% of URLs and text       | [Legal-mC4 card](https://huggingface.co/datasets/joelniklaus/legal-mc4), [filter script](https://raw.githubusercontent.com/JoelNiklaus/LegalDatasets/main/pretrain/mc4_legal/filter_mc4.py)         |
| Web crawls           | none, all selected | `fineweb2`, `culturax`, `c4`, `hplt`, `cc100`, `classla_web_sl`, `macocu_sl`        | overlaps; measured 10–76%                      | [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2), [CulturaX](https://huggingface.co/datasets/uonlp/CulturaX), [HPLT 2.0](https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned), [CLASSLA-web 2.0 paper](https://arxiv.org/html/2601.11170) |
| Wikipedia            | none               | `classlawiki_sl` with `c4`, `cc100`, `culturax`, `fineweb2`, `hplt`, `macocu_sl`    | overlaps; measured 12–63%                      | [FineWeb-2 card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)                                                                                                                           |
| News                 | none               | `slovenian_news` with `gigafida` and the web crawls                                 | overlaps; measured 14–29%                      | measured only                                                                                                                                                                                       |
| Academic             | none               | `kas` with `oss`; `kas`, `kzb` with `finepdf`                                       | overlaps; measured 37–39%, 11–13%              | [OSS 1.0](https://www.clarin.si/repository/xmlui/handle/11356/1774), [KAS 2.0](https://www.clarin.si/repository/xmlui/handle/11356/1448)                                                            |
| Legal                | none               | `coleslaw` with `classla_web_sl`, `macocu_sl`, `c4`, `finepdf`, `legal_mc4`         | overlaps; measured 12–36%                      | measured only                                                                                                                                                                                       |
| Gigafida and Trendi  | —                  | `gigafida`, `trendi`                                                                | disjoint by date (to 2018; from 2019)          | [Gigafida 2.0 paper](https://aclanthology.org/2020.lrec-1.409.pdf), [Trendi](https://www.clarin.si/repository/xmlui/handle/11356/1681)                                                              |
| metaFida             | `metafida` (off)   | `janes_*`, `solar`, `oss`, `classlawiki_sl`; `siparl`, `gigafida` in other versions | contains; overlaps                             | [metaFida 1.0](https://www.clarin.si/repository/xmlui/handle/11356/1775)                                                                                                                            |
| Sloleks              | —                  | `sloleks`, `sloleks_relations`                                                      | overlaps: both from Sloleks 2.0 entries        | [Sloleks 3.1](https://www.clarin.si/repository/xmlui/handle/11356/2080), [word relations](https://www.clarin.si/repository/xmlui/handle/11356/1986)                                                 |

**SUK.** SUK 1.1 is built from both parts of ssj500k 2.3 (500,247 of its
500,295 words) plus Ambiga, ElexisWSD and SentiCoref. Both are benchmarks, so
the pretraining selection skips them by role; on the task route `ner/suk`
excludes the ssj500k documents (below). SentiCoref's documents were drawn from
SentiNews, so `suk` also overlaps `sentinews`; no task pairs the two, so the
isolation check has nothing to enforce there.

**Parliament.** ParlaMint-SI was built from siParl, and its 2000–2022 minutes
sit inside siParl 4.0's 1990–2022. Every sampled ParlaMint-SI sentence recurs in
siParl (31% of siParl's recur in ParlaMint-SI), so `siparl` is the
representative and `--all` reports `parlamint_si` as `skipped: contained in
siparl`. Name `parlamint_si` positionally to curate it alone.

**mC4.** Legal-mC4's filter script reads the `train` split of `mc4`, now an
alias of `allenai/c4`; every one of its URLs and sampled sentences recurs in
`c4`, its validation file included. `c4` is the representative and
`legal_mc4` is skipped.

**Web crawls.** The Common Crawl family (FineWeb-2: 96 snapshots 2013–2024;
mC4: 86 dumps; CulturaX: mC4 plus four OSCAR releases; CC100: 2018; HPLT 2.0:
mostly Internet Archive plus Common Crawl) and the two .si crawls are different
filterings of overlapping crawls, so none contains another; choosing between
them is a curation decision about filter quality, not containment. All stay
selected and every pair is declared `overlaps`, which prints a warning per
pair and leaves the shared text to exact and sentence dedup. CLASSLA-web.sl 2.0
is a fresh 2024 crawl, not the MaCoCu-sl 2.0 crawl its 1.0 release reused.

**Wikipedia, news, academic and legal.** Each pair shares text without either
side holding the other: web crawls carry Wikipedia pages and news articles,
FinePDFs carries theses and legal PDFs, and KAS (theses to 2018) and OSS (the
same portal, to 2022) share about 38% each way, short of containment because
their extractions differ. All stay selected under an `overlaps` warning.
`gigafida`, enabled only through the local overlay, overlaps `slovenian_news`
(14% each way; the news crawl starts in 2011) and shares under 10% with every
web crawl.

**Gigafida and Trendi.** Trendi continues Gigafida from January 2019, so they
are disjoint; neither is downloadable without a request.

**metaFida.** metaFida 1.0 bundles 34 corpora, paragraph-deduplicated,
including the same releases of the Janes corpora, Šolar, OSS and CLASSLAWiki,
and older siParl (3.0) and Gigafida (2.0) releases. It is not downloadable, so
it stays disabled and its members remain their own representatives; declaring
the relation makes obtaining it a one-line change.

**Unconfirmed, left disjoint.** Gigafida against SentiNews (same portals,
2007–2016) and against ssj500k (sampled from FidaPLUS, Gigafida's predecessor):
neither publisher says the texts were kept, and the benchmarks were not
measured. KZB against OSS measured 1%, Šolar under 1% against everything, and
PoVeJMo-VeMo-Med under 10%; SuperGLUE-SL is translated from English.

**User-generated text.** Only the Janes corpora are documented as
user-generated, and they are disabled for their register. The web crawls
carry an unmeasured share of forums and comments (CLASSLA-web labels a Forum
genre that could filter it), `slovenian_news` lists a few community sites, and
Šolar is non-standard learner writing; none is disabled here.

### Every declared pair

One row per relation the registry declares, `a` before `b` as written there;
measured figures read "`a` in `b` / `b` in `a`". Every pair not listed is
`disjoint`, including those the [unconfirmed](#containment-map) paragraph names.

| `a` | Relation | `b` | Evidence |
| --- | --- | --- | --- |
| `classla_web_sl` | overlaps | `c4` | measured 23% / 14% (URLs 9% / 5%) |
| `classla_web_sl` | overlaps | `cc100` | measured 10% / 17% |
| `classla_web_sl` | overlaps | `coleslaw` | measured 5% / 28% (URLs 0% / 0%) |
| `classla_web_sl` | overlaps | `culturax` | measured 28% / 15% (URLs 12% / 9%) |
| `classla_web_sl` | overlaps | `fineweb2` | measured 30% / 13% (URLs 19% / 8%) |
| `classla_web_sl` | overlaps | `hplt` | measured 22% / 18% (URLs 12% / 7%) |
| `classla_web_sl` | overlaps | `legal_mc4` | measured 1% / 29% (URLs 0% / 4%) |
| `classla_web_sl` | overlaps | `macocu_sl` | [CLASSLA-web 2.0 paper](https://arxiv.org/html/2601.11170); measured 30% / 37% (URLs 18% / 13%) |
| `classla_web_sl` | overlaps | `slovenian_news` | measured 8% / 20% (URLs 7% / 7%) |
| `classlawiki_sl` | overlaps | `c4` | measured 63% / 1% |
| `classlawiki_sl` | overlaps | `cc100` | measured 41% / 1% |
| `classlawiki_sl` | overlaps | `culturax` | measured 54% / 1% |
| `classlawiki_sl` | overlaps | `fineweb2` | [FineWeb-2 card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2); measured 41% / 0% |
| `classlawiki_sl` | overlaps | `hplt` | measured 58% / 2% |
| `classlawiki_sl` | overlaps | `macocu_sl` | measured 12% / 0% |
| `macocu_sl` | overlaps | `c4` | measured 43% / 21% (URLs 16% / 12%) |
| `macocu_sl` | overlaps | `cc100` | measured 20% / 28% |
| `macocu_sl` | overlaps | `coleslaw` | measured 4% / 18% (URLs 0% / 0%) |
| `macocu_sl` | overlaps | `culturax` | measured 49% / 22% (URLs 18% / 17%) |
| `macocu_sl` | overlaps | `fineweb2` | measured 41% / 16% (URLs 21% / 12%) |
| `macocu_sl` | overlaps | `hplt` | measured 38% / 27% (URLs 17% / 14%) |
| `macocu_sl` | overlaps | `legal_mc4` | measured 1% / 32% (URLs 0% / 12%) |
| `kas` | overlaps | `finepdf` | measured 11% / 5% |
| `kas` | overlaps | `oss` | [OSS 1.0](https://www.clarin.si/repository/xmlui/handle/11356/1774); measured 39% / 37% |
| `coleslaw` | overlaps | `c4` | measured 12% / 1% (URLs 9% / 0%) |
| `coleslaw` | overlaps | `finepdf` | measured 13% / 2% (URLs 0% / 0%) |
| `coleslaw` | overlaps | `legal_mc4` | measured 4% / 36% (URLs 6% / 5%) |
| `siparl` | contains | `parlamint_si` | [ParlaMint paper §2.1.14](https://epubl.ktu.edu/object/elaba:119556500/119556500.pdf); measured 31% / 100% |
| `kzb` | overlaps | `finepdf` | measured 13% / 0% |
| `finepdf` | overlaps | `legal_mc4` | measured 0% / 16% (URLs 0% / 0%) |
| `fineweb2` | overlaps | `c4` | measured 41% / 49% (URLs 27% / 35%) |
| `fineweb2` | overlaps | `cc100` | measured 14% / 49% |
| `fineweb2` | overlaps | `culturax` | measured 62% / 69% (URLs 35% / 56%) |
| `fineweb2` | overlaps | `hplt` | measured 41% / 63% (URLs 30% / 43%) |
| `fineweb2` | overlaps | `legal_mc4` | measured 0% / 32% (URLs 0% / 25%) |
| `fineweb2` | overlaps | `slovenian_news` | measured 5% / 29% (URLs 5% / 11%) |
| `culturax` | overlaps | `c4` | [CulturaX card](https://huggingface.co/datasets/uonlp/CulturaX); measured 58% / 64% (URLs 50% / 40%) |
| `culturax` | overlaps | `cc100` | measured 20% / 58% |
| `culturax` | overlaps | `hplt` | measured 40% / 59% (URLs 34% / 30%) |
| `culturax` | overlaps | `legal_mc4` | [CulturaX card](https://huggingface.co/datasets/uonlp/CulturaX); measured 1% / 64% (URLs 0% / 57%) |
| `culturax` | overlaps | `slovenian_news` | measured 5% / 23% (URLs 5% / 8%) |
| `legal_mc4` | overlaps | `cc100` | measured 24% / 1% |
| `legal_mc4` | overlaps | `hplt` | measured 32% / 1% (URLs 22% / 0%) |
| `c4` | contains | `legal_mc4` | [filter script](https://raw.githubusercontent.com/JoelNiklaus/LegalDatasets/main/pretrain/mc4_legal/filter_mc4.py); measured 1% / 100% (URLs 0% / 100%) |
| `c4` | overlaps | `cc100` | measured 27% / 76% |
| `c4` | overlaps | `hplt` | measured 33% / 50% (URLs 22% / 25%) |
| `c4` | overlaps | `slovenian_news` | measured 4% / 14% (URLs 2% / 5%) |
| `hplt` | overlaps | `cc100` | measured 32% / 52% |
| `hplt` | overlaps | `slovenian_news` | measured 4% / 14% (URLs 2% / 4%) |
| `gigafida` | overlaps | `slovenian_news` | measured 14% / 14% |
| `metafida` | contains | `janes_blog` | [metaFida 1.0](https://www.clarin.si/repository/xmlui/handle/11356/1775) |
| `metafida` | contains | `janes_forum` | [metaFida 1.0](https://www.clarin.si/repository/xmlui/handle/11356/1775) |
| `metafida` | contains | `janes_news` | [metaFida 1.0](https://www.clarin.si/repository/xmlui/handle/11356/1775) |
| `metafida` | contains | `solar` | [metaFida 1.0](https://www.clarin.si/repository/xmlui/handle/11356/1775) |
| `metafida` | contains | `oss` | [metaFida 1.0](https://www.clarin.si/repository/xmlui/handle/11356/1775) |
| `metafida` | contains | `classlawiki_sl` | [metaFida 1.0](https://www.clarin.si/repository/xmlui/handle/11356/1775) |
| `metafida` | overlaps | `siparl` | [metaFida 1.0](https://www.clarin.si/repository/xmlui/handle/11356/1775) |
| `metafida` | overlaps | `gigafida` | [metaFida 1.0](https://www.clarin.si/repository/xmlui/handle/11356/1775) |
| `suk` | contains | `ssj500k` | [SUK 1.1](https://www.clarin.si/repository/xmlui/handle/11356/1959) |
| `suk` | overlaps | `sentinews` | [SentiCoref 1.0](https://www.clarin.si/repository/xmlui/handle/11356/1285) |
| `sloleks` | overlaps | `sloleks_relations` | [word relations](https://www.clarin.si/repository/xmlui/handle/11356/1986) |

### Measured overlap

Measured on 2026-10-05 with `curate_pretraining_corpus.py overlap` (see
[pretraining-corpus.md](pretraining-corpus.md#measuring-overlap-between-extracted-corpora))
over the 20 extracted pretraining corpora: seed 0, a 1,000,000-document sample
of each measured side (all documents of smaller corpora), and a 1-in-16 hash
slice of sentence units on both sides. A cell is the percentage of the row
corpus's sampled sentences found in the column corpus, with the share of its
distinct URLs in brackets where both record one. Repeated boilerplate counts as
shared, so web cells read somewhat high.

| `a` \ found in `b` | `fineweb2` | `culturax` | `c4` | `hplt` | `cc100` | `classla_web_sl` | `macocu_sl` | `slovenian_news` | `classlawiki_sl` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fineweb2` | — | 62 (35) | 41 (27) | 41 (30) | 14 | 13 (8) | 16 (12) | 5 (5) | 0 |
| `culturax` | 69 (56) | — | 58 (50) | 40 (34) | 20 | 15 (9) | 22 (17) | 5 (5) | 1 |
| `c4` | 49 (35) | 64 (40) | — | 33 (22) | 27 | 14 (5) | 21 (12) | 4 (2) | 1 |
| `hplt` | 63 (43) | 59 (30) | 50 (25) | — | 32 | 18 (7) | 27 (14) | 4 (2) | 2 |
| `cc100` | 49 | 58 | 76 | 52 | — | 17 | 28 | 3 | 1 |
| `classla_web_sl` | 30 (19) | 28 (12) | 23 (9) | 22 (12) | 10 | — | 30 (18) | 8 (7) | 0 |
| `macocu_sl` | 41 (21) | 49 (18) | 43 (16) | 38 (17) | 20 | 37 (13) | — | 5 (2) | 0 |
| `slovenian_news` | 29 (11) | 23 (8) | 14 (5) | 14 (4) | 4 | 20 (7) | 10 (3) | — | 0 |
| `classlawiki_sl` | 41 | 54 | 63 | 58 | 41 | 5 | 12 | 1 | — |

Other pairs, row in column (sentences, URLs): `parlamint_si` in `siparl` 100%;
`legal_mc4` in `c4` 100% (100%), in `culturax` 64% (57%), in `coleslaw` 36%
(5%); `kas` in `oss` 39%, `oss` in `kas` 37%; `coleslaw` in `classla_web_sl`
28%, in `macocu_sl` 18%, in `finepdf` 13%, in `c4` 12% (9%); `kzb` in
`finepdf` 13%; `kas` in `finepdf` 11%; `gigafida` in `slovenian_news` 14% and
back 14%; `kzb` in `oss` 1%; `solar` under 1% anywhere.

### SUK held out against ssj500k

SUK 1.1 integrates ssj500k 2.3, and `ner/ssj500k` trains on ssj500k while
`ner/suk` is held out, so the held-out splits would contain training
documents. SUK keeps the ssj500k document ids (`ssj1` … `ssj1655`; all 1,655
recur in SUK, 1,654 with identical text), so `ner/suk` drops them with
`source.exclude: [ssj500k]` in `configs/data/tasks.yaml`. The task registry
refuses to load a `finetune_and_eval` / `held_out` pair of one task whose
sources are equal or where one contains the other without that exclusion.

Documents per split, from the extracted tier of 2026-10-04:

| Entry         | Exclusion      | train | val | test |
| ------------- | -------------- | ----: | --: | ---: |
| `ner/ssj500k` | —              | 1,164 | 258 |  233 |
| `ner/suk`     | none (before)  |     — | 641 |  733 |
| `ner/suk`     | ssj500k by id  |     — | 364 |  393 |

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

## Candidate sources

Datasets the [Slovene data landscape](../experiments/data/data-landscape-slovenian/README.md)
catalogued (`tables/catalogue.csv`) that could join a later corpus build. The
shortlist keeps rows that are not in the registry, are openly downloadable,
were written in Slovene (not translated or machine-made), and are mostly edited,
grammatical prose. That record found science already clears its targets on such
supply ([F3]), medicine's new native supply large in words but small in
documents ([F2]), and most of the catalogue general-web surplus not worth
ingesting ([F4]). Nothing here is in the registry yet; adding an entry is a
separate change, and each would need the containment check against its likely
overlap.

| Dataset                                                                                            | Licence      | Size                         | Domain        | Why it fits                                                                       |
| -------------------------------------------------------------------------------------------------- | ------------ | ---------------------------- | ------------- | --------------------------------------------------------------------------------- |
| [MARCELL Slovenian legislative subcorpus v2](https://live.european-language-grid.eu/catalogue/corpus/19460) | CC BY 4.0    | 25,002 docs, 148M tokens     | legal         | Edited legislation 1974–2020; likely overlaps COLESLAW's PISRS part.               |
| [MultiLegalPile, Slovene](https://huggingface.co/datasets/joelniklaus/MultiLegalPile_Wikipedia_Filtered) | CC BY 4.0    | 294,171 rows                 | legal         | Case law, legislation and contracts; check against COLESLAW and Legal-mC4.        |
| [JezKor](https://hdl.handle.net/11356/1755)                                                        | CC BY 4.0    | 338 docs, 9.3M words         | scientific    | Linguistics journal articles and papers, long native academic prose.              |
| [Zdravniški vestnik](https://vestnik.szd.si/index.php/ZdravVest/issue/archive)                     | CC BY-NC 4.0 | 1,616 articles, 6.8M words   | medical       | The national medical journal, most of medicine's new native words ([F2]); per-article fetch, no bulk file. |
| [EMMediaTopic 1.0](https://hdl.handle.net/11356/1991)                                              | CC BY-SA 4.0 | 21,000 docs, 5.4M words      | news          | Edited news articles; likely overlaps `slovenian_news`.                           |
| [NewsSLO](https://doi.org/10.5281/zenodo.12518387)                                                 | CC BY 4.0    | unsized                      | news          | COVID-19 coverage from major dailies, 2020; likely inside `slovenian_news`.       |
| [Slovenian Wikipedia 20231101.sl](https://huggingface.co/datasets/wikimedia/wikipedia)             | CC BY-SA 3.0 | 183,006 articles             | encyclopaedic | A dump three years newer than CLASSLAWiki-sl's; would replace it, not add to it.  |
| [Maj68 3.0](https://hdl.handle.net/11356/1970)                                                     | CC BY-NC-SA 4.0 | 1,521 texts, 1.0M words   | literary      | Published literature around 1968, the only sized modern literary prose found.     |

Left out on the same test: historical corpora (sPeriodika, IMP, PriLit,
ELTeC-slv, Kranjska, SI-IUS), whose pre-1950 orthography and OCR are not
contemporary prose; SlovParl 2.0, already inside siParl; FinePDFs-Edu, a filter
of `finepdf`; learner and student writing (KOST, KOŠ); dictionaries, lexicons
and label-only sets; and every forum or social-media source, per the register
exclusion above.

[F2]: ../experiments/data/data-landscape-slovenian/README.md#f2--medicines-new-native-supply-is-large-in-words-and-small-in-documents--key
[F3]: ../experiments/data/data-landscape-slovenian/README.md#f3--science-clears-both-kpis-on-native-prose-alone--key
[F4]: ../experiments/data/data-landscape-slovenian/README.md#f4--the-catalogue-is-mostly-supply-the-project-does-not-use--supporting-f3
