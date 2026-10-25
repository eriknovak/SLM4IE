# Findings

What SLM4IE has tried and what it showed — one entry per key finding, linked to
the experiment record behind it, with terms in [GLOSSARY.md](GLOSSARY.md).

`data/pretraining-corpus-slovenian/` and `methods/tokenizer-sweep-slovenian/`
are reserved slugs with no hypothesis yet. The settings they used to hold are
now shared registries under `configs/`, the 2026-06 sweep runs they produced are
parked in MLflow under `slm4ie/archive/`, and what those runs showed is written
up in [`docs/notes/`](../docs/notes/).

## data

### [Open Slovene data landscape — KPI coverage](data/data-landscape-slovenian/) · refuted · 2026-09-28

The first survey of what Slovene text exists openly beyond the project's own downloads. It covers science on both KPIs and medicine on tokens, but not medicine on examples.

- **[F1](data/data-landscape-slovenian/#f1--open-slovene-medicine-is-mostly-not-written-in-slovene) — Open Slovene medicine is mostly not written in Slovene.** Medicine holds 3.04M open documents, of which 24,757 were written in Slovene; the rest is translation and machine output.
- **[F2](data/data-landscape-slovenian/#f2--medicines-new-native-supply-is-large-in-words-and-small-in-documents) — Medicine's new native supply is large in words and small in documents.** The four new native medical sources counted by this experiment hold about 7.9M words, thirty times KPI 4, in under four thousand documents. Together with F1, this places the medical gap in documents, not in text.
- **[F3](data/data-landscape-slovenian/#f3--science-clears-both-kpis-on-native-prose-alone) — Science clears both KPIs on native prose alone.** Native science outside the download registry reaches 147,291 open documents and 33.6M estimated tokens, both far past their thresholds.

### [Curation pipeline quality — Slovenian](data/curation-quality-slovenian/) · refuted · 2026-09-30

The first check of the pretraining-corpus pipeline against labels: an LLM judge, calibrated against a person, read what each curation stage kept and dropped. What the pipeline keeps is clean, but it throws away mostly good text, and the medical and scientific sources lose most of theirs to a quality filter built for web crawls — the domains the data landscape found enough open text for.

- **[F1](data/curation-quality-slovenian/#f1--every-content-filter-drops-mostly-text-worth-keeping) — Every content filter drops mostly text worth keeping.** More than half of what each content filter drops is text the judge would keep.
- **[F2](data/curation-quality-slovenian/#f2--a-filters-precision-depends-heavily-on-the-source) — A filter's precision depends heavily on the source.** The spam filter drops mostly real spam on the fineweb2 web crawl and almost nothing but good text on every curated source.
- **[F3](data/curation-quality-slovenian/#f3--what-the-stages-keep-is-clean) — What the stages keep is clean.** Fewer than one in ten documents that any stage keeps is bad text at the lenient bar.
- **[F7](data/curation-quality-slovenian/#f7--curated-domain-sources-lose-most-of-their-text-to-a-filter-the-judge-does-not-confirm) — Curated domain sources lose most of their text to a filter the judge does not confirm.** The quality filter removes most of the medical source, and the judge calls none of the sampled drops bad text.

The repairs these findings call for, per-source filter settings and a spam filter that counts distinct words, are the subject of the threshold experiment below.

### [Curation filter thresholds — Slovenian](data/curation-thresholds-slovenian/) · inconclusive · 2026-10-20

The repair the quality audit called for: each source takes one of three routes through the content filters, and the filters' thresholds are loosened per route on the judged evidence. Whether each filter now drops what it targets cannot be called, because every judge run agrees with the person less closely than the gate asks. The question the record answers instead is whether the loosened settings keep more good text without a dirtier kept corpus, and the answer is yes. This repairs the quality audit's main loss, good text discarded by filters built for English web crawls. Simple rules still leave web junk that no tested measure separates from good text.

- **[F12](data/curation-thresholds-slovenian/#f12--the-loosened-build-keeps-more-good-text-and-the-draws-cannot-place-its-rise-in-bad-text-above-the-bar) — The loosened build keeps more good text, and the draws cannot place its rise in bad text above the bar.** The loosened build keeps about a quarter more good documents than the baseline, and the interval on its rise in bad text spans the bar.

### [KPI coverage of the curated Slovene corpus](data/kpi-coverage-slovenian/) · confirmed · 2026-10-21

The check of the corpus the project built against its four data targets, read once with every document carrying its source's domain and once with each document labelled by a classifier that passed a quality gate on judged gold. It reads the loosened build the [threshold experiment](data/curation-thresholds-slovenian/) produced, and answers what the [data landscape](data/data-landscape-slovenian/) left open: whether medical text hidden in the web crawls changes the medical verdict. All four targets clear. Medicine misses the examples target under source labels only because its one source never held ten thousand distinct texts, and clears it a hundredfold once the medical documents inside the web crawls are counted. That contradicts the landscape survey's reading that medical examples are missing, for documents, though not for annotated extraction examples, which still have to be built.

- **[F1](data/kpi-coverage-slovenian/#f1--every-data-target-clears-under-document-level-labels-and-under-source-labels-only-the-medical-examples-target-misses) — Every data target clears under document-level labels, and under source labels only the medical examples target misses.** Medical documents miss the examples target under source labels and clear it a hundredfold under document labels, while every other count clears under both.
- **[F2](data/kpi-coverage-slovenian/#f2--the-medical-shortfall-under-source-labels-is-a-source-gap-since-the-medical-source-never-held-enough-distinct-texts) — The medical shortfall under source labels is a source gap, since the medical source never held enough distinct texts.** The medical source falls below the examples target at exact dedup, which removes only copies, so only a new dataset can close the gap.

## methods

_No experiments recorded yet._

## validation

### [Whether the Sloleks 3.2 stem-ending split ranks Slovene tokenizers as the lemma-alignment gold does](validation/morph-gold-slovenian/) · refuted · 2026-10-25

The first experiment of the [Slovene tokenizer choice](programmes/tokenizer-choice-slovenian.md) programme, and the first check of the morphological gold that every morph metric of the tokenizer sweep rests on. The project's own gold cuts an inflected form where it first differs from its lemma; the Sloleks 3.2 authors released a stem-ending split of their own. Twenty-four tokenizers, six algorithms at four vocabulary sizes, were retrained on a held-out corpus split and ranked under each gold on morpheme-boundary F1. The two golds rank them alike, so the gold does not decide which tokenizer looks best. The manually checked forms rank them differently from all forms, through their word-class mix, so the morph score is reported on manual forms first from here on. The record reads refuted because that second claim fails.

- **[F1](validation/morph-gold-slovenian/#f1--the-heuristic-gold-and-the-sloleks-gold-rank-the-tokenizers-alike) — The heuristic gold and the Sloleks gold rank the tokenizers alike.** The two golds rank the tokenizers with a Kendall's τ of 0.86, and the whole bootstrap interval sits above the 0.8 bar.
- **[F2](validation/morph-gold-slovenian/#f2--the-manually-checked-forms-rank-the-tokenizers-differently-from-all-forms) — The manually checked forms rank the tokenizers differently from all forms.** Ranking the tokenizers on the manually checked forms alone agrees with the all-forms ranking at τ 0.69, below the 0.9 bar.
- **[F3](validation/morph-gold-slovenian/#f3--on-all-forms-only-unigrams-morpheme-boundary-f1-still-rises-at-the-largest-vocabulary-while-compression-and-morph-edit-distance-keep-improving-for-every-algorithm) — On all forms only Unigram's morpheme-boundary F1 still rises at the largest vocabulary, while compression and Morph-Edit-Distance keep improving for every algorithm.** From 64k to 128k every algorithm still gains on fertility and Morph-Edit-Distance, but on all forms morpheme-boundary F1 rises for Unigram alone.
