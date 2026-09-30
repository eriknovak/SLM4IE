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

The repairs these findings call for, per-source filter settings and a spam filter that counts distinct words, are the subject of the planned threshold experiment, curation-thresholds-slovenian.

## methods

_No experiments recorded yet._

## validation

_No experiments recorded yet._
