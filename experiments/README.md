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

## methods

_No experiments recorded yet._

## validation

_No experiments recorded yet._
