# Findings

What SLM4IE has tried and what it showed — one entry per key finding, linked to
the experiment record behind it, with terms in [GLOSSARY.md](GLOSSARY.md).

`data/pretraining-corpus-slovenian/` and `methods/tokenizer-sweep-slovenian/`
are reserved slugs with no hypothesis yet. The settings they used to hold are
now shared registries under `configs/`, the 2026-06 sweep runs they produced are
parked in MLflow under `slm4ie/archive/`, and what those runs showed is written
up in [`docs/notes/`](../docs/notes/).

## data

**[Curation pipeline quality — Slovenian](data/curation-quality-slovenian/README.md)** — refuted, 2026-09-30. The first check of the pretraining-corpus pipeline against labels: an LLM judge, calibrated against a person, read what each curation stage kept and dropped. What the pipeline keeps is clean, but it throws away mostly good text, and the medical and scientific sources lose most of theirs to a quality filter built for web crawls.

- **[F1](data/curation-quality-slovenian/README.md) — Every content filter drops mostly text worth keeping.** More than half of what each content filter drops is text the judge would keep.
- **[F2](data/curation-quality-slovenian/README.md) — A filter's precision depends heavily on the source.** The spam filter drops mostly real spam on the fineweb2 web crawl and almost nothing but good text on every curated source.
- **[F3](data/curation-quality-slovenian/README.md) — What the stages keep is clean.** Fewer than one in ten documents that any stage keeps is bad text at the lenient bar.
- **[F7](data/curation-quality-slovenian/README.md) — Curated domain sources lose most of their text to a filter the judge does not confirm.** The quality filter removes most of the medical source, and the judge calls none of the sampled drops bad text.

The repairs these findings call for, per-source filter settings and a spam filter that counts distinct words, are the subject of the planned threshold experiment, curation-thresholds-slovenian.

## methods

_No experiments recorded yet._

## validation

_No experiments recorded yet._
