# Findings

What SLM4IE has tried and what it showed — one entry per key finding, linked to
the experiment record behind it, with terms in [GLOSSARY.md](GLOSSARY.md).

`data/pretraining-corpus-slovenian/` and `methods/tokenizer-sweep-slovenian/`
are reserved slugs with no hypothesis yet. The settings they used to hold are
now shared registries under `configs/`, the 2026-06 sweep runs they produced are
parked in MLflow under `slm4ie/archive/`, and what those runs showed is written
up in [`docs/notes/`](../docs/notes/).

## data

### [Open Slovene data landscape — KPI coverage](data/data-landscape-slovenian/) · running

- Medicine holds 3.04M open documents, of which 22,435 were written in Slovene; the rest is translation and machine output.
- No open Slovene medical dataset publishes a size; counted by this experiment, the new native sources hold about 7.9M words, thirty times KPI 4.
- Science reaches 299,961 openly downloadable documents written in Slovene and 5.28B estimated tokens, both far past their thresholds.

## methods

_No experiments recorded yet._

## validation

_No experiments recorded yet._
