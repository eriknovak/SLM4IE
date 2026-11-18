---
title: Joint relation extraction
kind: component
status: from-paper
summary: One forward pass returns entities and the typed relations between them, scoring each pair of found entities against each relation type named at the call.
variant_of:
part_of: information-extraction/gliner
couples_to: []
---

## Description

A relation model takes entity types and relation types at the call, such as
"method" and "dataset" with the relation "evaluated on". It first finds the
entities, as any GLiNER model does. It then scores which pairs of entities
are linked, and scores each linked pair against each relation type. Three
thresholds cut the entities, the linked pairs and the relations. The output
is the list of entities and a list of relations, each naming its head entity,
its tail entity, the type and a score.

## What the source adds

The library has two relation architectures, one over spans and one over
tokens, and release v0.2.26 improved them. The released checkpoints come
from Knowledgator, about 0.5 billion parameters, under the Apache-2.0
licence. Their cards name no backbone and no languages, and every example is
English.

- **In short**: entities and typed relations from one pass, with English checkpoints of 0.5B parameters

## Why it matters here

The scientific data is built around methods, datasets, metrics and results.
Its value lies largely in which method scored which result on which dataset,
which is a relation. No Slovene relation data exists yet, so this entry is a
later step.

- **Scientific data (#47)**: decide now whether the annotation guide marks
  relations as well as spans. Adding them later means a second annotation
  pass over the same texts.
- **Training data format**: a relation example adds pairs of entity
  indices with a type to the span example the task data already has.

## Facts

- **Stage**: architecture — chosen before training; the model must be trained with relations
- **Architectures**: span relation, token relation — two architectures
- **Checkpoints**: knowledgator/gliner-relex-large-v0.5 and v1.0 — 0.5B, Apache-2.0, languages unstated
- **Since**: before v0.2.26 — improved in v0.2.26
- **Combines with**: label descriptions, entity vectors, export — v1.0 takes descriptions, and relation models export to ONNX
- **Conflicts with**: none known — prompt compression also covers relation labels
- **Training data**: spans plus head-tail pairs — pairs index into the example's entity list

## Sources

- **Release notes**: GLiNER v0.2.26, <https://github.com/urchade/GLiNER/releases/tag/v0.2.26> — relation model improvements
- **Model card**: knowledgator/gliner-relex-large-v1.0, <https://huggingface.co/knowledgator/gliner-relex-large-v1.0> — inference call, thresholds, output format
- **Docs**: GLiNER training guide, <https://urchade.github.io/GLiNER/training.html> — relation training data
- **Paper vs code**: no paper is cited on the card; this repository holds no GLiNER code yet, so no difference can be found

## Where it sits

- **Steps**: find entities → **score entity pairs and relation types** → return entities and relations

Entity finding is the ordinary GLiNER head. Building a knowledge graph from
the relations is not part of the library.

## Limits

### No Slovene relation data

No Slovene dataset with typed relations sits in the task registry. Without
one, neither training nor scoring is possible in Slovene.

- **Question**: does the English relation checkpoint find any usable method-dataset relations in Slovene scientific abstracts zero-shot, or must Slovene relations be annotated first?
