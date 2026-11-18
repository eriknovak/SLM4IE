---
title: Label descriptions
kind: component
status: from-paper
summary: Each entity type is passed as a name with a sentence that says what it covers, and the sentence is written into the prompt in place of the bare name.
variant_of:
part_of: information-extraction/gliner
couples_to: []
---

## Description

A GLiNER model normally reads only a type's name, such as "measurement",
and has to guess what it covers. With label descriptions the user gives each
type a short sentence instead, such as "a value measured on the patient, with
its unit". The model writes the sentence into its prompt where the name would
stand, and the spans it returns still carry the short name. A batch can give
each text its own descriptions. Every description must be a distinct string,
because the model tells the types apart by their prompt.

## What the source adds

The library added descriptions in release v0.2.29 for every architecture. The
release notes add that only models trained with descriptive labels benefit. A
model trained on bare names reads a long sentence as an unfamiliar name. The
relation-extraction checkpoint v1.0 is one model that takes descriptions.

- **In short**: types passed as name-plus-sentence, since v0.2.29; useful only after training with descriptions

## Why it matters here

The medical data labels spans with OMOP (Observational Medical Outcomes
Partnership) domains such as "Observation", "Measurement" and "Procedure".
These names are vague, and they overlap in everyday use. The scientific data
has a similar pair in "method" and "result".

- **Medical and scientific data (#46, #47)**: write one description per type
  into the annotation guide, and keep it in every training example, so a
  model can learn from them.
- **Domain fine-tuning (#51)**: training with descriptions against training
  with names is a cheap ablation on the same silver data.
- **Backbone comparison (#45)**: description-trained zero-shot checkpoints
  may favour backbones that read long prompts well.

## Facts

- **Stage**: training and scoring — the gain needs descriptions in the training data too
- **Architectures**: any — uni-encoder, bi-encoder, relation and streaming models accept the dictionary
- **Checkpoints**: relation-extraction v1.0 shows them — no Slovene checkpoint is trained with them
- **Since**: v0.2.29 — a dictionary from label to description
- **Combines with**: calibration, nested spans, relation extraction — descriptions change only the prompt, so the other features are untouched
- **Conflicts with**: prompt compression — a compressed prompt is fixed per label name and cannot hold a description
- **Prompt language**: open — a Slovene model could take Slovene or English descriptions

## Sources

- **Release notes**: GLiNER v0.2.29, <https://github.com/urchade/GLiNER/releases/tag/v0.2.29> — the descriptions dictionary and the note on description-trained models
- **Docs**: GLiNER usage, label descriptions, <https://urchade.github.io/GLiNER/usage.html#label-descriptions>
- **Model card**: knowledgator/gliner-relex-large-v1.0, <https://huggingface.co/knowledgator/gliner-relex-large-v1.0> — descriptions in a released checkpoint
- **Paper vs code**: no paper; this repository holds no GLiNER code yet, so no difference can be found

## Where it sits

- **Steps**: write the annotation guide → **describe each type in the prompt** → encode prompt and text → score spans

The annotation guides come from the medical and scientific data experiments.
The scoring step is the GLiNER head itself.

## Limits

### Language of the description

The multilingual GLiNER checkpoints were trained mostly on English prompts.
A Slovene description may read worse to them than an English one, even for a
Slovene text.

- **Question**: on the Slovene NER test sets, does a Slovene description of each type score higher or lower than an English one?

### Long prompts take context

Each description takes tokens from the encoder's context window, which is
512 tokens for many backbones. With a dozen types, the text left per window
shrinks.

- **Question**: at what number of described types does the shorter text window cost more span F1 than the descriptions gain?
