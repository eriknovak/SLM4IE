---
title: Multitask pipelines
kind: component
status: from-paper
summary: Ready-made pipelines pose classification, question answering, relation extraction, open extraction and summarisation as span or label matching for one GLiNER model trained on all of them.
variant_of:
part_of: information-extraction/gliner
couples_to: []
---

## Description

Many extraction tasks can be posed as finding spans. Question answering
finds the answer span for a question used as the prompt. Summarisation picks
the sentences worth keeping. Classification matches the whole text against
each label name. The library wraps each task in a pipeline that builds the
prompt and reads the output, all on one ordinary GLiNER model. The model must
be trained on the tasks, which the multitask checkpoint is.

## What the paper adds

The GLiNER multi-task paper trains one compact encoder on synthetic data for
all these tasks. It reports zero-shot NER at the state of the art for its
size, with question answering, summarisation and relation extraction from the
same weights.

- **In short**: one small encoder trained for several extraction tasks posed as span matching

## Why it matters here

The project's zero-shot programme has measured only NER so far. Question
answering and classification are open: no frontier or open LLM has been
scored on them, and no small model either. These pipelines offer a small
model for the same tasks without a new architecture.

- **Question answering and sentiment gap**: a Slovene multitask GLiNER,
  fine-tuned on the SuperGLUE-SL question answering splits and the
  sentiment data, could be the small-model arm once those test files exist.
- **Shared backbone**: the same Slovene backbone chosen in #45 could carry
  NER and these tasks, which keeps one model to release.

## Facts

- **Stage**: architecture — the model must be trained on each task it serves
- **Architectures**: uni-encoder — an ordinary GLiNER model behind task-specific prompts
- **Checkpoints**: knowledgator/gliner-multitask-large-v0.5 — English; backbone not named on the card
- **Since**: before v0.2.26 — fixes to multitask inference in v0.2.28
- **Combines with**: calibration, export — the model is an ordinary uni-encoder
- **Conflicts with**: none known — task prompts are text, so compression would only fit a fixed task
- **Tasks**: classification, question answering, relation extraction, open extraction, summarisation — five pipelines

## Sources

- **Paper**: GLiNER multi-task: Generalist Lightweight Model for Various Information Extraction Tasks, Stepanov and Shtopko, 2024, arXiv:2406.12925 — the multitask training and its claims
- **Model card**: knowledgator/gliner-multitask-large-v0.5, <https://huggingface.co/knowledgator/gliner-multitask-large-v0.5> — pipelines and training data
- **Paper vs code**: this repository holds no GLiNER code yet, so no difference can be found

## Where it sits

- **Steps**: pick a task → **build the task prompt and run the GLiNER model** → read spans or labels as the task's answer

The task data comes from the task registry. The backbone comes from the
backbone comparison.

## Limits

### Synthetic English training

The released model learned its tasks from English synthetic data. Slovene
question answering may need Slovene training examples first.

- **Question**: does a multitask GLiNER fine-tuned on Slovene SuperGLUE-SL question answering come within reach of an open LLM on the same split?
