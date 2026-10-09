---
title: GLiNER2
kind: algorithm
status: from-paper
summary: GLiNER's span matching behind a schema prompt that composes entity types, classification labels and record fields in one input, with a count head for how many records to fill and a label head for classification, on a 2,048-token context.
variant_of: information-extraction/gliner
part_of:
couples_to: [encoder-transformers/deberta-v3]
---

## Description

GLiNER2 keeps GLiNER's idea, one encoder over prompt and text and a dot
product between request vectors and span vectors, and widens what a request
may be. The request is a schema: a list of entity types, a classification task
with its labels, or a record type with named fields, and several of these at
once. Each part of the schema becomes a bracketed group in the prompt with its
own marker tokens, the groups are joined by a separator, and the text follows.
One forward pass then serves every group.

- **Entities work as in GLiNER.** Each type marker's vector is matched against
  every span of at most 8 words.
- **Records are entities with a count.** A record type such as "employment"
  has fields such as "employee" and "employer". A small head reads the
  group's prompt marker and predicts how many records the text holds, from 0
  to 19. For each record, a recurrent layer stamps the field vectors with the
  record's index, and each stamped field vector is matched against the spans
  as an entity type would be. The result is a list of filled records rather
  than a flat list of spans.
- **Classification reads the label markers.** Each label's marker vector goes
  through a small classifier that outputs one logit. A single-label task takes
  the largest; a multi-label task keeps every label above 0.5.
- **Descriptions and examples ride in the prompt.** A type or label may carry
  a short description and a task may carry a few labelled examples, written
  after the group's name behind their own marker tokens.

## What the paper adds

The paper's claim is that one encoder under 500M parameters can replace a
large language model for entity recognition, text classification and
structured extraction at once, on a CPU. It trains a 205M-parameter model on
254,334 examples, 135,698 from news, Wikipedia, legal text, arXiv and PubMed
annotated by GPT-4o and 118,636 synthetic ones from emails, messages, resumes
and e-commerce text, and reports results against GLiNER and against prompted
large models on zero-shot NER, classification and record extraction. The
released package adds a schema builder and LoRA fine-tuning.

## Why it matters here

GLiNER2 is the first entry whose output covers more than entity spans, which
is what a Slovenian extraction benchmark beyond NER would need: sentiment and
the SuperGLUE-SL tasks this project prepares are classification, and a
record schema fits event-like annotations. Its parts are separable from the
backbone in the same way GLiNER's are, so a Slovenian encoder can take the
span head, the count head and the label head as they stand. The multilingual
release again sits on mDeBERTa-v3-base.

## Facts

- **Tasks**: named entity recognition, single- and multi-label text classification, record extraction with named fields, and any combination in one pass
- **Prompt tokens**: each schema group is `( [P] name ( marker field … ) )`, with `[E]` before an entity type, `[C]` before a record field and `[L]` before a label; groups are joined by `[SEP_STRUCT]` and `[SEP_TEXT]` precedes the text; `[DESCRIPTION]`, `[EXAMPLE]` and `[OUTPUT]` introduce a description, an example and its answer inside a group
- **Span candidates**: every span of at most 8 words, scored against every entity type and every stamped record field
- **Longest span**: 8 words
- **Scoring**: GLiNER's sigmoid dot product between the span vector and the marker vector; record fields are matched by the marker vector stamped with the record index; a label's logit is a classifier over its marker vector
- **Decoding**: spans above 0.5 kept greedily without overlap per type; a count head picks the number of records from 0 to 19 and each field takes its best span, or every span above threshold for a list field; single-label classification takes the argmax, multi-label keeps labels above 0.5
- **Backbone**: DeBERTa-v3-base for the base model, mDeBERTa-v3-base for the multilingual one
- **Context length**: 2,048 tokens
- **Training data**: 254,334 examples, 135,698 real texts annotated by GPT-4o and 118,636 synthetic
- **Training**: binary cross-entropy over span and field pairs with half the negatives masked at random, cross-entropy over the record count, binary cross-entropy over label logits; AdamW at 1e-5 for the encoder and 2e-5 for the heads, weight decay 0.01, 1,000 warmup steps, gradient clipping at 1, 5 epochs
- **Sizes**: base 205M, multilingual about 300M parameters

## Sources

- **Paper**: GLiNER2: An Efficient Multi-Task Information Extraction System with Schema-Driven Interface, Zaratiana, Pasternak, Boyd, Hurn-Maloney and Lewis, 2025, arXiv:2507.18546
- **Code**: none in this repository; the reference implementation is the gliner2 package of the fastino-ai/GLiNER2 repository, version 2.1.0 read for this entry: processor.py builds the prompt, models/span/model.py holds the span grid, the count head and the losses, layers.py the span marker and the count stamping
- **Model**: fastino/gliner2-base-v1 and fastino/gliner2-multi-v1 on the Hugging Face hub; the width of 8 and the count layer above are read from their configuration
- **Example**: `experiments/reference/information-extraction/examples.py`
- **Paper vs code**: not checked against project code. The paper writes the prompt as `[P] name ( [E] e … ) [SEP] text`; the code wraps each group in parentheses, joins groups with `[SEP_STRUCT]`, closes the prompt with `[SEP_TEXT]`, and lowercases the text. The paper says the count is a classification over 0 to 19; the code clamps the gold count to 19 and stamps records with learned position embeddings through a GRU and a two-layer transformer. The paper gives one learning rate for task layers; the code also masks half of the negative span pairs in the structure loss.

## Difference from predecessor

- **Line**: 1 and 4 to 11 of Algorithm 1 against GLiNER's Algorithm 1
- **Predecessor**: one prompt of type names, every span of at most 12 words scored against every type, greedy decoding
- **Here**: a prompt of bracketed schema groups with four marker tokens, every span of at most 8 words scored against type and stamped field vectors, a count head and a label head beside the span scores
- **Change**: a schema prompt, record extraction with a predicted count, and classification, in one pass.
- **Effect**: One model answers an entity, a classification and a record request together, and the output is structured rather than a flat list of spans.

## Algorithm

```algorithm
\caption{GLiNER2 inference: one forward pass over a composed schema}
\Require words $w_1, \dots, w_N$, schema groups $g_1, \dots, g_G$ each with a name and fields $f_{g,1}, \dots, f_{g,k}$, encoder $E$, span width $K = 8$
\Ensure entities per type, filled records per record type, labels per classification task
\Changed \State $x \gets \mathrm{group}(g_1)\, [\mathrm{SEP\_STRUCT}] \dots \mathrm{group}(g_G)\, [\mathrm{SEP\_TEXT}]\, w_1 \dots w_N$, with $\mathrm{group}(g) = (\, [\mathrm{P}]\ \mathrm{name}\ (\, \mu\ f_{g,1} \dots \mu\ f_{g,k} \,)\,)$ and $\mu$ one of $[\mathrm{E}]$, $[\mathrm{C}]$, $[\mathrm{L}]$
\State $H \gets E(x)$; $p_g \gets$ the vector at the $[\mathrm{P}]$ of group $g$; $u_{g,k} \gets$ the vector at its $k$-th field marker; $h_i \gets$ the vector at the first subword of $w_i$
\ForAll{entity groups $g$}
  \State score every span $(i, j)$ with $j - i < K$ against every $u_{g,k}$ as GLiNER's Algorithm 1, lines 4 to 12
\EndFor
\Changed \ForAll{record groups $g$}
  \Changed \State $n \gets \arg\max_{c \in 0..19} \mathrm{MLP}_c(p_g)$
  \Changed \State $u^{(r)}_{g,k} \gets T(\mathrm{GRU}(e_r, u_{g,k}) + u_{g,k})$ for $r = 1..n$, with $e_r$ a learned embedding of the record index and $T$ a small transformer
  \Changed \State record $r$, field $k \gets$ the span with the highest $\sigma(s_{ij} \cdot u^{(r)}_{g,k})$, or every span above $0.5$ for a list field
\EndFor
\Changed \ForAll{classification groups $g$}
  \Changed \State $z_{g,k} \gets \mathrm{MLP}_l(u_{g,k})$; single-label: the $k$ with the largest $z$; multi-label: every $k$ with $\sigma(z_{g,k}) > 0.5$
\EndFor
\Return the entities, records and labels
```

- **1**: Every request becomes a bracketed group. The group's name follows the prompt marker, its fields follow their own marker, and the marker's kind says what the fields are: entity types, record fields or labels. Several groups sit in one prompt, which is what lets one pass serve several tasks.
- **2**: As in GLiNER, vectors are read at the markers. Each group gives one prompt vector and one vector per field.
- **3** and **4**: Entity groups are plain GLiNER with a shorter width. Every span of at most 8 words is scored against every type vector and decoded greedily.
- **5** and **6**: A record group first decides how many records the text holds. The classifier over the prompt vector has 20 outputs, 0 to 19, and the largest wins.
- **7**: Each record needs its own field vectors, or every record would fill the same spans. A learned embedding of the record index is fed with the field vector through a GRU, the field vector is added back, and a two-layer transformer lets the fields of one record see each other.
- **8**: A stamped field vector is matched against the spans exactly as a type vector is. The best span fills the field; a field declared as a list takes every span above threshold.
- **9** and **10**: Classification never looks at spans. Each label's marker vector goes through a classifier that outputs one logit. A single-label task takes the largest logit; a multi-label task thresholds each one.
- **11**: One structured answer: lists of spans per type, lists of records per record type, labels per task.

## Worked example

The prompt-layouts table shows the input the released processor builds for
the toy sentence with an entity group, a classification group, a record group
and all of them composed, and the candidate-counts table shows how many spans
the width of 8 yields.

![The prompt each generation builds for the toy sentence](tables/prompt-layouts.csv)

![Spans scored per type by text length](tables/candidate-counts.csv)

The shorter width cuts the grid by about a third against GLiNER at every
length, from 318 to 228 spans per type at 32 words.

## Limits

### Eight words is a harder ceiling than twelve

A span of nine or more words cannot be returned. Slovenian organisation names
and titles exceed it more often than English ones.

### Nineteen records at most

The count head has 20 outputs. A text with more records than that fills
nineteen and drops the rest.

### No relations

Two spans can share a record but the model never scores a pair of spans as
such. A relation type with head and tail roles arrives only in GLiNER2.5.

### Tasks do not see each other's answers

Each group is decoded on its own. A label combination the schema forbids,
such as a harm type under a "safe" verdict, can be returned, which is what
the type-safe decoding of the successor exists to prevent.
