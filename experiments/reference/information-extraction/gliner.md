---
title: GLiNER
kind: algorithm
status: from-paper
summary: Entity type names and the text go through one bidirectional encoder, every span of up to 12 words is scored against every type name by a sigmoid dot product, and the spans above 0.5 are kept greedily without overlap.
variant_of:
part_of:
couples_to: [encoder-transformers/deberta-v3]
---

## Description

GLiNER, a generalist model for named entity recognition, finds entities of
types it was never trained on. The user writes the type names, such as
"person" or "organisation", and the model prepends them to the text. One
bidirectional encoder reads prompt and text together, so the vector of each
type name has seen the text and the vector of each word has seen the type
names. The model then treats recognition as matching: every span of at most 12
words gets one vector, every type name gets one vector, and the sigmoid of
their dot product says whether the span is an entity of that type. All spans
and all types are scored in one pass, which is what makes it faster than a
generative model that writes entities out one token at a time.

- **The prompt is part of the input.** A marker token stands before each type
  name and a separator token closes the prompt. Both markers are new tokens
  whose embeddings are learned during training.
- **A span is its two ends.** The span vector is built from the vector of its
  first word and the vector of its last word, each passed through a small
  feed-forward layer, then joined and projected back to the encoder width.
- **Scoring is a dot product.** No head knows the type names in advance; a
  type is whatever its name's vector says, which is what lets the model take
  new types at inference.
- **Decoding is greedy.** Candidates above the threshold are sorted by score
  and taken in order, each one skipped if it overlaps a span already taken.

## What the paper adds

The paper's claim is that a 300M-parameter encoder, trained once on a corpus
of 13k entity types produced by prompting a large language model, beats
ChatGPT and fine-tuned generative models on zero-shot entity recognition at a
fraction of the cost. It trains on Pile-NER, 44,889 passages annotated by
GPT-3.5 with 240k spans, and evaluates on 20 benchmarks the model never saw,
in English and in seven other languages through the multilingual DeBERTa
backbone. Two regularisations matter for the claim: entity types are shuffled
and some are dropped at random so the model cannot learn their order, and
negative types drawn from other examples teach it to return nothing for a
type the text does not contain.

## Why it matters here

GLiNER is the extraction head a small Slovenian encoder would carry first,
because its output is exactly what this project's task data holds. The span
converter in `slm4ie/data/tasks/converters/spans.py:1-14` writes every
Slovenian NER dataset as GLiNER-style rows, text plus `start`, `end` and
`label` per span, so a GLiNER head can be fine-tuned and evaluated on them
without another conversion. The head is also separable from the backbone: the
released multilingual model sits on mDeBERTa-v3-base, and the encoders of the
encoder-transformers topic are the candidates to put under it.

## Facts

- **Tasks**: named entity recognition, flat or nested, with any type names given at inference
- **Prompt tokens**: `[ENT]` before each type name and `[SEP]` before the text in the paper; `<<ENT>>` and `<<SEP>>` in the released models
- **Span candidates**: every span whose first and last word lie in the text and whose width is at most 12 words, scored against every type
- **Longest span**: 12 words
- **Scoring**: sigmoid of the dot product between the span vector and the type vector; the span vector is the projected first-word and last-word vectors joined, the type vector is the projected marker vector
- **Decoding**: candidates above 0.5 sorted by score, kept greedily without overlap; the nested mode allows a span fully inside another
- **Backbone**: DeBERTa-v3 small, base or large for English; mDeBERTa-v3-base for the multilingual model; the span and type projections are 512 wide
- **Context length**: 384 subword tokens in the released models, prompt included
- **Training data**: Pile-NER, 44,889 passages with 240k spans of 13k types, annotated by GPT-3.5 over the Pile
- **Training**: at most 25 types per example after adding negatives, types shuffled and dropped at random, binary cross-entropy over every span and type pair, AdamW at 1e-5 for the encoder and 5e-5 for the rest, 30,000 steps with 10% warmup and cosine decay, dropout 0.4
- **Sizes**: small 50M, medium 90M, large 300M, multilingual 209M parameters

## Sources

- **Paper**: GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer, Zaratiana, Tomeh, Holat and Charnois, NAACL 2024, arXiv:2311.08526
- **Paper**: DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing, He, Gao and Chen, 2021, arXiv:2111.09543, the backbone
- **Code**: none in this repository; the reference implementation is the gliner package of the urchade/GLiNER repository, version 0.2.29 read for this entry: data_processing/processor.py builds the prompt, modeling/span_rep.py holds the span marker, modeling/base.py the scoring and loss, decoding/decoder.py the greedy search
- **Code**: `slm4ie/data/tasks/converters/spans.py` writes the GLiNER-style task rows this project evaluates on; `slm4ie/data/schema.py:202-214` fixes their shape
- **Model**: urchade/gliner_multi-v2.1 on the Hugging Face hub, the multilingual release; its configuration gives the width, length and markers above
- **Example**: `experiments/reference/information-extraction/examples.py`
- **Paper vs code**: not checked against project code. Three differences against the released package. The paper names the markers `[ENT]` and `[SEP]`; the code uses `<<ENT>>` and `<<SEP>>`. The paper builds a span vector by one feed-forward layer over the joined end vectors; the default span mode projects each end separately, joins them, applies ReLU and projects once more. The paper trains with binary cross-entropy; the code's default is focal loss, which reduces to binary cross-entropy at its default settings, with half of the negative span and type pairs dropped from the loss at random.

## Algorithm

```algorithm
\caption{GLiNER inference: type names and text through one encoder, every span scored against every type}
\Require words $w_1, \dots, w_N$, type names $t_1, \dots, t_M$, encoder $E$, span width $K = 12$, threshold $\tau = 0.5$
\Ensure a set $R$ of typed, non-overlapping spans $(i, j, m)$
\State $x \gets [\mathrm{ENT}]\, t_1 \, \dots \, [\mathrm{ENT}]\, t_M \, [\mathrm{SEP}]\, w_1 \, \dots \, w_N$
\State $H \gets E(x)$; $p_m \gets$ the vector at the $m$-th $[\mathrm{ENT}]$; $h_i \gets$ the vector at the first subword of $w_i$
\State $q_m \gets \mathrm{FFN}_t(p_m)$ for every type $m$
\ForAll{$(i, j)$ with $1 \le i \le j \le \min(i + K - 1, N)$}
  \State $s_{ij} \gets \mathrm{FFN}_o(\mathrm{ReLU}(\mathrm{FFN}_s(h_i) \oplus \mathrm{FFN}_e(h_j)))$
  \State $\phi(i, j, m) \gets \sigma(s_{ij} \cdot q_m)$ for every type $m$
\EndFor
\State $C \gets \{(i, j, m) : \phi(i, j, m) > \tau\}$, sorted by $\phi$ descending
\State $R \gets \emptyset$
\ForAll{$(i, j, m) \in C$}
  \If{no span in $R$ overlaps $[i, j]$}
    \State $R \gets R \cup \{(i, j, m)\}$
  \EndIf
\EndFor
\Return $R$
```

- **1**: The prompt is text. Each type name follows its own marker, and the separator keeps the encoder from reading the last type name into the first word. Nothing about the types is stored in the model.
- **2**: One encoder pass gives every token a vector that has seen prompt and text. The type vector is read at the marker, not at the name, so a type of several words still gives one vector. A word of several subwords is read at its first subword.
- **3**: A two-layer feed-forward network turns the marker vector into the type vector. The same network serves every type.
- **4**: Candidate spans are every start with every width up to $K$, so a text of $N$ words has about $NK$ of them. The limit keeps the count linear in the length and discards nothing the benchmarks contain, where entities rarely exceed 12 words.
- **5**: The span vector comes from its two ends. Each end is projected on its own, the two are joined, and one more projection brings the pair back to the encoder width. A span of one word uses the same vector twice.
- **6**: The score is a dot product squashed to a probability. Every span and type pair is scored at once as one matrix product, which is where the parallel extraction comes from.
- **7**: Only pairs above the threshold are candidates. Sorting by score decides which of two overlapping candidates wins.
- **8** to **12**: Greedy decoding. A candidate is kept unless it overlaps a kept span; in the nested mode a span fully inside a kept span is also allowed. The set returned is the set of entities.

```algorithm
\caption{GLiNER training: one batch}
\Require examples $(w, G)$ with gold spans $G \subseteq \{(i, j, m)\}$, the types of the other examples in the batch, limits of 25 types per example
\Ensure updated encoder and projection weights
\State for each example, take its gold types, add negative types drawn from the batch at a ratio of at most 1 to 1, shuffle, drop some at random, and cut to 25
\State compute $\phi(i, j, m)$ for every span and type pair as Algorithm 1, lines 1 to 6
\State $y(i, j, m) \gets 1$ if $(i, j, m) \in G$ else $0$
\State $L \gets -\sum_{(i, j, m)} \big[\, y \log \phi + (1 - y) \log (1 - \phi) \,\big]$
\Return one AdamW step on $L$, at $10^{-5}$ for the encoder and $5 \cdot 10^{-5}$ for the projections
```

- **1**: The type list is rebuilt for every example. Negative types teach the model to return nothing for a type the text lacks. Shuffling and dropping keep it from learning a type's position.
- **2**: Training scores the same span grid inference does, so there is no train-and-test gap in what is scored.
- **3** and **4**: Every span and type pair is a binary decision. Gold spans are positives; all other pairs, including gold spans under the wrong type, are negatives.
- **5**: The encoder moves slowly and the fresh projections faster. The paper runs 30,000 such steps with a cosine schedule.

## Worked example

The example script lays out the prompt the released multilingual model builds
for one Slovenian sentence with two type names, and counts the spans it scores
at several text lengths.

![The prompt each generation builds for the toy sentence](tables/prompt-layouts.csv)

![Spans scored per type by text length](tables/candidate-counts.csv)

A text of 32 words gives 318 spans per type. At 512 words the grid is 6,078
spans per type, still linear in the length because the width is capped, but
each span is scored against every type, so a prompt of 20 types at 512 words
scores over 120,000 pairs.

## Limits

### Twelve words is a hard ceiling

A span of 13 or more words cannot be returned, whatever the score. Titles of
works, addresses and legal citations in Slovenian text exceed it.

### The prompt eats the context

Type names and the text share 384 subword tokens. Twenty Slovenian type names
spelled by an English-trained tokenizer take a large share of it before the
text begins.

### Spans only

One span, one type. A relation between two spans, a label for the whole text,
or an attribute of a span has no place in the output.

### Silver training data

Pile-NER was annotated by GPT-3.5 with no human check, and its types are
whatever the annotator named. What the model learned as "person" is that
annotator's notion of a person.
