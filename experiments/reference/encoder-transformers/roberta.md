---
title: RoBERTa
kind: algorithm
status: from-paper
summary: BERT's encoder unchanged, retrained with a fresh mask on every pass, full sentences packed across document boundaries, no next-sentence prediction, a byte-level BPE vocabulary, and ten times the data at a batch of 8,000 sequences.
variant_of: encoder-transformers/bert
part_of:
couples_to: [subword-tokenizers/byte-level-bpe]
---

## Description

RoBERTa is BERT trained differently. The layers, the widths, the absolute
position table and the masking rule are all BERT's. What changes is how the
training examples are made and how long and how large the training runs.
The paper's point is that these choices, which BERT left under-explored, are
worth as much as a new architecture.

- **The mask is drawn anew every time a sequence is seen.** BERT fixed the
  mask when the corpus was prepared. RoBERTa draws it when the sequence is
  fed to the model, so every pass over the data hides different tokens.
- **Inputs are full sentences packed to 512 tokens.** Sentences are taken in
  order from the stream of documents until the input is full. When a document
  ends, a separator is added and the next document continues in the same
  input. There is no second segment and no next-sentence label.
- **Text is cut by byte-level BPE.** The vocabulary of 50,000 pieces is
  GPT-2's, learned over bytes, so no character is ever unknown.
- **Training is longer and wider.** 160GB of text instead of 16GB, batches of
  8,000 sequences instead of 256, 500,000 steps, and every step at the full
  length of 512 tokens.

## What the paper adds

The paper is a controlled study of BERT's training recipe. It reruns BERT with
one change at a time and shows which of them move the benchmarks: dynamic
masking slightly, dropping next-sentence prediction slightly, larger batches
and more data and longer training considerably. The byte-level vocabulary
costs a little on some tasks but removes unknown tokens, which the authors
take as worth it. The resulting recipe became the default for most encoders
that followed.

## Why it matters here

The Slovenian encoder most often used as a baseline, SloBERTa, is a RoBERTa
trained with the CamemBERT recipe, so RoBERTa is the closest published design
to what a Slovenian-only encoder has looked like. Its lesson for a small
model is that data, batch size and training length are the first dials to
set, before any change to the layers. Its byte-level vocabulary is also the
byte-level BPE of this project's tokenizer sweep, so the sweep's finding on
Slovenian morphology applies to this design directly.

## Facts

- **Position encoding**: learned absolute, as in BERT
- **Attention span**: every token attends to every token, as in BERT
- **Block normalisation**: post-norm, as in BERT
- **Activation**: GELU, as in BERT
- **Pretraining objective**: masked language modelling on 15% of the tokens with a fresh mask on every pass; no next-sentence prediction
- **Tokenizer**: byte-level BPE, 50,000 pieces, the GPT-2 vocabulary
- **Context length**: 512 tokens, every training step at full length
- **Input**: full sentences taken in order across document boundaries up to 512 tokens, a separator between documents, one segment for all tokens
- **Sizes**: Base and Large as in BERT, 125M and 355M parameters with the larger vocabulary
- **Training data**: BookCorpus and Wikipedia 16GB, CC-News 76GB, OpenWebText 38GB, Stories 31GB, in all over 160GB
- **Optimizer**: Adam with $\beta_2 = 0.98$ and $\epsilon = 10^{-6}$ for stability at the large batch, peak learning rate 6e-4 for Base and 4e-4 for Large, warmup over 30,000 and 24,000 steps, linear decay, weight decay 0.01
- **Training length**: 500,000 steps of 8,000 sequences of 512 tokens, about two trillion tokens seen

## Sources

- **Paper**: RoBERTa: A Robustly Optimized BERT Pretraining Approach, Liu et al., 2019, arXiv:1907.11692
- **Paper**: Language Models are Unsupervised Multitask Learners, Radford et al., 2019, whose byte-level BPE vocabulary RoBERTa reuses
- **Code**: none in this repository; the reference implementation is the `RobertaModel` of the HuggingFace transformers library and the `roberta` model of fairseq
- **Paper vs code**: not checked; the project has no encoder code yet

## Difference from predecessor

- **Line**: 1 to 3 and the return of Algorithm 1, against lines 1 to 3 and 9 of BERT's Algorithm 2
- **Predecessor**: two segments, half of them unrelated, a mask drawn once when the corpus is prepared, and the sum of the masked-token and next-sentence losses
- **Here**: full sentences packed across documents, a mask drawn each time the sequence is fed, and the masked-token loss alone
- **Change**: dynamic masking, full-sentence packing, no next-sentence prediction, a byte-level BPE vocabulary, and ten times the data at a batch of 8,000.
- **Effect**: Every pass over the data shows a different masking, every input is 512 tokens of real text, and the encoder itself is untouched.

## Algorithm

```algorithm
\caption{RoBERTa pretraining example: dynamic masking over packed full sentences}
\Require a stream of documents, byte-level BPE vocabulary $V$ of 50,000 pieces, maximum length $T = 512$
\Ensure one input sequence and its masked-token targets
\Changed \State take full sentences in order from the stream until $T$ tokens are reached; when a document ends, add a separator token and continue with the next document
\Changed \State $x \gets$ `<s>` sentences `</s>`, with one segment id for every token
\Changed \State $M \gets$ 15% of the token positions of $x$, drawn at random now, each time $x$ is fed to the model
\ForAll{$i \in M$}
  \State with probability $0.8$ replace $x_i$ by `<mask>`; with $0.1$ by a random token of $V$; with $0.1$ keep $x_i$
\EndFor
\State $h \gets$ the BERT encoder (BERT, Algorithm 1) on $x$
\State $L_{MLM} \gets \sum_{i \in M}$ cross-entropy of the prediction head against the original $x_i$
\Changed \Return $L_{MLM}$
```

- **1**: An input is filled with consecutive sentences until the length is reached, so almost every input is 512 tokens of text that belongs together. A document boundary inside the input is marked by a separator and not avoided. The paper also tried stopping at the document boundary, which scored slightly higher but gave batches of uneven size, and kept the packed form for the fixed batch size.
- **2**: There is one segment, so the segment embedding carries no information. The special tokens are the start and end markers of the byte-level vocabulary.
- **3-5**: The mask is drawn at the moment the sequence enters the model. Across the 500,000 steps the same sentence is hidden in a different place each time it is seen. The replacement rule inside the mask is BERT's.
- **6-7**: The encoder and the prediction head are BERT's. Only the masked positions are scored.
- **8**: There is no second loss. The paper found that removing next-sentence prediction matched or improved every task it measured.

## Limits

### The encoder is BERT's

Position is still absolute and told once, attention still spans every token,
and the context is still 512. The paper changes nothing a later entry would
call architecture.

### A byte-level vocabulary spends pieces on bytes

The 256 byte symbols and the merges built from them are learned on English
web text. A character of a language that is not English costs several bytes
before it becomes a piece of its own, so a vocabulary learned on English
spells Slovenian text in more tokens than a vocabulary learned on Slovenian.

### The recipe is expensive

A batch of 8,000 sequences of 512 tokens for 500,000 steps is about two
trillion tokens seen, over 1,000 GPUs. A small model trained on a small
corpus repeats the data many times instead, and the paper does not say what
happens then.
