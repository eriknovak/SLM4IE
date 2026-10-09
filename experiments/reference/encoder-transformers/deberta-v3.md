---
title: DeBERTa V3
kind: algorithm
status: from-paper
summary: DeBERTa V2 trained as the discriminator of replaced token detection, with a half-depth generator whose token embeddings it shares while its own gradient is kept from them.
variant_of: encoder-transformers/deberta-v2
part_of:
couples_to: [subword-tokenizers/unigram]
---

## Description

DeBERTa V3 keeps the V2 encoder and changes what it is trained to do. Instead
of predicting masked tokens, the encoder is trained as a discriminator: a
smaller generator fills the masked positions with plausible tokens, and the
discriminator says, for every position, whether the token there is the
original or a replacement. This is the replaced token detection of ELECTRA.
The encoder learns from every position rather than from the 15% that were
masked.

- **Two models train side by side.** The generator is a V2 encoder of the
  same width and half the depth, trained with masked language modelling. The
  discriminator is the full V2 encoder with a two-way classifier on every
  token. Only the discriminator is kept.
- **Token embeddings are shared with a one-way gradient.** The discriminator
  uses the generator's embedding table plus a residual table of its own. The
  generator's table is updated by the masked-token loss only. The residual
  table is updated by the detection loss only. After training the two tables
  are added and saved as the discriminator's.
- **Everything else is V2.** Disentangled attention with bucketed distances,
  shared projections, the 128,000-piece SentencePiece vocabulary, and the
  160GB RoBERTa data. The released base and large models carry no
  convolution.

## What the paper adds

The paper's diagnosis is a tug of war. In ELECTRA the two models share one
embedding table, and the two losses pull it in opposite directions: masked
language modelling draws similar tokens together, detection pushes them apart
to make replacements easier to spot. Separate tables train faster but give a
worse discriminator, because the generator's embeddings are the better ones.
Gradient-disentangled embedding sharing keeps the generator's table as the
base and lets the discriminator add only a correction, which the paper shows
trains as fast as separate tables and scores higher than either.

## Why it matters here

Replaced token detection is the training objective that lets a small encoder
make the most of a small corpus, since every token of every sequence yields a
training signal. For Slovenian, where the corpus is a hundredth of what the
English models saw, that is the lever this entry offers. The objective is
independent of the encoder's layers, so it can be paired with any entry of
this topic, and an experiment that varies the objective would name this entry.

## Facts

- **Position encoding**: relative, bucketed, as in DeBERTa V2
- **Attention span**: every token attends to every token, as in BERT
- **Block normalisation**: post-norm, as in BERT
- **Activation**: GELU, as in BERT
- **Pretraining objective**: replaced token detection; a generator trained with masked language modelling on 15% of the tokens fills the masks, the discriminator classifies every token as original or replaced, loss $L_{MLM} + 50\, L_{RTD}$
- **Tokenizer**: SentencePiece, 128,000 pieces, the DeBERTa V2 vocabulary; mDeBERTa uses 250,000 pieces
- **Context length**: 512 tokens
- **Generator**: same width as the discriminator, half its depth
- **Embedding sharing**: the discriminator's table is the generator's, with the gradient stopped, plus a residual table started at zero; after training the two are added and saved
- **Sizes**: Base 12 layers, width 768, 86M backbone and 98M embedding parameters; Large 24 layers, width 1024, 304M backbone; Small 6 layers, width 768, 44M; XSmall 12 layers, width 384, 22M
- **Training data**: 160GB, the RoBERTa sources; mDeBERTa on the 2.5TB CC100 multilingual corpus
- **Optimizer**: AdamW with $\beta_1 = 0.9$, $\beta_2 = 0.98$, learning rate 6e-4 for Base and Small and 3e-4 for Large, warmup over 10,000 steps, linear decay
- **Training length**: 500,000 steps of 8,192 sequences
- **Convolution**: none in the released base and large models

## Sources

- **Paper**: DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing, He, Gao and Chen, ICLR 2023, arXiv:2111.09543
- **Paper**: ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators, Clark, Luong, Le and Manning, ICLR 2020, arXiv:2003.10555, the replaced token detection objective
- **Code**: none in this repository; the reference implementation is the `DebertaV2Model` of the HuggingFace transformers library for the released discriminator, and the authors' DeBERTa repository for the training loop
- **Paper vs code**: not checked; the project has no encoder code yet. The released HuggingFace models are the discriminators alone, with the summed embedding table.

## Difference from predecessor

- **Line**: Algorithm 2 of RoBERTa's lineage, the pretraining step, against Algorithm 1 here
- **Predecessor**: one encoder, trained to name the 15% masked tokens
- **Here**: a half-depth generator names the masked tokens and fills them in; the encoder is trained to tell original tokens from the replacements, at every position, with a shared embedding table whose gradient flows one way
- **Change**: replaced token detection in place of masked language modelling, with gradient-disentangled embedding sharing.
- **Effect**: Every token of every sequence trains the encoder, and the embedding table is shaped by the generator alone with a correction learned by the discriminator.

## Algorithm

```algorithm
\caption{One training step of replaced token detection with gradient-disentangled embedding sharing}
\Require a sequence $x$ of $n$ tokens, generator $G$ with parameters $\theta_G$ and embedding table $E_G$, discriminator $D$ with parameters $\theta_D$ and residual embedding table $E_\Delta$ started at zero, weight $\lambda = 50$
\Ensure updated $\theta_G$, $E_G$, $\theta_D$, $E_\Delta$
\State $\tilde x_G \gets x$ with 15% of its positions $M$ masked, replaced as in BERT
\State $L_{MLM} \gets \sum_{i \in M}$ cross-entropy of $G(\tilde x_G)_i$ against $x_i$
\State update $\theta_G$ and $E_G$ by the gradient of $L_{MLM}$
\Changed \State $\tilde x_D \gets x$ with each $x_i$, $i \in M$, replaced by a token sampled from the distribution $G(\tilde x_G)_i$, without gradient
\Changed \State $E_D \gets \mathrm{sg}(E_G) + E_\Delta$
\Changed \State $L_{RTD} \gets \sum_{i=1}^{n}$ binary cross-entropy of $D(\tilde x_D)_i$ against $\mathbb{1}[\tilde x_{D,i} \ne x_i]$, with $D$ embedding its input through $E_D$
\Changed \State update $\theta_D$ and $E_\Delta$ by the gradient of $\lambda L_{RTD}$
\Return after the last step, save $E_G + E_\Delta$ as the discriminator's embedding table
```

- **1-2**: The generator sees BERT's corrupted input and is scored as BERT is, on the masked positions only. It is a V2 encoder of half the depth with the usual prediction head.
- **3**: The generator and its embedding table learn from the masked-token loss alone. This is the step that, in ELECTRA, also received the detection gradient.
- **4**: The discriminator's input is the original sequence with the masked positions filled by what the generator proposes. A proposal that happens to equal the original counts as original. Sampling is detached, so no gradient flows from the discriminator into the generator through the tokens.
- **5**: The discriminator's embedding table is the generator's, read through a stop-gradient so that its own loss cannot change it, plus a residual table of its own. At the first step the residual is zero and the two models embed tokens identically.
- **6**: Every position is scored, not only the masked ones. The label is whether the token differs from the original. A sequence of 512 tokens gives 512 training signals where masked language modelling gives about 77.
- **7**: The detection loss, weighted by 50 as in ELECTRA, updates the discriminator's layers and the residual embedding table. The generator's table receives nothing from it, which is what breaks the tug of war.
- **8**: What is released is the discriminator with one embedding table, the sum of the two, so a downstream user sees an ordinary encoder.

## Worked example

The topic's example script holds no V3 example. The training step has no
rule to tabulate: its inputs are draws from the generator, and its effect is
in the trained weights.

## Limits

### Two encoders in memory during training

The generator adds half an encoder's layers and a second forward pass per
step. The paper counts the cost of a step as the same across sharing methods,
but it is higher than masked language modelling alone.

### The weight 50 is inherited

$\lambda = 50$ is ELECTRA's value, taken over without a sweep. Whether a
smaller model or a smaller corpus wants another weight is not reported.

### Masked language modelling is lost as a probe

The released discriminator cannot fill in a blank, so a zero-shot use that
asks the model to name a masked word, which a masked language model can do,
is not available from this encoder.

### A generator the same width as a small model

For a small discriminator the half-depth generator is still a sizeable
model, and ELECTRA found that a generator too strong for its discriminator
hurts. The paper reports base and larger, not what happens below XSmall.
