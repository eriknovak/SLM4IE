---
title: DeBERTa
kind: algorithm
status: from-paper
summary: RoBERTa's recipe with attention that scores content and relative position separately, no position at the input, and absolute position added only in the last layers before the masked-token prediction.
variant_of: encoder-transformers/roberta
part_of:
couples_to: [subword-tokenizers/byte-level-bpe]
---

## Description

DeBERTa keeps RoBERTa's data, vocabulary and training recipe and changes how
the encoder handles position. In BERT and RoBERTa a token's vector is the sum
of what it is and where it is, and attention compares those sums. DeBERTa
keeps the two apart. Every token has a content vector, and every pair of
tokens has a relative position vector that depends only on how far apart they
are. Attention adds up three comparisons: content against content, content
against the other token's relative position, and this token's relative
position against the other's content.

- **Position is not added at the input.** The token embedding enters the
  first layer alone. Distance between tokens is read inside every layer from a
  shared table of relative position vectors with one row per distance up to
  512 either way.
- **Absolute position is added at the end.** Where a token stands in the
  sentence still matters for predicting a masked word, so the model adds the
  absolute position vectors to the states after the second-to-last layer and
  runs the last layer twice on that sum. The paper calls this the enhanced mask
  decoder.
- **Fine-tuning can add scale-invariant perturbation.** SiFT, used only for
  the 1.5B model, normalises the word embeddings before adding the adversarial
  noise of virtual adversarial training, so the noise is the same size for
  every token.

## What the paper adds

The paper argues that the position-to-content term, which earlier relative
position schemes dropped, matters: how much a token attends to a neighbour
depends on their distance and on which of the two carries the content. It
shows by ablation that removing either position term or the enhanced mask
decoder costs accuracy on every benchmark measured. The 1.5B-parameter model
was the first to pass the human baseline on SuperGLUE.

## Why it matters here

Slovenian word order is freer than English word order, and the case endings
carry what the position carries in English. An encoder that reads distance
rather than absolute place is a plausible fit for that, and this is the first
design in the lineage that does so. The relative position table is also the
part a small model can shrink or share across layers without touching the
rest. DeBERTa's released vocabulary is RoBERTa's byte-level BPE, so the
tokenizer finding carries over.

## Facts

- **Position encoding**: relative, a shared table with one row per signed distance up to 512, read inside every layer; no position at the input; absolute position added before the masked-token prediction only
- **Attention span**: every token attends to every token, as in BERT
- **Block normalisation**: post-norm, as in BERT
- **Activation**: GELU, as in BERT
- **Pretraining objective**: masked language modelling on 15% of the tokens, as in RoBERTa
- **Tokenizer**: byte-level BPE, 50,000 pieces, the GPT-2 vocabulary RoBERTa uses
- **Context length**: 512 tokens
- **Attention score**: content-to-content plus content-to-position plus position-to-content, divided by $\sqrt{3d}$ with $d$ the head width
- **Relative position table**: $2k$ rows of width $d$ with $k = 512$, shared by all layers, projected by their own query and key matrices
- **Enhanced mask decoder**: absolute position embeddings added after layer $L - 1$; the last layer is run twice with that sum as the query
- **Sizes**: Base 12 layers, width 768, 12 heads, 100M backbone parameters; Large 24 layers, width 1024, 16 heads; 1.5B with 48 layers, width 1536, 24 heads
- **Training data**: Wikipedia 12GB, BookCorpus 6GB, OpenWebText 38GB, Stories 31GB, 78GB after deduplication; the 1.5B model used 160GB
- **Optimizer**: Adam, learning rate 2e-4 for Base and Large, 1.5e-4 for 1.5B, warmup over 10,000 steps, linear decay, weight decay 0.01, dropout 0.1
- **Training length**: 1,000,000 steps of 2,000 sequences

## Sources

- **Paper**: DeBERTa: Decoding-enhanced BERT with Disentangled Attention, He, Liu, Gao and Chen, ICLR 2021, arXiv:2006.03654
- **Paper**: Self-Attention with Relative Position Representations, Shaw, Uszkoreit and Vaswani, NAACL 2018, arXiv:1803.02155, the relative position scheme DeBERTa extends
- **Code**: none in this repository; the reference implementation is the `DebertaModel` of the HuggingFace transformers library and the authors' DeBERTa repository
- **Example**: `experiments/reference/encoder-transformers/examples.py`
- **Paper vs code**: not checked against project code. The paper places the enhanced mask decoder "right after all the Transformer layers"; the authors' released code runs the last encoder layer twice over the states of layer $L - 1$ with the absolute position embeddings added to the query, which is what Algorithm 2 shows.

## Difference from predecessor

- **Line**: 1 of BERT's Algorithm 1 and line 3 of it, against Algorithm 1 here
- **Predecessor**: the token embedding, the segment embedding and the absolute position embedding are summed at the input, and attention compares those sums
- **Here**: the token embedding enters alone; attention adds a content-to-content, a content-to-position and a position-to-content score read from a shared table of relative distances; absolute position is added before the prediction head
- **Change**: disentangled attention over content and relative position, with absolute position moved from the input to the decoder.
- **Effect**: Every layer reads how far two tokens are apart, and where a token stands is used only to predict the masked word.

## Algorithm

```algorithm
\caption{Disentangled self-attention in one layer, one head}
\Require content states $H \in \mathbb{R}^{n \times d}$, relative position table $P \in \mathbb{R}^{2k \times d}$ shared by all layers with $k = 512$, content projections $W_{q,c}, W_{k,c}, W_{v,c}$, position projections $W_{q,r}, W_{k,r}$
\Ensure attention output $H_o \in \mathbb{R}^{n \times d}$
\State $Q_c \gets H W_{q,c}$, $K_c \gets H W_{k,c}$, $V_c \gets H W_{v,c}$
\Changed \State $Q_r \gets P W_{q,r}$, $K_r \gets P W_{k,r}$
\Changed \State $\delta(i, j) \gets 0$ if $i - j \le -k$; $2k - 1$ if $i - j \ge k$; $i - j + k$ otherwise
\Changed \State $\tilde A_{ij} \gets Q_{c,i} K_{c,j}^\top + Q_{c,i} K_{r,\delta(i,j)}^\top + K_{c,j} Q_{r,\delta(j,i)}^\top$ for every $i, j$
\Changed \State $H_o \gets \mathrm{softmax}(\tilde A / \sqrt{3d})\, V_c$
\Return $H_o$
```

- **1**: The content side is BERT's attention: a query, a key and a value for every token, from its content vector alone.
- **2**: The position side projects the shared table once per layer. Row $r$ of $Q_r$ and $K_r$ stands for the signed distance $r - k$, so the table holds every distance from $-k$ to $k - 1$. It is one table for the whole model, and each layer applies its own projections to it.
- **3**: The signed distance from token $i$ to token $j$ is turned into a row index. Distances beyond $k$ either way share the two outer rows.
- **4**: Three scores are added. The first asks how well the content of $j$ answers the content of $i$. The second asks how well the distance to $j$ answers the content of $i$. The third asks how well the content of $j$ answers the distance from $j$ back to $i$, which is why its index is $\delta(j, i)$. The fourth possible term, position against position, is left out because a distance compared with itself carries nothing.
- **5**: The sum of three dot products has three times the spread of one, so the scale is $\sqrt{3d}$ rather than BERT's $\sqrt{d}$. The paper reports this factor matters for stable training of large models.
- **6**: The output has the same shape as BERT's and feeds the same residual, norm and feed-forward steps.

```algorithm
\caption{Masked language modelling with the enhanced mask decoder}
\Require corrupted token ids $x$ with masked positions $M$, $L$ encoder layers of Algorithm 1, absolute position table $E_{pos}$, prediction head of BERT
\Ensure the masked-token loss
\Changed \State $h^0 \gets \mathrm{LN}(E_{tok}[x])$, with no position embedding
\State $h^{L-1} \gets$ layers $1$ to $L - 1$ of Algorithm 1 on $h^0$
\Changed \State $z \gets h^{L-1} + E_{pos}[1..n]$
\Changed \For{two passes}
  \Changed \State $z \gets$ layer $L$ with query from $z$ and key and value from $h^{L-1}$
\EndFor
\State $L_{MLM} \gets \sum_{i \in M}$ cross-entropy of the prediction head on $z_i$ against the original $x_i$
\Return $L_{MLM}$
```

- **1-2**: The encoder runs with content only. Through layer $L - 1$ the model knows distances between tokens and nothing about where in the sentence they stand.
- **3**: The absolute position vectors are added once, to the output of the second-to-last layer. The paper's example is "a new store opened beside the new mall" with both nouns masked: both follow "new" at the same distance, and only the absolute place tells the subject from the object.
- **4-5**: The last layer is applied twice. The query comes from the position-aware sum, the keys and values from the position-free states, and the second pass reads the first pass's output as its query. The weights of the two passes are the last layer's.
- **6-7**: The head and the loss are BERT's, on the decoder's output. In fine-tuning the decoder is dropped and the encoder's own last layer is used, so a downstream model never sees absolute position.

## Worked example

The topic's example script computes the row index of Algorithm 1, line 3,
for a set of distances, next to the bucketed index DeBERTa V2 replaces it
with. The table is shown and read in the DeBERTa V2 entry. For V1 every
distance from $-511$ to $511$ has its own row and the two outer rows gather
everything beyond.

## Limits

### The relative table is fixed at 512 rows

With $k = 512$ the table has 1,024 rows, and an input longer than 512 is
not reachable through the position vectors either way, since every distance
past 511 reads the same row. DeBERTa V2 compresses long distances onto a log
scale instead.

### Three scores cost three products

Each layer computes three attention matrices where BERT computes one, and the
position side needs two more projections. Inference is slower than
RoBERTa's at the same size, and the paper measures accuracy, not speed.

### Absolute position reaches only the decoder

A downstream task that depends on where a token stands in the input, rather
than on its distance to others, has to recover that from the relative pattern
alone, because the fine-tuned encoder never adds the absolute vectors.

### Dropping either position term costs accuracy

The paper's ablation shows every benchmark falls when the content-to-position
or the position-to-content term is removed, so neither can be dropped to save
compute without a loss.
