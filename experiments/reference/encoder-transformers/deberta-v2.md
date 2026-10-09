---
title: DeBERTa V2
kind: algorithm
status: from-paper
summary: DeBERTa with a 128,000-piece SentencePiece vocabulary learned on the training data, relative distances compressed onto log-spaced buckets, position projections shared with the content projections, and a convolution beside the first layer.
variant_of: encoder-transformers/deberta
part_of:
couples_to: [subword-tokenizers/unigram]
---

## Description

DeBERTa V2 is the architecture the 900M and 1.5B DeBERTa models were trained
with, released with the authors' code rather than a paper of its own. It
keeps disentangled attention and the enhanced mask decoder and changes four
things around them.

- **A larger vocabulary, learned on the training data.** 128,000 pieces from
  SentencePiece replace the 50,000 pieces of GPT-2's byte-level BPE. The
  embedding table grows to almost half the parameters of the base model.
- **Relative distances are bucketed.** Distances up to 128 either way keep a
  row each. Longer distances share rows spaced on a log scale, up to 255 rows
  each way, so 512 rows cover what V1 covered with 1,024 and a longer input
  still finds a row.
- **The position projections are the content projections.** The query and
  key matrices that V1 kept apart for the relative position table are the
  ones used for content, which removes two matrices per layer.
- **A convolution beside the first layer.** A one-dimensional convolution of
  width three over the input embeddings is added to the output of the first
  Transformer layer, so that the first layer's output carries the three-token
  neighbourhood of every token directly. The authors call this n-gram induced
  input encoding, nGiE.

## What the paper adds

There is no V2 paper. The DeBERTa paper's section on the 1.5B model describes
the shared projections and the convolution with an ablation on base models
showing the sharing costs no accuracy. The repository's release notes add the
vocabulary and the buckets. The DeBERTa V3 paper names V2 as the architecture
it inherits.

## Why it matters here

Two of V2's changes are the ones a Slovenian model would weigh first. The
vocabulary is learned on the training text with SentencePiece, which is the
Unigram model of this project's tokenizer sweep, so a Slovenian vocabulary
drops in without changing the encoder. The bucketed distances are what makes
a context past 512 tokens possible for the disentangled design. The
convolution is a cheap way to give the first layer the character of a
three-token window, which for an inflected language puts a word and its
neighbours' endings together.

## Facts

- **Position encoding**: relative, bucketed; distances up to 128 keep their own row, longer ones share rows on a log scale, 512 rows in all; absolute position added before the masked-token prediction only
- **Attention span**: every token attends to every token, as in BERT
- **Block normalisation**: post-norm, as in BERT, with a layer norm on the relative position table
- **Activation**: GELU, as in BERT
- **Pretraining objective**: masked language modelling on 15% of the tokens, as in RoBERTa
- **Tokenizer**: SentencePiece, 128,000 pieces learned on the training data
- **Context length**: 512 tokens in training; the bucketed table reaches any distance
- **Position buckets**: 256 per side; the scale is stretched so that distance 511 reaches the last bucket
- **Projections**: the query and key matrices of the content are applied to the relative position table too
- **Convolution**: kernel 3 over the input embeddings, GELU, added to the first layer's output and normalised; released in the XLarge and XXLarge models
- **Sizes**: XLarge 24 layers, width 1536, 24 heads, 710M parameters; XXLarge 48 layers, width 1536, 24 heads, 1.32B parameters
- **Training data**: 160GB, the RoBERTa sources
- **Optimizer**: as in DeBERTa

## Sources

- **Paper**: DeBERTa: Decoding-enhanced BERT with Disentangled Attention, He, Liu, Gao and Chen, ICLR 2021, arXiv:2006.03654, section 5.1 and appendix on the 1.5B model, for the shared projections and the convolution
- **Paper**: DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing, He, Gao and Chen, ICLR 2023, arXiv:2111.09543, section 4, for the vocabulary and data V2 and V3 share
- **Paper**: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer, Raffel et al., 2020, arXiv:1910.10683, whose log-spaced relative position buckets V2 adopts
- **Code**: none in this repository; the reference implementation is the `DebertaV2Model` of the HuggingFace transformers library, whose `make_log_bucket_position` the example script ports
- **Example**: `experiments/reference/encoder-transformers/examples.py`
- **Paper vs code**: the release notes say "log bucket similar to T5". The released code keeps exact distances up to half the bucket count and places the rest with $\lceil \log(|r| / 128) / \log(511 / 128) \cdot 127 \rceil + 128$, which the worked example computes. T5 uses a fixed number of exact buckets and a different scale.

## Difference from predecessor

- **Line**: 2, 3 and the first layer of Algorithm 1 of DeBERTa
- **Predecessor**: the position table has $2k = 1024$ rows, each distance up to 512 its own row, projected by its own query and key matrices; the first layer is a plain Transformer layer
- **Here**: 512 rows reached through log-spaced buckets, projected by the content's own matrices; a width-three convolution of the input embeddings is added to the first layer's output
- **Change**: log-bucketed relative distances, shared position projections, a convolution beside the first layer, and a 128,000-piece SentencePiece vocabulary.
- **Effect**: Any distance finds a row at half the table size, two projections per layer disappear, and the first layer sees every token's three-token neighbourhood.

## Algorithm

```algorithm
\caption{DeBERTa V2 encoder: convolution beside the first layer and bucketed relative attention}
\Require token ids $x_1, \dots, x_n$, embedding table $E_{tok}$, relative position table $P \in \mathbb{R}^{2b \times d}$ with $b = 256$ buckets per side, stretch $k = 512$, $L$ layers whose attention follows DeBERTa Algorithm 1 with the changes below
\Ensure hidden states $h^L$
\State $h^0 \gets \mathrm{LN}(E_{tok}[x])$
\Changed \State $h^1 \gets \mathrm{LN}(\mathrm{layer}_1(h^0) + \mathrm{GELU}(\mathrm{Conv}_3(h^0)))$
\For{$\ell = 2, \dots, L$}
  \Changed \State $Q_r \gets \mathrm{LN}(P) W_{q,c}$, $K_r \gets \mathrm{LN}(P) W_{k,c}$, with the content's own matrices of layer $\ell$
  \Changed \State $r \gets i - j$, $m \gets b / 2$; $\beta(r) \gets r$ if $|r| \le m$, else $\mathrm{sign}(r) \cdot (\lceil \log(|r| / m) / \log((k - 1) / m) \cdot (m - 1) \rceil + m)$
  \Changed \State $\delta(i, j) \gets \beta(i - j) + b$, clamped to $[0, 2b - 1]$
  \State $h^\ell \gets$ DeBERTa Algorithm 1 with these $Q_r$, $K_r$ and $\delta$, then residual, norm and feed-forward as in BERT
\EndFor
\Return $h^L$
```

- **1**: As in DeBERTa, the token embedding enters alone and no position is added at the input.
- **2**: The first layer runs as in DeBERTa. Beside it a convolution of kernel three slides over the input embeddings, so each output mixes a token with its left and right neighbour. The two results are added and normalised together. The convolution runs on the input only, so layers 2 to $L$ do not have it.
- **3-4**: In every layer the relative position table is first normalised and then projected by the same query and key matrices the layer uses for content. The two position-only matrices of DeBERTa are gone.
- **5**: A distance of 128 or less is its own bucket. A longer distance is mapped onto a log scale that reaches bucket 255 at distance 511, so distances from 129 to 511 share 127 buckets, with a bucket covering more distances the further out it lies. The sign is kept, so left and right neighbours never share a row.
- **6**: The bucket is shifted by $b$ to index the table, and clamped, so a distance beyond 511 reads the outermost row rather than failing.
- **7-8**: The attention itself is DeBERTa's three-term score with the scale $\sqrt{3d}$, and everything after it is BERT's layer.

## Worked example

The example script ports the bucket rule of the released code to plain Python
and applies it to a set of distances, next to the row DeBERTa V1 would read
with $k = 512$. The last column counts how many distances from 0 to 511 land
in the same row.

![Row of the relative position table read for a distance between two tokens, DeBERTa V1 against V2](tables/deberta-v2-relative-buckets.csv)

Up to distance 128 the two agree except for the offset of the table. From
129 on, V2 starts merging: 150 lands in bucket 143, 200 in 169 with two
neighbours, and 400 in 233 with four. V1 keeps every distance up to 511
apart and sends 512, 600 and 1000 to the same outer row. V2 sends them to
its outer row too, since the table ends at bucket 255, but a model trained
with a stretch $k$ larger than 512 would keep them apart.

## Limits

### The convolution is a release detail

The released base and large V3 models carry no convolution, only the V2
XLarge and XXLarge do. Which sizes it helps is not reported.

### Half the parameters of a small model are embeddings

At width 768 the 128,000-piece table holds 98M parameters against 86M in the
layers. A small Slovenian model would set the vocabulary size by the
tokenizer sweep, not by this release.

### Long distances share a row from 129 on

A bucket near 400 covers five distances, so two tokens 398 and 402 apart look
the same to the position side. What this costs at a context of 512 is not
measured; it is the price of reaching beyond.
