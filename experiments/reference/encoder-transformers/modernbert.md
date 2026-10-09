---
title: ModernBERT
kind: algorithm
status: from-paper
summary: RoBERTa's objective on an encoder rebuilt with rotary position, pre-norm blocks without bias, a gated GELU feed-forward, and attention that alternates every third layer between the whole text and a 128-token window, trained on two trillion tokens to a context of 8,192.
variant_of: encoder-transformers/roberta
part_of:
couples_to: [subword-tokenizers/byte-level-bpe]
---

## Description

ModernBERT keeps RoBERTa's training objective, masked language modelling
without next-sentence prediction, and rebuilds the encoder with the parts
decoder language models settled on between 2019 and 2024. The released base
model has 22 layers of width 768 and 149M parameters, the large model 28
layers of width 1024 and 395M. Both read up to 8,192 tokens.

- **Position is rotary.** Instead of a learned table, each query and key
  vector is rotated by an angle that grows with the token's position, so the
  dot product of two of them depends only on their distance. No position
  enters the input.
- **Attention alternates.** Every third layer, starting with the first, lets
  every token see the whole text. The other layers let a token see 64 tokens
  to each side. The global layers use a rotary base of 160,000, the local
  layers 10,000.
- **The block is pre-norm, without bias.** The layer norm sits before the
  attention and before the feed-forward part, not after the residual
  addition. Linear layers and norms have no bias term except the final
  decoder. The feed-forward part is a gated GELU: the input is projected to
  two halves, one passes through GELU and multiplies the other.
- **Training is long and staged.** Two trillion tokens of English web text,
  code and scientific text, 30% of tokens masked, a batch warmed up to 4,608
  sequences, a learning rate held constant for most of the run and decayed at
  the end. The first 1.7 trillion tokens run at length 1,024, then 250
  billion at 8,192, then 50 billion of decay.
- **Padding is removed.** Sequences in a batch are concatenated into one
  stream and attention is told where each begins and ends, so no compute is
  spent on padding tokens.

## What the paper adds

The paper's claim is that encoders had stood still since RoBERTa while
decoders moved, and that porting the decoder changes gives a faster and
longer encoder with equal or better scores. The ablations are reported in
aggregate; the paper's own comparison is against BERT, RoBERTa, DeBERTa V3
and the 2024 retrieval encoders on classification, retrieval and code, with
a memory and speed table at lengths 512 and 8,192. It also sizes the layers
so that every matrix tiles the tensor cores of common GPUs, which it calls
hardware-aware design.

## Why it matters here

ModernBERT is the current encoder design a small Slovenian model would
copy if it copied one. Its parts are separable: rotary position, the local
window, the pre-norm block and the gated feed-forward can each be taken or
left, and the entry names the line of the algorithm each one sits on. The
8,192-token context matters for document-level extraction, where a relation
can span paragraphs. The tokenizer is a byte-level BPE learned on English,
so the tokenizer sweep's finding applies, and a Slovenian vocabulary would
be the first thing to swap.

## Facts

- **Position encoding**: rotary, applied to queries and keys in every layer; base 160,000 in global layers, 10,000 in local layers; no position at the input
- **Attention span**: every third layer, counting from the first, over the whole input; the others over a window of 64 tokens to each side
- **Block normalisation**: pre-norm, a layer norm before the attention and before the feed-forward part, without bias; a norm after the embedding and after the last layer; the first layer's attention norm removed
- **Activation**: gated GELU, two halves of width 1,152 in base and 2,624 in large
- **Pretraining objective**: masked language modelling on 30% of the tokens, no next-sentence prediction
- **Tokenizer**: byte-level BPE, 50,368 pieces, a modified OLMo vocabulary with 83 unused slots
- **Context length**: 8,192 tokens
- **Bias**: none in linear layers or norms, except the decoder of the prediction head
- **Sizes**: base 22 layers, width 768, 12 heads, 149M parameters; large 28 layers, width 1024, 16 heads, 395M parameters
- **Training data**: 2 trillion tokens of mostly English web text, code and scientific text
- **Optimizer**: StableAdamW, AdamW with per-parameter update clipping; learning rate 8e-4 for base held constant after a 3 billion token warmup, 5e-4 for large after 2 billion; decay by $1 - \sqrt{t}$ over the last 50 billion tokens
- **Batch size**: warmed up from 768 to 4,608 sequences over 50 billion tokens for base, 448 to 4,928 over 10 billion for large
- **Stages**: 1.7 trillion tokens at length 1,024 with rotary base 10,000; 250 billion at length 8,192 at learning rate 3e-4 with the global base raised to 160,000; 50 billion of decay
- **Initialisation**: Megatron's for base; large starts from base's weights
- **Prediction head**: dense, GELU, layer norm, then a decoder tied to the input embeddings with a bias
- **Pooling**: mean of the token vectors for sentence tasks

## Sources

- **Paper**: Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference, Warner et al., 2024, arXiv:2412.13663
- **Paper**: RoFormer: Enhanced Transformer with Rotary Position Embedding, Su et al., 2021, arXiv:2104.09864, the rotary position
- **Paper**: GLU Variants Improve Transformer, Shazeer, 2020, arXiv:2002.05202, the gated GELU
- **Paper**: On Layer Normalization in the Transformer Architecture, Xiong et al., ICML 2020, arXiv:2002.04745, the pre-norm block
- **Code**: none in this repository; the reference implementation is the `ModernBertModel` of the HuggingFace transformers library and the authors' FlexBERT repository
- **Example**: `experiments/reference/encoder-transformers/examples.py`
- **Paper vs code**: not checked against project code. The paper says a 128-token window; the released code gives a token 64 neighbours to each side, 129 tokens in all. The first layer is global, since the code takes a layer whose index from zero is divisible by three.

## Difference from predecessor

- **Line**: 1, 3 to 6 and the return of BERT's Algorithm 1, against Algorithm 1 here
- **Predecessor**: token, segment and absolute position embeddings summed at the input; attention over every token; a layer norm after each residual addition; a GELU feed-forward with bias
- **Here**: the token embedding alone; rotary position applied to queries and keys; attention over the whole text in every third layer and over a 64-token window either side otherwise; a layer norm before each part; a gated GELU without bias
- **Change**: rotary position, alternating global and local attention, pre-norm blocks without bias, a gated GELU, and a context of 8,192.
- **Effect**: Distance rather than place reaches every layer, two thirds of the layers cost linear time in the length, and the block trains stably at depth 22 to 28.

## Algorithm

```algorithm
\caption{ModernBERT encoder: from token ids to one vector per token}
\Require token ids $x_1, \dots, x_n$ with $n \le 8192$, embedding table $E_{tok}$, $L$ layers of width $d$, window $w = 64$, rotary bases $\theta_g = 160000$ and $\theta_l = 10000$
\Ensure hidden states $h^L$
\Changed \State $h^0 \gets \mathrm{LN}(E_{tok}[x])$, with no position or segment embedding and no bias in the norm
\For{$\ell = 1, \dots, L$}
  \Changed \State $g \gets$ true if $(\ell - 1) \bmod 3 = 0$, the global layers
  \Changed \State $u \gets h^{\ell-1}$ if $\ell = 1$, else $\mathrm{LN}(h^{\ell-1})$
  \Changed \State $q, k, v \gets W_{qkv} u$ without bias; rotate $q_i$ and $k_i$ by the angle of position $i$ with base $\theta_g$ if $g$ else $\theta_l$
  \Changed \State $a_i \gets \mathrm{softmax}_j(q_i \cdot k_j / \sqrt{d / A})\, v_j$ over all $j$ if $g$, else over $j$ with $|i - j| \le w$
  \State $h \gets h^{\ell-1} + W_o\, a$
  \Changed \State $(p, z) \gets W_{in}\, \mathrm{LN}(h)$ split into two halves; $h^\ell \gets h + W_{out}(\mathrm{GELU}(p) \odot z)$
\EndFor
\Changed \Return $\mathrm{LN}(h^L)$
```

- **1**: The input is the token embedding alone, normalised once. Nothing says where a token stands; that enters through the rotation in every layer.
- **2**: Layers 1, 4, 7 and so on are global, eight of the 22 in base. The rest are local.
- **3**: Pre-norm: the layer norm is applied to the input of the attention, not to its output. The first layer skips it because the embedding was just normalised.
- **4**: Queries, keys and values come from one projection without bias. The rotation turns each pair of coordinates of $q_i$ and $k_i$ by an angle proportional to $i$, with a frequency per pair set by the base. Because a rotation by $i$ followed by one by $-j$ is a rotation by $i - j$, the dot product of a rotated query and key depends on their distance only. The larger base of the global layers makes the angles grow more slowly, so distances up to 8,192 stay apart.
- **5**: A global layer compares every token with every token, at quadratic cost in the length. A local layer compares a token with its 64 neighbours on each side, at linear cost. A token's view widens by 64 to each side per local layer and covers the text at every global one.
- **6**: The attention output is added to the layer input. No norm follows.
- **7**: The feed-forward part is gated. One projection produces two halves; GELU of the first half multiplies the second, and a projection narrows the product back. The halves are 1,152 wide in base, so the part has about as many parameters as BERT's feed-forward at inner width 3,072 would at a smaller depth.
- **8**: A final norm closes the stack, which pre-norm blocks need because no norm follows the last residual addition.

## Worked example

The example script lists every layer of the base model with the rule the
released code applies: a layer whose index from zero is divisible by three
is global, the rest are local with 64 tokens to each side.

![Attention span and rotary base of every layer of ModernBERT-base](tables/modernbert-attention-layout.csv)

Eight of the 22 layers see the whole text, 14 see a window of 129 tokens.
At length 8,192 a local layer compares each token with 129 others where a
global layer compares it with 8,192, so the local layers are about one
sixtieth of the attention cost at full length.

## Limits

### The vocabulary is English

The 50,368-piece byte-level BPE was learned on English and code. Slovenian
text is spelled in more tokens per word than with a vocabulary learned on
Slovenian, which lengthens every input and spends the context on pieces.

### The recipe assumes two trillion tokens

The staged schedule, the batch warmup and the constant learning rate are
tuned to a run that no Slovenian corpus can fill. What the design gives at
a hundredth of the data is not in the paper.

### Ablations are not broken out

The paper reports the parts together and names the earlier work each came
from. Which of the five changes carries the gain over RoBERTa is not
separated, which is the question an experiment here would ask.

### Local layers see 129 tokens

A dependency longer than 64 tokens is carried only by the eight global
layers, or by stacking local layers. The paper measures retrieval and
classification, not a task where one long-range link decides the answer.

### Speed needs the unpadded kernels

The memory and speed figures rely on Flash Attention with variable-length
sequences. Without those kernels the alternating pattern still saves
compute in principle but not the measured factor.
