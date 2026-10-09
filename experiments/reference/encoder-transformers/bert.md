---
title: BERT
kind: algorithm
status: from-paper
summary: A stack of Transformer encoder layers over token, segment and learned absolute position embeddings, pretrained to recover 15% masked tokens and to tell whether two segments were adjacent.
variant_of:
part_of:
couples_to: [subword-tokenizers/wordpiece]
---

## Description

BERT is an encoder: it reads a whole text at once and gives every token one
vector that depends on the tokens to its left and to its right. The model is
a stack of identical Transformer layers. Each layer lets every token gather
information from every other token through attention, then passes the result
through a small feed-forward network. Position is told to the model only once,
at the input, by adding a learned vector for each of the 512 positions to the
token's embedding.

- **Pretraining teaches the encoder two things.** Masked language modelling
  hides 15% of the tokens and asks the model to name them from the rest of the
  text. Next-sentence prediction shows the model two segments and asks whether
  the second followed the first in the source document.
- **Fine-tuning adds one layer on top.** For a tagging task such as named
  entity recognition, a linear layer over every token vector gives the tag.
  For a sentence task, the vector of the leading `[CLS]` token stands for the
  whole input.
- **Two sizes were released.** Base has 12 layers of width 768 with 12
  attention heads and 110M parameters. Large has 24 layers of width 1024 with
  16 heads and 340M parameters.

## What the paper adds

Before BERT, pretrained language models read text in one direction, so a
token's vector could only see what came before it. The paper's claim is that
masking makes a bidirectional encoder trainable: the model cannot cheat by
copying the token it is asked to predict, because the token is hidden. The
paper also fixes the input convention that later encoders kept, a `[CLS]`
token in front, a `[SEP]` token between segments, and a segment embedding that
tells the two apart.

## Why it matters here

Every encoder in this topic is a change to BERT, and the Slovenian encoders
that benchmark results are usually reported with, CroSloEngual BERT and the
multilingual BERT, are BERT itself trained on other text. An information
extraction model that tags spans reads the per-token vectors BERT produces,
so BERT is the baseline design any small Slovenian encoder is measured
against.

## Facts

- **Position encoding**: learned absolute, one vector per position added to the token embedding at the input, 512 positions
- **Attention span**: every token attends to every token of the input
- **Block normalisation**: post-norm, a layer norm after each of the two residual additions
- **Activation**: GELU in a feed-forward network of inner width four times the hidden width
- **Pretraining objective**: masked language modelling on 15% of the tokens, plus next-sentence prediction
- **Tokenizer**: WordPiece, 30,000 pieces
- **Context length**: 512 tokens
- **Input**: `[CLS]`, segment A, `[SEP]`, segment B, `[SEP]`, with a learned segment embedding for A and for B
- **Sizes**: Base 12 layers, width 768, 12 heads, 110M parameters; Large 24 layers, width 1024, 16 heads, 340M parameters
- **Masking**: of the 15% chosen positions, 80% become `[MASK]`, 10% a random token, 10% stay unchanged
- **Training data**: BooksCorpus, 800M words, and English Wikipedia, 2,500M words
- **Optimizer**: Adam, learning rate 1e-4, warmup over 10,000 steps then linear decay, weight decay 0.01, dropout 0.1
- **Training length**: 1,000,000 steps of 256 sequences, about 40 passes over the data, 90% of the steps at length 128 and the last 10% at 512
- **Prediction head**: a dense layer with GELU and layer norm, then a softmax over the vocabulary whose weights are the input embeddings

## Sources

- **Paper**: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, Devlin et al., NAACL 2019, arXiv:1810.04805
- **Paper**: Attention Is All You Need, Vaswani et al., NeurIPS 2017, arXiv:1706.03762, the encoder layer BERT stacks
- **Code**: none in this repository; the reference implementation is the `BertModel` of the HuggingFace transformers library
- **Example**: `experiments/reference/encoder-transformers/examples.py`
- **Paper vs code**: not checked; the project has no encoder code yet

## Algorithm

```algorithm
\caption{BERT encoder: from token ids to one vector per token}
\Require token ids $x_1, \dots, x_n$ with $n \le 512$, segment ids $s_i \in \{A, B\}$, embedding tables $E_{tok}$, $E_{seg}$, $E_{pos}$, $L$ layers of width $d$ with $A$ heads
\Ensure hidden states $h^L_1, \dots, h^L_n$
\State $h^0_i \gets \mathrm{Dropout}(\mathrm{LN}(E_{tok}[x_i] + E_{seg}[s_i] + E_{pos}[i]))$ for every $i$
\For{$\ell = 1, \dots, L$}
  \State $a \gets$ multi-head attention over $h^{\ell-1}$ with $A$ heads: for each head, $\mathrm{softmax}_j(q_i \cdot k_j / \sqrt{d/A})\, v_j$ summed over all $j$
  \State $u \gets \mathrm{LN}(h^{\ell-1} + \mathrm{Dropout}(W_o\, a))$
  \State $f \gets W_2\, \mathrm{GELU}(W_1 u + b_1) + b_2$ with inner width $4d$
  \State $h^\ell \gets \mathrm{LN}(u + \mathrm{Dropout}(f))$
\EndFor
\Return $h^L$
```

- **1**: Each token starts as the sum of three learned vectors: what the token is, which of the two segments it belongs to, and where it stands. The sum passes through a layer norm, which rescales every vector to a fixed spread. Position enters only here, so a later layer knows where a token is only through what this sum carried.
- **2-3**: Every layer begins with attention. A token builds a query from its own vector and compares it with a key built from every other token, so the weights say how much each token matters to it. The weighted sum of value vectors is what the token gathers. The division by the square root of the head width keeps the comparisons from growing with the width.
- **4**: The gathered information is added back to the token's own vector, and the sum is normalised. The norm sits after the addition, which is the post-norm arrangement.
- **5-6**: A feed-forward network, the same for every token, widens the vector four times, applies the GELU activation and narrows it back. Its output is added to the token's vector and normalised again. This is where most of the model's parameters sit.
- **7**: After the last layer every token has a vector that reflects the whole input. A tagging model reads these vectors one by one.

```algorithm
\caption{BERT pretraining example: masked language modelling with next-sentence prediction}
\Require a corpus of documents split into sentences, WordPiece vocabulary $V$, maximum length $T = 512$
\Ensure one input sequence, its masked-token targets and its next-sentence label
\State draw a run of sentences $A$ from one document; with probability $0.5$ let $B$ be the sentences that follow $A$ (label IsNext), otherwise sentences drawn from another document (label NotNext); trim so that $|A| + |B| \le T - 3$
\State $x \gets$ `[CLS]` $A$ `[SEP]` $B$ `[SEP]`, with segment id A for the first part and B for the second
\State $M \gets$ 15% of the token positions of $x$, drawn at random once, excluding the three special tokens
\ForAll{$i \in M$}
  \State with probability $0.8$ replace $x_i$ by `[MASK]`; with $0.1$ by a random token of $V$; with $0.1$ keep $x_i$
\EndFor
\State $h \gets$ Algorithm 1 on $x$
\State $L_{MLM} \gets \sum_{i \in M}$ cross-entropy of $\mathrm{softmax}(E_{tok}\, \mathrm{LN}(\mathrm{GELU}(W h_i)) + b)$ against the original $x_i$
\State $L_{NSP} \gets$ cross-entropy of a two-way classifier on $\tanh(W_p h_1)$ against the label
\Return $L_{MLM} + L_{NSP}$
```

- **1**: Half of the examples pair a segment with its true continuation, half with text from elsewhere. The label is what next-sentence prediction has to recover. The two segments together fill the 512 positions, minus the three special tokens.
- **2**: The two segments are laid out with the special tokens between them. The segment ids are the only thing that tells the model where A ends and B begins, besides the separator.
- **3-5**: The positions to hide are chosen once per example, when the corpus is prepared, and the same mask is reused on every pass over the data. Most hidden positions get the `[MASK]` token. One in ten gets a random token, and one in ten is left as it is, so that the model cannot assume an unmasked token is right and keeps a useful vector for every position. `[MASK]` never appears in fine-tuning, which is the mismatch the two smaller cases soften.
- **6**: The encoder of Algorithm 1 runs over the corrupted sequence.
- **7**: Only the chosen positions are scored. The head transforms the token vector once more and scores it against the input embedding table, so the output and input vocabularies share one set of vectors.
- **8**: The vector of the leading `[CLS]` token is passed through a small pooling layer and a two-way classifier that says IsNext or NotNext.
- **9**: The two losses are added with equal weight.

## Worked example

The topic's example script holds no BERT example. BERT's two mechanisms that
later entries change, the absolute position table and the fixed mask, have
nothing to compute: the position of token $i$ is row $i$ of the table, and the
mask is a draw. The DeBERTa V2 entry shows the table that replaces the first,
and the RoBERTa entry explains what replaces the second.

## Limits

### Position is told once and absolutely

The model learns one vector per position and never sees the distance between
two tokens directly. A pattern learned at positions 10 and 12 has to be
learned again at 300 and 302, and no input can be longer than the 512 rows of
the table.

### The mask is fixed when the corpus is prepared

Every pass over the data hides the same tokens of the same sentence, so forty
passes give forty views of one masking. RoBERTa draws the mask anew each time.

### Next-sentence prediction costs half the input

Half of the examples pair a segment with unrelated text, so the second half of
those inputs teaches the encoder little about language. RoBERTa measured no
loss from dropping the task.

### The `[MASK]` token is never seen downstream

Pretraining sees `[MASK]` at 12% of the positions and fine-tuning never does,
so the model is trained on an input distribution it will not meet. The 10%
random and 10% unchanged cases reduce the mismatch but do not remove it.

### Short training at full length

Nine tenths of the steps run on sequences of 128 tokens, so the position
vectors for positions 129 to 512 are trained on one tenth of the data.
