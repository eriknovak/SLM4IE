---
title: MorphBPE
kind: algorithm
status: checked
summary: Character BPE that may not join two characters across a morpheme boundary while it trains; encoding afterwards is ordinary BPE.
variant_of: subword-tokenizers/character-bpe
---

## Description

Byte-pair encoding (BPE) builds a vocabulary by joining the most frequent
pair of adjacent symbols, again and again. MorphBPE adds one rule to the
training. Every word found in a morpheme lexicon is cut into its morphemes
first, and pairs are counted inside each piece only. A pair that spans a
boundary never gets a count, so it can never be joined.

- **The result is a plain list of merges.** Nothing about morphemes is stored
  in the trained tokenizer.
- **Encoding is ordinary BPE.** A new word is cut by replaying the merges, and
  no lexicon is consulted. The boundary is therefore learned, not guaranteed.
- **The lexicon decides what is protected.** Here it is derived from Sloleks,
  a lexicon of Slovenian word forms, and splits a form into stem and ending.
  The sweep can add splits of derived words when that source is configured.

## What the paper adds

The paper proposes two scores for how well tokens follow morphemes. Both are
used by this project's tokenizer sweep.

- **Morphological consistency** asks whether words that share a morpheme also
  share a token, and the reverse. It is reported as an F1 score.
- **Morphological edit distance** counts the edits needed to turn the token
  cuts of a word into its morpheme cuts. Lower is closer.

The paper tests English, Russian, Hungarian and Arabic, on language models of
300 million and 1 billion parameters. It reports lower training loss and
faster convergence than plain BPE. Those are the paper's results, not this
project's.

## Why it matters here

Slovenian marks case, number and gender with endings, so one noun has many
forms that share a stem. A tokenizer that keeps the stem whole lets a small
model reuse what it learned about `hiš` across `hiša`, `hiše` and `hišami`.

The sweep trains MorphBPE and Character BPE at the same vocabulary size. The
two differ in one training line, so a gap in their scores is the effect of
the boundary rule alone.

## Facts

- **Base unit**: Unicode characters
- **Chunk while training**: morpheme for a word in the lexicon, whole word otherwise
- **Picks next token by**: highest pair count, a tie going to the pair that sorts last
- **Encoding**: standard BPE merges, no lexicon
- **Needs lexicon**: training only
- **Setting**: vocabulary size, which includes the special tokens
- **Repeatable**: yes, training has no random step
- **Export**: a standard `tokenizer.json`, loaded as a fast tokenizer
- **Registry key**: `morphbpe`

## Sources

- **Paper**: MorphBPE: A Morpho-Aware Tokenizer Bridging Linguistic Complexity for Efficient LLM Training Across Morphologies, arXiv:2502.00894
- **Code**:
  - `slm4ie/tokenizers/backends/morph_bpe.py`
  - `slm4ie/tokenizers/bpe_core.py`
  - `slm4ie/tokenizers/backends/_hf_base.py`
- **Lexicon**: `slm4ie/tokenizers/morphology.py`
- **Example**: `experiments/reference/subword-tokenizers/examples.py`
- **Paper vs code**:
  - The paper does not say what happens to a word without a segmentation. The code keeps it whole.
  - The paper's pseudocode speaks of byte pairs but starts from characters. The code works on characters.
  - The paper does not say how a tie between two pairs is broken. The code takes the pair that sorts last.

## Difference from predecessor

- **Line**: 3
- **Predecessor**: $(c_1) \gets (w)$
- **Here**: $(c_1, \dots, c_k) \gets L(w)$ if $w \in L$, otherwise $(w)$
- **Change**: pairs are counted inside morphemes, not inside whole words.
- **Effect**: An ending is never joined to its stem during training, so the vocabulary fills with stems and endings.

## Algorithm

```algorithm
\caption{MorphBPE training}
\Require corpus $D$, morpheme lexicon $L$, vocabulary size $B$, special tokens $S$
\Ensure vocabulary $V$, ordered merges $M$
\State $f(w) \gets$ count of each word $w$ in $D$ \Comment{`slm4ie/tokenizers/backends/morph_bpe.py:53-55`}
\ForAll{words $w$}
  \Changed \State $(c_1, \dots, c_k) \gets L(w)$ if $w \in L$, otherwise $(w)$ \Comment{`slm4ie/tokenizers/backends/morph_bpe.py:59-60`}
  \State add $f(w)$ to $f(c_i)$ for every $i$ \Comment{`slm4ie/tokenizers/backends/morph_bpe.py:61-62`}
\EndFor
\State $V \gets$ sorted characters of all chunks, $M \gets ()$ \Comment{`slm4ie/tokenizers/bpe_core.py:116-117`}
\State $n(a, b) \gets \sum_c f(c)$ times the places where $a$ is followed by $b$ inside $c$ \Comment{`slm4ie/tokenizers/bpe_core.py:120-123`}
\While{$|V| < B - |S|$ and some pair has a count} \Comment{`slm4ie/tokenizers/bpe_core.py:126`}
  \State $(a^*, b^*) \gets$ the pair with the highest $n$ \Comment{`slm4ie/tokenizers/bpe_core.py:127`}
  \State join $a^* b^*$ in every chunk that holds it, and update $n$ \Comment{`slm4ie/tokenizers/bpe_core.py:129-135`}
  \State append $(a^*, b^*)$ to $M$, add the joined symbol to $V$ \Comment{`slm4ie/tokenizers/bpe_core.py:138-141`}
\EndWhile
\Return a standard BPE model holding $S$, $V$ and $M$ \Comment{`slm4ie/tokenizers/backends/_hf_base.py:118`}
```

- **1**: Words are counted once, so the rest works on distinct words with a weight.
- **2-4**: Each word becomes chunks. Line 3 is where the lexicon enters. A word it does not know stays one chunk.
- **5**: The vocabulary starts as every character seen, sorted so that two runs give the same result.
- **6**: Pairs are counted inside chunks only. This is what keeps a boundary from being crossed.
- **7-10**: Each round joins one pair everywhere and records it. The loop ends when the vocabulary is full or no pair is left.
- **11**: Only the vocabulary and the merges leave training. The lexicon is not part of the model.

```algorithm
\caption{BPE encoding, shared with Character BPE}
\Require text $t$, vocabulary $V$, ordered merges $M$
\Ensure the tokens of $t$
\State normalise $t$ to the composed Unicode form \Comment{`slm4ie/tokenizers/backends/_hf_base.py:119`}
\State split $t$ into words and runs of punctuation \Comment{`slm4ie/tokenizers/backends/_hf_base.py:120`}
\ForAll{pieces $p$}
  \State $s \gets$ characters of $p$, a character outside $V$ becoming the unknown token
  \While{some adjacent pair of $s$ is in $M$}
    \State join the adjacent pair that comes earliest in $M$
  \EndWhile
\EndFor
\Return every $s$ in order \Comment{`slm4ie/tokenizers/backends/_hf_base.py:42`}
```

- **1-2**: The text is cleaned and cut into pieces. No lexicon is consulted.
- **3-6**: Merges are replayed in the order they were learned. These lines run inside the HuggingFace `tokenizers` library, so they cite no project code.
- **7**: What is left when no learned pair remains is the token sequence.

## Worked example

The corpus holds six forms of two nouns: `hiša` (house) five times, `hiše`
four, `hišo` three, `miza` (table) four, `mize` three, `mizo` twice. The
lexicon splits each into stem and ending, such as `hiš` and `a`. The
vocabulary size is 15: eight characters, the unknown token, and room for six
merges. Both tokenizers are the project's own backends.

![Merges learned on the toy corpus, and the word hiša after each](tables/morphbpe-merges.csv)

Character BPE spends its fifth and sixth merge on joining an ending to the
stem. MorphBPE has no pair left after four merges, because every chunk is
already one symbol.

![Encoding of two corpus forms and two forms the tokenizers never saw](tables/morphbpe-encoding.csv)

MorphBPE keeps the boundary after the stem in all four forms. It still cuts
the ending `ami` into `a` and `mi`. The merge of `m` and `i` was learned
inside the stem `miz`, and encoding applies it wherever the pair appears.

## Limits

### Words outside the lexicon are unprotected

A word the lexicon does not know is one chunk, so its merges may cross any
boundary. The rule covers only as much text as the lexicon does.

### Boundaries are learned, not guaranteed

Encoding knows no morphemes. A merge learned inside one stem fires inside an
ending of another word, as `m` and `i` do in the worked example.

### The lexicon marks inflection

The Sloleks-derived splits separate stem from ending. Prefixes and suffixes
that build new words stay inside the stem unless the second, derivational
source is configured.

### The vocabulary may not fill

Short chunks run out of pairs sooner than whole words. On the toy corpus
training stops with two of the six merge slots unused.

### Training and encoding read text differently

Training counts words from the raw text and treats each punctuation mark as
its own word. Encoding first normalises the text and keeps a run of
punctuation together. A character written in decomposed form, or a run such
as `...`, is therefore seen differently by the two.
