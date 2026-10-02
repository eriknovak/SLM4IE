---
title: Morfessor pre-tokenization
kind: algorithm
status: from-paper
summary: Every word is first cut into morph-like pieces by Morfessor, a segmenter that learns from the word list alone; BPE or Unigram then trains and encodes inside those pieces, so no token spans a learned boundary.
---

## Description

Morfessor is a family of unsupervised segmenters that cut words into morphs,
the surface pieces of a word that recur across many words, such as a stem or
an ending. It needs no lexicon. It learns from the list of distinct words in
a corpus by looking for the set of morphs that makes the whole list cheapest
to write down: a short lexicon of morphs, and a short description of every
word as a sequence of them. In this method Morfessor's cuts are used only as
pre-tokenization. Text is split into words, each word into morphs, and the
usual subword tokenizer, BPE or Unigram, is trained on and applied inside the
morphs. The subword tokenizer still controls the vocabulary size and still
splits a rare morph further.

- **No lexicon is needed.** Morfessor learns from the words themselves, so it
  works for a language without a morpheme resource.
- **A boundary holds at encoding time too.** MorphBPE forgets its lexicon
  after training. Here Morfessor runs on every new word as well, so a cut it
  makes is never crossed by a merge.
- **Morphs are statistical, not linguistic.** Morfessor finds pieces that
  recur, which often are morphemes and sometimes are not. Its cuts are a
  silver standard.

## What the paper adds

The paper separates tokenization into pre-tokenization, vocabulary
construction and segmentation, and changes the first step from words to
Morfessor morphs. It adds two more methods on top: a segmentation that picks
the subwords whose embeddings best match the word's embedding, and a distilled
subword bigram model that reproduces that segmentation without running
Morfessor or storing embeddings at inference. The table gives the paper's
numbers for Czech, the tested language closest to Slovenian, at a vocabulary
of 32,000. Those are the paper's results, not this project's.

| Pre-tokenization and vocabulary | Boundary precision on Czech | Rényi efficiency |
| ------------------------------- | --------------------------- | ---------------- |
| Words, BPE                      | 76.5                        | 0.419            |
| Words, Unigram                  | 84.3                        | 0.424            |
| Morfessor, BPE                  | 88.4                        | 0.449            |
| Morfessor, Unigram              | 89.4                        | 0.457            |

Boundary precision is the share of a tokenizer's cuts that fall on a morpheme
boundary of the SIGMORPHON 2022 test words. Rényi efficiency is how evenly
the tokens are used, a quantity shown elsewhere to track downstream quality.
Morfessor pre-tokenization raised both in all eight languages tested, English,
Finnish, German, Hungarian, Latvian, Russian, Turkish and Xhosa among them,
and lowered boundary recall. Part-of-speech tagging gained consistently.
Translation did not: with Morfessor the chrF score fell below the word-based
pipelines, by about 0.7 BLEU on average over 18 language pairs.

## Why it matters here

Slovenian has no gold morpheme segmentation. The sweep's lexicon is derived
from Sloleks and marks inflection only, so every morpheme-aware backend in the
sweep inherits that limit. Morfessor would give a boundary rule that needs no
lexicon at all, and its cuts can be compared with the Sloleks-derived ones to
see where the two disagree. Czech, the paper's closest language, gained the
most in boundary precision, which makes Slovenian a likely case.

## Facts

- **Base unit**: Unicode characters; Morfessor cuts characters, and the inner tokenizer may start from characters or bytes
- **Chunk while training**: a Morfessor morph, for every word
- **Picks next token by**: the inner tokenizer's rule inside each morph, highest pair count for BPE or likelihood for Unigram
- **Encoding**: Morfessor's most probable cut of the word, then the inner tokenizer inside each morph; or the distilled bigram model alone
- **Needs lexicon**: no; Morfessor learns from the distinct words of the corpus
- **Setting**: Morfessor's corpus weight, which trades fewer and longer morphs against more and shorter ones; how word counts are dampened, the paper keeps the defaults; the inner vocabulary size
- **Repeatable**: only with a fixed seed; Morfessor visits the words in a random order each pass
- **Export**: no standard file; a Morfessor model plus the inner tokenizer, or the bigram model after distillation
- **Registry key**: none, the method has no backend in this project

## Sources

- **Paper**: Lexically Grounded Subword Segmentation, Libovický and Helcl, EMNLP 2024, arXiv:2406.13560
- **Paper**: Morfessor 2.0: Python Implementation and Extensions for Morfessor Baseline, Virpioja, Smit, Grönroos and Kurimo, Aalto University 2013, ISBN 978-952-60-5501-5
- **Paper**: Unsupervised models for morpheme segmentation and morphology learning, Creutz and Lagus, ACM TSLP 2007, doi:10.1145/1187415.1187418
- **Code**: the Morfessor package at github.com/aalto-speech/morfessor and the authors' segmenter at github.com/ufal/legros; nothing in this repository
- **Paper vs code**: no code in this repository to compare

## Definition

Morfessor Baseline scores a lexicon of morphs $\Lambda$ with a cut for every
word by

$$\text{cost}(\Lambda) = L(\Lambda) + \alpha\, C(\Lambda)$$

where $L$ is the cost of writing the lexicon, the letters of every morph plus a
prior on how the counts are shared out, and $C$ is the cost of writing the
corpus as morphs, $-\sum_m f(m) \log \frac{f(m)}{N}$ with $f(m)$ the count of
morph $m$ and $N$ the total of all morph counts and word ends. The corpus
weight $\alpha$ is the one dial: a larger value makes every morph use cheap to
write and so yields fewer, longer morphs.

## Algorithm

```algorithm
\caption{Morfessor Baseline training by recursive splitting}
\Require distinct words $w$ with counts $f(w)$, corpus weight $\alpha$
\Ensure morph lexicon $\Lambda$ with counts, a cut for every word
\State $\Lambda \gets$ every word as one morph
\While{the last pass lowered the cost by more than a small share of the word count}
  \ForAll{words $w$ in random order}
    \State remove the morphs of $w$ from the counts
    \State score no cut and every cut position of $w$ by the cost with the two halves added
    \If{some cut scores at or below no cut}
      \State keep the best cut, add both halves with the count of $w$, and repeat from line 4 on each half
    \Else
      \State add $w$ back as one morph
    \EndIf
  \EndFor
\EndWhile
\Return $\Lambda$ and the cuts
```

- **1**: Training starts from whole words, so a word stays whole unless cutting it makes the list cheaper to write.
- **2**: A pass visits every word once. Passes stop when the cost no longer falls, a few thousandths of the word count being the paper's default margin.
- **3-5**: For one word, every binary cut is tried against leaving it whole. A cut pays off when the two halves are already in the lexicon or recur elsewhere, so the lexicon cost falls more than the corpus cost rises.
- **6-7**: A winning cut is applied and each half is treated as a word of its own. This recursion gives a word with several morphs.
- **8-9**: A word that is cheaper whole is kept whole.
- **10**: What leaves training is the lexicon of morphs with counts, used for new words below.

```algorithm
\caption{Encoding with Morfessor pre-tokenization}
\Require text $t$, morph lexicon $\Lambda$ with counts, inner tokenizer trained on morphs
\Ensure the tokens of $t$
\State split $t$ into words and punctuation
\ForAll{words $w$}
  \State cut $w$ into the sequence of lexicon morphs with the lowest total cost, where a morph costs $\log N - \log(f(m) + \delta)$ and a string outside the lexicon is allowed at a smoothing count $\delta$
  \ForAll{morphs $m$ of $w$}
    \State cut $m$ with the inner tokenizer
  \EndFor
\EndFor
\Return every token in order
```

- **1**: Words are cut from each other as the inner tokenizer would do on its own.
- **2-3**: Morfessor's Viterbi search, a dynamic programme over the positions of the word, picks the cut whose morphs are most probable. The smoothing count lets a word with an unseen piece still be cut, at a price.
- **4-5**: The inner tokenizer sees each morph as if it were a word. A frequent morph is one token; a rare one is split further by the merges or the unigram lattice.
- **6**: Tokens leave in text order, with every Morfessor boundary intact.

## Where it sits

Morfessor is a pre-tokenizer. In the three-step view of tokenization it
replaces the step that cuts text into words, and the next two steps, building
the vocabulary and segmenting, are those of Character BPE, Byte-level BPE,
WordPiece or Unigram unchanged. The paper also distils the whole pipeline into
a subword bigram model, which segments with a beam search over bigram
probabilities and needs neither Morfessor nor embeddings at inference.

## Limits

### Recall of boundaries falls

Morfessor cuts fewer times than a morphological reference does, and the inner
tokenizer cannot restore a boundary Morfessor missed. The paper reports lower
boundary recall with Morfessor pre-tokenization at every vocabulary size.

### Translation did not gain

On 18 translation pairs the word-based pre-tokenization beat Morfessor by
about 0.7 BLEU on average. The paper confirms gains on part-of-speech tagging
only and leaves the reason open.

### Two models at inference

A new word must pass through Morfessor and then the inner tokenizer, so the
tokenizer is two models and runs slower. The paper's distilled bigram model
removes this at a small cost in boundary precision.

### Morphs are not morphemes

A cut is made where it saves description length, not where a linguist would
cut. The corpus weight moves the cuts, and the paper used the defaults; what
the defaults do on Slovenian is not known.

### Training is slow and randomised

Each pass tries every cut of every word, and the words are visited in a random
order, so training a large word list takes time and a seed to repeat.
