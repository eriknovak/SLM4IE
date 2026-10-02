---
title: BPE-knockout
kind: algorithm
status: from-paper
summary: Byte-level BPE whose finished merge list is pruned afterwards; every merge that more often than not glues two morphemes together is removed, and the merges built on it take its parts instead.
variant_of: subword-tokenizers/byte-level-bpe
---

## Description

Byte-pair encoding (BPE) learns its merges from counts alone, so many merges
glue the end of one morpheme to the start of the next, where a morpheme is the
smallest meaning-bearing part of a word. BPE-knockout takes a finished BPE
tokenizer and a reference list of words already cut at their morpheme
boundaries. It runs the tokenizer over every reference word and blames the
merge that deleted each boundary the reference keeps. A merge that is blamed
in at least half of its uses is knocked out. Its token leaves the vocabulary,
but every later merge that consumed that token now consumes the token's parts,
so the rest of the tokenizer keeps working. The result is a byte-tuple
tokenizer: a merge may join more than two pieces at once.

- **Nothing is retrained.** The method edits an existing tokenizer and keeps
  most of its merges. The paper applied it to the tokenizer of a trained Dutch
  language model and continued that model's pretraining.
- **The lexicon is read once, after training.** Encoding replays the pruned
  merges and consults no lexicon.
- **The vocabulary shrinks.** Each knocked-out merge frees one slot. In the
  paper about a tenth of a 40,000-token vocabulary went.

## What the paper adds

The paper writes BPE as a graph of merges with two invariants, every merge has
two parents and every token is made by exactly one merge, and shows that the
second invariant is what lets a token be removed without breaking the merges
above it. It defines the blame ratio and reports the gains below on the CELEX
lexicons of English, Dutch and German, with word tokens weighted by their
frequency in the OSCAR web corpus. Those are the paper's results, not this
project's.

| Change against plain BPE, 40,000 tokens | Word types | Word tokens |
| --------------------------------------- | ---------- | ----------- |
| Morpheme-boundary precision             | about +10  | +5 to +45   |
| Morpheme-boundary recall                | about +25  | +50 to +60  |
| Morpheme-boundary F1                    | about +15  | +50 to +60  |
| Compound-boundary recall                | about +10  | +50 to +70  |

Two Dutch RoBERTa models were pretrained from scratch, one per tokenizer. The
plain BPE model kept a lower pseudo-perplexity and won most fine-tuning tasks
except named-entity recognition. Switching the plain BPE model's tokenizer to
the pruned one and pretraining for 5,000 more batches then beat both on the
token-level tasks, with part-of-speech accuracy rising from 93.8 to 96.0 and
named-entity F1 from 84.0 to 87.5, while sequence-level tasks stayed level.
Random dropout of merges, the usual regulariser, lowered precision instead.

## Why it matters here

Byte-level BPE is the baseline of the tokenizer sweep, and this method edits it
after training with no new training run. It enforces a morpheme boundary at a
third step: MorphBPE does so while training, MorphPiece while encoding, and
BPE-knockout once training is over. The Sloleks-derived lexicon the sweep
already holds is a flat list of forms cut into stem and ending, which is the
only input the blame step needs.

The paper's adaptation result also matters on its own. A Slovenian model that
already exists could keep its weights and get a tokenizer that follows
morphology, at the cost of a short continued pretraining.

## Facts

- **Base unit**: UTF-8 bytes, each shown as one printable character, as in the predecessor
- **Chunk while training**: whole word, with the space before it, as in the predecessor
- **Picks next token by**: highest pair count, followed by a pruning pass that removes every merge whose blame ratio is at least one half
- **Encoding**: the pruned merges replayed in their original priority, a merge joining all its parts at once; no lexicon
- **Needs lexicon**: once, after training, a word list cut at morpheme boundaries
- **Setting**: the blame threshold, one half in the paper; whether blame counts each reference word once or weights it by frequency, once in the paper
- **Vocabulary**: the predecessor's size less the knocked-out tokens, about 9 to 11% fewer in the paper
- **Round trip**: as the predecessor, since the byte symbols and the decoder are unchanged
- **Repeatable**: yes; the order in which merges are knocked out does not change the result
- **Export**: not a standard `tokenizer.json`, because a merge with more than two parts has no place in the HuggingFace BPE model; the paper's own package applies the pruned merges
- **Registry key**: none, the method has no backend in this project

## Sources

- **Paper**: BPE-knockout: Pruning Pre-existing BPE Tokenisers with Backwards-compatible Morphological Semi-supervision, Bauwens and Delobelle, NAACL 2024, doi:10.18653/v1/2024.naacl-long.324
- **Code**: the authors' package at github.com/bauwenst/BPE-knockout; nothing in this repository
- **Lexicon**: CELEX in the paper; here the Sloleks-derived cuts of `slm4ie/tokenizers/morphology.py` would serve
- **Paper vs code**: no code in this repository to compare

## Difference from predecessor

- **Line**: after line 11 of the training algorithm
- **Predecessor**: return a standard BPE model holding $S$, $V$ and $M$
- **Here**: blame every merge on the reference lexicon, knock out each merge with $R(m) \ge 1/2$, return the pruned tuple merges
- **Change**: a pruning pass guided by a morpheme lexicon follows training.
- **Effect**: Merges that glue morphemes together disappear, the merges above them take their parts, and the cuts fall on morpheme boundaries more often.

## Definition

For a merge $m$, let $N(m)$ be how many times it was applied while encoding
the reference words and $B(m)$ how many of those applications deleted a
boundary the reference keeps. The blame ratio is

$$R(m) = \frac{B(m)}{N(m)}$$

and a merge is knocked out when $R(m) \ge 1/2$. The threshold is a heuristic
the paper keeps fixed.

## Algorithm

```algorithm
\caption{Blame: choosing the merges to knock out}
\Require trained merges $M$ in priority order, reference words $W$ each cut at its morpheme boundaries
\Ensure the set $K$ of merges to remove
\State $N(m) \gets 0$ and $B(m) \gets 0$ for every merge $m$
\ForAll{words $w$ in $W$}
  \State encode $w$ by replaying $M$, and note for every deleted space which merge deleted it
  \ForAll{merges $m$ applied in $w$}
    \State $N(m) \gets N(m) + 1$
    \If{$m$ deleted a space that the reference cut of $w$ keeps}
      \State $B(m) \gets B(m) + 1$
    \EndIf
  \EndFor
\EndFor
\Return $K \gets \{m : B(m) / N(m) \ge 1/2\}$
```

- **1**: Every merge starts with a clean record.
- **2-3**: The tokenizer runs as usual over each reference word. Each space deleted between two characters was deleted by exactly one merge, so blame has one owner.
- **4-7**: A merge is counted every time it fires, and blamed when it fired across a boundary. It is enough that the merge glued one character of each morpheme; once a boundary is gone no later merge can restore it.
- **8**: The ratio compares harm to use. Each word counts once, so rare words weigh as much as common ones.

```algorithm
\caption{Knockout: removing one token from the merge graph}
\Require vocabulary $V$, for every token its forming merge $M_i(t)$ and the merges $M_o(t)$ that consume it, the token $t$ to remove
\Ensure the graph without $t$
\If{$t$ has no forming merge}
  \Return the graph unchanged, $t$ is a byte symbol
\EndIf
\State $m_{old} \gets$ the one merge in $M_i(t)$, with parts $p_1, \dots, p_n$
\ForAll{parts $p_i$}
  \State remove $m_{old}$ from $M_o(p_i)$
\EndFor
\ForAll{merges $q$ in $M_o(t)$}
  \State replace $t$ among the parts of $q$ by $p_1, \dots, p_n$, keeping the priority of $q$
  \ForAll{parts $p_i$}
    \State add $q$ to $M_o(p_i)$
  \EndFor
\EndFor
\State remove $t$ from $V$, and empty $M_i(t)$ and $M_o(t)$
\Return the graph
```

- **1-2**: A byte symbol was never merged into being, so it cannot be knocked out.
- **3-5**: The merge that made $t$ is cut loose from its parents.
- **6-9**: Every merge that needed $t$ now needs the parts of $t$ instead, at the same priority. This is the step that stops the removal from cascading: a merge above $t$ keeps working, only with more parts.
- **10-11**: The token is gone. Its parts and its consumers stay.

```algorithm
\caption{Byte-tuple encoding}
\Require word $w$ as byte symbols, pruned merges $M'$ in priority order
\Ensure the tokens of $w$
\While{some merge in $M'$ matches adjacent symbols of $w$}
  \State apply the matching merge of highest priority, joining all of its parts into one symbol
\EndWhile
\Return the symbols of $w$
```

- **1-2**: Encoding is the predecessor's with one change: a merge may have more than two parts and joins them in one step. Priorities are the original ones, so a word with no knocked-out merge is cut exactly as before.
- **3**: What is left when no merge applies is the token sequence.

## Limits

### The reference sets the ceiling

Blame finds only the boundaries the reference knows. The Sloleks-derived cuts
of this project mark the boundary between stem and ending, so a merge across a
prefix, a suffix that builds a new word, or a compound seam goes unblamed
unless the derivational source is configured.

### Recall rises, whole-word precision falls

More boundaries are kept, so more cuts land inside what the reference treats
as one word. The paper reports drops in compound-boundary precision of up to
10 points on frequency-weighted tokens, and the morpheme-boundary F1 still
rises.

### Gains from scratch were not uniform

The Dutch model pretrained from scratch with the pruned tokenizer lost to
plain BPE on perplexity and on most fine-tuning tasks. The clear gains came
from adapting an existing model, and the paper offers only a speculative
reason.

### The tokenizer leaves the standard format

A merge with three or more parts cannot be stored in a HuggingFace BPE model,
so the pruned tokenizer needs the authors' package or a custom encoder at
training and inference time.

### The threshold is a guess

One half was chosen as a plain heuristic. A different threshold prunes more
or less, and the paper's appendix explores it without fixing a better value.
