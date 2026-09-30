---
title: Character BPE
kind: algorithm
status: checked
summary: Joins the most frequent pair of adjacent characters inside words, again and again, until the vocabulary is full.
---

## Description

Byte-pair encoding (BPE) builds a vocabulary of word pieces from a corpus. It
starts from single characters and repeatedly joins the pair of adjacent
symbols that occurs most often, until the vocabulary holds the wanted number
of pieces. Each join is recorded as a merge. A new word is then cut by
replaying the merges in the order they were learned, so a rare word is spelled
from pieces the model has seen.

- **The unit is the Unicode character.** Text is first put into the composed
  Unicode form, so `š` is always one character and never `s` plus an accent.
  The byte-level variant in the sweep starts from bytes instead.
- **Pairs never span a word.** The corpus is split into words and runs of
  punctuation, and pairs are counted inside those pieces only.
- **It is the predecessor of MorphBPE.** MorphBPE changes one line of the
  training and keeps everything else, including encoding.

## Why it matters here

The tokenizer sweep trains Character BPE and MorphBPE at the same vocabulary
size. Both work on characters and share the encoder, so the gap between their
scores is the effect of MorphBPE's morpheme rule alone. The byte-level BPE in
the sweep is the usual baseline of language-model pretraining, but it differs
from MorphBPE in its base unit as well, so it cannot isolate the rule.

## Facts

- **Base unit**: Unicode characters
- **Chunk while training**: whole word
- **Picks next token by**: highest pair count, a tie going to the pair whose symbols entered the vocabulary first
- **Encoding**: standard BPE merges, no lexicon
- **Needs lexicon**: no
- **Setting**: vocabulary size, which includes the special tokens; the characters seen in training are always kept in full, even when they alone exceed it
- **Repeatable**: yes, training has no random step
- **Export**: a standard `tokenizer.json`, loaded as a fast tokenizer
- **Registry key**: `charbpe`

## Sources

- **Paper**: Neural Machine Translation of Rare Words with Subword Units, arXiv:1508.07909
- **Code**:
  - `slm4ie/tokenizers/backends/char_bpe.py`
  - `slm4ie/tokenizers/backends/_hf_base.py`
- **Example**: `experiments/reference/subword-tokenizers/examples.py`
- **Paper vs code**:
  - The paper ends every word with a marker `</w>`, so a piece at the end of a word is a different symbol from the same letters inside a word. The code adds no marker, so one piece serves both positions.
  - The paper's listing breaks a tie between equally frequent pairs by whichever the dictionary yields first, which it does not fix. The code takes the pair whose symbols entered the vocabulary first. The example script checks this on three ties.
  - The paper counts words split on whitespace. The code first normalises the text and separates runs of punctuation from words.
  - The paper has no special tokens. The code counts them toward the vocabulary size, and it keeps every character seen in training even when the characters alone exceed the size.
  - The merge loop itself runs inside the HuggingFace `tokenizers` library. The code sets its inputs and reads its output, so the loop lines cite no project code, and the example script checks each learned merge against a replay of the pair counts.

## Algorithm

```algorithm
\caption{Character BPE training}
\Require corpus $D$, vocabulary size $B$, special tokens $S$
\Ensure vocabulary $V$, ordered merges $M$
\State $f(w) \gets$ count of each word $w$ in $D$, after normalising $D$ and splitting it into words and runs of punctuation \Comment{`slm4ie/tokenizers/backends/char_bpe.py:39-40`}
\ForAll{words $w$}
  \State $(c_1) \gets (w)$
  \State add $f(w)$ to $f(c_1)$
\EndFor
\State $V \gets S$ followed by the sorted characters of all chunks, $M \gets ()$ \Comment{`slm4ie/tokenizers/backends/char_bpe.py:43`}
\State $n(a, b) \gets \sum_c f(c)$ times the places where $a$ is followed by $b$ inside $c$
\While{$|V| < B$ and some pair has a count} \Comment{`slm4ie/tokenizers/backends/char_bpe.py:42`}
  \State $(a^*, b^*) \gets$ the pair with the highest $n$, a tie going to the pair whose symbols entered $V$ first
  \State join $a^* b^*$ in every chunk that holds it, and update $n$
  \State append $(a^*, b^*)$ to $M$, add the joined symbol to $V$
\EndWhile
\Return a standard BPE model holding $S$, $V$ and $M$ \Comment{`slm4ie/tokenizers/backends/char_bpe.py:46-47`}
```

- **1**: Words are counted once, so the rest works on distinct words with a weight. The text is read exactly as encoding reads it, through the same normaliser and splitter.
- **2-4**: Every word is one chunk. The loop is written out so that the line numbers match MorphBPE, which replaces line 3 with a cut at morpheme boundaries.
- **5**: The special tokens take the first places. The characters follow in sorted order, so two runs give the same result.
- **6**: Pairs are counted inside chunks only, so no merge ever spans two words.
- **7-10**: Each round joins one pair everywhere and records it. The loop ends when the vocabulary is full or no pair is left. The count is the only criterion, so an ending is joined to its stem as soon as that pair is frequent.
- **11**: The trained model is the vocabulary and the merges, saved as a standard file.

```algorithm
\caption{BPE encoding}
\Require text $t$, vocabulary $V$, ordered merges $M$
\Ensure the tokens of $t$
\State normalise $t$ to the composed Unicode form \Comment{`slm4ie/tokenizers/backends/char_bpe.py:39`}
\State split $t$ into words and runs of punctuation \Comment{`slm4ie/tokenizers/backends/char_bpe.py:40`}
\ForAll{pieces $p$}
  \State $s \gets$ characters of $p$, a character outside $V$ becoming the unknown token
  \While{some adjacent pair of $s$ is in $M$}
    \State join the adjacent pair that comes earliest in $M$
  \EndWhile
\EndFor
\Return every $s$ in order \Comment{`slm4ie/tokenizers/backends/_hf_base.py:42`}
```

- **1-2**: The text is cleaned and cut into pieces the same way as in training.
- **3-6**: Merges are replayed in the order they were learned. These lines run inside the HuggingFace `tokenizers` library, so they cite no project code.
- **7**: What is left when no learned pair remains is the token sequence. Each unknown character stays its own unknown token.

## Worked example

The corpus holds six forms of two nouns: `hiša` (house) five times, `hiše`
four, `hišo` three, `miza` (table) four, `mize` three, `mizo` twice. The
vocabulary size is 15: eight characters, the unknown token, and room for six
merges. The tokenizer is the project's own backend, and the table is written
by the example script, which also checks that every merge is a most frequent
pair.

![Merges learned on the toy corpus, the count that chose each, and the word hiša after it](tables/character-bpe-merges.csv)

The first four merges rebuild the two stems, because every form of a noun
contributes to the pairs inside its stem. Three of the six choices are ties.
The tie at merge 1 falls to `hi` over `iš`, and the tie at merge 6 to `hiše`
over `miza`, in both cases the pair whose first symbol entered the vocabulary
first. Merges 5 and 6 then join an ending to a stem, since after the stems are
whole those are the most frequent pairs left. This is the step MorphBPE
forbids.

![Encoding of two corpus forms, two forms the tokenizer never saw, and a form with an unseen character](tables/character-bpe-encoding.csv)

The unseen form `hišami` is cut as `hiša`, `m`, `i`. The piece `hiša` was
learned as a whole word and now fires at the start of a longer one, because no
marker tells a piece where a word ends. The letter `q` never occurred in the
corpus, so it becomes the unknown token.

## Limits

### Frequency alone decides

A pair is joined because it is common, not because it forms a unit of
meaning. In the worked example the stem and the ending of `hiša` become one
piece at merge 5, and the stem then cannot be shared with the ending `ami`.

### A piece does not know where a word ends

Without an end-of-word marker, a piece learned as a whole word also matches
the start of a longer word. Byte-level BPE in the sweep marks the start of a
word with a leading space instead, and the paper marks the end.

### An unseen character is lost

A character absent from the training sample cannot be spelled and becomes the
unknown token. A byte-level tokenizer never has this case, since every
character is a sequence of bytes it knows.

### The vocabulary can overshoot

The characters seen in training are all kept, so a sample with more distinct
characters than the vocabulary size produces a larger vocabulary than asked
and learns no merge at all.
