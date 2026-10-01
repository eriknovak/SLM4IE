---
title: Byte-level BPE
kind: algorithm
status: checked
summary: Character BPE run over the bytes of the text instead of its characters, with the space before a word kept as part of the word.
variant_of: subword-tokenizers/character-bpe
---

## Description

Byte-pair encoding (BPE) builds a vocabulary of word pieces by joining the
most frequent pair of adjacent symbols, again and again, until the vocabulary
is full. Byte-level BPE runs the same procedure over the UTF-8 bytes of the
text rather than over its characters. Every byte is shown as one printable
character, so the pieces are still strings, and the 256 byte symbols are the
starting vocabulary. A new word is cut by replaying the merges in the order
they were learned.

- **Nothing is ever unknown.** Every character is a sequence of bytes, and all
  256 byte symbols are in the vocabulary, so any text can be spelled.
- **A space belongs to the word after it.** Text is cut into words, numbers and
  runs of punctuation, and each keeps the space in front of it, shown as `Ġ`.
  A piece therefore knows whether it begins a word, and decoding returns the
  exact text, spaces included.
- **A Slovenian letter outside ASCII is two symbols.** `š`, `č` and `ž` are
  two bytes each, shown as `Å¡`, `Äį` and `Å¾`, so a merge is spent on
  rebuilding each before a stem can form.

## What the paper adds

The GPT-2 report introduced this scheme to get the open vocabulary of bytes
without the long byte sequences that plain byte BPE produces. Its one rule is
to stop merges from crossing between letters, digits and punctuation, with the
space before a word as the only exception. The report's language-model results
are not this project's.

## Why it matters here

Byte-level BPE is the usual tokenizer of language-model pretraining, used by
GPT-2, RoBERTa and ModernBERT, and it is the fallback path of MorphPiece in
this project's sweep. Its scores set the bar that every morpheme-aware
tokenizer has to beat. It differs from MorphBPE in its base unit as well as in
the boundary rule, so the pair that isolates the rule is Character BPE against
MorphBPE, not this entry.

## Facts

- **Base unit**: UTF-8 bytes, each shown as one printable character
- **Chunk while training**: whole word, with the space before it
- **Picks next token by**: highest pair count, a tie going to the pair whose symbols entered the vocabulary first
- **Encoding**: standard BPE merges, no lexicon
- **Needs lexicon**: no
- **Setting**: vocabulary size, which includes the special tokens and the 256 byte symbols
- **Unknown token**: none, every character is spelled from its bytes
- **Round trip**: decoding returns the exact text, spaces included
- **Repeatable**: yes, training has no random step
- **Export**: a standard `tokenizer.json`, loaded as a fast tokenizer
- **Registry key**: `bpe`

## Sources

- **Paper**: Language Models are Unsupervised Multitask Learners, Radford et al., OpenAI, 2019, section 2.2
- **Paper**: Neural Machine Translation with Byte-Level Subwords, arXiv:1909.03341, which names the method and studies it on its own
- **Code**:
  - `slm4ie/tokenizers/backends/hf_bpe.py`
  - `slm4ie/tokenizers/backends/_hf_base.py`
- **Example**: `experiments/reference/subword-tokenizers/examples.py`
- **Paper vs code**:
  - The report cuts text with one pattern that keeps letters, digits and punctuation apart and attaches the space before a word to it. The code uses the same pattern through the byte-level pre-tokenizer of the HuggingFace `tokenizers` library.
  - The report puts no space before the first word of a text, so that word is a different symbol from the same word inside a text. The code adds the space, so every word at the start of a line is spelled like one inside it.
  - The report has one special token. The code adds the sweep's five and counts them toward the vocabulary size.
  - The report does not say how a tie between equally frequent pairs is broken. The code takes the pair whose symbols entered the vocabulary first. The example script checks this on five ties.
  - The merge loop itself runs inside the HuggingFace `tokenizers` library. The code sets its inputs and reads its output, so the loop lines cite no project code, and the example script checks each learned merge against a replay of the pair counts.
  - The report returns tokens only. The code also returns the span of text each token covers, with the space before a word left out, which is what the morpheme metrics read.

## Difference from predecessor

- **Line**: 1 and 5
- **Predecessor**: words are counted after normalising the text and splitting it into words and punctuation, and $V$ starts from the characters seen
- **Here**: words keep the space before them and are written as their bytes, and $V$ starts from the 256 byte symbols
- **Change**: the symbols are bytes, and the space before a word is part of it.
- **Effect**: Every character can be spelled, so there is no unknown token, and a piece knows whether it begins a word.

## Algorithm

```algorithm
\caption{Byte-level BPE training}
\Require corpus $D$, vocabulary size $B$, special tokens $S$
\Ensure vocabulary $V$, ordered merges $M$
\Changed \State $f(w) \gets$ count of each word $w$ in $D$, each word keeping the space before it and written as its byte symbols \Comment{`slm4ie/tokenizers/backends/hf_bpe.py:40`}
\ForAll{words $w$}
  \State $(c_1) \gets (w)$
  \State add $f(w)$ to $f(c_1)$
\EndFor
\Changed \State $V \gets S$ followed by the 256 byte symbols, $M \gets ()$ \Comment{`slm4ie/tokenizers/backends/hf_bpe.py:45`}
\State $n(a, b) \gets \sum_c f(c)$ times the places where $a$ is followed by $b$ inside $c$
\While{$|V| < B$ and some pair has a count} \Comment{`slm4ie/tokenizers/backends/hf_bpe.py:42-43`}
  \State $(a^*, b^*) \gets$ the pair with the highest $n$, a tie going to the pair whose symbols entered $V$ first
  \State join $a^* b^*$ in every chunk that holds it, and update $n$
  \State append $(a^*, b^*)$ to $M$, add the joined symbol to $V$
\EndWhile
\Return a standard BPE model holding $S$, $V$ and $M$, with a byte-level decoder \Comment{`slm4ie/tokenizers/backends/hf_bpe.py:41-49`}
```

- **1**: Words are counted once, so the rest works on distinct words with a weight. A space is put before every line, and the text is cut into words, numbers and runs of punctuation, each keeping the space before it. Every piece is then written as the printable form of its UTF-8 bytes. The text is not normalised, so a letter written in decomposed form keeps its bytes.
- **2-4**: Every word is one chunk, as in Character BPE.
- **5**: The special tokens take the first places and the 256 byte symbols follow, whether or not a byte occurs in the corpus. The order fixes how a tie is broken.
- **6**: Pairs are counted inside chunks only. The space before a word is inside the chunk, so a merge may absorb it, and no merge ever spans two words.
- **7-10**: Each round joins one pair everywhere and records it. The loop ends when the vocabulary is full or no pair is left.
- **11**: The trained model is the vocabulary and the merges. The decoder turns the printable byte symbols back into text.

```algorithm
\caption{Byte-level BPE encoding}
\Require text $t$, vocabulary $V$, ordered merges $M$
\Ensure the tokens of $t$, each with the span of $t$ it covers
\Changed \State put a space before $t$ and split it into words, numbers and runs of punctuation, each keeping its space \Comment{`slm4ie/tokenizers/backends/hf_bpe.py:40`}
\Changed \State write every piece as its byte symbols
\ForAll{pieces $p$}
  \State $s \gets$ the byte symbols of $p$
  \While{some adjacent pair of $s$ is in $M$}
    \State join the adjacent pair that comes earliest in $M$
  \EndWhile
\EndFor
\Return every $s$ in order, each with the span of $t$ it covers \Comment{`slm4ie/tokenizers/backends/_hf_base.py:54-55`}
```

- **1-2**: The text is cut into pieces the same way as in training. No piece can hold a byte outside $V$, so there is no unknown token.
- **3-6**: Merges are replayed in the order they were learned. These lines run inside the HuggingFace `tokenizers` library, so they cite no project code.
- **7**: What is left when no learned pair remains is the token sequence. Each token also reports which characters of the text it covers, with the space before a word left out. That span is how the morpheme metrics find its cuts without reading the byte symbols.

## Worked example

The corpus holds six forms of two nouns: `hiša` (house) five times, `hiše`
four, `hišo` three, `miza` (table) four, `mize` three, `mizo` twice. The
vocabulary size is 263: the 256 byte symbols, the unknown token the sweep
reserves, and room for six merges. The tokenizer is the project's own backend,
and the tables are written by the example script, which also checks that every
merge is a most frequent pair. In the tables `Ġ` is the space before a word
and `Å¡` the two bytes of `š`.

![Merges learned on the toy corpus, the count that chose each, and the word hiša after it](tables/byte-level-bpe-merges.csv)

The first four merges rebuild the stem `hiš` with its space: the two letters
`hi`, the two bytes of `š`, the space onto `hi`, and the whole. Five of the
six choices are ties, and each falls to the pair whose symbols entered the
vocabulary first. At merge 1 the pair `hi` beats the pair of space and `h`,
because the printable ASCII bytes enter the vocabulary before the space
marker. Merges 5 and 6 rebuild `miz`, but no slot is left to join its space,
so every form of `miza` starts with a lone space token.

![Encoding of two corpus forms, three forms the tokenizer never saw, and the cuts the morpheme metrics read from the spans](tables/byte-level-bpe-encoding.csv)

The letter `q` never occurred in the corpus and is spelled as its own byte, not
as an unknown token. The unseen word `čas` costs five tokens for three letters,
because `č` is two bytes and neither pair was ever merged. The morpheme
metrics see `č` as one piece, since both byte tokens report the same span. The
form `mize` shows a cut after `m` that is not in the tokens. The lone space
token reports the span of the first letter, and the metrics read a boundary
from it.

## Limits

### A letter outside ASCII costs two symbols

`š`, `č` and `ž` are two bytes each, so a merge is spent on rebuilding every
such letter before a stem that holds it can form. Until that merge is learned,
a token can be half a character and cannot be read on its own.

### The same word is spelled two ways

A word keeps the space before it, so `hiša` after a space and `hiša` right
after an opening bracket or a quote are different symbol strings and may cut
differently. Character BPE does not see the space and spells both alike.

### A lone space token adds a false cut

When the space before a word was never merged into a piece, the token holding
it reports the span of the word's first letter. The morpheme metrics then read
a boundary after that letter, as `mize` shows in the worked example.

### The vocabulary never falls below 256 symbols

All byte symbols are kept, including the many that only ever appear as the
second byte of a letter. A small vocabulary spends a fixed share on symbols
that never stand alone in Slovenian text.
