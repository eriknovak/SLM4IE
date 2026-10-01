---
title: WordPiece
kind: algorithm
status: checked
summary: Character BPE training with a marker on every character inside a word, keeping only the vocabulary; a new word is cut by taking the longest piece in the vocabulary from each position.
variant_of: subword-tokenizers/character-bpe
---

## Description

WordPiece is the tokenizer of BERT. It learns a vocabulary of word pieces from
a corpus and then cuts a new word greedily: from the start of the word it takes
the longest piece the vocabulary holds, then the longest piece from where that
one ended, and so on. A piece that continues a word carries the prefix `##`,
so `miz` and `##e` spell `mize` and the prefix tells the reader where the word
began.

- **Training is Character BPE with a marked alphabet.** The library this
  project uses trains the vocabulary by joining the most frequent pair of
  adjacent symbols, again and again, exactly as byte-pair encoding (BPE) does.
  The only difference is that a character inside a word is a different symbol
  from the same character at its start. The merges are thrown away afterwards.
- **Encoding does not replay merges.** The vocabulary alone decides the cut,
  through the longest-match rule. Two pieces that were never joined in
  training can still sit side by side in a token sequence.
- **A word is cut whole or not at all.** When no piece in the vocabulary fits
  at some position, the entire word becomes the unknown token, even the part
  that was already matched.

## What the paper adds

The 2012 paper introduced the method for Japanese and Korean voice search,
where no spaces mark the words. Its rule picks the pair of pieces whose join
most raises the likelihood of the corpus under a unigram model, which favours
a pair whose two halves are rare on their own. BERT took the vocabulary this
produces and fixed the greedy encoder, the `##` prefix and the limit of 100
characters per word in its released code. The paper's rule is not what this
project's trainer runs; see Paper vs code.

## Why it matters here

BERT and the encoders built on it, among them multilingual BERT and
CroSloEngual BERT, cut Slovenian text this way, so WordPiece is the tokenizer
of the encoder models Slovenian benchmarks are usually reported with. It is
also the one tokenizer in the sweep whose cut at encoding time comes from a
rule other than merge replay. The longest-match
rule takes a whole inflected form whenever the vocabulary holds it and falls
back to a stem only for a form it lacks, so WordPiece shows whether the cut
rule, and not only the vocabulary, moves the morpheme scores.

## Facts

- **Base unit**: Unicode characters, a character inside a word being a different symbol from the same character at its start
- **Chunk while training**: whole word
- **Picks next token by**: highest pair count, a tie going to the pair whose symbols entered the vocabulary first
- **Encoding**: longest piece in the vocabulary from each position, no merges, no lexicon
- **Needs lexicon**: no
- **Setting**: vocabulary size, which includes the special tokens; every character seen in training is kept, once as a word start and once more with the prefix when it also occurs inside a word
- **Unknown token**: one for the whole word when any part of it is not in the vocabulary, or when the word has more than 100 characters
- **Punctuation**: every mark is its own word, so `...` is three words
- **Repeatable**: the vocabulary is the same on each run of the toy corpus; the prefixed characters are numbered in a different order each run, so a tie between two pairs of prefixed characters may fall differently
- **Export**: a standard `tokenizer.json`, loaded as a fast tokenizer
- **Registry key**: `wordpiece`

## Sources

- **Paper**: Japanese and Korean Voice Search, Schuster and Nakajima, ICASSP 2012, doi:10.1109/ICASSP.2012.6289079, which introduces the method and its likelihood rule
- **Paper**: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, arXiv:1810.04805, section 3, whose released code fixes the greedy encoder, the `##` prefix and the 100-character limit
- **Paper**: Google's Neural Machine Translation System, arXiv:1609.08144, section 4.1, which describes the model between the two
- **Code**:
  - `slm4ie/tokenizers/backends/hf_wordpiece.py`
  - `slm4ie/tokenizers/backends/_hf_base.py`
- **Example**: `experiments/reference/subword-tokenizers/examples.py`
- **Paper vs code**:
  - The 2012 paper joins the pair whose join most raises the likelihood of the corpus under a unigram model. The code's trainer is the BPE trainer of the HuggingFace `tokenizers` library with the `##` prefix set, so it joins the most frequent pair and only then turns the result into a WordPiece vocabulary. The example script computes the paper's pick at every step: on the toy corpus the two rules agree at three merges, tie at two, and differ at the last.
  - The paper does not say how a tie between equally likely pairs is broken. The code takes the pair whose symbols entered the vocabulary first, and the prefixed characters enter in an order that changes from run to run.
  - BERT's code removes control characters, puts a space around every Chinese character, and in its uncased model lowercases the text and strips accents. The code only puts the text into the composed Unicode form, so `Š` and `š` stay different pieces and every accent is kept.
  - BERT's code cuts every punctuation character on its own and gives up on a word longer than 100 characters. The code does both, through the BERT pre-tokenizer of the library and the library's default limit.
  - BERT has five special tokens of its own. The code adds the sweep's five and counts them toward the vocabulary size.
  - The merge loop and the greedy encoder run inside the library. The code sets their inputs and reads their output, so those lines cite no project code. The example script checks each learned merge against a replay of the pair counts, and each encoding against a replay of the longest-match walk.

## Difference from predecessor

- **Line**: 1, 5, 9 and 11 of training, and the whole encoder
- **Predecessor**: the symbols are the characters of a word, and the trained model is the vocabulary with its merges, replayed to cut a new word
- **Here**: a character inside a word carries `##`, only the vocabulary leaves training, and a new word is cut by taking the longest piece in the vocabulary from each position
- **Change**: a position-marked alphabet, and a greedy longest-match encoder in place of merge replay.
- **Effect**: One piece cannot serve both the start of a word and its inside, and a word with any part the vocabulary cannot spell becomes one unknown token.

## Algorithm

```algorithm
\caption{WordPiece training}
\Require corpus $D$, vocabulary size $B$, special tokens $S$
\Ensure vocabulary $V$
\Changed \State $f(w) \gets$ count of each word $w$ in $D$, after normalising $D$ and splitting it into words and single punctuation marks, each word written as its first character followed by its other characters with `##` in front \Comment{`slm4ie/tokenizers/backends/hf_wordpiece.py:39-40`}
\ForAll{words $w$}
  \State $(c_1) \gets (w)$
  \State add $f(w)$ to $f(c_1)$
\EndFor
\Changed \State $V \gets S$ followed by the sorted word-start characters, then the prefixed characters in no fixed order, $M \gets ()$ \Comment{`slm4ie/tokenizers/backends/hf_wordpiece.py:44-45`}
\State $n(a, b) \gets \sum_c f(c)$ times the places where $a$ is followed by $b$ inside $c$
\While{$|V| < B$ and some pair has a count} \Comment{`slm4ie/tokenizers/backends/hf_wordpiece.py:43`}
  \State $(a^*, b^*) \gets$ the pair with the highest $n$, a tie going to the pair whose symbols entered $V$ first
  \Changed \State join $a^* b^*$ in every chunk that holds it, written as $a^*$ followed by $b^*$ without its `##`, and update $n$
  \State append $(a^*, b^*)$ to $M$, add the joined symbol to $V$
\EndWhile
\Changed \Return a WordPiece model holding $S$ and $V$; $M$ is discarded \Comment{`slm4ie/tokenizers/backends/hf_wordpiece.py:38-48`}
```

- **1**: Words are counted once, so the rest works on distinct words with a weight. The text is put into the composed Unicode form and cut on spaces, with every punctuation mark its own word. The first character of a word is a bare symbol and every later one carries `##`, so `hiša` is the four symbols `h`, `##i`, `##š`, `##a`.
- **2-4**: Every word is one chunk, as in Character BPE.
- **5**: The special tokens take the first places and the word-start characters follow in sorted order. The prefixed characters come last, in an order the library does not fix. The order decides how a tie is broken.
- **6**: Pairs are counted inside chunks only, so no merge ever spans two words. A bare symbol can only ever be the first half of a pair.
- **7-10**: Each round joins one pair everywhere and records it. The joined symbol keeps the prefix of its first half and drops the prefix of its second, so `hi` and `##š` give `hiš`, and `##a` and `##m` would give `##am`. The loop ends when the vocabulary is full or no pair is left.
- **11**: The trained model is the vocabulary with the unknown token and the prefix. The merges are not saved, because encoding never reads them.

```algorithm
\caption{WordPiece encoding}
\Require text $t$, vocabulary $V$
\Ensure the tokens of $t$, each with the span of $t$ it covers
\State normalise $t$ to the composed Unicode form \Comment{`slm4ie/tokenizers/backends/hf_wordpiece.py:39`}
\Changed \State split $t$ into words, every punctuation mark on its own \Comment{`slm4ie/tokenizers/backends/hf_wordpiece.py:40`}
\ForAll{words $p$}
  \Changed \State $s \gets ()$, $i \gets 0$; if $p$ has more than 100 characters, $s \gets$ (the unknown token) and $i \gets |p|$
  \Changed \While{$i < |p|$}
    \Changed \State $j \gets$ the largest end such that the characters $i$ to $j$ of $p$, with `##` in front when $i > 0$, are one piece in $V$
    \Changed \If{no such $j$ exists}
      \Changed \State $s \gets$ (the unknown token), $i \gets |p|$
    \Changed \Else
      \Changed \State append that piece to $s$, $i \gets j$
    \EndIf
  \EndWhile
\EndFor
\Return every $s$ in order, each with the span of $t$ it covers \Comment{`slm4ie/tokenizers/backends/_hf_base.py:54-55`}
```

- **1-2**: The text is cleaned and cut into words the same way as in training.
- **3-4**: Each word is cut on its own. A word of more than 100 characters is not tried at all and becomes the unknown token.
- **5-6**: From the current position, every candidate from the rest of the word down to one character is looked up, longest first, with `##` in front unless the position is the start of the word. The first hit is the longest piece that fits.
- **7-8**: When even the single character does not fit, the whole word becomes the unknown token, and the pieces already found are dropped.
- **9-10**: Otherwise the piece is kept and the search goes on from where it ended. These lines, like 5 to 8, run inside the HuggingFace `tokenizers` library and cite no project code.
- **11**: The tokens of all words, in order. Each token also reports which characters of the text it covers, which is how the morpheme metrics find its cuts. An unknown token covers its whole word.

## Worked example

The corpus holds six forms of two nouns: `hiša` (house) five times, `hiše`
four, `hišo` three, `miza` (table) four, `mize` three, `mizo` twice. The
vocabulary size is 21: eight word-start characters, six prefixed characters,
the unknown token, and room for six merges. Only six characters get a prefixed
form, because `h` and `m` never occur inside a word. The tokenizer is the
project's own backend. The tables are written by the example script, which
recovers the merge order by running the BPE trainer the WordPiece trainer
wraps, checks that both give the same vocabulary, and checks every merge
against a replay of the pair counts. A tied pair is shown as its two symbols
run together, so `##i##š` is the pair `##i` and `##š`.

![Merges learned on the toy corpus, the count that chose each, the word hiša after it, and the pair the paper's likelihood rule would have joined](tables/wordpiece-merges.csv)

The six merges are the six of Character BPE on the same corpus, with a prefix
on every second half, because the trainer is the same. Three of the six
choices are ties, and each falls to the pair whose symbols entered the
vocabulary first: at merge 1 the bare `h` entered before the prefixed `##i`,
and at merge 6 `hiš` entered before `miz`. The last column shows the paper's
rule. It agrees at merges 2, 4 and 5, where the most frequent pair is also the
most informative one. At merges 1 and 3 it ties between the same two pairs as
the count does. At merge 6 it would join `miz` and `##a` instead of `hiš` and
`##e`, because after merge 5 only four of the nine `##a` are left on their
own, so that pair's halves are rarer than `hiš` and `##e`, which both occur
seven times.

![Longest-match walk over a corpus form and a form the tokenizer never saw](tables/wordpiece-walk.csv)

The walk over `mize` tries the whole word, finds nothing, then finds `miz` and
continues with `##e`. The walk over `hišami` finds `hiša` as the longest piece
from the start, which skips past the stem boundary. From the fifth character
neither `##mi` nor `##m` exists, since `m` was never seen inside a word, so the
word becomes the unknown token and the `hiša` already found is dropped.

![Encoding of two corpus forms, four forms the tokenizer never saw, and the cuts the morpheme metrics read from the spans](tables/wordpiece-encoding.csv)

The form `mize` is cut at the stem boundary, because `mize` itself never
became a piece while `hiša` and `hiše` did. Both unseen forms with the ending
`ami` are unknown, as is `hišaq` with its unseen letter. The bare stem `hiš` is
one piece, since it was learned as a word start. The morpheme metrics see an
unknown token as one piece covering the whole form, so an unknown form shows
no cut at all rather than a wrong one.

## Limits

### The trainer is not the paper's

The vocabulary is chosen by pair count, as in BPE, not by the likelihood gain
the 2012 paper describes. On the toy corpus the two rules part at the last
merge, where the paper's rule prefers a pair whose halves are rare on their
own.

### Any failure loses the whole word

When one position has no piece in the vocabulary, the word becomes a single
unknown token, including the parts already matched. In the worked example
`hišami` loses its matched `hiša` because `##m` does not exist. A merge-based
tokenizer would have kept `hiša` and spelled the rest from characters.

### A character knows its position

Every character seen inside a word costs a second vocabulary slot with the
prefix, and a character seen only at a word start cannot be used inside one.
In the worked example `h` and `m` start every corpus word, so no unseen form
can hold them anywhere else.

### Longest match ignores the stem

The encoder takes a whole inflected form whenever it is in the vocabulary, and
the longest piece from each position otherwise. A stem that is a prefix of a
longer piece is never chosen while the longer piece fits, as `hiša` over `hiš`
in `hišami` shows.

### An unknown form is invisible to the morpheme metrics

An unknown token covers the whole form, so the metrics read no cut from it. A
tokenizer with many unknown forms is scored on the forms it can spell, not on
the ones it cannot.

### Punctuation is cut to single marks

Every punctuation character is its own word, so a run such as `...` is three
tokens. Character BPE keeps the run together.

### Ties among prefixed characters are not repeatable

The prefixed characters are numbered in an order that changes between runs,
and a tie is broken by that order. Two runs on the same corpus may therefore
differ on a tie between two pairs of prefixed characters.
