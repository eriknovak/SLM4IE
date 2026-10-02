---
title: Unigram
kind: algorithm
status: checked
summary: Starts from a large set of frequent substrings, gives each a probability, and prunes the set down to the vocabulary size; a new word is cut into its most probable sequence of pieces.
---

## Description

The Unigram tokenizer, as implemented in SentencePiece, treats a piece of text
as a sequence of independent draws from a vocabulary of pieces, each with its
own probability. Training works downward rather than upward. It begins with
a vocabulary far larger than wanted, made of the frequent substrings of the
corpus, fits a probability to every piece, and removes the pieces whose loss
hurts the fit least, round after round, until the vocabulary is small enough.
A new word is cut into the sequence of pieces with the highest product of
probabilities.

- **Pieces compete, they are not joined.** Byte-pair encoding grows a
  vocabulary by joining pairs. Unigram shrinks one, so a long piece and the
  shorter pieces that spell it exist side by side and the probabilities decide
  which is used.
- **A space belongs to the word after it.** The space before a word is written
  `▁` and is part of the first piece, so decoding returns the exact text.
- **The text is normalised first.** SentencePiece applies Unicode
  compatibility normalisation with its own rules for machine translation, so
  a ligature such as `ﬁ` becomes `fi`. Letter case is kept.

## What the paper adds

The paper introduces the tokenizer as a means to an end. Since every
segmentation of a word has a probability, the training of a translation model
can draw a different segmentation each time it sees a sentence, which the
paper calls subword regularization and shows to help translation. This
project uses only the single most probable segmentation. The second paper
describes SentencePiece, the library that implements the trainer and the
encoder used here, and its normalisation and space handling.

## Why it matters here

Unigram is the tokenizer of the T5 and mT5 encoder-decoder family and of
ALBERT and XLNet, so it is one of the two tokenizer families a Slovenian
benchmark model is likely to carry. In the sweep it is the one tokenizer that
chooses pieces by probability and by pruning rather than by joining. On the
worked example below it keeps a stem and single-letter endings without any
lexicon, so it is the baseline that shows how far the likelihood alone gets
toward morpheme boundaries.

## Facts

- **Base unit**: Unicode characters after compatibility normalisation, with `▁` for the space before a word
- **Chunk while training**: whole word with the space before it; a piece also never crosses between letters and digits or between two writing systems
- **Picks next token by**: nothing is joined; the pieces whose removal costs the corpus likelihood least are dropped, a quarter of them per round
- **Encoding**: the most probable sequence of pieces under the piece probabilities, no lexicon
- **Needs lexicon**: no
- **Setting**: vocabulary size, which includes the three control pieces and the special tokens; the model may hold fewer pieces than asked, since a piece the corpus does not use is dropped
- **Longest piece**: 16 characters
- **Unknown token**: a run of unseen characters is one unknown token, whose characters the encoder still returns
- **Repeatable**: yes, training has no random step when every sentence of the corpus is used, which is the setting here
- **Export**: a SentencePiece `spm.model`, rebuilt as a HuggingFace Unigram tokenizer at export
- **Registry key**: `unigram`

## Sources

- **Paper**: Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates, arXiv:1804.10959, section 3, which defines the model, the training and the encoding
- **Paper**: SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing, arXiv:1808.06226, which describes the library's normalisation and space handling
- **Code**:
  - `slm4ie/tokenizers/backends/sp_unigram.py`
  - `slm4ie/tokenizers/base.py`
  - `slm4ie/tokenizers/hf_export.py`
- **Example**: `experiments/reference/subword-tokenizers/examples.py`
- **Paper vs code**:
  - The paper fits piece probabilities by maximum likelihood. The library's trainer sets a piece's log probability to the digamma function of its expected count minus the digamma of the total, a Bayesian variant that discounts rare pieces. The example script reproduces every learned score from the piece counts with this rule.
  - The paper says the seed vocabulary is the frequent substrings of the corpus. The library takes the substrings found through a suffix array, ranked by count times length, at most one million of them and none longer than 16 characters, and always every character.
  - The paper drops a fixed share of pieces per round and does not say what else is dropped. The library also drops, at every fit, a piece whose expected count falls below one half, so the vocabulary can end smaller than asked. The code accepts that instead of failing.
  - The paper does not say what an unseen character becomes. The library makes a run of them one unknown token and still returns the characters as its surface, so the metrics see them as pieces.
  - The paper draws a segmentation at random during model training. The code encodes with the single best segmentation only.
  - The library reserves the unknown, start and end pieces itself. The code passes the sweep's other special tokens as user-defined symbols, which are matched before segmentation, and keeps every character of the sample where the library's default drops the rarest.
  - The seed, the fit, the pruning and the search for the best segmentation run inside the SentencePiece library. The code sets their inputs and reads their output, so those lines cite no project code. The example script checks the scores and reproduces every encoding by scoring all segmentations of the word.

## Algorithm

```algorithm
\caption{Unigram training}
\Require corpus $D$, vocabulary size $B$, special tokens $S$
\Ensure pieces $V$, each with a log probability $\log p(x)$
\State normalise every sentence of $D$, put $▁$ before it and in place of every space \Comment{`slm4ie/tokenizers/backends/sp_unigram.py:63-64`}
\State $V \gets$ every character of $D$, and the most frequent substrings that cross no space, no change of writing system and no letter-digit boundary, at most 16 characters long, ranked by count times length \Comment{`slm4ie/tokenizers/backends/sp_unigram.py:68`}
\While{$|V| > 1.1\,B$}
  \For{two rounds}
    \State $c(x) \gets$ expected count of each piece $x$ over every segmentation of $D$, each weighted by its probability under the current $\log p$
    \State $\log p(x) \gets \psi(c(x)) - \psi(\sum_y c(y))$, dropping every $x$ with $c(x) < 1/2$
  \EndFor
  \State $\ell(x) \gets$ the fall in the corpus log-likelihood when $x$ is removed and the words that used it take their next best segmentation
  \State keep every single character, and of the other pieces the three quarters with the largest $\ell$
\EndWhile
\State keep every character and the highest-scoring other pieces so that, with the control pieces and $S$, there are $B$ \Comment{`slm4ie/tokenizers/backends/sp_unigram.py:66-71`}
\Return $V$ with $\log p$ \Comment{`slm4ie/tokenizers/backends/sp_unigram.py:74-75`}
```

- **1**: The text is normalised and the space before a word becomes part of it. The trainer reads the sentences from memory, and every sentence is used.
- **2**: The seed is large on purpose. Every character is in it so that any text of the corpus can be spelled, and the substrings are the candidates the pruning will choose among.
- **3**: Rounds continue until the vocabulary is within ten percent of the target. The final trim on line 9 removes the rest.
- **4-6**: Two rounds of expectation-maximisation refit the probabilities to the current vocabulary. Line 5 sums, over all ways to cut each sentence, how often each piece is used, weighting each way by its probability. Line 6 turns the counts into log probabilities with the digamma function $\psi$, which behaves like the logarithm for large counts and discounts small ones. A piece the corpus has stopped using is dropped here.
- **7-8**: Each piece is scored by how much the corpus likelihood would fall without it. Characters are never removed, so every text stays spellable. The quarter of pieces that matter least go.
- **9**: The vocabulary is cut to size, characters first. The control pieces are the unknown, start and end pieces, and $S$ holds the sweep's other special tokens.
- **10**: The model is the pieces and their log probabilities. No merges exist.

```algorithm
\caption{Unigram encoding}
\Require text $t$, pieces $V$ with $\log p$
\Ensure the tokens of $t$, each with the span of $t$ it covers
\State normalise $t$, put $▁$ before it and in place of every space \Comment{`slm4ie/tokenizers/backends/sp_unigram.py:87`}
\State build the lattice of $t$: an edge from position $i$ to $j$ for every piece equal to the characters $i$ to $j$, and one edge over every run of characters no piece covers, scored as the unknown token
\State $s \gets$ the path from the start of $t$ to its end with the largest sum of $\log p$ \Comment{`slm4ie/tokenizers/backends/sp_unigram.py:87`}
\State write every unknown edge of $s$ as the characters it covers, with the id of the unknown token
\Return every piece of $s$ in order, each aligned to the characters of $t$ it spells, with $▁$ left out \Comment{`slm4ie/tokenizers/base.py:245-255`}
```

- **1**: The text is prepared exactly as in training, so a word at the start of a text is spelled like one inside it.
- **2-3**: Every way to cut the text is a path through the lattice, and the best path is found by dynamic programming. These lines run inside the SentencePiece library and cite no project code.
- **4**: An unseen character cannot be spelled. Its run is one token with the unknown id, but the characters are kept, so the output still covers the text.
- **5**: Each piece also reports which characters it spells. The project aligns the pieces to the text left to right after removing the `▁` marker, which is what the morpheme metrics read. A lone `▁` spells nothing and gets an empty span.

## Worked example

The corpus holds six forms of two nouns: `hiša` (house) five times, `hiše`
four, `hišo` three, `miza` (table) four, `mize` three, `mizo` twice. The
vocabulary size asked for is 21, as for WordPiece on the same corpus. The
tokenizer is the project's own backend. The tables are written by the example
script, which reproduces every learned log probability from the piece counts
and every encoding from a scoring of all segmentations.

![The learned pieces, their log probabilities, and the digamma rule applied to their counts](tables/unigram-vocabulary.csv)

The model keeps 14 pieces, not 21: the three control pieces, eleven learned.
The seed held every substring of the six forms, whole forms included. The
fitting rounds moved the expected count of the whole forms to the two stems
`▁hiš` and `▁miz` and the single-letter endings, and a piece whose expected
count fell below one half was dropped. Every piece with a count has the log
probability the digamma rule gives it. The six characters with no count are
there because a character is never removed, and they carry scores next to the
lowest learned one, so that they are used only when nothing longer fits.

![Every segmentation of two forms over the learned pieces, with its log probability](tables/unigram-segmentations.csv)

With this vocabulary each form has two ways to be spelled, as stem plus ending
or as single characters. The stem path wins by a wide margin, because the
characters that are never used in training carry the lowest scores. The unseen
form `hišami` is cut at the stem boundary and then letter by letter, since no
piece for `ami` or `mi` survived.

![Encoding of two corpus forms, five forms the tokenizer never saw, and the cuts the morpheme metrics read from the spans](tables/unigram-encoding.csv)

The letter `q` in `hišaq` is returned as a token of its own with the unknown id
0, so the morpheme metrics still see a cut before and after it. The form
`Hiša...` shows two more cases: the uppercase `H` is unseen, since the text is
not lowercased, and the three dots form a single unknown token, since a run of
unseen characters is one.

## Limits

### The vocabulary may stay below the target

A piece the corpus stops using is dropped at every fit, whatever the target.
On the toy corpus the model keeps 14 pieces of the 21 asked for. A sweep that
compares tokenizers at one vocabulary size has to read the actual size from
the model.

### Pruning follows the fit, not the best vocabulary

Each round refits the probabilities to the pieces that are left and then
removes the least useful ones. A piece that loses its count early is gone for
good, so the result is one that the rounds settle on, not the vocabulary that
would give the corpus the highest likelihood.

### Unseen characters are returned, but not known

A run of unseen characters is one token with the unknown id. The characters
survive in the output, so a text can be read back, but the model has one id
for all of them.

### A piece cannot cross a writing system or a digit

The trainer never proposes a piece that spans letters and digits or two
writing systems, so `covid19` is at least two pieces whatever the corpus.

### The space is a piece of its own when nothing absorbs it

A word whose first piece was never learned with the space in front starts with
a lone `▁` token, as `Hiša...` shows. That token spells nothing, and the
morpheme metrics read no cut from it.
