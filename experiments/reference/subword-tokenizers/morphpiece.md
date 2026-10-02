---
title: MorphPiece
kind: algorithm
status: checked
summary: Byte-level BPE whose encoder first looks every word up in a morpheme table and emits the table's morphemes when the word is there; only a word outside the table reaches the merges.
variant_of: subword-tokenizers/byte-level-bpe
---

## Description

MorphPiece is a tokenizer with two paths. A table maps a word form to its
morphemes, the smallest meaning-bearing parts of the word, such as a stem and
an ending. When a word of the text is in the table, it becomes exactly those
morphemes, one token each. When it is not, an ordinary byte-level BPE trained
alongside cuts it by replaying merges. The vocabulary is the union of the two:
the BPE pieces and the morphemes the table uses.

- **The table is consulted at encoding time.** MorphBPE lets a lexicon shape
  training and then forgets it, so its boundaries are learned. MorphPiece
  carries the table inside the trained tokenizer, so a known word is always
  cut at its morpheme boundaries.
- **Two kinds of token share one vocabulary.** A morpheme token is the plain
  string of the morpheme, such as `hiš`. A BPE token is a string of byte
  symbols with the space marker, such as `ĠhiÅ¡`. When the two spell the same
  string, as `a` does, they are one token with one id.
- **The table decides what is protected.** Here it is derived from Sloleks, a
  lexicon of Slovenian word forms, and splits a form into stem and ending. The
  paper built it from MorphyNet for English.

## What the paper adds

The paper builds the table from MorphyNet, a database of derivational and
inflectional morphology, and keeps the entries whose affixes and stems occur
at least five times in the training text. The BPE path is trained with every
table word removed from the text, so it only ever learns what it will encode.
A GPT-2 Base model trained with the tokenizer, called MorphGPT, matches or
exceeds GPT-2 on language-modeling perplexity, zero-shot GLUE and the MTEB
embedding benchmark after about half the training steps. The paper also
reports that MorphPiece writes a text in about 17% more tokens than BPE. Those
are the paper's results, not this project's.

| Quantity in the paper          | Value   |
| ------------------------------ | ------- |
| Words in MorphyNet             | 346,340 |
| Table entries kept             | 134,943 |
| Morpheme tokens                | 18,304  |
| BPE vocabulary                 | 32,000  |
| Final vocabulary               | 50,006  |

## Why it matters here

MorphPiece is the only backend in the sweep whose encoder guarantees the
boundary of a known word, so its morpheme scores are the ceiling a lexicon can
buy. Against MorphBPE, which only steers training, it shows whether a
guaranteed boundary is worth the second path. Against byte-level BPE, its own
fallback, it shows what the table changes on its own, because the merges are
learned from the same text.

It is also the one backend that cannot be saved as a standard tokenizer file,
since its encoder is a table lookup, so it is exported as a custom slow
tokenizer.

## Facts

- **Base unit**: UTF-8 bytes on the BPE path, each shown as one printable character; Unicode characters for a morpheme token
- **Chunk while training**: whole word, with the space before it, on the BPE path; the table does not train
- **Picks next token by**: highest pair count on the BPE path
- **Encoding**: table lookup for every word, byte-level BPE merges for a word not in the table
- **Needs lexicon**: training and encoding; the table is saved with the model
- **Setting**: vocabulary size; half of the budget left after the special tokens is offered to morphemes, ranked by how often the words using them occur
- **Unknown token**: none on the BPE path; a morpheme outside the vocabulary cannot occur, since a table entry needs every morpheme kept
- **Round trip**: not guaranteed; two table words in a row cannot be told apart in the token stream
- **Repeatable**: yes, training has no random step
- **Export**: a custom slow tokenizer; on disk a `tokenizer.json` for the BPE path, the table in `morpheme_table.jsonl.gz` and the combined vocabulary in `morphpiece.json`
- **Registry key**: `morphpiece`

## Sources

- **Paper**: MorphPiece: A Linguistic Tokenizer for Large Language Models, arXiv:2307.07262
- **Code**:
  - `slm4ie/tokenizers/backends/morph_piece.py`
  - `slm4ie/tokenizers/hf_morphpiece.py`
  - `slm4ie/tokenizers/base.py`
- **Lexicon**: `slm4ie/tokenizers/morphology.py`
- **Example**: `experiments/reference/subword-tokenizers/examples.py`
- **Paper vs code**:
  - The paper removes every table word from the text before training the BPE path. The code trains it on the whole text.
  - The paper adds every affix and stem of the trimmed table to the vocabulary. The code offers half the budget and keeps the most frequent morphemes; a table entry that uses a dropped morpheme is dropped too.
  - The paper marks a prefix or suffix with `#` and separates the parts of a compound with a `#` token. The code emits bare morphemes with no marker, so a morpheme spelled like a BPE token shares its id.
  - The paper's table comes from MorphyNet and holds derivational and inflectional cuts of English. The code's comes from Sloleks and holds stem and ending, with derivational cuts only when the second source is configured.
  - The paper's detokenizer classifies each token by its surface marks and reverses the table to rebuild words. The code's wrapper treats each run of morpheme tokens as one word with a space in front, and cannot split two table words that follow each other.
  - The paper reaches its target vocabulary size. The code falls short of it by the number of morphemes that are already BPE tokens.
  - The paper does not say what happens to a word whose table cut is a single morpheme. The code leaves such words to the BPE path.

## Difference from predecessor

- **Line**: Algorithm 2, line 5
- **Predecessor**: $s \gets$ the byte symbols of $p$, then the merges of $M$
- **Here**: $s \gets T(p)$ when $p$ is in the table $T$, otherwise the BPE tokens of $p$
- **Change**: a word in the morpheme table is cut by the table, not by the merges.
- **Effect**: A known word is always cut at its morpheme boundaries, and the vocabulary gains the morphemes as tokens of their own.

## Algorithm

```algorithm
\caption{MorphPiece training}
\Require corpus $D$, morpheme lexicon $L$, vocabulary size $B$, special tokens $S$
\Ensure morpheme table $T$, byte-level BPE $(V_b, M)$, vocabulary $V$
\State $f(w) \gets$ count of each word $w$ in $D$, words split from punctuation \Comment{`slm4ie/tokenizers/backends/morph_piece.py:71-73`}
\State $g(m) \gets \sum f(w)$ over every word $w$ in $L$ whose cut $L(w)$ holds the morpheme $m$ \Comment{`slm4ie/tokenizers/backends/morph_piece.py:74-79`}
\State $K \gets$ the $\lfloor (B - |S|) / 2 \rfloor$ morphemes with the highest $g$ \Comment{`slm4ie/tokenizers/backends/morph_piece.py:81-83`}
\State $T \gets \{w \mapsto L(w)\}$ for every $w$ in $L$ with two or more morphemes, all of them in $K$ \Comment{`slm4ie/tokenizers/backends/morph_piece.py:85-89`}
\State $(V_b, M) \gets$ byte-level BPE trained on $D$ with vocabulary size $\max(|S| + 256,\ B - |K|)$ \Comment{`slm4ie/tokenizers/backends/morph_piece.py:91-101`}
\State $V \gets V_b$, then every $m$ in $K$ not already in $V_b$, each with the next free id \Comment{`slm4ie/tokenizers/backends/morph_piece.py:103`}
\Return $T$, $(V_b, M)$ and $V$ \Comment{`slm4ie/tokenizers/backends/morph_piece.py:112-118`}
```

- **1**: Words are counted once. The splitter keeps letters together and makes each punctuation mark its own word, the same splitter the lexicon was built with.
- **2**: A morpheme is as frequent as the words it occurs in. The count comes from the corpus, so a stem used by many common forms ranks high.
- **3**: Half of the budget is the most morphemes the table may use. When the lexicon has fewer, all are kept.
- **4**: Only a word whose every morpheme survived is kept in the table. A word with a dropped morpheme falls to the BPE path, so no token outside the vocabulary can ever be emitted.
- **5**: The BPE path is the byte-level BPE of this topic, trained on the whole text with the budget left over. The floor keeps room for the 256 byte symbols.
- **6-7**: The two vocabularies are joined. A morpheme already spelled by a BPE token is not added again, so the final size can fall below $B$.

```algorithm
\caption{MorphPiece encoding}
\Require text $t$, table $T$, byte-level BPE $(V_b, M)$
\Ensure the tokens of $t$, each with the span of $t$ it covers
\State encode $t$ with the byte-level BPE, keeping each token's span and which word it belongs to \Comment{`slm4ie/tokenizers/backends/morph_piece.py:131-132`}
\ForAll{words $p$ of $t$, as runs of tokens with the same word} \Comment{`slm4ie/tokenizers/backends/morph_piece.py:136-145`}
  \State $u \gets$ the characters of $t$ that the run covers, the space before the word left out \Comment{`slm4ie/tokenizers/backends/morph_piece.py:145-146`}
  \If{$u$ is in $T$}
    \Changed \State emit the morphemes $T(u)$, each spanning its own characters of $t$ \Comment{`slm4ie/tokenizers/backends/morph_piece.py:147-152`}
  \Else
    \State emit the run's BPE tokens unchanged \Comment{`slm4ie/tokenizers/backends/morph_piece.py:153-155`}
  \EndIf
\EndFor
\Return the emitted tokens in order \Comment{`slm4ie/tokenizers/backends/morph_piece.py:169`}
```

- **1**: The BPE path runs first over the whole text. It supplies the word boundaries, the spans and the fallback tokens in one pass.
- **2-3**: The tokens are regrouped into words. The lookup key is the surface text of the word, exactly as written, so a capital letter or an unseen form misses the table.
- **4-5**: A hit replaces the word's BPE tokens by its morphemes. This is the line that differs from byte-level BPE. The spans are laid end to end, so the morpheme metrics read the table's cuts.
- **6-7**: A miss keeps the BPE tokens as they are.
- **8**: Tokens of both kinds leave in text order. A table word carries no space marker, so the stream alone does not say where it began.

```algorithm
\caption{Decoding a token stream back to text}
\Require tokens $s_1, \dots, s_n$, the set $K'$ of morpheme tokens that are not BPE tokens
\Ensure the text
\ForAll{tokens $s_i$}
  \If{$s_i$ is in $K'$}
    \State if the previous token was not in $K'$, decode the pending BPE tokens and start a new word with a space \Comment{`slm4ie/tokenizers/hf_morphpiece.py:130-133`}
    \State append $s_i$ as it is \Comment{`slm4ie/tokenizers/hf_morphpiece.py:134-135`}
  \Else
    \State hold $s_i$ for the byte-level decoder \Comment{`slm4ie/tokenizers/hf_morphpiece.py:137-138`}
  \EndIf
\EndFor
\Return the parts joined, with the pending BPE tokens decoded \Comment{`slm4ie/tokenizers/hf_morphpiece.py:139-140`}
```

- **1-2**: A token is a morpheme token only when it is not also a BPE token. A shared token, such as `a`, is decoded as bytes.
- **3-4**: A run of morpheme tokens is one word. Where two table words follow each other, they come out as one.
- **5-6**: BPE tokens are decoded by the byte-level decoder of the predecessor, so their spaces come back exactly.
- **7**: The paper reverses the table with a heuristic over surface marks. The code has no marks to read, so this is a best effort.

## Worked example

The corpus holds six forms of two nouns: `hiša` (house) five times, `hiše`
four, `hišo` three, `miza` (table) four, `mize` three, `mizo` twice. The
lexicon splits each into stem and ending, such as `hiš` and `a`, so it holds
five morphemes: `hiš`, `miz`, `a`, `e`, `o`. The vocabulary size asked is 268,
the byte-level entry's 263 plus the five morphemes. The tokenizer is the
project's own backend, and the tables are written by the example script, which
checks that every table word is encoded as its lexicon cut and every other
word exactly as the BPE path encodes it.

![Vocabulary sizes: asked, the BPE path, the morphemes kept and the final size](tables/morphpiece-sizes.csv)

All five morphemes fit the budget, yet the final vocabulary is 264, not 268.
Four of the morphemes are already BPE tokens: `a`, `e` and `o` are byte
symbols, and `miz` is the sixth merge. Only `hiš` gets a new id, because its
BPE spelling `hiÅ¡` is a different string.

![The morpheme tokens, how often the corpus uses them, and whether they double as a BPE token](tables/morphpiece-vocabulary.csv)

![The six merges the BPE path learned](tables/morphpiece-merges.csv)

The BPE path learns the same six merges as the byte-level BPE entry, since it
sees the same text. The two tokenizers differ only in what the table does
afterwards.

![Encoding of two corpus forms, two unseen forms, a capitalised form and the bare stem](tables/morphpiece-encoding.csv)

The two corpus forms are cut by the table into stem and ending, two tokens
each. The unseen form `hišami` misses the table and goes to the BPE path,
where the stem comes out whole but with its space marker as `ĠhiÅ¡`. The form
`mizami` shows the lone-space cut of byte-level BPE, which the morpheme metrics
read as a boundary after `m`. The capitalised `Hiša` misses the table and costs
five tokens. The bare stem `hiš` is one morpheme, so it is not in the table
and is encoded by BPE as id 260, while the same letters emitted by the table
are id 263.

## Limits

### Only the exact form is protected

The lookup key is the word as written. A capital letter, a typo or an inflected
form the lexicon lacks sends the word to the BPE path, where no boundary is
guaranteed. In the worked example `Hiša` and `hišami` both miss.

### One string, two ids

A morpheme emitted by the table and the same letters emitted by BPE are
different tokens when their spellings differ, as `hiš` and `ĠhiÅ¡` do. The
model sees the stem under two ids depending on whether the form was in the
table.

### A morpheme that spells a BPE token merges with it

`a`, `e`, `o` and `miz` share their ids with BPE tokens. The decoder therefore
takes them for byte-level pieces, and the vocabulary falls short of the size
asked by one slot per shared morpheme.

### Word starts are unmarked on the table path

A morpheme token carries no space marker. Decoding guesses that a run of
morpheme tokens is one word, so two table words in a row come back as one, and
the exact text is not recoverable from the tokens alone.

### The BPE path has less room

The morphemes take their share of the budget before BPE trains, so the
fallback path has a smaller vocabulary than a plain byte-level BPE of the same
size. The paper measured about 17% more tokens per text for this reason.
