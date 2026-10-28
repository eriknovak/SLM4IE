"""Generate the worked-example tables of the subword-tokenizers topic.

Every table is produced by training the project's own tokenizer backends on a
toy corpus, so an entry never shows an example its code would not reproduce.

    uv run --group tokenizers python experiments/reference/subword-tokenizers/examples.py
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Callable
from pathlib import Path

from slm4ie.tokenizers.backends.char_bpe import CharBpeTokenizer
from slm4ie.tokenizers.backends.hf_bpe import BpeTokenizer
from slm4ie.tokenizers.backends.hf_wordpiece import WordPieceTokenizer
from slm4ie.tokenizers.backends.morph_bpe import MorphBpeTokenizer
from slm4ie.tokenizers.backends.morph_piece import MorphPieceTokenizer
from slm4ie.tokenizers.backends.sp_unigram import UnigramTokenizer
from slm4ie.tokenizers.base import TrainContext
from slm4ie.tokenizers.bpe_core import encode_bpe, merge_ranks
from slm4ie.tokenizers.metrics import _segments, _token_spans
from slm4ie.tokenizers.morphology import MorphemeSegmentation, MorphLexicon

# ===============================
# Defaults
# ===============================

TABLES = Path(__file__).resolve().parent / "tables"

#: Toy corpus: inflected forms of `hiša` (house) and `miza` (table), with counts.
CORPUS: dict[str, int] = {"hiša": 5, "hiše": 4, "hišo": 3, "miza": 4, "mize": 3, "mizo": 2}
#: The corpus as training sentences, each form repeated by its count.
SENTENCES: list[str] = [form for form, count in CORPUS.items() for _ in range(count)]
#: Stem and ending of every corpus form, plus the held-out forms encoded below.
GOLD: dict[str, list[str]] = {
    "hiša": ["hiš", "a"],
    "hiše": ["hiš", "e"],
    "hišo": ["hiš", "o"],
    "miza": ["miz", "a"],
    "mize": ["miz", "e"],
    "mizo": ["miz", "o"],
    "hišami": ["hiš", "ami"],
    "mizami": ["miz", "ami"],
}
#: Forms absent from the corpus, encoded to show what the tokenizer generalizes.
HELD_OUT: list[str] = ["hišami", "mizami"]
SPECIAL_TOKENS: list[str] = ["<unk>"]
#: Eight characters, six merges and the unknown token.
VOCAB_SIZE = 15
#: The 256 byte symbols, six merges and the unknown token.
BYTE_VOCAB_SIZE = 256 + 6 + 1
#: Eight word-start characters, six continuation characters, six merges and the unknown token.
WORDPIECE_VOCAB_SIZE = 8 + 6 + 6 + 1
#: WordPiece marks a piece that continues a word with this prefix.
CONTINUATION = "##"
#: The WordPiece target again; Unigram keeps fewer pieces than asked on this corpus.
UNIGRAM_VOCAB_SIZE = WORDPIECE_VOCAB_SIZE
#: The byte-level target plus the five morphemes of the toy lexicon, which MorphPiece adds as tokens.
MORPHPIECE_VOCAB_SIZE = BYTE_VOCAB_SIZE + 5
#: SentencePiece marks the space before a word with this character.
SPACE_MARKER = "▁"
#: Euler's constant, for the digamma function at an integer.
EULER_GAMMA = 0.5772156649015329
TRACED_WORD = "hiša"
Replay = Callable[[str, list[tuple[str, str]]], list[str]]

# ===============================
# Helper functions
# ===============================


def toy_lexicon() -> MorphLexicon:
    """Build the lexicon the toy corpus is segmented with.

    Returns:
        Stem and ending for every corpus form.
    """
    # add the stem and ending of every corpus form
    lexicon = MorphLexicon()
    for form in CORPUS:
        lexicon.by_form[form] = MorphemeSegmentation(form, GOLD[form], ["stem", "suffix"], lemma=form)
    return lexicon


def learned_merges(tokenizer: CharBpeTokenizer | MorphBpeTokenizer) -> list[tuple[str, str]]:
    """Read the ordered merges out of a trained backend.

    Args:
        tokenizer: A trained backend wrapping a HuggingFace BPE model.

    Returns:
        Merge pairs in the order they were learned.
    """
    merges = json.loads(tokenizer._tokenizer.to_str())["model"]["merges"]
    return [tuple(pair) if isinstance(pair, list) else tuple(pair.split(" ")) for pair in merges]


def trace(word: str, merges: list[tuple[str, str]], steps: int) -> list[str]:
    """Show `word` after each of the first `steps` merges.

    Args:
        word: The word to follow.
        merges: Ordered learned merges.
        steps: Number of rows to produce; a row past the last merge is empty.

    Returns:
        The pieces of `word`, space-separated, one entry per merge.
    """
    return [" ".join(encode_bpe(word, merge_ranks(merges[: k + 1]))) if k < len(merges) else "" for k in range(steps)]


def write_csv(name: str, header: list[str], rows: list[list[str]]) -> None:
    """Write one table under `tables/`.

    Args:
        name: File name without extension.
        header: Column titles.
        rows: Table body.
    """
    # write the header and the rows
    TABLES.mkdir(exist_ok=True)
    with (TABLES / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def replay_bpe(word: str, merges: list[tuple[str, str]]) -> list[str]:
    """Apply `merges` to the characters of `word` as plain BPE does.

    Args:
        word: The word to cut.
        merges: Ordered learned merges.

    Returns:
        The pieces of `word`.
    """
    return encode_bpe(word, merge_ranks(merges))


def wordpiece_symbols(word: str) -> list[str]:
    """Spell `word` as the WordPiece trainer sees it: a bare first character, then prefixed ones.

    Args:
        word: The word to spell.

    Returns:
        One symbol per character.
    """
    return [word[0], *(CONTINUATION + char for char in word[1:])]


def replay_wordpiece(word: str, merges: list[tuple[str, str]]) -> list[str]:
    """Apply `merges` in order to the prefixed symbols of `word`, each everywhere it occurs.

    A join keeps the prefix of its first symbol and drops the prefix of its
    second, which is how the trainer spells a joined symbol.

    Args:
        word: The word to cut.
        merges: Ordered learned merges over prefixed symbols.

    Returns:
        The pieces of `word`, prefixed where they continue it.
    """
    # apply the merges in order
    symbols = wordpiece_symbols(word)
    for first, second in merges:
        joined = first + second[len(CONTINUATION) :]
        out: list[str] = []
        index = 0
        while index < len(symbols):
            if index + 1 < len(symbols) and (symbols[index], symbols[index + 1]) == (first, second):
                out.append(joined)
                index += 2
            else:
                out.append(symbols[index])
                index += 1
        symbols = out

    return symbols


def pair_counts(
    words: dict[str, int], merges: list[tuple[str, str]], replay: Replay = replay_bpe
) -> dict[tuple[str, str], int]:
    """Count adjacent symbol pairs over `words` after replaying `merges`.

    Args:
        words: Word to corpus count.
        merges: Merges applied before counting.
        replay: How a word is cut after the merges so far.

    Returns:
        Pair to weighted count.
    """
    # count the pairs weighted by word frequency
    counts: dict[tuple[str, str], int] = {}
    for word, freq in words.items():
        symbols = replay(word, merges)
        for pair in zip(symbols, symbols[1:]):
            counts[pair] = counts.get(pair, 0) + freq

    return counts


def merge_rows(
    merges: list[tuple[str, str]],
    words: dict[str, int],
    vocab: dict[str, int],
    traced: str,
    replay: Replay = replay_bpe,
) -> list[list[str]]:
    """Replay the pair counts behind every learned merge and check the merge rule.

    The trainer runs inside the HuggingFace `tokenizers` library, so the rule
    is checked here instead: every learned pair must be the most frequent one at
    its step, and a tie must go to the pair whose symbols entered the vocabulary
    first.

    Args:
        merges: Ordered learned merges.
        words: Training word to corpus count, in the symbols the
            trainer saw.
        vocab: Token to id, which fixes the tie order.
        traced: The word to show after each merge, in the same symbols.
        replay: How a word is cut after the merges so far.

    Returns:
        One row per merge: number, pair, count, the tied pairs
            it beat, and `traced` after it.

    Raises:
        AssertionError: If a learned merge is not a most frequent pair, or a tie
            is broken another way.
    """
    rows: list[list[str]] = []
    for k, pair in enumerate(merges):
        # find the most frequent pairs at this step
        counts = pair_counts(words, merges[:k], replay)
        top = max(counts.values())
        tied = sorted(p for p, n in counts.items() if n == top)

        # check the merge rule
        assert counts[pair] == top, (k, pair, counts)
        assert pair == min(tied, key=lambda p: (vocab[p[0]], vocab[p[1]])), (k, pair, tied)

        # record the merge
        rows.append(
            [
                str(k + 1),
                " + ".join(pair),
                str(top),
                ", ".join(a + b for a, b in tied if (a, b) != pair) or "none",
                " ".join(replay(traced, merges[: k + 1])),
            ]
        )

    return rows


def likelihood_picks(words: dict[str, int], merges: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Return the pairs the original WordPiece rule would join next, more than one when they tie.

    The rule joins the pair whose merge most raises the log-likelihood of the
    corpus under a unigram model over the current symbols. The gain is computed
    exactly from the symbol counts before and after the join.

    Args:
        words: Training word to corpus count.
        merges: Merges learned so far.

    Returns:
        Every pair whose gain is within rounding of the largest.

    Raises:
        AssertionError: If a pair repeats a symbol, whose overlapping occurrences
            the count would not handle.
    """
    # count the current symbols
    symbol_counts: dict[str, int] = {}
    for word, freq in words.items():
        for symbol in replay_wordpiece(word, merges):
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + freq
    total = sum(symbol_counts.values())

    def log_likelihood(counts: dict[str, int], size: int) -> float:
        return sum(count * math.log(count / size) for count in counts.values() if count > 0)

    # compute the gain of every candidate join
    before = log_likelihood(symbol_counts, total)
    gains: dict[tuple[str, str], float] = {}
    for (first, second), count in pair_counts(words, merges, replay_wordpiece).items():
        assert first != second, (first, second)
        after = dict(symbol_counts)
        after[first] -= count
        after[second] -= count
        after[first + second[len(CONTINUATION) :]] = count
        gains[(first, second)] = log_likelihood(after, total - count) - before

    # keep the pairs within rounding of the best gain
    best = max(gains.values())
    return sorted(pair for pair, gain in gains.items() if math.isclose(gain, best))


def greedy_walk(vocab: dict[str, int], word: str) -> tuple[list[list[str]], list[str]]:
    """Replay WordPiece encoding of `word` one position at a time.

    From each position the longest piece in `vocab` is taken, prefixed when it
    does not start the word. When no piece fits, the whole word is the unknown
    token.

    Args:
        vocab: Token to id.
        word: The word to cut.

    Returns:
        One row per position (word, start,
            the candidates tried longest first, the pick) and the tokens.
    """
    rows: list[list[str]] = []
    tokens: list[str] = []
    start = 0
    while start < len(word):
        # take the longest piece in the vocabulary
        tried: list[str] = []
        pick = ""
        for end in range(len(word), start, -1):
            candidate = (CONTINUATION if start else "") + word[start:end]
            tried.append(candidate)
            if candidate in vocab:
                pick = candidate
                break

        # record the position
        rows.append([word, str(start + 1), " ".join(tried), pick or "none, the word becomes <unk>"])
        if not pick:
            return rows, ["<unk>"]

        tokens.append(pick)
        start = end

    return rows, tokens


def digamma(count: int) -> float:
    """Return the digamma function at a positive integer.

    Args:
        count: The argument, at least 1.

    Returns:
        Minus Euler's constant plus the harmonic number of `count - 1`.
    """
    return -EULER_GAMMA + sum(1 / k for k in range(1, count))


def segmentations(form: str, scores: dict[str, float]) -> list[tuple[list[str], float]]:
    """Enumerate every way to spell `form` from the scored pieces, with its log probability.

    Args:
        form: The text to spell, space marker included.
        scores: Piece to log probability.

    Returns:
        Each segmentation and the sum of its
            pieces' log probabilities, best first.
    """
    if not form:
        return [([], 0.0)]

    # spell the rest of the form after each leading piece
    found: list[tuple[list[str], float]] = []
    for end in range(1, len(form) + 1):
        piece = form[:end]
        if piece in scores:
            for rest, score in segmentations(form[end:], scores):
                found.append(([piece, *rest], scores[piece] + score))

    return sorted(found, key=lambda item: -item[1])


# ===============================
# Table functions
# ===============================


def morphbpe() -> None:
    """Write the merge trace and the held-out encodings for the MorphBPE entry.

    Raises:
        AssertionError: If replaying the merges disagrees with the backend's own
            encoder, which would make the trace misleading.
    """
    # train both backends and read their merges
    context = TrainContext(special_tokens=SPECIAL_TOKENS, lexicon=toy_lexicon())
    char, morph = CharBpeTokenizer(), MorphBpeTokenizer()
    char.train(SENTENCES, VOCAB_SIZE, config=context)
    morph.train(SENTENCES, VOCAB_SIZE, config=context)
    merges = {"char": learned_merges(char), "morph": learned_merges(morph)}

    # check the replayed merges against the backends' encoders
    for key, tokenizer in (("char", char), ("morph", morph)):
        for word in [*CORPUS, *HELD_OUT]:
            replayed = encode_bpe(word, merge_ranks(merges[key]))
            assert replayed == tokenizer.encode(word), (key, word, replayed, tokenizer.encode(word))

    # write the merge trace
    steps = max(len(m) for m in merges.values())
    traces = {key: trace(TRACED_WORD, m, steps) for key, m in merges.items()}
    write_csv(
        "morphbpe-merges",
        [
            "Merge",
            "Character BPE joins",
            f"{TRACED_WORD} in Character BPE",
            "MorphBPE joins",
            f"{TRACED_WORD} in MorphBPE",
        ],
        [
            [
                str(k + 1),
                " + ".join(merges["char"][k]) if k < len(merges["char"]) else "no pair left",
                traces["char"][k],
                " + ".join(merges["morph"][k]) if k < len(merges["morph"]) else "no pair left",
                traces["morph"][k] or traces["morph"][len(merges["morph"]) - 1],
            ]
            for k in range(steps)
        ],
    )

    # write the held-out encodings
    write_csv(
        "morphbpe-encoding",
        ["Form", "In the corpus", "Stem and ending", "Character BPE", "MorphBPE"],
        [
            [
                word,
                "yes" if word in CORPUS else "no",
                " ".join(GOLD[word]),
                " ".join(char.encode(word)),
                " ".join(morph.encode(word)),
            ]
            for word in ["hiša", "mize", *HELD_OUT]
        ],
    )


def character_bpe() -> None:
    """Write the merge trace and the encodings for the Character BPE entry.

    Raises:
        AssertionError: If replaying the merges disagrees with the backend's own
            encoder.
    """
    # train the backend and read its merges
    context = TrainContext(special_tokens=SPECIAL_TOKENS)
    char = CharBpeTokenizer()
    char.train(SENTENCES, VOCAB_SIZE, config=context)
    merges = learned_merges(char)

    # write the merge trace
    rows = merge_rows(merges, CORPUS, char.vocab, TRACED_WORD)
    write_csv("character-bpe-merges", ["Merge", "Joins", "Count", "Tied pairs left", f"{TRACED_WORD} after it"], rows)

    # check the replayed merges against the backend's encoder
    ranks = merge_ranks(merges)
    for word in [*CORPUS, *HELD_OUT]:
        assert encode_bpe(word, ranks) == char.encode(word), (word, encode_bpe(word, ranks), char.encode(word))

    # write the encodings
    write_csv(
        "character-bpe-encoding",
        ["Form", "In the corpus", "Tokens", "Token count"],
        [
            [word, "yes" if word in CORPUS else "no", " ".join(char.encode(word)), str(len(char.encode(word)))]
            for word in ["hiša", "mize", *HELD_OUT, "hišaq"]
        ],
    )


def byte_level_bpe() -> None:
    """Write the merge trace and the encodings for the Byte-level BPE entry.

    The trainer sees every word as the printable form of its UTF-8 bytes with
    the space marker in front, so the pair counts are replayed on those symbols.

    Raises:
        AssertionError: If a learned merge breaks the merge rule, or if
            replaying the merges disagrees with the backend's own encoder.
    """
    # train the backend and read its merges
    bpe = BpeTokenizer()
    bpe.train(SENTENCES, BYTE_VOCAB_SIZE, config=TrainContext(special_tokens=SPECIAL_TOKENS))
    merges = learned_merges(bpe)

    def as_bytes(word: str) -> str:
        return "".join(piece for piece, _ in bpe._tokenizer.pre_tokenizer.pre_tokenize_str(word))

    # write the merge trace
    words = {as_bytes(form): count for form, count in CORPUS.items()}
    rows = merge_rows(merges, words, bpe.vocab, as_bytes(TRACED_WORD))
    write_csv("byte-level-bpe-merges", ["Merge", "Joins", "Count", "Tied pairs left", f"{TRACED_WORD} after it"], rows)

    # check the replayed merges against the backend's encoder
    ranks = merge_ranks(merges)
    for word in [*CORPUS, *HELD_OUT]:
        replayed = encode_bpe(as_bytes(word), ranks)
        assert replayed == bpe.encode(word), (word, replayed, bpe.encode(word))

    # write the encodings
    encoding_rows: list[list[str]] = []
    for word in ["hiša", "mize", *HELD_OUT, "hišaq", "čas"]:
        spans = _token_spans(bpe, word)
        assert spans is not None, word
        encoding_rows.append(
            [
                word,
                "yes" if word in CORPUS else "no",
                " ".join(bpe.encode(word)),
                str(len(bpe.encode(word))),
                " ".join(_segments(spans, word)),
            ]
        )
    write_csv(
        "byte-level-bpe-encoding",
        ["Form", "In the corpus", "Tokens", "Token count", "Cuts seen by the morph metrics"],
        encoding_rows,
    )


def wordpiece() -> None:
    """Write the merge trace, the encoding walk and the encodings for the WordPiece entry.

    The WordPiece trainer keeps only the vocabulary, so the merge order is
    recovered by running the BPE trainer it wraps with the same settings and
    checking that both produce the same vocabulary.

    Raises:
        AssertionError: If the two trainers disagree, a learned merge breaks the
            merge rule, or the replayed encoding disagrees with the backend.
    """
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers

    # train the WordPiece backend
    wordpiece_tokenizer = WordPieceTokenizer()
    wordpiece_tokenizer.train(SENTENCES, WORDPIECE_VOCAB_SIZE, config=TrainContext(special_tokens=SPECIAL_TOKENS))

    # recover the merge order with the BPE trainer it wraps
    bpe = Tokenizer(models.BPE(unk_token=SPECIAL_TOKENS[0], continuing_subword_prefix=CONTINUATION))
    bpe.normalizer = normalizers.NFC()
    bpe.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=WORDPIECE_VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        continuing_subword_prefix=CONTINUATION,
        show_progress=False,
    )
    bpe.train_from_iterator(SENTENCES, trainer=trainer)
    assert set(bpe.get_vocab()) == set(wordpiece_tokenizer.vocab), (bpe.get_vocab(), wordpiece_tokenizer.vocab)
    merges = [tuple(pair) for pair in json.loads(bpe.to_str())["model"]["merges"]]

    # write the merge trace
    rows = merge_rows(merges, CORPUS, wordpiece_tokenizer.vocab, TRACED_WORD, replay_wordpiece)
    for k, row in enumerate(rows):
        row.append(", ".join(" + ".join(pair) for pair in likelihood_picks(CORPUS, merges[:k])))
    write_csv(
        "wordpiece-merges",
        ["Merge", "Joins", "Count", "Tied pairs left", f"{TRACED_WORD} after it", "The paper's rule would join"],
        rows,
    )

    # write the encoding walk
    walk_rows: list[list[str]] = []
    for word in ["mize", "hišami"]:
        word_rows, tokens = greedy_walk(wordpiece_tokenizer.vocab, word)
        assert tokens == wordpiece_tokenizer.encode(word), (word, tokens, wordpiece_tokenizer.encode(word))
        walk_rows.extend(word_rows)
    write_csv("wordpiece-walk", ["Form", "From character", "Tried, longest first", "Picked"], walk_rows)

    # write the encodings
    encoding_rows: list[list[str]] = []
    for word in ["hiša", "mize", *HELD_OUT, "hišaq", "hiš"]:
        _, tokens = greedy_walk(wordpiece_tokenizer.vocab, word)
        assert tokens == wordpiece_tokenizer.encode(word), (word, tokens, wordpiece_tokenizer.encode(word))
        spans = _token_spans(wordpiece_tokenizer, word)
        assert spans is not None, word
        encoding_rows.append(
            [
                word,
                "yes" if word in CORPUS else "no",
                " ".join(tokens),
                str(len(tokens)),
                " ".join(_segments(spans, word)),
            ]
        )
    write_csv(
        "wordpiece-encoding",
        ["Form", "In the corpus", "Tokens", "Token count", "Cuts seen by the morph metrics"],
        encoding_rows,
    )


def unigram() -> None:
    """Write the vocabulary, the candidate segmentations and the encodings for the Unigram entry.

    Training runs inside the SentencePiece library, so two of its claims are
    checked here instead: each learned log probability is reproduced from the
    piece's count in the best segmentation of the corpus through the digamma
    rule, and each encoding is the highest-scoring segmentation over the
    vocabulary.

    Raises:
        AssertionError: If a score or an encoding is not reproduced.
    """
    from sentencepiece import sentencepiece_model_pb2 as model_pb2

    # train the backend and read its piece scores
    unigram_tokenizer = UnigramTokenizer()
    unigram_tokenizer.train(SENTENCES, UNIGRAM_VOCAB_SIZE, config=TrainContext(special_tokens=SPECIAL_TOKENS))
    model = model_pb2.ModelProto()
    model.ParseFromString(unigram_tokenizer._model_bytes)
    scores = {piece.piece: piece.score for piece in model.pieces if piece.type == piece.NORMAL}

    # count the pieces in the best segmentation of the corpus
    counts: dict[str, int] = {}
    for form, freq in CORPUS.items():
        for token in unigram_tokenizer.encode(form):
            counts[token] = counts.get(token, 0) + freq
    total = sum(counts.values())

    # check each score against the digamma rule and write the vocabulary
    vocab_rows: list[list[str]] = []
    for piece, score in scores.items():
        if piece in counts:
            expected = digamma(counts[piece]) - digamma(total)
            assert math.isclose(score, expected, abs_tol=1e-3), (piece, score, expected)
            vocab_rows.append([piece, f"{score:.3f}", str(counts[piece]), f"{expected:.3f}"])
        else:
            vocab_rows.append([piece, f"{score:.3f}", "0", "kept as a character"])
    write_csv(
        "unigram-vocabulary",
        ["Piece", "Log probability", "Count in the corpus", "Digamma of count minus digamma of total"],
        vocab_rows,
    )

    # write the candidate segmentations
    segmentation_rows: list[list[str]] = []
    for word in ["mize", "hišami"]:
        candidates = segmentations(SPACE_MARKER + word, scores)
        assert candidates[0][0] == unigram_tokenizer.encode(word), (word, candidates[0], unigram_tokenizer.encode(word))
        for rank, (pieces, score) in enumerate(candidates):
            segmentation_rows.append([word, " ".join(pieces), f"{score:.3f}", "yes" if rank == 0 else "no"])
    write_csv("unigram-segmentations", ["Form", "Segmentation", "Log probability", "Picked"], segmentation_rows)

    # write the encodings
    encoding_rows: list[list[str]] = []
    for word in ["hiša", "mize", *HELD_OUT, "hišaq", "hiš", "Hiša..."]:
        tokens = unigram_tokenizer.encode(word)
        ids = unigram_tokenizer.encode_ids(word)
        spans = _token_spans(unigram_tokenizer, word)
        assert spans is not None, word
        encoding_rows.append(
            [
                word,
                "yes" if word in CORPUS else "no",
                " ".join(tokens),
                " ".join(str(i) for i in ids),
                str(len(tokens)),
                " ".join(_segments(spans, word)),
            ]
        )
    write_csv(
        "unigram-encoding",
        ["Form", "In the corpus", "Tokens", "Ids", "Token count", "Cuts seen by the morph metrics"],
        encoding_rows,
    )


def morphpiece() -> None:
    """Write the vocabulary split and the encodings for the MorphPiece entry.

    The backend keeps a table of lexicon forms and a byte-level BPE for every
    other word, so two claims are checked here: a form in the table is encoded
    as exactly its lexicon morphemes, and any other word is encoded exactly as
    the inner BPE encodes it.

    Raises:
        AssertionError: If an encoding does not follow the path the entry
            describes.
    """
    # train the backend
    lexicon = toy_lexicon()
    piece = MorphPieceTokenizer()
    piece.train(SENTENCES, MORPHPIECE_VOCAB_SIZE, config=TrainContext(special_tokens=SPECIAL_TOKENS, lexicon=lexicon))

    # count the morphemes and write the vocabulary split
    morpheme_freqs: dict[str, int] = {}
    for form, count in CORPUS.items():
        for morpheme in GOLD[form]:
            morpheme_freqs[morpheme] = morpheme_freqs.get(morpheme, 0) + count
    bpe_vocab = piece._bpe.get_vocab()
    vocabulary_rows = [
        [morpheme, str(morpheme_freqs[morpheme]), "yes" if morpheme in bpe_vocab else "no", str(piece.vocab[morpheme])]
        for morpheme in sorted(piece.morpheme_tokens | (set(morpheme_freqs) & set(bpe_vocab)))
    ]
    write_csv(
        "morphpiece-vocabulary",
        ["Morpheme token", "Count in the corpus", "Also a BPE token", "Id"],
        vocabulary_rows,
    )

    # write the vocabulary sizes
    shared = len(vocabulary_rows) - len(piece.morpheme_tokens)
    write_csv(
        "morphpiece-sizes",
        ["Asked", "BPE path", "Morphemes kept", "Of which already BPE tokens", "Final"],
        [
            [
                str(MORPHPIECE_VOCAB_SIZE),
                str(len(bpe_vocab)),
                str(len(vocabulary_rows)),
                str(shared),
                str(len(piece.vocab)),
            ]
        ],
    )

    # write the inner BPE merges
    merges = [
        tuple(pair) if isinstance(pair, list) else tuple(pair.split(" "))
        for pair in json.loads(piece._bpe.to_str())["model"]["merges"]
    ]
    write_csv(
        "morphpiece-merges", ["Merge", "Joins"], [[str(k + 1), " + ".join(pair)] for k, pair in enumerate(merges)]
    )

    # check each encoding follows its path and write the encodings
    encoding_rows: list[list[str]] = []
    for word in ["hiša", "mize", *HELD_OUT, "Hiša", "hiš"]:
        tokens = piece.encode(word)
        in_table = word in piece._table
        expected = GOLD[word] if in_table else piece._bpe.encode(word).tokens
        assert tokens == expected, (word, tokens, expected)
        spans = _token_spans(piece, word)
        assert spans is not None, word
        encoding_rows.append(
            [
                word,
                "yes" if in_table else "no",
                " ".join(tokens),
                " ".join(str(i) for i in piece.encode_ids(word)),
                str(len(tokens)),
                " ".join(_segments(spans, word)),
            ]
        )
    write_csv(
        "morphpiece-encoding",
        ["Form", "In the table", "Tokens", "Ids", "Token count", "Cuts seen by the morph metrics"],
        encoding_rows,
    )


if __name__ == "__main__":
    character_bpe()
    byte_level_bpe()
    morphbpe()
    wordpiece()
    unigram()
    morphpiece()
