"""Generate the worked-example tables of the subword-tokenizers topic.

Every table is produced by training the project's own tokenizer backends on a
toy corpus, so an entry never shows an example its code would not reproduce.

    uv run --group tokenizers python experiments/reference/subword-tokenizers/examples.py
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from slm4ie.tokenizers.backends.char_bpe import CharBpeTokenizer
from slm4ie.tokenizers.backends.hf_bpe import BpeTokenizer
from slm4ie.tokenizers.backends.hf_wordpiece import WordPieceTokenizer
from slm4ie.tokenizers.backends.morph_bpe import MorphBpeTokenizer
from slm4ie.tokenizers.base import TrainContext
from slm4ie.tokenizers.bpe_core import encode_bpe, merge_ranks
from slm4ie.tokenizers.metrics import _segments, _token_spans
from slm4ie.tokenizers.morphology import MorphemeSegmentation, MorphLexicon

TABLES = Path(__file__).resolve().parent / "tables"

#: Toy corpus: inflected forms of `hiša` (house) and `miza` (table), with counts.
CORPUS: Dict[str, int] = {"hiša": 5, "hiše": 4, "hišo": 3, "miza": 4, "mize": 3, "mizo": 2}
#: Stem and ending of every corpus form, plus the held-out forms encoded below.
GOLD: Dict[str, List[str]] = {
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
HELD_OUT: List[str] = ["hišami", "mizami"]
SPECIAL_TOKENS: List[str] = ["<unk>"]
#: Eight characters, six merges and the unknown token.
VOCAB_SIZE = 15
#: The 256 byte symbols, six merges and the unknown token.
BYTE_VOCAB_SIZE = 256 + 6 + 1
#: Eight word-start characters, six continuation characters, six merges and the unknown token.
WORDPIECE_VOCAB_SIZE = 8 + 6 + 6 + 1
#: WordPiece marks a piece that continues a word with this prefix.
CONTINUATION = "##"
TRACED_WORD = "hiša"
Replay = Callable[[str, List[Tuple[str, str]]], List[str]]


def toy_lexicon() -> MorphLexicon:
    """Build the lexicon the toy corpus is segmented with.

    Returns:
        MorphLexicon: Stem and ending for every corpus form.
    """
    lexicon = MorphLexicon()
    for form in CORPUS:
        lexicon.by_form[form] = MorphemeSegmentation(form, GOLD[form], ["stem", "suffix"], lemma=form)
    return lexicon


def learned_merges(tokenizer: CharBpeTokenizer | MorphBpeTokenizer) -> List[Tuple[str, str]]:
    """Read the ordered merges out of a trained backend.

    Args:
        tokenizer: A trained backend wrapping a HuggingFace BPE model.

    Returns:
        List[Tuple[str, str]]: Merge pairs in the order they were learned.
    """
    merges = json.loads(tokenizer._tokenizer.to_str())["model"]["merges"]
    return [tuple(pair) if isinstance(pair, list) else tuple(pair.split(" ")) for pair in merges]


def trace(word: str, merges: List[Tuple[str, str]], steps: int) -> List[str]:
    """Show `word` after each of the first `steps` merges.

    Args:
        word (str): The word to follow.
        merges (List[Tuple[str, str]]): Ordered learned merges.
        steps (int): Number of rows to produce; a row past the last merge is empty.

    Returns:
        List[str]: The pieces of `word`, space-separated, one entry per merge.
    """
    return [" ".join(encode_bpe(word, merge_ranks(merges[: k + 1]))) if k < len(merges) else "" for k in range(steps)]


def write_csv(name: str, header: List[str], rows: List[List[str]]) -> None:
    """Write one table under `tables/`.

    Args:
        name (str): File name without extension.
        header (List[str]): Column titles.
        rows (List[List[str]]): Table body.
    """
    TABLES.mkdir(exist_ok=True)
    with (TABLES / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def morphbpe() -> None:
    """Write the merge trace and the held-out encodings for the MorphBPE entry.

    Raises:
        AssertionError: If replaying the merges disagrees with the backend's own
            encoder, which would make the trace misleading.
    """
    sentences = [form for form, count in CORPUS.items() for _ in range(count)]
    context = TrainContext(special_tokens=SPECIAL_TOKENS, lexicon=toy_lexicon())
    char, morph = CharBpeTokenizer(), MorphBpeTokenizer()
    char.train(sentences, VOCAB_SIZE, config=context)
    morph.train(sentences, VOCAB_SIZE, config=context)
    merges = {"char": learned_merges(char), "morph": learned_merges(morph)}

    for key, tokenizer in (("char", char), ("morph", morph)):
        for word in [*CORPUS, *HELD_OUT]:
            replayed = encode_bpe(word, merge_ranks(merges[key]))
            assert replayed == tokenizer.encode(word), (key, word, replayed, tokenizer.encode(word))

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


def replay_bpe(word: str, merges: List[Tuple[str, str]]) -> List[str]:
    """Apply `merges` to the characters of `word` as plain BPE does.

    Args:
        word (str): The word to cut.
        merges (List[Tuple[str, str]]): Ordered learned merges.

    Returns:
        List[str]: The pieces of `word`.
    """
    return encode_bpe(word, merge_ranks(merges))


def wordpiece_symbols(word: str) -> List[str]:
    """Spell `word` as the WordPiece trainer sees it: a bare first character, then prefixed ones.

    Args:
        word (str): The word to spell.

    Returns:
        List[str]: One symbol per character.
    """
    return [word[0], *(CONTINUATION + char for char in word[1:])]


def replay_wordpiece(word: str, merges: List[Tuple[str, str]]) -> List[str]:
    """Apply `merges` in order to the prefixed symbols of `word`, each everywhere it occurs.

    A join keeps the prefix of its first symbol and drops the prefix of its
    second, which is how the trainer spells a joined symbol.

    Args:
        word (str): The word to cut.
        merges (List[Tuple[str, str]]): Ordered learned merges over prefixed symbols.

    Returns:
        List[str]: The pieces of `word`, prefixed where they continue it.
    """
    symbols = wordpiece_symbols(word)
    for first, second in merges:
        joined = first + second[len(CONTINUATION) :]
        out: List[str] = []
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
    words: Dict[str, int], merges: List[Tuple[str, str]], replay: Replay = replay_bpe
) -> Dict[Tuple[str, str], int]:
    """Count adjacent symbol pairs over `words` after replaying `merges`.

    Args:
        words (Dict[str, int]): Word to corpus count.
        merges (List[Tuple[str, str]]): Merges applied before counting.
        replay (Replay): How a word is cut after the merges so far.

    Returns:
        Dict[Tuple[str, str], int]: Pair to weighted count.
    """
    counts: Dict[Tuple[str, str], int] = {}
    for word, freq in words.items():
        symbols = replay(word, merges)
        for pair in zip(symbols, symbols[1:]):
            counts[pair] = counts.get(pair, 0) + freq
    return counts


def merge_rows(
    merges: List[Tuple[str, str]],
    words: Dict[str, int],
    vocab: Dict[str, int],
    traced: str,
    replay: Replay = replay_bpe,
) -> List[List[str]]:
    """Replay the pair counts behind every learned merge and check the merge rule.

    The trainer runs inside the HuggingFace `tokenizers` library, so the rule
    is checked here instead: every learned pair must be the most frequent one at
    its step, and a tie must go to the pair whose symbols entered the vocabulary
    first.

    Args:
        merges (List[Tuple[str, str]]): Ordered learned merges.
        words (Dict[str, int]): Training word to corpus count, in the symbols the
            trainer saw.
        vocab (Dict[str, int]): Token to id, which fixes the tie order.
        traced (str): The word to show after each merge, in the same symbols.
        replay (Replay): How a word is cut after the merges so far.

    Returns:
        List[List[str]]: One row per merge: number, pair, count, the tied pairs
            it beat, and `traced` after it.

    Raises:
        AssertionError: If a learned merge is not a most frequent pair, or a tie
            is broken another way.
    """
    rows: List[List[str]] = []
    for k, pair in enumerate(merges):
        counts = pair_counts(words, merges[:k], replay)
        top = max(counts.values())
        tied = sorted(p for p, n in counts.items() if n == top)
        assert counts[pair] == top, (k, pair, counts)
        assert pair == min(tied, key=lambda p: (vocab[p[0]], vocab[p[1]])), (k, pair, tied)
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


def character_bpe() -> None:
    """Write the merge trace and the encodings for the Character BPE entry.

    Raises:
        AssertionError: If replaying the merges disagrees with the backend's own
            encoder.
    """
    sentences = [form for form, count in CORPUS.items() for _ in range(count)]
    context = TrainContext(special_tokens=SPECIAL_TOKENS)
    char = CharBpeTokenizer()
    char.train(sentences, VOCAB_SIZE, config=context)
    merges = learned_merges(char)

    rows = merge_rows(merges, CORPUS, char.vocab, TRACED_WORD)
    write_csv("character-bpe-merges", ["Merge", "Joins", "Count", "Tied pairs left", f"{TRACED_WORD} after it"], rows)

    ranks = merge_ranks(merges)
    for word in [*CORPUS, *HELD_OUT]:
        assert encode_bpe(word, ranks) == char.encode(word), (word, encode_bpe(word, ranks), char.encode(word))
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
    sentences = [form for form, count in CORPUS.items() for _ in range(count)]
    bpe = BpeTokenizer()
    bpe.train(sentences, BYTE_VOCAB_SIZE, config=TrainContext(special_tokens=SPECIAL_TOKENS))
    merges = learned_merges(bpe)

    def as_bytes(word: str) -> str:
        return "".join(piece for piece, _ in bpe._tokenizer.pre_tokenizer.pre_tokenize_str(word))

    words = {as_bytes(form): count for form, count in CORPUS.items()}
    rows = merge_rows(merges, words, bpe.vocab, as_bytes(TRACED_WORD))
    write_csv("byte-level-bpe-merges", ["Merge", "Joins", "Count", "Tied pairs left", f"{TRACED_WORD} after it"], rows)

    ranks = merge_ranks(merges)
    for word in [*CORPUS, *HELD_OUT]:
        replayed = encode_bpe(as_bytes(word), ranks)
        assert replayed == bpe.encode(word), (word, replayed, bpe.encode(word))
    encoding_rows: List[List[str]] = []
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


def likelihood_picks(words: Dict[str, int], merges: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Return the pairs the original WordPiece rule would join next, more than one when they tie.

    The rule joins the pair whose merge most raises the log-likelihood of the
    corpus under a unigram model over the current symbols. The gain is computed
    exactly from the symbol counts before and after the join.

    Args:
        words (Dict[str, int]): Training word to corpus count.
        merges (List[Tuple[str, str]]): Merges learned so far.

    Returns:
        List[Tuple[str, str]]: Every pair whose gain is within rounding of the largest.

    Raises:
        AssertionError: If a pair repeats a symbol, whose overlapping occurrences
            the count would not handle.
    """
    symbol_counts: Dict[str, int] = {}
    for word, freq in words.items():
        for symbol in replay_wordpiece(word, merges):
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + freq
    total = sum(symbol_counts.values())

    def log_likelihood(counts: Dict[str, int], size: int) -> float:
        return sum(count * math.log(count / size) for count in counts.values() if count > 0)

    before = log_likelihood(symbol_counts, total)
    gains: Dict[Tuple[str, str], float] = {}
    for (first, second), count in pair_counts(words, merges, replay_wordpiece).items():
        assert first != second, (first, second)
        after = dict(symbol_counts)
        after[first] -= count
        after[second] -= count
        after[first + second[len(CONTINUATION) :]] = count
        gains[(first, second)] = log_likelihood(after, total - count) - before
    best = max(gains.values())
    return sorted(pair for pair, gain in gains.items() if math.isclose(gain, best))


def greedy_walk(vocab: Dict[str, int], word: str) -> Tuple[List[List[str]], List[str]]:
    """Replay WordPiece encoding of `word` one position at a time.

    From each position the longest piece in `vocab` is taken, prefixed when it
    does not start the word. When no piece fits, the whole word is the unknown
    token.

    Args:
        vocab (Dict[str, int]): Token to id.
        word (str): The word to cut.

    Returns:
        Tuple[List[List[str]], List[str]]: One row per position (word, start,
            the candidates tried longest first, the pick) and the tokens.
    """
    rows: List[List[str]] = []
    tokens: List[str] = []
    start = 0
    while start < len(word):
        tried: List[str] = []
        pick = ""
        for end in range(len(word), start, -1):
            candidate = (CONTINUATION if start else "") + word[start:end]
            tried.append(candidate)
            if candidate in vocab:
                pick = candidate
                break
        rows.append([word, str(start + 1), " ".join(tried), pick or "none, the word becomes <unk>"])
        if not pick:
            return rows, ["<unk>"]
        tokens.append(pick)
        start = end
    return rows, tokens


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

    sentences = [form for form, count in CORPUS.items() for _ in range(count)]
    wordpiece_tokenizer = WordPieceTokenizer()
    wordpiece_tokenizer.train(sentences, WORDPIECE_VOCAB_SIZE, config=TrainContext(special_tokens=SPECIAL_TOKENS))

    bpe = Tokenizer(models.BPE(unk_token=SPECIAL_TOKENS[0], continuing_subword_prefix=CONTINUATION))
    bpe.normalizer = normalizers.NFC()
    bpe.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=WORDPIECE_VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        continuing_subword_prefix=CONTINUATION,
        show_progress=False,
    )
    bpe.train_from_iterator(sentences, trainer=trainer)
    assert set(bpe.get_vocab()) == set(wordpiece_tokenizer.vocab), (bpe.get_vocab(), wordpiece_tokenizer.vocab)
    merges = [tuple(pair) for pair in json.loads(bpe.to_str())["model"]["merges"]]

    rows = merge_rows(merges, CORPUS, wordpiece_tokenizer.vocab, TRACED_WORD, replay_wordpiece)
    for k, row in enumerate(rows):
        row.append(", ".join(" + ".join(pair) for pair in likelihood_picks(CORPUS, merges[:k])))
    write_csv(
        "wordpiece-merges",
        ["Merge", "Joins", "Count", "Tied pairs left", f"{TRACED_WORD} after it", "The paper's rule would join"],
        rows,
    )

    walk_rows: List[List[str]] = []
    for word in ["mize", "hišami"]:
        word_rows, tokens = greedy_walk(wordpiece_tokenizer.vocab, word)
        assert tokens == wordpiece_tokenizer.encode(word), (word, tokens, wordpiece_tokenizer.encode(word))
        walk_rows.extend(word_rows)
    write_csv("wordpiece-walk", ["Form", "From character", "Tried, longest first", "Picked"], walk_rows)

    encoding_rows: List[List[str]] = []
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


if __name__ == "__main__":
    character_bpe()
    byte_level_bpe()
    morphbpe()
    wordpiece()
