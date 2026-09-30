"""Generate the worked-example tables of the subword-tokenizers topic.

Every table is produced by training the project's own tokenizer backends on a
toy corpus, so an entry never shows an example its code would not reproduce.

    uv run --group tokenizers python experiments/reference/subword-tokenizers/examples.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

from slm4ie.tokenizers.backends.char_bpe import CharBpeTokenizer
from slm4ie.tokenizers.backends.morph_bpe import MorphBpeTokenizer
from slm4ie.tokenizers.base import TrainContext
from slm4ie.tokenizers.bpe_core import encode_bpe, merge_ranks
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
TRACED_WORD = "hiša"


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


if __name__ == "__main__":
    morphbpe()
