"""Fixtures shared by the pretokenize tests: a tiny exported tokenizer and corpus."""

import gzip
import json
from pathlib import Path

import pytest

from slm4ie.data.pretokenize.config import PretokenizeEntry

_SPECIAL = ["<pad>", "<unk>", "<s>", "</s>", "<mask>"]
_TRAINING_TEXT = [
    "hiša ob cesti je velika",
    "mačka spi v hiši",
    "pes teče po travniku in laja",
    "mesto ima veliko hiš in cest",
] * 50

# documents per source; each line is one sentence
CORPUS = {
    "c4": [
        ("c4:doc-0", "hiša ob cesti\nmačka spi v hiši\npes teče"),
        ("c4:doc-1", "mesto ima veliko hiš in cest " * 12 + "\nkratko"),
        ("c4:doc-2", "pes"),
    ],
    "kas": [
        ("kas:doc-0", "velika hiša\n\nmačka laja\npes spi po travniku"),
        ("kas:doc-1", "travnik"),
    ],
}


@pytest.fixture(autouse=True)
def _logs_in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Send the per-shard run logs to `tmp_path`, not the repo's `logs/`."""
    monkeypatch.setattr("slm4ie.data.pretokenize.run.stamped_log_dir", lambda name: tmp_path / "logs" / name)


@pytest.fixture(scope="session")
def tokenizer_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train a tiny byte-level BPE and export it as the ModernBERT target would.

    Args:
        tmp_path_factory: Session-wide temporary directories.

    Returns:
        The exported tokenizer folder.
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    from transformers import PreTrainedTokenizerFast

    # train the byte-level BPE
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=300, special_tokens=_SPECIAL, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    tokenizer.train_from_iterator(_TRAINING_TEXT, trainer=trainer)

    # wrap every input in <s> ... </s> and save it
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> $B:1 </s>:1",
        special_tokens=[("<s>", tokenizer.token_to_id("<s>")), ("</s>", tokenizer.token_to_id("</s>"))],
    )
    exported = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        add_prefix_space=True,
        pad_token="<pad>",
        unk_token="<unk>",
        cls_token="<s>",
        sep_token="</s>",
        mask_token="<mask>",
    )
    out = tmp_path_factory.mktemp("tokenizer")
    exported.save_pretrained(str(out))
    return out


def write_corpus(corpus_dir: Path, documents: dict[str, list[tuple[str, str]]] = CORPUS) -> list[Path]:
    """Write documents as one gzipped JSONL shard per source.

    Args:
        corpus_dir: The corpus stage folder.
        documents: Source to its `(doc_id, text)` documents.

    Returns:
        The shards written, in source order.
    """
    shards = []
    for source, docs in documents.items():
        shard = corpus_dir / source / "00000.jsonl.gz"
        shard.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(shard, "wt", encoding="utf-8") as handle:
            for doc_id, text in docs:
                record = {"id": doc_id, "text": text, "metadata": {"dataset": source}}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        shards.append(shard)
    return shards


def make_entry(tmp_path: Path, tokenizer_dir: Path, **overrides) -> PretokenizeEntry:
    """Build an entry over the test corpus and tokenizer.

    Args:
        tmp_path: Test directory, holding `corpus/` and `out/`.
        tokenizer_dir: The exported tokenizer folder.
        **overrides: Entry fields to change.

    Returns:
        The entry.
    """
    settings = {
        "key": "corpus/tiny-16",
        "corpus_dir": tmp_path / "corpus",
        "tokenizer_dir": tokenizer_dir,
        "output_dir": tmp_path / "out",
        "max_seq_len": 16,
        "min_chunk_tokens": 4,
        "formats": ("mds", "mds-streaming", "parquet"),
        "validation_share": 0.4,
        "min_validation_docs": 1,
        "seed": 13,
        "shard_size_bytes": 1 << 20,
        "language": "sl",
    }
    return PretokenizeEntry(**{**settings, **overrides})
