"""Export trained tokenizer artifacts as HuggingFace tokenizers.

Bridges the training-time backends to the HuggingFace ecosystem for the
LM-pretraining phase. `bpe`, `charbpe`, `wordpiece`, and `morphbpe` wrap their
`tokenizer.json` as a `PreTrainedTokenizerFast`; `unigram` is converted from its
SentencePiece model; `morphpiece` returns the custom slow tokenizer. Every
returned object supports `decode` and offset mapping (native for the fast ones,
via `encode_with_offsets` for MorphPiece).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from slm4ie.tokenizers.backends.morph_piece import MorphPieceTokenizer
from slm4ie.tokenizers.hf_morphpiece import SPECIAL_TOKEN_ROLES, MorphPieceHFTokenizer


def _read_name(artifact_dir: Path) -> str:
    """Return the backend name recorded in the artifact metadata.

    Args:
        artifact_dir (Path): Directory holding `metadata.json`.

    Returns:
        str: The registry name of the tokenizer.

    Raises:
        FileNotFoundError: If no metadata sidecar is present.
    """
    meta_path = artifact_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No tokenizer metadata under {artifact_dir}")
    return json.loads(meta_path.read_text(encoding="utf-8"))["name"]


def _roles_present(tokenizer_obj: Any) -> Dict[str, str]:
    """Map the project's special tokens that exist in `tokenizer_obj` to roles.

    Args:
        tokenizer_obj (Any): A `tokenizers.Tokenizer`.

    Returns:
        Dict[str, str]: HuggingFace role to token string.
    """
    vocab = tokenizer_obj.get_vocab()
    return {role: token for token, role in SPECIAL_TOKEN_ROLES.items() if token in vocab}


def _unigram_to_tokenizer(artifact_dir: Path) -> Any:
    """Rebuild a `tokenizers` Unigram tokenizer from a SentencePiece model.

    Args:
        artifact_dir (Path): Directory holding `spm.model`.

    Returns:
        Any: A configured `tokenizers.Tokenizer` (Unigram, Metaspace).
    """
    import sentencepiece as spm
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    processor = spm.SentencePieceProcessor(model_file=str(artifact_dir / "spm.model"))
    vocab = [(processor.id_to_piece(i), processor.get_score(i)) for i in range(processor.get_piece_size())]
    unk_id = processor.unk_id()
    tokenizer = Tokenizer(models.Unigram(vocab, unk_id=unk_id if unk_id >= 0 else None, byte_fallback=False))
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    tokenizer.decoder = decoders.Metaspace()
    return tokenizer


def to_hf(artifact_dir: Path) -> Any:
    """Load a trained artifact as a HuggingFace tokenizer.

    Args:
        artifact_dir (Path): Directory holding a trained tokenizer artifact.

    Returns:
        Any: A `PreTrainedTokenizerFast` for the fast backends, or a
            `MorphPieceHFTokenizer` (slow) for `morphpiece`.
    """
    from transformers import PreTrainedTokenizerFast

    artifact_dir = Path(artifact_dir)
    name = _read_name(artifact_dir)

    if name == "morphpiece":
        return MorphPieceHFTokenizer(MorphPieceTokenizer.load(artifact_dir))

    if name == "unigram":
        tokenizer_obj = _unigram_to_tokenizer(artifact_dir)
    else:
        from tokenizers import Tokenizer

        tokenizer_obj = Tokenizer.from_file(str(artifact_dir / "tokenizer.json"))

    return PreTrainedTokenizerFast(tokenizer_object=tokenizer_obj, **_roles_present(tokenizer_obj))


def load_pretrained(artifact_dir: Path) -> Any:
    """Explicit loader returning the HuggingFace tokenizer for any backend.

    This is the entry point the pretraining code uses; it works uniformly for
    the fast backends and for MorphPiece (which is not loadable via
    `AutoTokenizer`).

    Args:
        artifact_dir (Path): Directory holding a trained tokenizer artifact.

    Returns:
        Any: The HuggingFace tokenizer object.
    """
    return to_hf(artifact_dir)


def save_pretrained_dir(artifact_dir: Path, out_dir: Optional[Path] = None) -> Path:
    """Write a HuggingFace tokenizer directory next to (or into) the artifact.

    For the fast backends this enables `AutoTokenizer.from_pretrained(out_dir)`.
    MorphPiece is saved too but must be reloaded with `load_pretrained` (its
    custom class is not registered with `AutoTokenizer`).

    Args:
        artifact_dir (Path): Directory holding a trained tokenizer artifact.
        out_dir (Path): Destination; defaults to `artifact_dir`.

    Returns:
        Path: The directory written.
    """
    artifact_dir = Path(artifact_dir)
    destination = Path(out_dir) if out_dir is not None else artifact_dir
    destination.mkdir(parents=True, exist_ok=True)
    to_hf(artifact_dir).save_pretrained(str(destination))
    return destination
