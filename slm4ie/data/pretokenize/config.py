"""The pretokenize registry `configs/data/pretokenize.yaml` loaded into `PretokenizeConfig`."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from slm4ie.data.pretokenize.formats import FORMATS
from slm4ie.utils.config import load_yaml
from slm4ie.utils.io import resolve_project_path

# ===============================
# Constants
# ===============================

# an entry key is its output path: `<corpus>/<tokenizer>-<length>`
_KEY_PATTERN = re.compile(r"^[a-z0-9_-]+/[a-z0-9_.-]+$")
_ROOTS = ("corpus", "tokenizers", "output")
_ENTRY_FIELDS = frozenset({"corpus", "tokenizer", "max_seq_len", "language"})
_DEFAULTS = {
    "formats": ["mds", "parquet"],
    "min_chunk_tokens": 8,
    "validation_share": 0.001,
    "min_validation_docs": 10_000,
    "seed": 13,
    "shard_size_mb": 64,
}
BYTES_PER_MB = 1 << 20

# ===============================
# Config classes
# ===============================


@dataclass(frozen=True)
class PretokenizeEntry:
    """One conversion: a corpus stage, an exported tokenizer and a length.

    Attributes:
        key: The entry key, which is also its path under the output root.
        corpus_dir: The corpus stage read, holding `<dataset>/*.jsonl.gz`.
        tokenizer_dir: The exported HuggingFace tokenizer folder.
        output_dir: Where the formats and the manifest are written.
        max_seq_len: Longest sample in tokens, special tokens included.
        min_chunk_tokens: Shorter chunks are dropped and counted.
        formats: Output writers to run, by registered name.
        validation_share: Share of documents in the validation split.
        min_validation_docs: Fewest validation documents; raises the share
            when the share alone would give fewer.
        seed: Seed of the document hash that assigns the split.
        shard_size_bytes: Target size of one output shard.
        language: Language tag logged with the lineage.
    """

    key: str
    corpus_dir: Path
    tokenizer_dir: Path
    output_dir: Path
    max_seq_len: int
    min_chunk_tokens: int
    formats: tuple[str, ...]
    validation_share: float
    min_validation_docs: int
    seed: int
    shard_size_bytes: int
    language: str


@dataclass
class PretokenizeConfig:
    """Resolved contents of `configs/data/pretokenize.yaml`.

    Attributes:
        entries: Conversions by key.
        mlflow: The `mlflow` section, as read.
    """

    entries: dict[str, PretokenizeEntry] = field(default_factory=dict)
    mlflow: dict[str, Any] = field(default_factory=dict)


# ===============================
# Helper functions
# ===============================


def expected_output_name(tokenizer: str, max_seq_len: int) -> str:
    """Return the output folder name an entry's tokenizer and length give.

    The tokenizer path `<run>/hf-<target>` becomes `<run>-<target>`, followed by
    the length, so `bpe-32000/hf-modernbert` at 1,024 gives
    `bpe-32000-modernbert-1024`.

    Args:
        tokenizer: The entry's tokenizer path under the tokenizers root.
        max_seq_len: The entry's maximum sequence length.

    Returns:
        The name the entry key must end in.
    """
    return f"{tokenizer.strip('/').replace('/hf-', '-').replace('/', '-')}-{max_seq_len}"


def _positive_int(value: Any, where: str) -> int:
    """Return `value` as a positive int.

    Args:
        value: A config value.
        where: The config location, named in the error.

    Returns:
        The int.

    Raises:
        ValueError: If `value` is not a positive int.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{where} must be a positive integer, got {value!r}")
    return value


def _entry_from_dict(
    key: str, raw: dict[str, Any], roots: dict[str, Path], defaults: dict[str, Any]
) -> PretokenizeEntry:
    """Build one entry from its mapping, the roots and the defaults.

    Args:
        key: The entry key.
        raw: The entry mapping.
        roots: Resolved `corpus`, `tokenizers` and `output` roots.
        defaults: Settings an entry does not override.

    Returns:
        The entry.

    Raises:
        ValueError: If the key, a required field or a setting is malformed.
    """
    where = f"pretokenize entry {key!r}"
    if not _KEY_PATTERN.match(key):
        raise ValueError(f"{where}: key must be `<corpus>/<tokenizer>-<length>` in lowercase")
    missing = [name for name in ("corpus", "tokenizer", "max_seq_len") if not raw.get(name)]
    if missing:
        raise ValueError(f"{where} missing {', '.join(missing)}")
    unknown = sorted(set(raw) - _ENTRY_FIELDS - set(_DEFAULTS))
    if unknown:
        raise ValueError(f"{where} has unknown fields {unknown}")
    settings = {**defaults, **{name: raw[name] for name in _DEFAULTS if name in raw}}

    # check the settings
    formats = settings["formats"]
    if not isinstance(formats, list) or not formats or not all(isinstance(name, str) for name in formats):
        raise ValueError(f"{where}: formats must be a non-empty list of names, got {formats!r}")
    unknown_formats = sorted(set(formats) - set(FORMATS))
    if unknown_formats:
        raise ValueError(f"{where}: unknown formats {unknown_formats}; registered: {sorted(FORMATS)}")
    share = settings["validation_share"]
    if not isinstance(share, (int, float)) or not 0.0 < share < 1.0:
        raise ValueError(f"{where}: validation_share must be in (0, 1), got {share!r}")
    max_seq_len = _positive_int(raw["max_seq_len"], f"{where}: max_seq_len")
    expected = expected_output_name(str(raw["tokenizer"]), max_seq_len)
    if key.split("/", 1)[1] != expected:
        raise ValueError(f"{where}: the key must end in the tokenizer and length, `<corpus>/{expected}`")
    min_chunk_tokens = _positive_int(settings["min_chunk_tokens"], f"{where}: min_chunk_tokens")
    if min_chunk_tokens > max_seq_len:
        raise ValueError(f"{where}: min_chunk_tokens {min_chunk_tokens} exceeds max_seq_len {max_seq_len}")

    return PretokenizeEntry(
        key=key,
        corpus_dir=roots["corpus"] / str(raw["corpus"]),
        tokenizer_dir=roots["tokenizers"] / str(raw["tokenizer"]),
        output_dir=roots["output"] / key,
        max_seq_len=max_seq_len,
        min_chunk_tokens=min_chunk_tokens,
        formats=tuple(formats),
        validation_share=float(share),
        min_validation_docs=_positive_int(settings["min_validation_docs"], f"{where}: min_validation_docs"),
        seed=int(settings["seed"]),
        shard_size_bytes=_positive_int(settings["shard_size_mb"], f"{where}: shard_size_mb") * BYTES_PER_MB,
        language=str(raw.get("language", "sl")),
    )


# ===============================
# Loading
# ===============================


def load_pretokenize_config(config_path: Path) -> PretokenizeConfig:
    """Load and validate the pretokenize registry, applying any local overlay.

    Args:
        config_path: Path to `configs/data/pretokenize.yaml`.

    Returns:
        The entries by key, with project-relative roots resolved.

    Raises:
        FileNotFoundError: If `config_path` does not exist.
        ValueError: If a root, an entry or a setting is missing or malformed.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Pretokenize config not found: {config_path}")
    raw = load_yaml(config_path)

    # resolve the roots
    roots_raw = raw.get("roots") or {}
    missing = [name for name in _ROOTS if not roots_raw.get(name)]
    if missing:
        raise ValueError(f"Pretokenize config {config_path} missing roots: {', '.join(missing)}")
    roots = {name: resolve_project_path(roots_raw[name]) for name in _ROOTS}

    # build the entries
    defaults = {**_DEFAULTS, **(raw.get("defaults") or {})}
    unknown = sorted(set(defaults) - set(_DEFAULTS))
    if unknown:
        raise ValueError(f"Pretokenize config {config_path}: unknown defaults {unknown}")
    entries = {
        str(key): _entry_from_dict(str(key), value or {}, roots, defaults)
        for key, value in (raw.get("entries") or {}).items()
    }
    if not entries:
        raise ValueError(f"Pretokenize config {config_path} declares no entries")
    return PretokenizeConfig(entries=entries, mlflow=dict(raw.get("mlflow") or {}))
