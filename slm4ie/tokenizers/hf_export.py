"""Export trained tokenizer artifacts as HuggingFace tokenizers.

Bridges the training-time backends to the HuggingFace ecosystem for the
LM-pretraining phase. `bpe`, `charbpe`, `wordpiece`, and `morphbpe` wrap their
`tokenizer.json` as a `PreTrainedTokenizerFast`; `unigram` is converted from its
SentencePiece model; `morphpiece` returns the custom slow tokenizer. Every
returned object supports `decode` and offset mapping (native for the fast ones,
via `encode_with_offsets` for MorphPiece). `export_sweep_runs` writes those files for
a whole selection of trained runs.

Model training runs in other projects, and each model expects its own special
tokens and input wrapping. An `ExportTarget` from the sweep registry states one
model's needs (special-token roles, single and pair templates, maximum length);
with a target, every tokenizer wraps its input as the templates say and is
written to `<run>/hf-<target>/`. MorphPiece's export carries its tokenizer code
(`hf_morph_piece.py`), so `AutoTokenizer` loads it with `trust_remote_code=True`.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import shutil
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

import slm4ie.tokenizers.backends  # noqa: F401  (registers backends on import)
from slm4ie.tokenizers.config import ExportTarget
from slm4ie.tokenizers.corpus import iter_sample
from slm4ie.tokenizers.evaluate import EVAL_UNITS_FILENAME, METRICS_FILENAME
from slm4ie.tokenizers.hf_morph_piece import SPECIAL_TOKEN_ROLES, MorphPieceHFTokenizer
from slm4ie.tokenizers.registry import get_tokenizer
from slm4ie.tokenizers.train import MLFLOW_LINK_FILENAME

# ===============================
# Constants
# ===============================

# characters of a mismatching text quoted in a parity error
PARITY_EXCERPT_CHARS = 80
# float32 steps within which SentencePiece cannot tell two path scores apart
TIE_FLOAT32_STEPS = 2
PARTIAL_SUFFIX = ".partial"
# digests of the artifact and target an export directory was built from
LINEAGE_FILE = "export_lineage.json"
# the summary of every export, written under the sweep's report directory
EXPORT_SUMMARY_FILENAME = "export-summary.json"
DIGEST_CHUNK_BYTES = 1 << 20
# files evaluation and MLflow tracking write into a run directory after training
NON_TOKENIZER_FILES = frozenset({METRICS_FILENAME, EVAL_UNITS_FILENAME, MLFLOW_LINK_FILENAME})

# ===============================
# Logging
# ===============================

logger = logging.getLogger(__name__)

# ===============================
# Errors
# ===============================


class ParityError(ValueError):
    """Raised when an exported tokenizer encodes text differently from its trained backend."""


# ===============================
# Helper functions
# ===============================


def _read_name(artifact_dir: Path) -> str:
    """Return the backend name recorded in the artifact metadata.

    Args:
        artifact_dir: Directory holding `metadata.json`.

    Returns:
        The registry name of the tokenizer.

    Raises:
        FileNotFoundError: If no metadata sidecar is present.
    """
    meta_path = artifact_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No tokenizer metadata under {artifact_dir}")
    return json.loads(meta_path.read_text(encoding="utf-8"))["name"]


def _roles_present(tokenizer_obj: Any) -> dict[str, str]:
    """Map the project's special tokens that exist in `tokenizer_obj` to roles.

    Args:
        tokenizer_obj: A `tokenizers.Tokenizer`.

    Returns:
        HuggingFace role to token string.
    """
    vocab = tokenizer_obj.get_vocab()
    return {role: token for token, role in SPECIAL_TOKEN_ROLES.items() if token in vocab}


def _prefix_space_option(tokenizer_obj: Any) -> dict[str, bool]:
    """Return the trained `add_prefix_space` setting of a byte-level pre-tokenizer.

    `PreTrainedTokenizerFast` resets a byte-level pre-tokenizer's
    `add_prefix_space` to its own keyword, which defaults to False, so the
    trained value must be passed through or the first word encodes differently.

    Args:
        tokenizer_obj: A `tokenizers.Tokenizer`.

    Returns:
        `{"add_prefix_space": <trained value>}` for a byte-level pre-tokenizer,
        else an empty mapping.
    """
    pre_tokenizer = json.loads(tokenizer_obj.to_str()).get("pre_tokenizer") or {}
    if pre_tokenizer.get("type") == "ByteLevel":
        return {"add_prefix_space": bool(pre_tokenizer.get("add_prefix_space"))}
    return {}


def _unigram_scores(tokenizer_obj: Any) -> list[float] | None:
    """Return the log-probability of every piece of a Unigram model, by id.

    Args:
        tokenizer_obj: A `tokenizers.Tokenizer`.

    Returns:
        The score of each vocabulary id, or None when the model is not Unigram.
    """
    model = json.loads(tokenizer_obj.to_str()).get("model") or {}
    if model.get("type") != "Unigram":
        return None
    return [float(score) for _, score in model["vocab"]]


def scores_tie_in_float32(total: float, other_total: float) -> bool:
    """Return True when two Unigram path scores are equal at float32 precision.

    SentencePiece sums a path's piece scores in float32, so two paths whose
    totals differ by less than a float32 step at that size look equal to it,
    and it may pick either. The comparison allows `TIE_FLOAT32_STEPS` steps.

    Args:
        total: Total piece score of one split.
        other_total: Total piece score of another split of the same text.

    Returns:
        True when the totals are within `TIE_FLOAT32_STEPS` float32 steps.
    """
    step = float(np.spacing(np.float32(max(abs(total), abs(other_total)))))
    return abs(total - other_total) <= TIE_FLOAT32_STEPS * step


def _is_score_tie(ids: list[int], other_ids: list[int], scores: list[float] | None) -> bool:
    """Return True when two Unigram splits tie at SentencePiece's float32 precision.

    Args:
        ids: One split of a text, as vocabulary ids.
        other_ids: Another split of the same text.
        scores: Piece scores by id, as `_unigram_scores` returns, or None.

    Returns:
        True when the total scores tie per `scores_tie_in_float32`; always
        False without scores.
    """
    if scores is None:
        return False
    return scores_tie_in_float32(sum(scores[i] for i in ids), sum(scores[i] for i in other_ids))


def _apply_template(tokenizer_obj: Any, target: ExportTarget) -> None:
    """Set the post-processor that wraps every input as the target's templates say.

    Args:
        tokenizer_obj: A `tokenizers.Tokenizer`; its post-processor is replaced.
        target: The export target whose `single` and `pair` templates apply.

    Raises:
        ValueError: If a template token is missing from the vocabulary.
    """
    from tokenizers.processors import TemplateProcessing

    # look up the id of every special token the templates name
    special_tokens = [(name, tokenizer_obj.token_to_id(name)) for name in sorted(target.template_tokens)]
    missing = [name for name, token_id in special_tokens if token_id is None]
    if missing:
        raise ValueError(f"Export target {target.name!r}: template tokens {missing} missing from the vocabulary")

    tokenizer_obj.post_processor = TemplateProcessing(
        single=target.single, pair=target.pair, special_tokens=special_tokens
    )


def _unigram_to_tokenizer(artifact_dir: Path) -> Any:
    """Rebuild a `tokenizers` Unigram tokenizer from a SentencePiece model.

    Args:
        artifact_dir: Directory holding `spm.model`.

    Returns:
        A configured `tokenizers.Tokenizer` (SentencePiece normalizer, Unigram,
        Metaspace).
    """
    import sentencepiece as spm
    from sentencepiece import sentencepiece_model_pb2
    from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers

    # rebuild the vocabulary and scores
    model_path = artifact_dir / "spm.model"
    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    vocab = [(processor.id_to_piece(i), processor.get_score(i)) for i in range(processor.get_piece_size())]
    unk_id = processor.unk_id()

    # rebuild the normalizer as transformers' SpmConverter does
    proto = sentencepiece_model_pb2.ModelProto()
    proto.ParseFromString(model_path.read_bytes())
    steps = [normalizers.Strip(), normalizers.Replace(Regex(" {2,}"), " ")]
    if proto.normalizer_spec.precompiled_charsmap:
        steps.insert(0, normalizers.Precompiled(proto.normalizer_spec.precompiled_charsmap))

    # assemble the tokenizer
    tokenizer = Tokenizer(models.Unigram(vocab, unk_id=unk_id if unk_id >= 0 else None, byte_fallback=False))
    tokenizer.normalizer = normalizers.Sequence(steps)
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    tokenizer.decoder = decoders.Metaspace()
    return tokenizer


# ===============================
# Export functions
# ===============================


def to_hf(artifact_dir: Path, target: ExportTarget | None = None) -> Any:
    """Load a trained artifact as a HuggingFace tokenizer.

    Args:
        artifact_dir: Directory holding a trained tokenizer artifact.
        target: The training project's export settings. When given, the
            tokenizer takes the target's special-token roles, wraps every input
            in its templates and carries its `model_max_length`. When omitted,
            the project's special tokens keep their default roles and no
            input is wrapped.

    Returns:
        A `PreTrainedTokenizerFast` for the fast backends, or a slow
        `MorphPieceHFTokenizer` for `morphpiece`.
    """
    from transformers import PreTrainedTokenizerFast

    artifact_dir = Path(artifact_dir)
    name = _read_name(artifact_dir)

    # morphpiece has its own slow tokenizer, with the templates as settings
    if name == "morphpiece":
        settings = (
            {}
            if target is None
            else {
                "single_template": target.single,
                "pair_template": target.pair,
                "model_max_length": target.model_max_length,
                **target.roles,
            }
        )
        return MorphPieceHFTokenizer.from_artifact(artifact_dir, **settings)

    # load or convert the tokenizer object
    if name == "unigram":
        tokenizer_obj = _unigram_to_tokenizer(artifact_dir)
    else:
        from tokenizers import Tokenizer

        tokenizer_obj = Tokenizer.from_file(str(artifact_dir / "tokenizer.json"))

    prefix_space = _prefix_space_option(tokenizer_obj)
    if target is None:
        return PreTrainedTokenizerFast(tokenizer_object=tokenizer_obj, **prefix_space, **_roles_present(tokenizer_obj))

    # set the target's template, roles and length
    _apply_template(tokenizer_obj, target)
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_obj, model_max_length=target.model_max_length, **prefix_space, **target.roles
    )


def load_pretrained(artifact_dir: Path) -> Any:
    """Explicit loader returning the HuggingFace tokenizer for any backend.

    It works uniformly for the fast backends and for MorphPiece, straight from
    a sweep run directory; an exported directory loads with `AutoTokenizer`.

    Args:
        artifact_dir: Directory holding a trained tokenizer artifact.

    Returns:
        The HuggingFace tokenizer object.
    """
    return to_hf(artifact_dir)


def check_parity(artifact_dir: Path, export_dir: Path, texts: Iterable[str]) -> int:
    """Check that an exported directory encodes texts as the trained backend does.

    The exported directory is reloaded with `AutoTokenizer`, so the check covers
    what a training project loads. Its ids without the template's special
    tokens must equal the backend's `encode_ids` on every text. For a Unigram
    model, two splits whose total scores tie at float32 precision also pass
    (`scores_tie_in_float32`): SentencePiece cannot tell them apart and
    `tokenizers` may pick the other one.

    Args:
        artifact_dir: Directory holding the trained tokenizer artifact.
        export_dir: Directory written by `save_pretrained_dir` for that artifact.
        texts: Texts to encode with both.

    Returns:
        The number of texts checked.

    Raises:
        ParityError: If any text encodes to different ids, naming how many
            differ and quoting the first.
    """
    # load both tokenizers
    backend = get_tokenizer(_read_name(artifact_dir)).load(Path(artifact_dir))
    exported = load_export(export_dir)

    # compare the ids text by text
    scores = _unigram_scores(exported.backend_tokenizer) if exported.is_fast else None
    checked = 0
    mismatches: list[str] = []
    for text in texts:
        checked += 1
        exported_ids = exported(text, add_special_tokens=False)["input_ids"]
        backend_ids = backend.encode_ids(text)
        if exported_ids != backend_ids and not _is_score_tie(exported_ids, backend_ids, scores):
            mismatches.append(text)
    if mismatches:
        raise ParityError(
            f"{export_dir}: {len(mismatches)} of {checked} texts encode differently from the trained backend; "
            f"first: {mismatches[0][:PARITY_EXCERPT_CHARS]!r}"
        )
    return checked


def load_export(export_dir: Path) -> Any:
    """Load an exported directory as a training project does.

    MorphPiece exports carry their tokenizer code, so custom code is allowed;
    it is the code `save_pretrained_dir` copied in from this package.

    Args:
        export_dir: Directory written by `save_pretrained_dir`.

    Returns:
        The tokenizer `AutoTokenizer` loads from it.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(export_dir), trust_remote_code=True)


def target_dir(artifact_dir: Path, target: ExportTarget) -> Path:
    """Return the directory a run is exported to for one export target.

    Args:
        artifact_dir: Directory holding a trained tokenizer artifact.
        target: The training project's export settings.

    Returns:
        `<artifact_dir>/hf-<target name>`.
    """
    return Path(artifact_dir) / f"hf-{target.name}"


def save_pretrained_dir(artifact_dir: Path, out_dir: Path | None = None, target: ExportTarget | None = None) -> Path:
    """Write a HuggingFace tokenizer directory next to (or into) the artifact.

    Every backend then loads with `AutoTokenizer.from_pretrained(out_dir)`.
    MorphPiece is registered for `AutoTokenizer` before saving, so its tokenizer
    code (`hf_morph_piece.py`) is copied into the directory and loading it needs
    `trust_remote_code=True`.

    Args:
        artifact_dir: Directory holding a trained tokenizer artifact.
        out_dir: Destination; defaults to `target_dir` with a target and to
            `artifact_dir` without one.
        target: The training project's export settings, as `to_hf` applies them.

    Returns:
        The directory written.
    """
    artifact_dir = Path(artifact_dir)
    if out_dir is not None:
        destination = Path(out_dir)
    else:
        destination = target_dir(artifact_dir, target) if target is not None else artifact_dir
    tokenizer = to_hf(artifact_dir, target)
    if isinstance(tokenizer, MorphPieceHFTokenizer):
        tokenizer.register_for_auto_class("AutoTokenizer")
    destination.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(destination))
    return destination


# ===============================
# Sweep export
# ===============================


@dataclass
class ExportSummary:
    """Outcome of an export run, each entry `<run key>/<target name>`.

    Attributes:
        exported: Exports written and checked.
        skipped: Exports built from the same artifact and target, left as they were.
        failed: Exports that raised, a failed id check included; an earlier
            directory, when there was one, is kept and recorded as stale.
    """

    exported: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)


def read_parity_texts(sample_path: Path, n_lines: int) -> list[str]:
    """Read the texts an export's ids are checked on.

    Args:
        sample_path: The sweep's evaluation sample, one document per line.
        n_lines: How many non-empty lines to read from its start.

    Returns:
        Exactly `n_lines` texts.

    Raises:
        FileNotFoundError: If the sample does not exist.
        ValueError: If the sample holds fewer than `n_lines` non-empty lines.
    """
    if not Path(sample_path).exists():
        raise FileNotFoundError(f"Evaluation sample not found: {sample_path}; run `sweep_tokenizers.py sample` first")
    texts = list(itertools.islice(iter_sample(sample_path), n_lines))
    if len(texts) < n_lines:
        raise ValueError(
            f"Evaluation sample {sample_path} holds {len(texts)} text lines, fewer than export.parity_lines={n_lines}"
        )
    return texts


def _artifact_digest(artifact_dir: Path) -> str:
    """Hash the files of a trained artifact, so a retrained tokenizer gets a new digest.

    Args:
        artifact_dir: Directory holding the trained tokenizer artifact.

    Returns:
        The SHA-256 hex digest over the name and bytes of every file directly
        in `artifact_dir`, in name order. Export subdirectories and the files
        evaluation and tracking write there (`NON_TOKENIZER_FILES`) are left
        out, so re-evaluating a run does not change its digest.
    """
    digest = hashlib.sha256()
    files = (p for p in Path(artifact_dir).iterdir() if p.is_file() and p.name not in NON_TOKENIZER_FILES)
    for path in sorted(files):
        digest.update(path.name.encode("utf-8") + b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(DIGEST_CHUNK_BYTES), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _target_digest(target: ExportTarget) -> str:
    """Hash a target's settings, so a changed target gets a new digest.

    Args:
        target: The training project's export settings.

    Returns:
        The SHA-256 hex digest of the settings as sorted JSON.
    """
    return hashlib.sha256(json.dumps(asdict(target), sort_keys=True).encode("utf-8")).hexdigest()


def _read_lineage(export_dir: Path) -> dict[str, str] | None:
    """Return the digests an export directory was built from.

    Args:
        export_dir: A `hf-<target>` directory.

    Returns:
        The `artifact_digest` and `target_digest` of its lineage file, or None
        when the directory has no lineage file or misses a file the lineage
        file lists as written.
    """
    lineage_path = export_dir / LINEAGE_FILE
    if not lineage_path.exists():
        return None
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    if not all((export_dir / name).exists() for name in lineage.get("files", [])):
        return None
    return {key: lineage[key] for key in ("artifact_digest", "target_digest")}


def _describe_export(export_dir: Path, target: ExportTarget, artifact_digest: str) -> dict[str, Any]:
    """Describe an export directory on disk, as the summary file records it.

    Args:
        export_dir: A complete `hf-<target>` directory.
        target: The training project's export settings.
        artifact_digest: The digest of the trained artifact as it is now.

    Returns:
        The export's path, vocabulary size, special-token ids by role, the
        digest of the artifact it was built from (`export_digest`) and the
        artifact's current digest; the two differ for a stale export.
    """
    tokenizer = load_export(export_dir)
    lineage = _read_lineage(export_dir) or {}
    return {
        "path": str(export_dir),
        "vocab_size": len(tokenizer),
        "special_token_ids": {role: tokenizer.convert_tokens_to_ids(token) for role, token in target.roles.items()},
        "export_digest": lineage.get("artifact_digest"),
        "artifact_digest": artifact_digest,
    }


def _export_one(artifact_dir: Path, target: ExportTarget, parity_texts: list[str], lineage: dict[str, str]) -> None:
    """Export one run for one target and check its ids.

    The files are written to a partial directory, with the lineage file, and
    moved into place only after the id check passes, so a failed export never
    replaces the directory that was there.

    Args:
        artifact_dir: Directory holding the trained tokenizer artifact.
        target: The training project's export settings.
        parity_texts: Texts the export must encode as the backend does.
        lineage: The artifact and target digests to record in the directory.
    """
    destination = target_dir(artifact_dir, target)
    partial = destination.with_name(destination.name + PARTIAL_SUFFIX)
    shutil.rmtree(partial, ignore_errors=True)

    # write and check, removing the partial directory when either fails
    try:
        save_pretrained_dir(artifact_dir, partial, target)
        check_parity(artifact_dir, partial, parity_texts)
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise

    # record the lineage with the files written, then swap into place
    record = {**lineage, "files": sorted(path.name for path in partial.iterdir() if path.is_file())}
    (partial / LINEAGE_FILE).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    shutil.rmtree(destination, ignore_errors=True)
    partial.rename(destination)


def _export_entry(
    artifact_dir: Path, target: ExportTarget, parity_texts: list[str], force: bool
) -> tuple[str, dict[str, Any]]:
    """Export one run for one target unless its lineage still matches, and record the result.

    Args:
        artifact_dir: Directory holding the trained tokenizer artifact.
        target: The training project's export settings.
        parity_texts: Texts the export must encode as the backend does.
        force: Export again even when the digests match.

    Returns:
        The outcome (`exported`, `skipped` or `failed`) and the
        summary record. A failed re-export that kept an earlier directory is
        recorded as `stale`.
    """
    entry = f"{artifact_dir.name}/{target.name}"
    destination = target_dir(artifact_dir, target)
    artifact_digest = _artifact_digest(artifact_dir)
    lineage = {"artifact_digest": artifact_digest, "target_digest": _target_digest(target)}

    # skip an export built from the same artifact and target
    if not force and _read_lineage(destination) == lineage:
        logger.info("Skipped %s: exported from the same artifact and target", entry)
        return "skipped", {"status": "exported", **_describe_export(destination, target, artifact_digest)}

    # export, keeping an earlier directory when the export fails
    try:
        _export_one(artifact_dir, target, parity_texts, lineage)
    except Exception as exc:
        logger.exception("Failed exporting %s", entry)
        if _read_lineage(destination) is None:
            return "failed", {"status": "failed", "reason": str(exc), "artifact_digest": artifact_digest}
        kept = _describe_export(destination, target, artifact_digest)
        return "failed", {"status": "stale", "reason": str(exc), **kept}
    logger.info("Exported %s -> %s", entry, destination)
    return "exported", {"status": "exported", **_describe_export(destination, target, artifact_digest)}


def _write_summary(summary_path: Path, records: dict[str, dict[str, Any]]) -> None:
    """Merge this run's records into the export summary file.

    Records of exports not touched by this run are kept, so exporting a subset
    of runs leaves the others listed.

    Args:
        summary_path: The JSON summary file.
        records: Record per `<run key>/<target name>`.
    """
    existing = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({**existing, **records}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def export_sweep_runs(
    output_root: Path,
    keys: list[str],
    targets: list[ExportTarget],
    parity_texts: list[str],
    summary_path: Path,
    force: bool = False,
) -> ExportSummary:
    """Export each selected trained run once per export target.

    Each export goes to `<run>/hf-<target>/` with a lineage file holding the
    digests of the trained artifact and the target settings, and must pass
    `check_parity` on `parity_texts`. An export whose digests still match is
    skipped; a retrained artifact or changed target is exported again. When a
    re-export fails, the earlier directory is kept and recorded as `stale`.
    Every record written describes the directory on disk, so the summary and
    the exports stay in sync for the selected runs; records of runs outside the
    selection are kept as they were and refreshed by the next export of them.

    Args:
        output_root: Directory holding per-run artifact subdirs.
        keys: Run keys (`<name>-<vocab>`) to export.
        targets: Export targets to export each run for.
        parity_texts: Texts every export must encode as its backend does.
        summary_path: JSON file with one record per `<run key>/<target name>`:
            status (`exported`, `stale` or `failed`), path,
            vocabulary size, special-token ids, digests and any failure reason.
        force: Export again even when the digests match.

    Returns:
        The exported, skipped and failed exports.
    """
    summary = ExportSummary()
    records: dict[str, dict[str, Any]] = {}
    outcomes = {
        "exported": summary.exported,
        "skipped": summary.skipped,
        "failed": summary.failed,
    }

    # export each run for each target; an error fails that export only
    for key in keys:
        for target in targets:
            entry = f"{key}/{target.name}"
            try:
                outcome, records[entry] = _export_entry(output_root / key, target, parity_texts, force)
            except Exception as exc:
                logger.exception("Failed exporting %s", entry)
                outcome, records[entry] = "failed", {"status": "failed", "reason": str(exc)}
            outcomes[outcome].append(entry)

    # record and report the outcome
    _write_summary(summary_path, records)
    logger.info(
        "Done. Exported %d, skipped %d, failed %s. Summary: %s",
        len(summary.exported),
        len(summary.skipped),
        summary.failed or "none",
        summary_path,
    )
    return summary
