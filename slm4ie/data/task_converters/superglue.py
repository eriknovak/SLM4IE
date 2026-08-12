"""SuperGLUE-SL (`to_superglue`) converter backend.

Turns the SuperGLUE-SL subtasks (NLI / QA / coref / WSD / COPA) into their
task-family `TypedDict` rows:

* `nli/cb` / `nli/rte`  -> `NliExample`
* `qa/boolq`              -> `QaBooleanExample`
* `qa/multirc`            -> `QaBooleanExample` (one row per passage/question/answer)
* `coref/wsc`             -> `CorefExample`
* `wsd/wic`               -> `WsdExample`
* `commonsense/copa`      -> `CommonsenseCopaExample`

The split policy is SOURCE: each record keeps its originating split
(`train` / `val` / `test`). The distribution is expected under
`<roots.raw>/superglue_sl/` (e.g. `SuperGLUE-HumanT/<Task>/`); the
`--variant` flag (default `humant`) selects the translated variant. Because
the split policy is SOURCE, the driver's role gate for a `held_out` entry simply
means its `train.jsonl` source file is never read.
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from slm4ie.data.schema import (
    CommonsenseCopaExample,
    CorefExample,
    NliExample,
    QaBooleanExample,
    WsdExample,
)
from slm4ie.data.task_converter import (
    ConvertContext,
    SplitPolicy,
    TaskConverter,
    label_allow_set,
    register_converter,
    synthesize_id,
)
from slm4ie.data.tasks import TaskEntry, TasksRoots
from slm4ie.data.task_writer import find_first_existing, iter_jsonl

logger = logging.getLogger(__name__)

#: Default SuperGLUE-SL variant when `--variant` is not supplied.
_DEFAULT_VARIANT = "humant"

#: Map each `<task>/<dataset>` to its SuperGLUE subtask directory.
SUBTASK_DIRS: Dict[str, str] = {
    "nli/cb": "CB",
    "nli/rte": "RTE",
    "qa/boolq": "BoolQ",
    "qa/multirc": "MultiRC",
    "coref/wsc": "WSC",
    "wsd/wic": "WiC",
    "commonsense/copa": "COPA",
}

#: Variant subdirectory candidates for each `--variant` value.
VARIANT_DIRS: Dict[str, Tuple[str, ...]] = {
    "humant": ("SuperGLUE-HumanT", "HumanT"),
    "googlemt": ("SuperGLUE-GoogleMT", "GoogleMT"),
}

#: Source filenames to try (per split) inside each subtask directory.
_SPLIT_FILENAMES: Dict[str, Tuple[str, ...]] = {
    "train": ("train.jsonl", "train.json"),
    "val": ("val.jsonl", "val.json"),
    "test": ("test.jsonl", "test.json"),
}


def stable_id(record: Dict[str, Any], dataset: str, index: int) -> str:
    """Synthesize a stable id for one SuperGLUE record.

    A thin family-local alias for `synthesize_id`, kept because SuperGLUE
    records carry their own `id` / `idx` fields (including compound dict ids
    for MultiRC) rather than the `uid` / `doc_id` of extracted records.

    Args:
        record: Source record.
        dataset: Entry dataset name, used as the id prefix.
        index: Zero-based record position; used as the last-resort fallback.

    Returns:
        A string id, unique within the originating split.
    """
    return synthesize_id(record, dataset, index)


def _find_variant_root(raw_dir: Path, variant: str) -> Path:
    """Return the directory holding subtask subdirectories.

    Args:
        raw_dir: Root of the SuperGLUE-SL raw bundle (`<roots.raw>/superglue_sl`).
        variant: One of `humant` / `googlemt`.

    Returns:
        Directory that contains per-subtask folders.

    Raises:
        FileNotFoundError: If no candidate matches on disk.
    """
    candidates = VARIANT_DIRS.get(variant, ())
    for name in candidates:
        path = raw_dir / name
        if path.is_dir():
            return path
    if (raw_dir / "BoolQ").is_dir():
        return raw_dir
    raise FileNotFoundError(
        f"Could not find SuperGLUE {variant!r} variant directory in {raw_dir}. Tried: {candidates}."
    )


def _find_subtask_dir(variant_root: Path, subtask: str) -> Optional[Path]:
    """Return the subtask subdirectory matching *subtask* case-insensitively.

    Args:
        variant_root: Variant root.
        subtask: Canonical subtask name (e.g. `BoolQ`).

    Returns:
        The matching directory, or None when absent.
    """
    if not variant_root.is_dir():
        return None
    for child in variant_root.iterdir():
        if child.is_dir() and child.name.lower() == subtask.lower():
            return child
    return None


def _source_path_for_split(subtask_dir: Path, split: str) -> Optional[Path]:
    """Return the source JSONL path for *split* inside *subtask_dir*.

    Args:
        subtask_dir: Per-subtask directory.
        split: Output split name (`train` / `val` / `test`).

    Returns:
        First existing candidate path, or None when absent.
    """
    names = _SPLIT_FILENAMES.get(split, ())
    return find_first_existing([subtask_dir / name for name in names])


def _coerce_bool(value: Any) -> Optional[bool]:
    """Coerce common boolean spellings to `bool`.

    Args:
        value: A value pulled from a record.

    Returns:
        `True`/`False` for recognized inputs, None otherwise.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "t"}:
            return True
        if lowered in {"false", "0", "no", "n", "f"}:
            return False
    return None


def _convert_nli(
    record: Dict[str, Any],
    dataset: str,
    index: int,
    allow: Optional[set],
) -> Optional[NliExample]:
    """Convert one CB/RTE record to an `NliExample`.

    Args:
        record: Source record.
        dataset: Entry dataset name.
        index: Zero-based record position.
        allow: Optional label allow-list.

    Returns:
        The `NliExample`, or None when required fields are missing or the label
        is filtered out.
    """
    premise = record.get("premise")
    hypothesis = record.get("hypothesis")
    label = record.get("label")
    if premise is None or hypothesis is None:
        return None
    label_str = str(label) if label is not None else ""
    if allow is not None and label_str and label_str not in allow:
        return None
    return NliExample(
        id=stable_id(record, dataset, index),
        premise=str(premise),
        hypothesis=str(hypothesis),
        label=label_str,
    )


def _convert_boolq(
    record: Dict[str, Any],
    dataset: str,
    index: int,
    allow: Optional[set],
) -> Optional[QaBooleanExample]:
    """Convert one BoolQ record to a `QaBooleanExample`.

    Args:
        record: Source record.
        dataset: Entry dataset name.
        index: Zero-based record position.
        allow: Unused (boolean labels are not filtered).

    Returns:
        The `QaBooleanExample`, or None when required fields are missing.
    """
    del allow
    passage = record.get("passage") or record.get("paragraph")
    question = record.get("question")
    if passage is None or question is None:
        return None
    label = _coerce_bool(record.get("label"))
    if label is None:
        label = False
    return QaBooleanExample(
        id=stable_id(record, dataset, index),
        passage=str(passage),
        question=str(question),
        label=label,
    )


def _convert_wsc(
    record: Dict[str, Any],
    dataset: str,
    index: int,
    allow: Optional[set],
) -> Optional[CorefExample]:
    """Convert one WSC record to a `CorefExample`.

    Args:
        record: Source record with `target` span1/span2 fields.
        dataset: Entry dataset name.
        index: Zero-based record position.
        allow: Unused.

    Returns:
        The `CorefExample`, or None when required fields are missing.
    """
    del allow
    text = record.get("text")
    target = record.get("target") or {}
    span1_text = target.get("span1_text") or record.get("span1_text")
    span2_text = target.get("span2_text") or record.get("span2_text")
    span1_index = target.get("span1_index", record.get("span1_index"))
    span2_index = target.get("span2_index", record.get("span2_index"))
    if text is None or span1_text is None or span2_text is None:
        return None

    span1: Dict[str, Any] = {"text": str(span1_text)}
    span2: Dict[str, Any] = {"text": str(span2_text)}
    if span1_index is not None:
        span1["start"] = int(span1_index)
        span1["end"] = int(span1_index) + len(str(span1_text).split())
    if span2_index is not None:
        span2["start"] = int(span2_index)
        span2["end"] = int(span2_index) + len(str(span2_text).split())

    label = _coerce_bool(record.get("label"))
    if label is None:
        label = False
    return CorefExample(
        id=stable_id(record, dataset, index),
        text=str(text),
        span1=span1,
        span2=span2,
        label=label,
    )


def _convert_wic(
    record: Dict[str, Any],
    dataset: str,
    index: int,
    allow: Optional[set],
) -> Optional[WsdExample]:
    """Convert one WiC record to a `WsdExample`.

    Args:
        record: Source record.
        dataset: Entry dataset name.
        index: Zero-based record position.
        allow: Unused.

    Returns:
        The `WsdExample`, or None when required fields are missing.
    """
    del allow
    sentence1 = record.get("sentence1")
    sentence2 = record.get("sentence2")
    word = record.get("word")
    if sentence1 is None or sentence2 is None or word is None:
        return None
    label = _coerce_bool(record.get("label"))
    if label is None:
        label = False
    return WsdExample(
        id=stable_id(record, dataset, index),
        sentence1=str(sentence1),
        sentence2=str(sentence2),
        word=str(word),
        label=label,
    )


def _convert_copa(
    record: Dict[str, Any],
    dataset: str,
    index: int,
    allow: Optional[set],
) -> Optional[CommonsenseCopaExample]:
    """Convert one COPA record to a `CommonsenseCopaExample`.

    Args:
        record: Source record.
        dataset: Entry dataset name.
        index: Zero-based record position.
        allow: Unused (the registry's COPA labels are `[0, 1]`).

    Returns:
        The `CommonsenseCopaExample`, or None when required fields are missing.
    """
    del allow
    premise = record.get("premise")
    choice1 = record.get("choice1")
    choice2 = record.get("choice2")
    question = record.get("question", "cause")
    if premise is None or choice1 is None or choice2 is None:
        return None
    raw_label = record.get("label")
    try:
        label_int = int(raw_label) if raw_label is not None else 0
    except (TypeError, ValueError):
        label_int = 0
    return CommonsenseCopaExample(
        id=stable_id(record, dataset, index),
        premise=str(premise),
        choice1=str(choice1),
        choice2=str(choice2),
        question=str(question),
        label=label_int,
    )


def _iter_multirc(
    record: Dict[str, Any],
    dataset: str,
    base_index: int,
) -> Iterator[QaBooleanExample]:
    """Flatten one MultiRC passage into per-answer `QaBooleanExample` rows.

    Args:
        record: Native MultiRC record with nested
            `passage.questions[].answers[]` structure.
        dataset: Entry dataset name.
        base_index: Index of the parent record, used to synthesize stable ids
            when explicit `idx` fields are absent.

    Yields:
        One `QaBooleanExample` per `(passage, question, answer)` triple.
    """
    passage = record.get("passage") or {}
    paragraph_text = passage.get("text", "")
    passage_idx = record.get("idx", base_index)
    for question in passage.get("questions") or []:
        q_text = question.get("question", "")
        q_idx = question.get("idx")
        for answer in question.get("answers") or []:
            ans_idx = answer.get("idx")
            parts = [str(passage_idx)]
            if q_idx is not None:
                parts.append(f"q{q_idx}")
            if ans_idx is not None:
                parts.append(f"a{ans_idx}")
            example_id = f"{dataset}:{'-'.join(parts)}"
            label = _coerce_bool(answer.get("label"))
            if label is None:
                label = False
            combined_question = f"{q_text}\n{answer.get('text', '')}".strip()
            yield QaBooleanExample(
                id=example_id,
                passage=str(paragraph_text),
                question=combined_question,
                label=label,
            )


#: Per-task-family record converter callable, keyed by `<task>/<dataset>`.
#: Each returns the example dict, or None when required fields are missing.
_RECORD_CONVERTERS: Dict[
    str,
    Callable[[Dict[str, Any], str, int, Optional[set]], Optional[Any]],
] = {
    "nli/cb": _convert_nli,
    "nli/rte": _convert_nli,
    "qa/boolq": _convert_boolq,
    "coref/wsc": _convert_wsc,
    "wsd/wic": _convert_wic,
    "commonsense/copa": _convert_copa,
}


@register_converter("to_superglue")
class SuperglueConverter(TaskConverter):
    """Converter for the SuperGLUE-SL subtask families."""

    name = "to_superglue"
    split_policy = SplitPolicy.SOURCE
    description = (
        "Convert SuperGLUE-SL subtasks into per-split task JSONL under <roots.tasks>/<task>/<dataset>/<split>.jsonl.gz."
    )

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add the `--variant` argument selecting the translated bundle.

        Args:
            parser: The shared argument parser to extend.
        """
        parser.add_argument(
            "--variant",
            choices=tuple(VARIANT_DIRS.keys()),
            default=_DEFAULT_VARIANT,
            help=f"SuperGLUE-SL variant to read (default: {_DEFAULT_VARIANT}).",
        )

    def context_from_args(self, args: argparse.Namespace) -> ConvertContext:
        """Capture the `--variant` selection into the run context.

        Args:
            args: Parsed CLI arguments.

        Returns:
            A context carrying `{"variant": <variant>}`.
        """
        return ConvertContext(options={"variant": args.variant})

    def iter_examples(
        self,
        entry: TaskEntry,
        roots: TasksRoots,
        ctx: ConvertContext,
        splits: List[str],
    ) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """Yield `(source_split, example)` pairs for one SuperGLUE entry.

        Only the source files for the splits in `splits` are read, so a
        `held_out` entry (whose `train` split has been gated out) never touches
        its `train.jsonl` source.

        Args:
            entry: Task entry assigned to `to_superglue`.
            roots: Filesystem roots.
            ctx: Per-run options carrying the `variant` selection.
            splits: Target split names after the role gate.

        Yields:
            `(source_split, example_dict)` pairs for the SOURCE split policy.

        Raises:
            FileNotFoundError: If the SuperGLUE-SL bundle is missing.
            ValueError: If the entry has no registered subtask mapping or a
                non-`raw` source kind.
        """
        variant = ctx.options.get("variant", _DEFAULT_VARIANT)
        key = f"{entry.task}/{entry.dataset}"
        subtask = SUBTASK_DIRS.get(key)
        if subtask is None:
            raise ValueError(f"No SuperGLUE subtask mapping registered for {key!r}.")

        if entry.source.kind != "raw":
            raise ValueError(f"to_superglue only supports source.kind='raw'; got {entry.source.kind!r} for {key}.")

        bundle = roots.raw / entry.source.keys[0]
        variant_root = _find_variant_root(bundle, variant)
        subtask_dir = _find_subtask_dir(variant_root, subtask)
        if subtask_dir is None:
            raise FileNotFoundError(f"Subtask directory {subtask!r} not found under {variant_root}.")

        allow = label_allow_set(entry)

        is_multirc = key == "qa/multirc"
        converter = None if is_multirc else _RECORD_CONVERTERS.get(key)

        for split in splits:
            src_path = _source_path_for_split(subtask_dir, split)
            if src_path is None:
                logger.warning(
                    "No source file for split %r of %s in %s; skipping split.",
                    split,
                    key,
                    subtask_dir,
                )
                continue
            for index, record in enumerate(iter_jsonl(src_path)):
                if is_multirc:
                    for example in _iter_multirc(record, entry.dataset, index):
                        yield split, dict(example)
                    continue
                assert converter is not None
                example = converter(record, entry.dataset, index, allow)
                if example is None:
                    continue
                yield split, dict(example)
