"""Shared driver, ABC, and registry for the task-family converters.

The three task converters (`to_spans`, `to_sentiment`, `to_superglue`) all read
the same `configs/data/tasks.yaml` registry and share one output convention --
one gzipped JSONL per declared split under
`<roots.tasks>/<task>/<dataset>/<split>.jsonl.gz`. Only the per-family record
transform genuinely differs. This module owns everything the three used to
copy-paste:

* the common CLI (`entries`/`--all`, `--config`, `--force`, `--mlflow`,
  `--max-workers`), extended per converter via `TaskConverter.add_arguments`;
* entry-key resolution, config-path resolution, and the parallel dispatch with
  its skip / write / count / log wrapper and MLflow tracking tail;
* the `TaskConverter` ABC and its `@register_converter` registry, mirroring the
  tokenizer route (`BaseTokenizer` + `@register_tokenizer`);
* the shared `synthesize_id` id helper.

Each family only provides `iter_examples`, a declared `SplitPolicy`, and (for
SuperGLUE) an argument hook for `--variant`. The thin scripts under
`scripts/data/` are one-liners calling `run_converter("<name>")`.

Role gating lives here too: a `held_out` entry never writes a `train` split --
its declared splits are trimmed to everything except `train` before any output
file is opened. Nothing is dropped, only re-bucketed. For hash-policy families
the population is spread across the remaining eval splits (`val`/`test` split
~50/50 when `train` is absent) rather than collapsing the whole train bucket
into one split; for source-policy families (SuperGLUE) the `train.jsonl` source
file is simply never read. Held-out eval uses the labeled `val` split, so
`train` -- not the blind, unlabeled `test` set -- is the split dropped.
"""

from __future__ import annotations

import abc
import argparse
import enum
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Type

from slm4ie.data.io_utils import find_project_root, iter_joined_records
from slm4ie.data.parallel import (
    configure_script_logging,
    cpu_default,
    resolve_workers,
    run_parallel,
)
from slm4ie.data.tasks import (
    TaskEntry,
    TasksRoots,
    filter_for_converter,
    load_tasks,
    resolve_output_dir,
)
from slm4ie.data.task_tracking import log_task_runs
from slm4ie.data.task_writer import (
    SPLIT_TRAIN,
    all_outputs_exist,
    hash_split,
    outputs_for_splits,
    write_jsonl_splits,
)

logger = logging.getLogger(__name__)

#: Zero-pad width for the `idx-<N>` fallback id, shared across families so a
#: synthesized id has one consistent shape regardless of the converter.
_IDX_PAD: int = 8

#: Val-split percent boundary used when `train` has been dropped from a
#: hash-policy entry (held-out re-bucketing): the population that would have
#: gone to `train`/`val`/`test` is spread ~50/50 across the remaining
#: `val`/`test` splits instead of collapsing into one.
_HELD_OUT_VAL_PCT: int = 50


# --- Split policy and per-run context ---


class SplitPolicy(enum.Enum):
    """How a converter's yielded split key maps to an output split.

    Attributes:
        HASH: The converter yields `(stable_id, example)`; the driver assigns
            the split by hashing the id across the entry's target splits.
        SOURCE: The converter yields `(source_split_name, example)`; the
            driver keeps that split verbatim.
    """

    HASH = "hash"
    SOURCE = "source"


class ConvertContext:
    """Per-run options derived from the CLI, handed to every `iter_examples`.

    The driver builds one context from parsed arguments (via
    `TaskConverter.context_from_args`) and passes it to every entry's
    conversion. It must be picklable because it crosses the process pool, so it
    holds only plain data.

    Attributes:
        options: Converter-specific options (e.g. `{"variant": "humant"}` for
            SuperGLUE). Empty for converters that take no extra arguments.
    """

    def __init__(self, options: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the context.

        Args:
            options: Converter-specific options, or None for an empty mapping.
        """
        self.options: Dict[str, Any] = dict(options or {})


# --- Shared record helpers (id synthesis, label sets, extracted reads) ---


def synthesize_id(
    record: Dict[str, Any],
    dataset: str,
    index: int,
    pad: int = _IDX_PAD,
) -> str:
    """Synthesize a stable id for one source record across task families.

    Resolution order, covering every case the three converters used to handle
    separately:

    1. A non-empty `uid` (document-shaped extracted records).
    2. An explicit `id` / `idx` (SuperGLUE records): kept as-is when it
       already contains a `":"`, otherwise prefixed with the record's `source`
       (falling back to `dataset`). A dict (e.g. MultiRC-style compound ids) is
       joined with `"-"` under the same prefix.
    3. `"<source>:<doc_id>"` when a `doc_id` is present.
    4. `"<source-or-dataset>:idx-<zero-padded-index>"` as a last resort.

    Args:
        record: Source record.
        dataset: Entry dataset name, used as the id prefix when the record
            carries no `source` field.
        index: Zero-based record position, used only for the last-resort id.
        pad: Zero-pad width for the last-resort `idx-<N>` id.

    Returns:
        A string id, stable for the record within its originating split.
    """
    uid = record.get("uid")
    if uid:
        return str(uid)

    source = record.get("source", dataset)

    raw = record.get("id") if record.get("id") is not None else record.get("idx")
    if isinstance(raw, bool):
        raw = None
    if isinstance(raw, (str, int)):
        text = str(raw)
        return text if ":" in text else f"{source}:{text}"
    if isinstance(raw, dict):
        parts = [str(v) for v in raw.values() if v is not None]
        if parts:
            return f"{source}:{'-'.join(parts)}"

    doc_id = record.get("doc_id")
    if doc_id is not None:
        return f"{source}:{doc_id}"
    return f"{source}:idx-{index:0{pad}d}"


def label_allow_set(entry: TaskEntry) -> Optional[Set[str]]:
    """Return the entry's label allow-list as a string set, or None.

    Shared by the label-filtering families so the `entry.labels` -> set
    coercion lives in one place.

    Args:
        entry: The task entry.

    Returns:
        A set of accepted label strings, or None when the entry declares no
        closed label set.
    """
    if entry.labels is None:
        return None
    return {str(lbl) for lbl in entry.labels}


def iter_extracted_records(entry: TaskEntry, roots: TasksRoots) -> Iterator[Dict[str, Any]]:
    """Yield joined records from an entry's `extracted` sources.

    Reads each source key's `<key>.jsonl` text file joined with its optional
    `<key>.annotations.jsonl.gz` sidecar. Shared by the document-shaped families
    (`to_spans`, `to_sentiment`) so the extraction-tree read lives in one place.

    Args:
        entry: A task entry whose `source.kind` is `extracted`.
        roots: Filesystem roots.

    Yields:
        Joined records (text plus, when present, an `annotations` field).

    Raises:
        FileNotFoundError: If a source `<key>.jsonl` is missing.
    """
    for key in entry.source.keys:
        text_path = roots.extracted / f"{key}.jsonl"
        ann_path = roots.extracted / f"{key}.annotations.jsonl.gz"
        if not text_path.exists():
            raise FileNotFoundError(f"Source for {entry.task}/{entry.dataset}: {text_path} does not exist.")
        ann_arg: Optional[Path] = ann_path if ann_path.exists() else None
        yield from iter_joined_records(text_path, ann_arg)


# --- Converter registry and abstract base ---

_CONVERTER_REGISTRY: Dict[str, Type["TaskConverter"]] = {}


def register_converter(name: str):
    """Return a decorator registering a `TaskConverter` subclass by name.

    Mirrors `slm4ie.tokenizers.registry.register_tokenizer`. The name must match
    the converter name resolved for entries in `tasks.yaml`.

    Args:
        name: Registry key (e.g. `"to_spans"`).

    Returns:
        A class decorator that registers the class and returns it unchanged.
    """

    def decorator(cls: Type["TaskConverter"]) -> Type["TaskConverter"]:
        """Register `cls` under the captured name.

        Args:
            cls: The `TaskConverter` subclass being registered.

        Returns:
            The same class, unchanged.
        """
        _CONVERTER_REGISTRY[name] = cls
        return cls

    return decorator


def get_converter(name: str) -> "TaskConverter":
    """Instantiate the registered converter for `name`.

    Args:
        name: Registry key (e.g. `"to_spans"`).

    Returns:
        A fresh instance of the registered converter class.

    Raises:
        KeyError: If no converter is registered under `name`.
    """
    if name not in _CONVERTER_REGISTRY:
        raise KeyError(f"Converter {name!r} not found. Available: {sorted(_CONVERTER_REGISTRY)}")
    return _CONVERTER_REGISTRY[name]()


class TaskConverter(abc.ABC):
    """Abstract base for a single task-family converter.

    A concrete converter declares its registry `name`, its `split_policy`, and
    a `description` for the CLI, then implements `iter_examples`. Converters that
    take extra CLI arguments (SuperGLUE's `--variant`) override
    `add_arguments` and `context_from_args`.

    Attributes:
        name: Converter name matching `tasks.yaml`; set by subclasses.
        split_policy: How yielded split keys map to output splits.
        description: One-line CLI description shown by argparse.
    """

    name: str = "base"
    split_policy: SplitPolicy = SplitPolicy.HASH
    description: str = "Convert task datasets into per-split JSONL."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add converter-specific CLI arguments to `parser`.

        The default adds nothing. SuperGLUE overrides this to add `--variant`.

        Args:
            parser: The shared argument parser to extend.
        """

    def context_from_args(self, args: argparse.Namespace) -> ConvertContext:
        """Build the per-run `ConvertContext` from parsed arguments.

        The default returns an empty context. Converters with extra arguments
        override this to capture them.

        Args:
            args: Parsed CLI arguments.

        Returns:
            The context passed to every `iter_examples` call.
        """
        return ConvertContext()

    @abc.abstractmethod
    def iter_examples(
        self,
        entry: TaskEntry,
        roots: TasksRoots,
        ctx: ConvertContext,
        splits: List[str],
    ) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """Yield `(split_key, example)` pairs for one entry.

        For `SplitPolicy.HASH`, `split_key` is a stable id the driver hashes
        into one of `splits`. For `SplitPolicy.SOURCE`, `split_key` is the
        output split name itself, and the converter must read only the source
        files for the splits in `splits` (so a `held_out` entry never reads its
        `train` source).

        Args:
            entry: The task entry to convert.
            roots: Filesystem roots.
            ctx: Per-run options from the CLI.
            splits: Target split names after the role gate (`train` removed
                for `held_out`).

        Yields:
            `(split_key, example_dict)` pairs.
        """


# --- Split assignment and the conversion driver ---


def target_splits(entry: TaskEntry) -> List[str]:
    """Return the split names to write for `entry`, applying the role gate.

    A `held_out` entry never writes a `train` split; every other declared split
    is kept. `finetune_and_eval` entries keep every declared split.

    Args:
        entry: The task entry.

    Returns:
        Declared split names in registry order, minus `train` when the entry is
        `held_out`.
    """
    names = list(entry.splits.keys())
    if entry.role == "held_out":
        return [name for name in names if name != SPLIT_TRAIN]
    return names


def assign_hash_split(key: str, targets: List[str]) -> str:
    """Assign `key` to one of `targets` by deterministic hash bucket.

    When `train` is among `targets`, the standard 70/15/15 train/val/test
    boundaries apply. When `train` has been dropped (held-out re-bucketing), the
    whole population is spread ~50/50 across the remaining `val`/`test` splits
    instead of collapsing the train bucket into the first split.

    Args:
        key: Stable id of the example.
        targets: Target split names (already role-gated).

    Returns:
        One of the names in `targets`.
    """
    if SPLIT_TRAIN in targets:
        return hash_split(key, targets)
    return hash_split(key, targets, train_pct=0, val_pct=_HELD_OUT_VAL_PCT)


def convert_entry(
    key: str,
    *,
    entry: TaskEntry,
    roots: TasksRoots,
    ctx: ConvertContext,
    converter: TaskConverter,
    force: bool = False,
) -> Optional[Dict[str, int]]:
    """Convert one entry, writing one gzipped JSONL per target split.

    Applies the role gate (`target_splits`), skips when every target output
    already exists (unless `force`), streams the converter's examples through
    the split policy, and writes them.

    Args:
        key: Entry key `"<task>/<dataset>"` (used for logging).
        entry: The parsed task entry.
        roots: Filesystem roots.
        ctx: Per-run options from the CLI.
        converter: The converter instance for this entry's family.
        force: Re-derive even when every target output already exists.

    Returns:
        Mapping `{split: written_count}`, or None when the outputs already
        existed and `force` is False.
    """
    output_dir = resolve_output_dir(entry, roots)
    targets = target_splits(entry)
    target_files = {split: entry.splits[split] for split in targets}
    outputs = outputs_for_splits(output_dir, target_files)

    if not force and all_outputs_exist(outputs):
        logger.info(
            "Skipping %s: every split already exists at %s (use --force to overwrite).",
            key,
            output_dir,
        )
        return None

    logger.info("Converting %s -> %s", key, output_dir)
    examples = converter.iter_examples(entry, roots, ctx, targets)
    if converter.split_policy is SplitPolicy.HASH:
        pairs: Iterator[Tuple[str, Dict[str, Any]]] = (
            (assign_hash_split(split_key, targets), example) for split_key, example in examples
        )
    else:
        pairs = examples

    counts = write_jsonl_splits(pairs, outputs)
    total = sum(counts.values())
    logger.info("Wrote %d records across splits %s for %s", total, counts, key)
    return counts


def resolve_keys(
    entries: List[TaskEntry],
    requested: List[str],
    use_all: bool,
    name: str,
) -> List[str]:
    """Resolve which entry keys to process from the CLI selection.

    Args:
        entries: Entries already filtered to this converter.
        requested: Positional `<task>/<dataset>` keys from the CLI.
        use_all: Whether `--all` was passed.
        name: Converter name, used only for error messages.

    Returns:
        List of entry keys, preserving registry order.

    Raises:
        SystemExit: If a requested key is unknown to this converter.
    """
    known = {f"{e.task}/{e.dataset}": e for e in entries}
    if use_all:
        return list(known.keys())
    unknown = [k for k in requested if k not in known]
    if unknown:
        logger.error(
            "Unknown entries for converter %s: %s. Known: %s",
            name,
            unknown,
            sorted(known.keys()),
        )
        sys.exit(1)
    return list(requested)


# --- CLI plumbing ---


def build_parser(converter: TaskConverter) -> argparse.ArgumentParser:
    """Build the argument parser for `converter`.

    Assembles the flag set common to every task converter, then lets the
    converter contribute its own arguments via `TaskConverter.add_arguments`.

    Args:
        converter: The converter whose CLI is being built.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(description=converter.description)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "entries",
        nargs="*",
        default=[],
        help="Entry keys to process, e.g. 'ner/ssj500k'. Mutually exclusive with --all.",
    )
    target.add_argument(
        "--all",
        action="store_true",
        help="Process every entry for this converter declared in tasks.yaml.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to tasks.yaml (default: configs/data/tasks.yaml).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-derive outputs even when every split already exists.",
    )
    parser.add_argument(
        "--mlflow",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable/disable MLflow per-dataset tracking, overriding "
            "tasks.yaml::mlflow.enabled. Default: defer to config."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=0,
        help=(
            "Process entries in parallel. 0=auto (cpu_count // 2), 1=serial, "
            "N=N workers. Capped at the number of entries."
        ),
    )
    converter.add_arguments(parser)
    return parser


def parse_args(
    converter: TaskConverter,
    argv: Optional[List[str]] = None,
) -> argparse.Namespace:
    """Parse command-line arguments for `converter`.

    Args:
        converter: The converter whose CLI is being parsed.
        argv: Optional argument list (defaults to `sys.argv`).

    Returns:
        The parsed namespace.

    Raises:
        SystemExit: If the selection is missing or combines `--all` with
            positional entries.
    """
    parser = build_parser(converter)
    args = parser.parse_args(argv)
    if args.all and args.entries:
        parser.error("argument --all: not allowed with positional entries")
    if not args.all and not args.entries:
        parser.error("one of the arguments entries --all is required")
    return args


# --- Main entry point ---


def run_converter(name: str, argv: Optional[List[str]] = None) -> None:
    """Run a task converter end to end from CLI arguments.

    The single entry point behind the three thin scripts. Resolves the
    converter, parses arguments, loads the registry, selects entries, and
    dispatches them through `convert_entry` with bounded concurrency before
    logging the MLflow tracking tail.

    Args:
        name: Converter name (e.g. `"to_spans"`).
        argv: Optional argument list (defaults to `sys.argv`).

    Raises:
        SystemExit: With code 2 when one or more entries fail to convert.
    """
    import slm4ie.data.task_converters  # noqa: F401  (registers converters on import)

    converter = get_converter(name)
    args = parse_args(converter, argv)
    ctx = converter.context_from_args(args)

    project_root = find_project_root()
    config_path = args.config if args.config is not None else project_root / "configs" / "data" / "tasks.yaml"

    tasks_config = load_tasks(config_path)
    entries = filter_for_converter(tasks_config, name)
    by_key = {f"{e.task}/{e.dataset}": e for e in entries}

    keys = resolve_keys(entries, args.entries, args.all, name)
    if not keys:
        logger.warning("No entries to process for converter %s.", name)
        return

    workers = resolve_workers(args.max_workers, len(keys), cpu_default(len(keys)))
    configure_script_logging(parallel=workers > 1)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = project_root / "logs" / name / stamp

    roots = tasks_config.roots

    def kwargs_for(key: str) -> Dict[str, Any]:
        """Return per-entry keyword arguments for `convert_entry`.

        Args:
            key: The entry key being dispatched.

        Returns:
            Keyword arguments for `convert_entry`.
        """
        return {
            "entry": by_key[key],
            "roots": roots,
            "ctx": ctx,
            "converter": converter,
            "force": args.force,
        }

    results, failures = run_parallel(
        convert_entry,
        keys,
        max_workers=workers,
        desc=name,
        pool="process",
        kwargs_for=kwargs_for,
        log_dir=log_dir,
    )

    skipped = [k for k, v in results.items() if v is None]
    total = sum(sum(v.values()) for v in results.values() if v is not None)
    logger.info(
        "Done. Processed %d entr(ies); %d skipped; %d records written. Failed: %s",
        len(results) - len(skipped),
        len(skipped),
        total,
        [k for k, _ in failures] or "none",
    )
    processed = [k for k in keys if k not in {f for f, _ in failures}]
    log_task_runs(tasks_config, by_key, processed, mlflow_enabled=args.mlflow, force=args.force)
    if failures:
        sys.exit(2)
