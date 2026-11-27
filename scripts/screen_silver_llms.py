r"""Screen hosted LLMs for making Slovene medical and scientific NER examples, one subcommand per step.

A thin CLI over `slm4ie.silver`. Each subcommand reads the experiment's configs
and the round's files, dispatches into the library, and writes the round's next
files. `make` and `rate` call paid models: both refuse to run without a tracking
server, take `--dry-run` to print the projected spend without any call, and stop
before the round's spend limit, counting the cost every stored call reported.

A run of any tier but `tracked` writes its interim folders with a `-scratch`
suffix and counts only the spend of scratch folders, so scratch calls are never
reused or counted by a tracked run, and never replace its MLflow runs. A tracked
`make` or `rate` refuses to start unless `--tag` names its `branch` and
`base_commit`.

    draw     <data-dir>/final/passages-round<n>/passages.jsonl
    preview  <data-dir>/interim/preview-round<n>.md
    make     <data-dir>/interim/examples-round<n>[-scratch]/<variant>.jsonl, one MLflow run per variant
    check    <data-dir>/interim/checks-round<n>[-scratch]/checks.jsonl
    rate     <data-dir>/interim/ratings-round<n>[-scratch]/{units,key,<judge>}.jsonl, one MLflow run per judge
    sheet    <data-dir>/interim/ratings-round<n>[-scratch]/person-sheet.md, or with --read person-ratings.jsonl

Shared options:
    --configs-dir   Folder of the five configs (default: the experiment's configs folder).
    --data-dir      Folder of the round's files (default: data/experiments/silver-llm-screen-slovenian).
    --round         Round number the output folders are named for (default: 1).
    --experiment    MLflow experiment of `make` and `rate` (default: slm4ie/silver-llm-screen-slovenian).
    --tag           MLflow tag KEY=VALUE, repeatable; tier defaults to scratch, language is always sl,
                    round is always R<n>. A tracked run needs tier=tracked, branch and base_commit.

Examples:
    Project the spend of round 1 without any call:

        uv run --group evaluation python scripts/screen_silver_llms.py make --dry-run

    Make a smoke run of one model on one passage and one brief:

        uv run --group evaluation python scripts/screen_silver_llms.py make \
            --variant claude-haiku-5-5 --limit-items 1 --tag tier=scratch

    Make the examples of round 1 as a tracked run, from a snapshot of the branch:

        uv run --group evaluation python scripts/screen_silver_llms.py make \
            --tag tier=tracked --tag branch=<branch> --tag base_commit=<sha>

    Rate the outputs that passed the rule checks, logged as a tracked run:

        uv run --group evaluation python scripts/screen_silver_llms.py rate \
            --tag tier=tracked --tag branch=<branch> --tag base_commit=<sha>

    Read the person's filled rating sheet of the tracked round:

        uv run --group evaluation python scripts/screen_silver_llms.py sheet \
            --read data/experiments/silver-llm-screen-slovenian/interim/ratings-round1/person-sheet.md
"""

import argparse
import json
import logging
import os
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import yaml

from slm4ie.silver.checks import CheckResult, check_output, check_shares
from slm4ie.silver.make import (
    JOBS,
    Item,
    SpendProjection,
    load_brief_items,
    load_labels,
    load_passage_items,
    make_examples,
    openrouter_factory,
    project_spend,
)
from slm4ie.silver.passages import draw_passages
from slm4ie.silver.rate import (
    DEFAULT_JUDGE_ESTIMATE,
    JUDGE,
    PERSON_SAMPLE_SIZE,
    draw_person_sample,
    rate_outputs,
    rating_pool,
    read_pool,
    read_rating_sheet,
    write_pool,
    write_rating_sheet,
)
from slm4ie.silver.tracking import MAKER, SCRATCH_SUFFIX, TRACKED_TIER, file_hashes, log_round_runs, template_hashes
from slm4ie.utils.config import load_yaml
from slm4ie.utils.io import find_project_root, open_text_stream
from slm4ie.utils.parallel import configure_script_logging

# ===============================
# Logging
# ===============================

logger = logging.getLogger(__name__)

# ===============================
# Defaults
# ===============================

ERROR_EXIT_CODE = 1
SLUG = "silver-llm-screen-slovenian"
DEFAULT_EXPERIMENT = f"slm4ie/{SLUG}"
DEFAULT_CONFIGS_DIR = ("experiments", "programmes", "slovene-training-data", SLUG, "configs")
DEFAULT_DATA_DIR = ("data", "experiments", SLUG)
DATA_ROOT = "data"

MODELS_CONFIG = "screen-models.yaml"
LABELS_CONFIG = "labels-domains.yaml"
PROMPTS_CONFIG = "prompts-jobs.yaml"
BRIEFS_CONFIG = "briefs-passages.yaml"
SOURCES_CONFIG = "passages-sources.yaml"

# tags every run carries unless --tag overrides them; language is always sl
DEFAULT_TAGS = {"tier": "scratch", "type": "baseline"}
LANGUAGE = "sl"
# tags a tracked run must be given, so it names the code it ran
TRACKED_REQUIRED_TAGS = ("branch", "base_commit")

# the medicine name in a product description's file name follows this marker
SMPC_MARKER = "_SmPC_"
PASSAGE_KEYS = ("set", "domain", "native", "passage_id", "doc_id", "group", "text", "words")
# the fields of a stored example copied to its row of rule-check flags
RECORD_KEYS = ("variant", "job", "item_id", "domain")

# ===============================
# Helper functions
# ===============================


def _parse_pairs(pairs: list[str], flag: str) -> dict[str, Any]:
    """Turn repeated `KEY=VALUE` arguments into a dict, each value read as YAML."""
    parsed = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise SystemExit(f"{flag} takes KEY=VALUE, got {pair!r}")
        parsed[key] = yaml.safe_load(value)
    return parsed


def _tracking_is_unset(experiment: str) -> bool:
    """Report and return whether `MLFLOW_TRACKING_URI` is missing for a run that logs to `experiment`.

    Without it, MLflow would write a local store inside the working tree, which a
    tracked run from a snapshot loses.
    """
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        logger.error("logging to %s needs MLFLOW_TRACKING_URI: launch through tracking-env.sh", experiment)
        return True
    return False


def _given_tags(args: argparse.Namespace) -> dict[str, str]:
    """Return the tags given by `--tag` over the defaults, every value as text."""
    given = {key: str(value) for key, value in _parse_pairs(args.tag, "--tag").items()}
    return {**DEFAULT_TAGS, **given}


def _run_tags(args: argparse.Namespace) -> dict[str, str]:
    """Build the MLflow tags of the round's runs from the defaults and `--tag`.

    Raises:
        SystemExit: If the tier is `tracked` and `--tag` gave no `branch` or no `base_commit`.
    """
    tags = _given_tags(args)
    missing = [key for key in TRACKED_REQUIRED_TAGS if not tags.get(key)]
    if tags["tier"] == TRACKED_TIER and missing:
        raise SystemExit(f"a tracked run needs --tag branch=<branch> --tag base_commit=<sha>; missing: {missing}")
    return {**tags, "language": LANGUAGE, "round": f"R{args.round}"}


def _is_tracked(args: argparse.Namespace) -> bool:
    """Return whether the run is of the tracked tier."""
    return _given_tags(args)["tier"] == TRACKED_TIER


def _passages_path(args: argparse.Namespace) -> Path:
    """Return the round's drawn passages file."""
    return args.data_dir / "final" / f"passages-round{args.round}" / "passages.jsonl"


def _interim_dir(args: argparse.Namespace, name: str, round_number: int | None = None) -> Path:
    """Return a round's interim folder of the run's tier: `examples-round1`, or `examples-round1-scratch`."""
    suffix = "" if _is_tracked(args) else f"-{SCRATCH_SUFFIX}"
    return args.data_dir / "interim" / f"{name}-round{round_number or args.round}{suffix}"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file; no rows when it does not exist."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write rows as a JSONL file, creating its folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def _recorded_spend(paths: Iterable[Path]) -> float:
    """Sum the `cost_usd` of every stored row in the given files; rows without one cost nothing."""
    return sum(row.get("cost_usd", 0.0) for path in paths for row in _read_jsonl(path))


def _round_files(args: argparse.Namespace, round_number: int) -> list[Path]:
    """List a round's example and rating files of the run's tier."""
    folders = (_interim_dir(args, "examples", round_number), _interim_dir(args, "ratings", round_number))
    return [path for folder in folders if folder.exists() for path in sorted(folder.glob("*.jsonl"))]


def _spend_budget(
    args: argparse.Namespace, limits: Mapping[str, float], counted: Iterable[Path]
) -> tuple[float, float]:
    """Return the limit and the spend to pass a runner that itself sums the files in `counted`.

    The limit keeps this round under `round_usd` and every round together under `experiment_usd`.
    Only the files of the run's tier count, so scratch spend never shrinks a tracked budget.
    """
    earlier = sum(_recorded_spend(_round_files(args, number)) for number in range(1, args.round))
    this_round = _recorded_spend(_round_files(args, args.round))
    limit = min(limits["round_usd"] + earlier, limits["experiment_usd"])
    return limit, earlier + this_round - _recorded_spend(counted)


def _print_projection(title: str, projection: SpendProjection, limit: float, spent: float) -> None:
    """Print the projected calls and spend per model and in total, beside the limit and the spend so far."""
    calls = projection.calls // len(projection.per_variant) if projection.per_variant else 0
    width = max([len("total"), *map(len, projection.per_variant)])
    print(title)
    print(f"{'model':<{width}}  {'calls':>6}  {'usd':>9}")
    for name, usd in projection.per_variant.items():
        print(f"{name:<{width}}  {calls:>6}  {usd:>9.4f}")
    print(f"{'total':<{width}}  {projection.calls:>6}  {projection.total_usd:>9.4f}")
    print(f"limit {limit:.2f} USD, already recorded {spent:.2f} USD, left {limit - spent:.2f} USD")


def _input_hashes(args: argparse.Namespace) -> dict[str, str]:
    """Hash the round's passages file and the briefs and labels configs, logged with every run."""
    return file_hashes(
        {
            "passages": _passages_path(args),
            "briefs": args.configs_dir / BRIEFS_CONFIG,
            "labels": args.configs_dir / LABELS_CONFIG,
        }
    )


def _load_items(args: argparse.Namespace) -> list[Item]:
    """Load the round's passages and the briefs as items, first the passages."""
    return [*load_passage_items(_passages_path(args)), *load_brief_items(args.configs_dir / BRIEFS_CONFIG)]


def _checked_records(
    args: argparse.Namespace, models: Mapping[str, Any], labels: Mapping[str, Any]
) -> list[tuple[dict[str, Any], CheckResult]]:
    """Run the rule checks on every stored example of the round, model by model.

    Raises:
        KeyError: If a stored example answers an item that is neither a drawn passage nor a brief.
    """
    items = {(item.job, item.item_id): item for item in _load_items(args)}
    examples_dir = _interim_dir(args, "examples")
    checked = []
    for variant in models:
        for record in _read_jsonl(examples_dir / f"{variant}.jsonl"):
            item = items.get((record["job"], record["item_id"]))
            if item is None:
                raise KeyError(f"{variant} answered unknown {record['job']} item {record['item_id']!r}")
            checked.append((record, check_output(record, labels[record["domain"]], item)))
    return checked


# ===============================
# draw
# ===============================


def _keep_document(spec: Mapping[str, Any]) -> Callable[[Mapping[str, Any]], bool]:
    """Build the predicate of a set's document filters: doctypes, UDC prefixes and metadata values."""
    doctypes = set(spec.get("doctypes") or ())
    prefixes = tuple(str(prefix) for prefix in spec.get("udc_prefix") or ())
    equals = dict(spec.get("metadata_equals") or {})

    def keep(document: Mapping[str, Any]) -> bool:
        """Return True when the document passes every filter the set lists."""
        metadata = document.get("metadata") or {}
        if doctypes and metadata.get("doctype") not in doctypes:
            return False
        if prefixes and not str(metadata.get("udc") or "").startswith(prefixes):
            return False
        return all(metadata.get(field) == value for field, value in equals.items())

    return keep


def _group_of(field: str | None) -> Callable[[Mapping[str, Any]], str]:
    """Build the group key of a set: the lowercased medicine name after `_SmPC_` in the field, else empty."""

    def group(document: Mapping[str, Any]) -> str:
        """Return the document's group, empty when the field names no medicine."""
        value = str((document.get("metadata") or {}).get(field) or "")
        if not field or SMPC_MARKER not in value:
            return ""
        name = value.split(SMPC_MARKER, 1)[1].split()
        return name[0].lower() if name else ""

    return group


def _iter_documents(folder: Path) -> Iterator[dict[str, Any]]:
    """Stream the documents of a source's shards, shard by shard in name order.

    Raises:
        FileNotFoundError: If the folder holds no shard.
    """
    shards = sorted(folder.glob("*.jsonl*"))
    if not shards:
        raise FileNotFoundError(f"no shards in {folder}")
    for shard in shards:
        with open_text_stream(shard) as handle:
            yield from (json.loads(line) for line in handle if line.strip())


def _add_draw_parser(subparsers: argparse._SubParsersAction, common: argparse.ArgumentParser) -> None:
    """Register the `draw` subcommand."""
    parser = subparsers.add_parser("draw", parents=[common], help="Draw the real passages from the corpus.")
    parser.add_argument(
        "--data-root", type=Path, default=None, help="Root of the data tiers (default: the repo's data folder)"
    )
    parser.add_argument("--force", action="store_true", help="Draw again over an existing passages file")


def _draw(args: argparse.Namespace) -> int:
    """Draw every set's passages by the written rule and write them to the round's passages file."""
    out_path = _passages_path(args)
    if out_path.exists() and not args.force:
        logger.info("%s exists; pass --force to draw again", out_path)
        return 0

    # draw each set from its source's shards
    config = load_yaml(args.configs_dir / SOURCES_CONFIG)
    data_root = args.data_root or find_project_root() / DATA_ROOT
    rows = []
    for name, spec in config["sets"].items():
        documents = _iter_documents(data_root / config["corpus_tier"] / spec["source"])
        passages = draw_passages(
            documents,
            spec["count"],
            config["seed"],
            keep=_keep_document(spec),
            group_of=_group_of(spec.get("group_field")),
            cue_words=spec.get("cue_words") or (),
            min_cues=spec.get("min_cues", 1),
            wrapped=spec.get("wrapped", False),
        )
        rows += [{"set": name, "domain": spec["domain"], "native": spec["native"], **asdict(p)} for p in passages]
        logger.info("set=%s source=%s passages=%d", name, spec["source"], len(passages))

    _write_jsonl(out_path, ({key: row[key] for key in PASSAGE_KEYS} for row in rows))
    logger.info("wrote %d passages to %s", len(rows), out_path)
    return 0


# ===============================
# preview
# ===============================


def _add_preview_parser(subparsers: argparse._SubParsersAction, common: argparse.ArgumentParser) -> None:
    """Register the `preview` subcommand."""
    subparsers.add_parser("preview", parents=[common], help="Write a page of each real passage and its brief.")


def _preview(args: argparse.Namespace) -> int:
    """Write one Markdown page listing every real passage followed by its paired brief."""
    passages = _read_jsonl(_passages_path(args))
    briefs = {brief["passage_id"]: brief for brief in load_yaml(args.configs_dir / BRIEFS_CONFIG)["briefs"]}

    lines = [
        f"# Round {args.round}: real passages and their writing briefs",
        "",
        f"{len(passages)} real passages, each followed by the brief a model writes a new passage from.",
    ]
    for number, passage in enumerate(passages, 1):
        origin = "written in Slovene" if passage["native"] else "probably translated"
        lines += ["", f"## {number}. {passage['passage_id']}", ""]
        lines += [f"- Set: {passage['set']} ({passage['domain']}, {origin})", f"- Words: {passage['words']}", ""]
        lines.append("\n>\n".join(f"> {paragraph}" for paragraph in passage["text"].split("\n")))
        brief = briefs.pop(passage["passage_id"], None)
        if brief is None:
            lines += ["", "No brief is paired with this passage."]
            continue
        lines += [
            "",
            f"**Brief {brief['brief_id']}**",
            "",
            f"- Text type: {brief['text_type']}",
            f"- Topic: {brief['topic']}",
            f"- Target words: {brief['words']}",
            f"- Required labels: {', '.join(brief['labels'])}",
        ]
    for passage_id, brief in briefs.items():
        logger.warning("brief %s names passage %s, which was not drawn", brief["brief_id"], passage_id)

    out_path = args.data_dir / "interim" / f"preview-round{args.round}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("wrote %s", out_path)
    return 0


# ===============================
# make
# ===============================


def _add_make_parser(subparsers: argparse._SubParsersAction, common: argparse.ArgumentParser) -> None:
    """Register the `make` subcommand."""
    parser = subparsers.add_parser("make", parents=[common], help="Ask every model to label and to write examples.")
    parser.add_argument("--dry-run", action="store_true", help="Print the projected calls and spend; call nothing")
    parser.add_argument(
        "--variant", action="append", default=[], help="Screened model to run, repeatable (default: every model)"
    )
    parser.add_argument(
        "--limit-items", type=int, default=None, help="Take only the first N items of each job, for a smoke run"
    )


def _make(args: argparse.Namespace) -> int:
    """Project the spend, or make the examples and log one MLflow run per variant."""
    config = load_yaml(args.configs_dir / MODELS_CONFIG)
    unknown = sorted(set(args.variant) - set(config["models"]))
    if unknown:
        logger.error("unknown variant %s; known: %s", unknown, sorted(config["models"]))
        return ERROR_EXIT_CODE
    variants = {name: spec for name, spec in config["models"].items() if not args.variant or name in args.variant}

    # load the items, the first N of each job for a smoke run
    items = _load_items(args)
    if args.limit_items is not None:
        items = [item for job in JOBS for item in [i for i in items if i.job == job][: args.limit_items]]

    # project the spend without building any backend
    if args.dry_run:
        limit, spent = _spend_budget(args, config["limits"], [])
        title = f"round {args.round}: projected spend of the makers at list prices, no call made"
        _print_projection(title, project_spend(variants, items), limit, spent)
        return 0
    tags = _run_tags(args)
    if _tracking_is_unset(args.experiment):
        return ERROR_EXIT_CODE

    # make the examples under the limit, counting the spend the runner does not sum itself
    examples_dir = _interim_dir(args, "examples")
    limit, spent = _spend_budget(args, config["limits"], [examples_dir / f"{name}.jsonl" for name in variants])
    prompts = load_yaml(args.configs_dir / PROMPTS_CONFIG)
    summary = make_examples(
        variants,
        items,
        load_labels(args.configs_dir / LABELS_CONFIG),
        prompts,
        examples_dir,
        limit,
        spent,
        backend_factory=openrouter_factory(config["sampling"]),
        max_tokens=config["sampling"]["max_tokens"],
    )
    logger.info(
        "calls=%d skipped=%d failed=%d spent_usd=%.4f",
        summary.calls,
        summary.skipped,
        summary.failed,
        summary.spent_usd,
    )
    if summary.capped:
        logger.warning("the spend limit of %.2f USD stopped the run before every call was made", limit)

    hashes = template_hashes(prompts, JOBS)
    inputs = _input_hashes(args)
    log_round_runs(args.experiment, MAKER, variants, examples_dir, config["sampling"], hashes, inputs, tags)
    return 0


# ===============================
# check
# ===============================


def _add_check_parser(subparsers: argparse._SubParsersAction, common: argparse.ArgumentParser) -> None:
    """Register the `check` subcommand."""
    subparsers.add_parser("check", parents=[common], help="Run the rule checks on every stored example.")


def _check(args: argparse.Namespace) -> int:
    """Write the rule-check flags of every stored example and print the passing shares per model and job."""
    models = load_yaml(args.configs_dir / MODELS_CONFIG)["models"]
    try:
        checked = _checked_records(args, models, load_labels(args.configs_dir / LABELS_CONFIG))
    except KeyError as exc:
        logger.error("Check failed: %s", exc)
        return ERROR_EXIT_CODE

    # write one row of flags per example
    out_path = _interim_dir(args, "checks") / "checks.jsonl"
    rows = [
        {**{key: record[key] for key in RECORD_KEYS}, "passes": result.passes, **result.checks()}
        for record, result in checked
    ]
    _write_jsonl(out_path, rows)
    logger.info("wrote %d rows to %s", len(rows), out_path)

    # print the passing shares per model and job
    groups: dict[tuple[str, str], list[CheckResult]] = {}
    for record, result in checked:
        groups.setdefault((record["variant"], record["job"]), []).append(result)
    width = max([len("model"), *(len(variant) for variant, _ in groups)])
    names = ("passes", *(field.name for field in fields(CheckResult)))
    print(f"{'model':<{width}}  {'job':<8}  {'n':>3}  " + "  ".join(f"{name:>14}" for name in names))
    for (variant, job), results in groups.items():
        shares = check_shares(results)
        cells = (f"{'-' if shares[name] is None else format(shares[name], '.2f'):>14}" for name in names)
        print(f"{variant:<{width}}  {job:<8}  {len(results):>3}  " + "  ".join(cells))
    return 0


# ===============================
# rate
# ===============================


def _add_rate_parser(subparsers: argparse._SubParsersAction, common: argparse.ArgumentParser) -> None:
    """Register the `rate` subcommand."""
    parser = subparsers.add_parser("rate", parents=[common], help="Have the judges rate the blind pool.")
    parser.add_argument("--dry-run", action="store_true", help="Print the judges' projected spend; call nothing")
    parser.add_argument(
        "--limit-units", type=int, default=None, help="Rate only the first N units of the pool, for a smoke run"
    )


def _rate(args: argparse.Namespace) -> int:
    """Project the judges' spend, or write the blind pool, rate it and log one MLflow run per judge."""
    config = load_yaml(args.configs_dir / MODELS_CONFIG)
    labels = load_labels(args.configs_dir / LABELS_CONFIG)
    judges = config["judges"]

    # pool the outputs that passed the rule checks with the real passages
    try:
        checked = _checked_records(args, config["models"], labels)
    except KeyError as exc:
        logger.error("Rating failed: %s", exc)
        return ERROR_EXIT_CODE
    families = {variant: spec["family"] for variant, spec in config["models"].items()}
    pool = rating_pool(checked, load_passage_items(_passages_path(args)), families)
    units = pool.units[: args.limit_units] if args.limit_units is not None else pool.units

    # project the spend without building any backend
    if args.dry_run:
        limit, spent = _spend_budget(args, config["limits"], [])
        calls = [Item(JUDGE, unit.unit_id, unit.domain) for unit in units]
        title = f"round {args.round}: projected spend of the judges on {len(units)} units at list prices, no call made"
        _print_projection(title, project_spend(judges, calls, {JUDGE: DEFAULT_JUDGE_ESTIMATE}), limit, spent)
        return 0
    tags = _run_tags(args)
    if _tracking_is_unset(args.experiment):
        return ERROR_EXIT_CODE

    # rate the pool under the limit, counting the spend the runner does not sum itself
    ratings_dir = _interim_dir(args, "ratings")
    write_pool(pool, ratings_dir)
    limit, spent = _spend_budget(args, config["limits"], [ratings_dir / f"{name}.jsonl" for name in judges])
    prompts = load_yaml(args.configs_dir / PROMPTS_CONFIG)
    summary = rate_outputs(
        units,
        judges,
        labels,
        prompts,
        ratings_dir,
        limit,
        spent,
        backend_factory=openrouter_factory(config["sampling"]),
        seed=config["sampling"]["seed"],
        max_tokens=config["sampling"]["max_tokens"],
    )
    logger.info(
        "calls=%d skipped=%d failed=%d dropped=%d spent_usd=%.4f",
        summary.calls,
        summary.skipped,
        summary.failed,
        summary.dropped,
        summary.spent_usd,
    )
    if summary.capped:
        logger.warning("the spend limit of %.2f USD stopped the run before every call was made", limit)

    hashes = template_hashes(prompts, (JUDGE,))
    inputs = _input_hashes(args)
    log_round_runs(args.experiment, JUDGE, judges, ratings_dir, config["sampling"], hashes, inputs, tags)
    return 0


# ===============================
# sheet
# ===============================


def _add_sheet_parser(subparsers: argparse._SubParsersAction, common: argparse.ArgumentParser) -> None:
    """Register the `sheet` subcommand."""
    parser = subparsers.add_parser("sheet", parents=[common], help="Write the person's rating sheet, or read it.")
    parser.add_argument(
        "--size",
        type=int,
        default=PERSON_SAMPLE_SIZE,
        help=f"Fewest units on the sheet (default: {PERSON_SAMPLE_SIZE})",
    )
    parser.add_argument("--read", type=Path, default=None, help="Filled sheet to read into person-ratings.jsonl")
    parser.add_argument("--force", action="store_true", help="Write the sheet over an existing one")


def _sheet(args: argparse.Namespace) -> int:
    """Write the blind sheet of the person's sample, or read a filled sheet into ratings."""
    if args.read is not None:
        return _read_sheet(args.read)

    ratings_dir = _interim_dir(args, "ratings")
    out_path = ratings_dir / "person-sheet.md"
    if out_path.exists() and not args.force:
        logger.info("%s exists; pass --force to write it again", out_path)
        return 0
    if not (ratings_dir / "units.jsonl").exists():
        logger.error("no rating pool in %s: run rate first", ratings_dir)
        return ERROR_EXIT_CODE

    # draw the sample from the pool the judges rate
    pool = read_pool(ratings_dir)
    seed = load_yaml(args.configs_dir / MODELS_CONFIG)["sampling"]["seed"]
    try:
        sample = draw_person_sample(pool.units, pool.key, args.size, seed=seed)
    except ValueError as exc:
        logger.error("Drawing the sample failed: %s", exc)
        return ERROR_EXIT_CODE

    labels = load_labels(args.configs_dir / LABELS_CONFIG)
    write_rating_sheet(sample, labels, load_yaml(args.configs_dir / PROMPTS_CONFIG), out_path)
    logger.info("wrote %d units to %s", len(sample), out_path)
    return 0


def _read_sheet(path: Path) -> int:
    """Read a filled sheet and write `person-ratings.jsonl` beside it, one row per unit."""
    try:
        ratings = read_rating_sheet(path)
    except ValueError as exc:
        logger.error("Reading the sheet failed: %s", exc)
        return ERROR_EXIT_CODE

    out_path = path.parent / "person-ratings.jsonl"
    _write_jsonl(out_path, ({"unit_id": unit_id, "rating": rating} for unit_id, rating in ratings.items()))
    filled = sum(rating is not None for rating in ratings.values())
    logger.info("wrote %d units, %d of them rated, to %s", len(ratings), filled, out_path)
    return 0


# ===============================
# Main function
# ===============================


def _common_parser() -> argparse.ArgumentParser:
    """Build the options every subcommand shares."""
    root = find_project_root()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=root.joinpath(*DEFAULT_CONFIGS_DIR),
        help="Folder of the experiment's configs (default: the experiment's configs folder)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=root.joinpath(*DEFAULT_DATA_DIR),
        help=f"Folder of the round's files (default: {'/'.join(DEFAULT_DATA_DIR)})",
    )
    parser.add_argument("--round", type=int, default=1, help="Round the output folders are named for (default: 1)")
    parser.add_argument(
        "--experiment", default=DEFAULT_EXPERIMENT, help=f"MLflow experiment name (default: {DEFAULT_EXPERIMENT})"
    )
    parser.add_argument(
        "--tag", action="append", default=[], metavar="KEY=VALUE", help="MLflow tag, repeatable, such as tier=tracked"
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to `sys.argv`).

    Returns:
        Parsed namespace, including the selected subcommand in `command`.
    """
    parser = argparse.ArgumentParser(description="Screen hosted LLMs for making Slovene NER examples.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = _common_parser()
    for add_parser in (
        _add_draw_parser,
        _add_preview_parser,
        _add_make_parser,
        _add_check_parser,
        _add_rate_parser,
        _add_sheet_parser,
    ):
        add_parser(subparsers, common)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Dispatch the selected subcommand.

    Args:
        argv: Optional argument list (defaults to `sys.argv`).

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    configure_script_logging(parallel=False)
    handlers = {"draw": _draw, "preview": _preview, "make": _make, "check": _check, "rate": _rate, "sheet": _sheet}
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
