"""Make silver NER examples with hosted LLMs: label real passages and write labelled passages from briefs.

Every screened model answers every item once. An item is one job: `annotate`
asks for the mentions of a real passage, `write` asks for a new passage and its
mentions from a brief. Each call is stored as one JSON line per model, with the
raw reply, its parse, the example in the GLiNER task schema and what the call
cost, so an interrupted run resumes where it stopped.

The spend cap is checked before every call: a call that could pass the limit,
judged by its worst case (the input estimate plus a reply of the full output
cap, never less than the model's costliest call so far), is never sent and the
run stops there.
"""

import hashlib
import json
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from slm4ie.evaluation.backends import ChatBackend, get_backend
from slm4ie.evaluation.prompts import FENCE, answer_schema, parse_answer
from slm4ie.evaluation.scoring import align_mentions
from slm4ie.utils.config import load_yaml

# ===============================
# Defaults
# ===============================

ANNOTATE = "annotate"
WRITE = "write"
JOBS = (ANNOTATE, WRITE)

TOKENS_PER_MILLION = 1_000_000

# hex characters of the sha256 kept as the prompt hash
PROMPT_HASH_CHARS = 16

BACKEND_NAME = "openrouter"

# ===============================
# Items and Token Estimates
# ===============================


@dataclass(frozen=True)
class Item:
    """One thing a model is asked to make.

    Attributes:
        job: `annotate` (label a real passage) or `write` (write a labelled passage from a brief).
        item_id: The passage id for `annotate`, the brief id for `write`.
        domain: `medical` or `scientific`; picks the label set.
        text: The passage to label; empty for `write`.
        text_type: The kind of text to write; empty for `annotate`.
        topic: What the text is about; empty for `annotate`.
        words: The passage's word count for `annotate`, the target length for `write`.
        required_labels: Labels the written text must mention; empty for `annotate`.
    """

    job: str
    item_id: str
    domain: str
    text: str = ""
    text_type: str = ""
    topic: str = ""
    words: int = 0
    required_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class TokenEstimate:
    """Tokens one call of a job is expected to use, for pricing a call before it is made.

    Attributes:
        input_tokens: Tokens the model reads.
        output_tokens: Tokens the model writes, reasoning included.
    """

    input_tokens: int
    output_tokens: int


# the per-call usage of the scoping projection (tables/scoping-cost-by-model.csv)
DEFAULT_TOKEN_ESTIMATES = {ANNOTATE: TokenEstimate(1_000, 750), WRITE: TokenEstimate(1_000, 750)}


@dataclass(frozen=True)
class MakeSummary:
    """What one run of `make_examples` did.

    Attributes:
        calls: Calls made in this run, failed ones included.
        skipped: Calls skipped because their record was already stored.
        failed: Calls in this run that raised an error.
        spent_usd: The running total: the spend passed in, plus every stored and new call's cost.
        capped: True when the spend cap stopped the run before every call was made.
    """

    calls: int
    skipped: int
    failed: int
    spent_usd: float
    capped: bool


@dataclass(frozen=True)
class SpendProjection:
    """The projected spend of a run from list prices, made without any call.

    Attributes:
        calls: Calls the run would make, over every model.
        per_variant: Variant name to its projected spend in US dollars.
        total_usd: The projected spend of every model together.
    """

    calls: int
    per_variant: dict[str, float]
    total_usd: float


# ===============================
# Loaders
# ===============================


def load_passage_items(path: Path) -> list[Item]:
    """Load the drawn passages as `annotate` items.

    Args:
        path: The passages JSONL, one row per passage with `domain`, `passage_id`, `text` and `words`.

    Returns:
        One item per passage, in file order.
    """
    items = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            items.append(Item(ANNOTATE, row["passage_id"], row["domain"], text=row["text"], words=int(row["words"])))
    return items


def load_brief_items(path: Path) -> list[Item]:
    """Load the writing briefs as `write` items.

    Args:
        path: The briefs YAML (`briefs-passages.yaml`).

    Returns:
        One item per brief, in file order.
    """
    return [
        Item(
            WRITE,
            str(brief["brief_id"]),
            brief["domain"],
            text_type=brief["text_type"],
            topic=brief["topic"],
            words=int(brief["words"]),
            required_labels=tuple(brief["labels"]),
        )
        for brief in load_yaml(path)["briefs"]
    ]


def load_labels(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Load the label sets of the domains.

    Args:
        path: The label YAML (`labels-domains.yaml`).

    Returns:
        Domain to label to its spec (`description`, `examples`), labels in prompt order.
    """
    return {domain: dict(spec["labels"]) for domain, spec in load_yaml(path)["domains"].items()}


# ===============================
# Prompts and Answers
# ===============================


def label_lines(label_specs: Mapping[str, Mapping[str, Any]]) -> str:
    """Write a domain's labels one per line, as `- <label>: <description> Examples: <examples>.`.

    Args:
        label_specs: Label to its spec, in prompt order.

    Returns:
        The label list for the prompt's `{labels}` placeholder.
    """
    lines = []
    for label, spec in label_specs.items():
        description = " ".join(str(spec["description"]).split())
        examples = ", ".join(str(example) for example in spec["examples"])
        lines.append(f"- {label}: {description} Examples: {examples}.")
    return "\n".join(lines)


def build_messages(
    item: Item, label_specs: Mapping[str, Mapping[str, Any]], prompts: Mapping[str, Mapping[str, str]]
) -> list[dict[str, str]]:
    """Build the chat turns of one item from its job's prompt.

    Args:
        item: The item to make.
        label_specs: The item's domain's labels, label to spec.
        prompts: Job to its `system` and `user` prompt (`prompts-jobs.yaml`).

    Returns:
        A system turn and a user turn, each `{"role": str, "content": str}`.
    """
    prompt = prompts[item.job]
    user = prompt["user"].format(
        labels=label_lines(label_specs),
        label_names=", ".join(label_specs),
        text=item.text,
        text_type=item.text_type,
        topic=item.topic,
        words=item.words,
        required_labels=", ".join(item.required_labels),
    )
    return [{"role": "system", "content": prompt["system"]}, {"role": "user", "content": user}]


def job_schema(job: str, labels: Iterable[str]) -> dict[str, Any]:
    """Build the JSON schema a job's answer must satisfy.

    Args:
        job: `annotate` or `write`.
        labels: The domain's label names.

    Returns:
        `{"mentions": [...]}` for `annotate`; for `write` the same object with a non-empty string `text` added.
    """
    schema = answer_schema(labels)
    if job == ANNOTATE:
        return schema
    return {
        **schema,
        "properties": {"text": {"type": "string", "minLength": 1}, **schema["properties"]},
        "required": ["text", *schema["required"]],
    }


def parse_annotation(answer: str, labels: Iterable[str]) -> dict[str, Any] | None:
    """Read an `annotate` answer, or `None` when it is off the schema.

    Args:
        answer: The reply text.
        labels: The domain's label names; no other label is accepted.

    Returns:
        `{"mentions": [{"text", "label"}, ...]}`, or `None` for an answer off the schema.
    """
    mentions = parse_answer(answer, labels)
    return None if mentions is None else {"mentions": mentions}


def parse_written(answer: str, labels: Iterable[str]) -> dict[str, Any] | None:
    """Read a `write` answer, or `None` when it is off the schema.

    One markdown code fence around the JSON is stripped first. The answer must
    be an object with exactly a non-empty string `text` and a `mentions` list
    that `parse_answer` accepts.

    Args:
        answer: The reply text.
        labels: The domain's label names; no other label is accepted.

    Returns:
        `{"text": str, "mentions": [{"text", "label"}, ...]}`, or `None` for an answer off the schema.
    """
    # strip one code fence and decode
    raw = answer.strip()
    fenced = FENCE.match(raw)
    if fenced:
        raw = fenced.group(1).strip()
    try:
        decoded = json.loads(raw)
    except ValueError:
        return None

    # check the text, then hand the mentions to the annotate parser
    if not isinstance(decoded, dict) or set(decoded) != {"text", "mentions"}:
        return None
    text = decoded["text"]
    if not isinstance(text, str) or not text:
        return None
    mentions = parse_answer(json.dumps({"mentions": decoded["mentions"]}), labels)
    return None if mentions is None else {"text": text, "mentions": mentions}


def prompt_hash(messages: list[dict[str, str]]) -> str:
    """Hash the system and user turns, so a record names the exact prompt it answered.

    Args:
        messages: The chat turns.

    Returns:
        The first 16 hex characters of the sha256 of the turns' contents.
    """
    contents = json.dumps([message["content"] for message in messages], ensure_ascii=False)
    return hashlib.sha256(contents.encode("utf-8")).hexdigest()[:PROMPT_HASH_CHARS]


# ===============================
# Pricing
# ===============================


def price_usd(spec: Mapping[str, Any], input_tokens: int, output_tokens: int) -> float:
    """Price a call from the model's list prices.

    Args:
        spec: The model spec, with `usd_per_million_input` and `usd_per_million_output`.
        input_tokens: Tokens the model read.
        output_tokens: Tokens the model wrote.

    Returns:
        The cost in US dollars.
    """
    usd = input_tokens * spec["usd_per_million_input"] + output_tokens * spec["usd_per_million_output"]
    return usd / TOKENS_PER_MILLION


def project_spend(
    variants: Mapping[str, Mapping[str, Any]],
    items: Iterable[Item],
    token_estimates: Mapping[str, TokenEstimate] = DEFAULT_TOKEN_ESTIMATES,
) -> SpendProjection:
    """Project the spend of a run from list prices, making no call.

    Args:
        variants: Variant name to its model spec (`screen-models.yaml`).
        items: The items every variant would answer.
        token_estimates: Job to the tokens one of its calls is expected to use.

    Returns:
        The number of calls and the projected spend per variant and in total.
    """
    items = list(items)
    per_variant = {}
    for variant, spec in variants.items():
        per_variant[variant] = sum(
            price_usd(spec, token_estimates[item.job].input_tokens, token_estimates[item.job].output_tokens)
            for item in items
        )
    return SpendProjection(len(items) * len(per_variant), per_variant, sum(per_variant.values()))


# ===============================
# Backends
# ===============================


def model_settings(sampling: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return the sampling settings one model is called with.

    The fixed settings hold for every model, except that a model which accepts no
    seed or no temperature is sent none, and a model spec may name the reasoning
    effort the model supports in place of the fixed one.

    Args:
        sampling: The `sampling` block of `screen-models.yaml` (`seed`, `temperature`, `reasoning_effort`,
            `max_tokens`).
        spec: The model's spec, with the optional `accepts_seed`, `accepts_temperature` and `reasoning_effort`.

    Returns:
        `seed` and `temperature` (each `None` when the model accepts none), `reasoning_effort` and `max_tokens`.
    """
    return {
        "seed": sampling["seed"] if spec.get("accepts_seed", True) else None,
        "temperature": sampling["temperature"] if spec.get("accepts_temperature", True) else None,
        "reasoning_effort": spec.get("reasoning_effort", sampling["reasoning_effort"]),
        "max_tokens": sampling["max_tokens"],
    }


def openrouter_factory(sampling: Mapping[str, Any]) -> Callable[[str, Mapping[str, Any]], ChatBackend]:
    """Build a backend factory that calls every model through OpenRouter with its sampling settings.

    Args:
        sampling: The `sampling` block of `screen-models.yaml`, applied per model by `model_settings`.

    Returns:
        A function of the variant name and its model spec that returns the model's backend.
    """

    def factory(variant: str, spec: Mapping[str, Any]) -> ChatBackend:
        """Build the OpenRouter backend of one model."""
        return get_backend(BACKEND_NAME, spec["model"], **model_settings(sampling, spec))

    return factory


# ===============================
# Making
# ===============================


def make_examples(
    variants: Mapping[str, Mapping[str, Any]],
    items: Iterable[Item],
    labels: Mapping[str, Mapping[str, Mapping[str, Any]]],
    prompts: Mapping[str, Mapping[str, str]],
    out_dir: Path,
    limit_usd: float,
    spent_usd: float = 0.0,
    *,
    backend_factory: Callable[[str, Mapping[str, Any]], ChatBackend],
    max_tokens: int,
    token_estimates: Mapping[str, TokenEstimate] = DEFAULT_TOKEN_ESTIMATES,
) -> MakeSummary:
    """Ask every variant for every item once and store one record per call.

    Calls go item by item across variants, so a cap stop loses whole items. A
    `(variant, job, item_id)` already stored in `<out_dir>/<variant>.jsonl` is
    skipped. A call that raises is stored with its error and not retried.

    Before each call the run stops when the running total plus a guard would
    pass `limit_usd`. The guard is the call's worst case: the job's input-token
    estimate plus `max_tokens` of output at list prices, raised to the costliest
    call the variant has made so far when that cost more. A reply that reports
    no cost counts at list prices from its token counts.

    Args:
        variants: Variant name to its model spec (`screen-models.yaml`).
        items: The items every variant answers.
        labels: Domain to label to its spec (`load_labels`).
        prompts: Job to its `system` and `user` prompt (`prompts-jobs.yaml`).
        out_dir: Folder of the per-variant record files; created when missing.
        limit_usd: The spend no call may pass.
        spent_usd: Spend already made outside `out_dir`'s records, such as an earlier round.
        backend_factory: Function of the variant name and its spec that returns its backend.
        max_tokens: The most tokens a reply may hold, the backend's reply cap (`sampling.max_tokens`).
        token_estimates: Job to the tokens one of its calls is expected to use.

    Returns:
        The calls made, skipped and failed, the running total and whether the cap stopped the run.
    """
    items = list(items)
    out_dir.mkdir(parents=True, exist_ok=True)

    # read what is stored and start the running total from it
    stored: dict[str, set[tuple[str, str]]] = {}
    costliest: dict[str, float] = {}
    total = spent_usd
    for variant in variants:
        stored[variant] = set()
        for record in _read_records(out_dir / f"{variant}.jsonl"):
            stored[variant].add((record["job"], record["item_id"]))
            total += record["cost_usd"]
            if record["error"] is None:
                costliest[variant] = max(costliest.get(variant, 0.0), record["cost_usd"])

    # list the calls still to make, item by item across variants
    pending = [
        (item, variant) for item in items for variant in variants if (item.job, item.item_id) not in stored[variant]
    ]
    skipped = len(items) * len(variants) - len(pending)
    backends = {variant: backend_factory(variant, spec) for variant, spec in variants.items()}

    # make each call unless it could pass the cap
    calls = failed = 0
    capped = False
    for item, variant in tqdm(pending, desc="making examples", unit="call"):
        spec = variants[variant]
        worst = price_usd(spec, token_estimates[item.job].input_tokens, max_tokens)
        guard = max(worst, costliest.get(variant, 0.0))
        if total + guard > limit_usd:
            capped = True
            break

        record = _make_record(variant, spec, item, labels[item.domain], prompts, backends[variant])
        _append_record(out_dir / f"{variant}.jsonl", record)
        calls += 1
        total += record["cost_usd"]
        if record["error"] is None:
            costliest[variant] = max(costliest.get(variant, 0.0), record["cost_usd"])
        else:
            failed += 1

    return MakeSummary(calls=calls, skipped=skipped, failed=failed, spent_usd=total, capped=capped)


# ===============================
# Helpers
# ===============================


def _make_record(
    variant: str,
    spec: Mapping[str, Any],
    item: Item,
    label_specs: Mapping[str, Mapping[str, Any]],
    prompts: Mapping[str, Mapping[str, str]],
    backend: ChatBackend,
) -> dict[str, Any]:
    """Call the backend once for one item and build its stored record."""
    messages = build_messages(item, label_specs, prompts)
    record: dict[str, Any] = {
        "variant": variant,
        "model": spec["model"],
        "job": item.job,
        "item_id": item.item_id,
        "domain": item.domain,
        "prompt_hash": prompt_hash(messages),
        "raw": None,
        "parsed": None,
        "example": None,
        "unaligned": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "cost_reported": False,
        "seconds": 0.0,
        "seed": backend.seed,
        "error": None,
    }

    # call once; a failed call is stored, not retried
    start = time.perf_counter()
    try:
        result = backend.chat(messages, job_schema(item.job, label_specs))
    except Exception as error:
        return {**record, "seconds": time.perf_counter() - start, "error": f"{type(error).__name__}: {error}"}

    # count an unreported cost at list prices
    read = result.input_tokens + result.cache_read_tokens + result.cache_write_tokens
    cost = result.cost_usd if result.cost_usd is not None else price_usd(spec, read, result.output_tokens)
    record.update(
        raw=result.text,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=cost,
        cost_reported=result.cost_usd is not None,
        seconds=result.seconds,
    )

    # parse the reply and align its mentions to the text they name
    parse = parse_annotation if item.job == ANNOTATE else parse_written
    parsed = parse(result.text, label_specs)
    if parsed is None:
        return record
    text = item.text if item.job == ANNOTATE else parsed["text"]
    alignment = align_mentions(text, parsed["mentions"])
    example = {"id": f"{variant}/{item.item_id}", "text": text, "spans": alignment.spans}
    return {**record, "parsed": parsed, "example": example, "unaligned": alignment.unaligned}


def _read_records(path: Path) -> list[dict[str, Any]]:
    """Read the stored records of one variant; none when the file does not exist yet."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _append_record(path: Path, record: Mapping[str, Any]) -> None:
    """Append one record as a JSON line and flush it, so a stop loses no stored call."""
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
