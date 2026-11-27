"""Rate made examples blind: by judge models, and by one person on a rating sheet.

Every output that passed the rule checks becomes one unit of a rating pool,
beside every real passage as the reference for natural Slovene. A unit shows a
rater the text and its numbered mentions under an id that says nothing of its
maker; the maker, the job and the item are kept apart in a key.

Each judge model rates every unit once with the same five judgements a person
gives: the language errors in the text, its naturalness and realism on 1 to 5,
whether each mention's label and span are correct, and the entities it missed.
A judge's answer off the schema is stored as a dropped rating. Ratings are
stored one JSON line per call, so an interrupted run resumes where it stopped,
under the same spend cap as `slm4ie.silver.make`.

The person rates a sample balanced over job, domain and maker family on one
Markdown sheet; `agreement` compares two raters' judgements.
"""

import hashlib
import json
import random
import re
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from slm4ie.evaluation.backends import ChatBackend
from slm4ie.evaluation.prompts import FENCE
from slm4ie.silver.checks import CheckResult
from slm4ie.silver.make import Item, TokenEstimate, label_lines, price_usd, prompt_hash

# ===============================
# Defaults
# ===============================

JUDGE = "judge"

# the maker, family and job of a real passage in the key
REAL_TEXT = "real-text"
REAL_FAMILY = "real"
REAL_JOB = "real"

# hex characters of the sha256 kept as the unit id
UNIT_ID_CHARS = 12

SCALE_MIN = 1
SCALE_MAX = 5
SCALES = ("naturalness", "realism")
RATING_KEYS = frozenset({"errors", "naturalness", "realism", "mentions", "missed"})
ERROR_KEYS = frozenset({"phrase", "correction"})
MENTION_KEYS = frozenset({"number", "label_correct", "span_correct"})
MISSED_KEYS = frozenset({"text", "label"})

# error counts of 3 or more share one bin for agreement
ERROR_BIN_TOP = 3
QUANTITIES = (*SCALES, "errors", "label_correct")

PERSON_SAMPLE_SIZE = 60

# a judge reads the label list, the text and the mentions; the scoping projection's usage
DEFAULT_JUDGE_ESTIMATE = TokenEstimate(2_000, 1_000)

# a numbered judgement line of the judge prompt, such as `1. errors: ...`
JUDGEMENT_LINE = re.compile(r"^\d+\. \S")

# the lines of a rating sheet the reader matches
SHEET_SEPARATOR = "=>"
UNIT_HEADING = re.compile(r"^## Unit (\S+)$")
MENTION_LINE = re.compile(r"^(\d+)\. ")
FIELD_LINE = re.compile(r"^\s*(label correct|span correct|Naturalness|Realism) \((?:y/n|1-5)\):(.*)$")
YES = frozenset({"y", "yes"})
NO = frozenset({"n", "no"})

# ===============================
# Units, Keys and Summaries
# ===============================


@dataclass(frozen=True)
class RatingUnit:
    """One text a rater sees: nothing in it names its maker.

    Attributes:
        unit_id: A short hash of the maker, job and item id.
        domain: `medical` or `scientific`; picks the label set.
        text: The text to rate.
        mentions: The annotated mentions as `(text, label)`, numbered from 1 in this order; empty for a real passage.
    """

    unit_id: str
    domain: str
    text: str
    mentions: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class KeyEntry:
    """Who made a unit, kept apart from what a rater sees.

    Attributes:
        maker: The variant that made the output, or `real-text` for a real passage.
        family: The maker's provider family, or `real` for a real passage.
        job: `annotate` or `write`, or `real` for a real passage.
        item_id: The passage or brief id the output answers.
    """

    maker: str
    family: str
    job: str
    item_id: str


@dataclass(frozen=True)
class RatingPool:
    """The units to rate and the key to their makers.

    Attributes:
        units: The units, outputs first, then real passages.
        key: Unit id to its key entry.
    """

    units: tuple[RatingUnit, ...]
    key: Mapping[str, KeyEntry]


@dataclass(frozen=True)
class RateSummary:
    """What one run of `rate_outputs` did.

    Attributes:
        calls: Calls made in this run, failed ones included.
        skipped: Calls skipped because their rating was already stored.
        failed: Calls in this run that raised an error.
        dropped: Calls in this run whose answer was off the schema or the scale.
        spent_usd: The running total: the spend passed in, plus every stored and new call's cost.
        capped: True when the spend cap stopped the run before every call was made.
    """

    calls: int
    skipped: int
    failed: int
    dropped: int
    spent_usd: float
    capped: bool


@dataclass(frozen=True)
class Agreement:
    """How two raters agree on one quantity.

    Attributes:
        quantity: The rated quantity, one of `QUANTITIES`.
        n: Paired judgements: units both rated, or for `label_correct` the mentions of those units.
        raw: The share of pairs with the same value; `None` when there is no pair.
        within_one: The share of pairs at most one point apart, for the 1 to 5 scales; `None` otherwise.
        kappa: Linearly weighted Cohen's kappa; `None` when either rater used a single value.
    """

    quantity: str
    n: int
    raw: float | None
    within_one: float | None
    kappa: float | None


# ===============================
# The Rating Pool
# ===============================


def rating_pool(
    checked: Iterable[tuple[Mapping[str, Any], CheckResult]], passages: Iterable[Item], families: Mapping[str, str]
) -> RatingPool:
    """Build the units to rate: every output that passed the rule checks and every real passage.

    Args:
        checked: Each stored record of `make_examples` with its rule checks (`check_output`).
        passages: The real passages, as `annotate` items; each is rated as bare text with no mentions.
        families: Variant name to its provider family (the `family` field of `screen-models.yaml`).

    Returns:
        The units and the key from unit id to maker, family, job and item id.
    """
    units = []
    key = {}

    # one unit per output that passed every rule check
    for record, result in checked:
        if not result.passes:
            continue
        entry = KeyEntry(record["variant"], families[record["variant"]], record["job"], record["item_id"])
        mentions = tuple((mention["text"], mention["label"]) for mention in record["parsed"]["mentions"])
        units.append(RatingUnit(_unit_id(entry), record["domain"], record["example"]["text"], mentions))
        key[units[-1].unit_id] = entry

    # one bare unit per real passage
    for passage in passages:
        entry = KeyEntry(REAL_TEXT, REAL_FAMILY, REAL_JOB, passage.item_id)
        units.append(RatingUnit(_unit_id(entry), passage.domain, passage.text, ()))
        key[units[-1].unit_id] = entry

    return RatingPool(tuple(units), key)


def write_pool(pool: RatingPool, out_dir: Path) -> None:
    """Write the units a rater sees and the key to their makers as two separate JSONL files.

    Args:
        pool: The rating pool.
        out_dir: Folder of `units.jsonl` and `key.jsonl`; created when missing.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "units.jsonl", "w", encoding="utf-8") as handle:
        for unit in pool.units:
            handle.write(json.dumps(asdict(unit), ensure_ascii=False) + "\n")
    with open(out_dir / "key.jsonl", "w", encoding="utf-8") as handle:
        for unit_id, entry in pool.key.items():
            handle.write(json.dumps({"unit_id": unit_id, **asdict(entry)}, ensure_ascii=False) + "\n")


def read_pool(out_dir: Path) -> RatingPool:
    """Read a rating pool written by `write_pool`.

    Args:
        out_dir: Folder of `units.jsonl` and `key.jsonl`.

    Returns:
        The units and the key, in file order.
    """
    units = tuple(
        RatingUnit(row["unit_id"], row["domain"], row["text"], tuple(tuple(mention) for mention in row["mentions"]))
        for row in _read_lines(out_dir / "units.jsonl")
    )
    key = {
        row["unit_id"]: KeyEntry(row["maker"], row["family"], row["job"], row["item_id"])
        for row in _read_lines(out_dir / "key.jsonl")
    }
    return RatingPool(units, key)


# ===============================
# Judge Prompts and Answers
# ===============================


def mention_lines(unit: RatingUnit) -> str:
    """Write a unit's mentions as numbered lines `<n>. "<text>" — <label>`, or `(none)`.

    Args:
        unit: The unit.

    Returns:
        The mention list for the judge prompt's `{mentions}` placeholder.
    """
    if not unit.mentions:
        return "(none)"
    return "\n".join(f'{number}. "{text}" — {label}' for number, (text, label) in enumerate(unit.mentions, 1))


def build_judge_messages(
    unit: RatingUnit, label_specs: Mapping[str, Mapping[str, Any]], prompts: Mapping[str, Mapping[str, str]]
) -> list[dict[str, str]]:
    """Build the chat turns that ask a judge to rate one unit.

    Args:
        unit: The unit to rate.
        label_specs: The unit's domain's labels, label to spec.
        prompts: Prompt name to its `system` and `user` prompt (`prompts-jobs.yaml`); uses `judge`.

    Returns:
        A system turn and a user turn, each `{"role": str, "content": str}`.
    """
    prompt = prompts[JUDGE]
    user = prompt["user"].format(
        labels=label_lines(label_specs),
        label_names=", ".join(label_specs),
        text=unit.text,
        mentions=mention_lines(unit),
    )
    return [{"role": "system", "content": prompt["system"]}, {"role": "user", "content": user}]


def judge_schema(labels: Iterable[str], mention_count: int) -> dict[str, Any]:
    """Build the JSON schema a judge's answer must satisfy.

    Args:
        labels: The domain's label names; a missed entity takes one of them.
        mention_count: The unit's number of mentions; the answer judges each once.

    Returns:
        A JSON schema of an object holding `errors`, `naturalness`, `realism`, `mentions` and `missed`.
    """

    def strict(properties: dict[str, Any]) -> dict[str, Any]:
        """Wrap properties in an object that requires each and allows no other."""
        return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}

    scale = {"type": "integer", "minimum": SCALE_MIN, "maximum": SCALE_MAX}
    number = {"type": "integer", "minimum": 1, "maximum": max(mention_count, 1)}
    error = strict({"phrase": {"type": "string", "minLength": 1}, "correction": {"type": "string"}})
    mention = strict({"number": number, "label_correct": {"type": "boolean"}, "span_correct": {"type": "boolean"}})
    missed = strict({"text": {"type": "string", "minLength": 1}, "label": {"type": "string", "enum": list(labels)}})
    return strict(
        {
            "errors": {"type": "array", "items": error},
            "naturalness": scale,
            "realism": scale,
            "mentions": {"type": "array", "items": mention, "minItems": mention_count, "maxItems": mention_count},
            "missed": {"type": "array", "items": missed},
        }
    )


def parse_rating(answer: str, labels: Iterable[str], mention_count: int) -> dict[str, Any] | None:
    """Read a judge's answer, or `None` when it is off the schema or the scale.

    One markdown code fence around the JSON is stripped first. The answer must
    hold exactly the five judgements; both ratings must be integers from 1 to 5,
    the mention numbers must be exactly 1 to `mention_count`, each once, and
    every missed entity must carry a label from `labels`.

    Args:
        answer: The reply text.
        labels: The domain's label names.
        mention_count: The unit's number of mentions.

    Returns:
        The rating with its mentions sorted by number, or `None` for an answer off the schema or the scale.
    """
    allowed = set(labels)

    # strip one code fence and decode
    raw = answer.strip()
    fenced = FENCE.match(raw)
    if fenced:
        raw = fenced.group(1).strip()
    try:
        decoded = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(decoded, dict) or set(decoded) != RATING_KEYS:
        return None

    # check the two ratings and the three lists
    if not all(_is_int(decoded[scale]) and SCALE_MIN <= decoded[scale] <= SCALE_MAX for scale in SCALES):
        return None
    errors_ok = _all_shaped(
        decoded["errors"], ERROR_KEYS, lambda e: _is_text(e["phrase"]) and isinstance(e["correction"], str)
    )
    mentions_ok = _all_shaped(
        decoded["mentions"],
        MENTION_KEYS,
        lambda m: _is_int(m["number"]) and isinstance(m["label_correct"], bool) and isinstance(m["span_correct"], bool),
    )
    missed_ok = _all_shaped(decoded["missed"], MISSED_KEYS, lambda m: _is_text(m["text"]) and m["label"] in allowed)
    if not (errors_ok and mentions_ok and missed_ok):
        return None

    # the mentions must judge each of the unit's mentions once
    mentions = sorted(decoded["mentions"], key=lambda mention: mention["number"])
    if [mention["number"] for mention in mentions] != list(range(1, mention_count + 1)):
        return None
    return {**decoded, "mentions": mentions}


def errors_per_100_words(rating: Mapping[str, Any], text: str) -> float:
    """Count a rating's listed errors per 100 whitespace-separated words of the rated text.

    Args:
        rating: A parsed rating, with its `errors` list.
        text: The rated text.

    Returns:
        The errors per 100 words.

    Raises:
        ValueError: If the text holds no word.
    """
    words = len(text.split())
    if not words:
        raise ValueError("the rated text holds no word, so errors per 100 words is undefined")
    return 100 * len(rating["errors"]) / words


# ===============================
# Rating by Judges
# ===============================


def rate_outputs(
    units: Iterable[RatingUnit],
    judges: Mapping[str, Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Mapping[str, Any]]],
    prompts: Mapping[str, Mapping[str, str]],
    out_dir: Path,
    limit_usd: float,
    spent_usd: float = 0.0,
    *,
    backend_factory: Callable[[str, Mapping[str, Any]], ChatBackend],
    seed: int,
    max_tokens: int,
    token_estimate: TokenEstimate = DEFAULT_JUDGE_ESTIMATE,
) -> RateSummary:
    """Ask every judge to rate every unit once and store one rating per call.

    Units go in a seeded shuffled order, each across judges, so a cap stop
    loses a balanced part of the pool. A `(judge, unit_id)` already stored in
    `<out_dir>/<judge>.jsonl` is skipped. A call that raises is stored with its
    error and not retried; an answer off the schema is stored with a null rating.

    Before each call the run stops when the running total plus a guard would
    pass `limit_usd`, as in `make_examples`. The guard is the call's worst case:
    the input tokens of `token_estimate` plus `max_tokens` of output at list
    prices, raised to the costliest call the judge has made so far when that
    cost more. A reply that reports no cost counts at list prices from its token counts.

    Args:
        units: The units to rate (`rating_pool`).
        judges: Judge name to its model spec (the `judges` block of `screen-models.yaml`).
        labels: Domain to label to its spec (`load_labels`).
        prompts: Prompt name to its `system` and `user` prompt (`prompts-jobs.yaml`); uses `judge`.
        out_dir: Folder of the per-judge rating files; created when missing.
        limit_usd: The spend no call may pass.
        spent_usd: Spend already made outside `out_dir`'s ratings, such as the made examples.
        backend_factory: Function of the judge name and its spec that returns its backend.
        seed: Seed of the order the units are rated in.
        max_tokens: The most tokens a reply may hold, the backend's reply cap (`sampling.max_tokens`).
        token_estimate: The tokens one judge call is expected to use.

    Returns:
        The calls made, skipped, failed and dropped, the running total and whether the cap stopped the run.
    """
    units = sorted(units, key=lambda unit: unit.unit_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    # read what is stored and start the running total from it
    stored: dict[str, set[str]] = {}
    costliest: dict[str, float] = {}
    total = spent_usd
    for judge in judges:
        stored[judge] = set()
        for rating in _read_lines(out_dir / f"{judge}.jsonl"):
            stored[judge].add(rating["unit_id"])
            total += rating["cost_usd"]
            if rating["error"] is None:
                costliest[judge] = max(costliest.get(judge, 0.0), rating["cost_usd"])

    # list the calls still to make, unit by unit in a seeded order across judges
    random.Random(seed).shuffle(units)
    pending = [(unit, judge) for unit in units for judge in judges if unit.unit_id not in stored[judge]]
    skipped = len(units) * len(judges) - len(pending)
    backends = {judge: backend_factory(judge, spec) for judge, spec in judges.items()}

    # make each call unless it could pass the cap
    calls = failed = dropped = 0
    capped = False
    for unit, judge in tqdm(pending, desc="rating outputs", unit="call"):
        spec = judges[judge]
        worst = price_usd(spec, token_estimate.input_tokens, max_tokens)
        guard = max(worst, costliest.get(judge, 0.0))
        if total + guard > limit_usd:
            capped = True
            break

        rating = _rate_unit(judge, spec, unit, labels[unit.domain], prompts, backends[judge])
        _append_line(out_dir / f"{judge}.jsonl", rating)
        calls += 1
        total += rating["cost_usd"]
        if rating["error"] is not None:
            failed += 1
            continue
        costliest[judge] = max(costliest.get(judge, 0.0), rating["cost_usd"])
        dropped += rating["rating"] is None

    return RateSummary(calls, skipped, failed, dropped, total, capped)


# ===============================
# Rating by a Person
# ===============================


def draw_person_sample(
    units: Iterable[RatingUnit], key: Mapping[str, KeyEntry], size: int = PERSON_SAMPLE_SIZE, *, seed: int
) -> tuple[RatingUnit, ...]:
    """Draw the units one person rates, balanced over job, domain and maker family.

    Units are grouped into strata by job, domain and family; real passages form
    the family `real`. Each round takes one seeded draw from every stratum that
    still holds a unit, and rounds go on until the sample holds `size` units, so
    the last round is whole and the sample can exceed `size`.

    Args:
        units: The rating pool's units.
        key: Unit id to its key entry.
        size: The fewest units the sample holds.
        seed: Seed of the draw within each stratum.

    Returns:
        The sample, in the order drawn.

    Raises:
        ValueError: If the pool holds fewer units than `size`.
    """
    units = sorted(units, key=lambda unit: unit.unit_id)
    if len(units) < size:
        raise ValueError(f"the rating pool holds {len(units)} units, fewer than the sample size {size}")

    # group the units into strata and shuffle each with the seed
    strata: dict[tuple[str, str, str], list[RatingUnit]] = {}
    for unit in units:
        entry = key[unit.unit_id]
        strata.setdefault((entry.job, unit.domain, entry.family), []).append(unit)
    queues = [strata[stratum] for stratum in sorted(strata)]
    rng = random.Random(seed)
    for queue in queues:
        rng.shuffle(queue)

    # take one unit per stratum per round until the sample is large enough
    sample: list[RatingUnit] = []
    while len(sample) < size:
        sample.extend(queue.pop() for queue in queues if queue)
    return tuple(sample)


def judgement_lines(prompts: Mapping[str, Mapping[str, str]]) -> tuple[str, ...]:
    """Read the numbered judgements out of the judge prompt, so a person reads the judges' wording.

    Args:
        prompts: Prompt name to its `system` and `user` prompt (`prompts-jobs.yaml`); uses `judge`.

    Returns:
        The judge prompt's user lines that open with a number and a full stop, in prompt order.

    Raises:
        ValueError: If the judge prompt holds no numbered judgement.
    """
    lines = tuple(line.strip() for line in prompts[JUDGE]["user"].splitlines() if JUDGEMENT_LINE.match(line.strip()))
    if not lines:
        raise ValueError("the judge prompt in prompts-jobs.yaml lists no numbered judgement such as '1. errors: ...'")
    return lines


def write_rating_sheet(
    sample: Iterable[RatingUnit],
    labels: Mapping[str, Mapping[str, Mapping[str, Any]]],
    prompts: Mapping[str, Mapping[str, str]],
    path: Path,
) -> None:
    """Write the sheet one person fills in by hand, one section per unit under its id only.

    The judgements, worded as in the judge prompt, and the label definitions of
    every domain come once at the top. Units go in the order of their ids, which
    as hashes reveal nothing of the maker. Each section shows the text, the
    numbered mentions and blank fields for the five judgements.

    Args:
        sample: The units to rate (`draw_person_sample`).
        labels: Domain to label to its spec (`load_labels`).
        prompts: Prompt name to its `system` and `user` prompt (`prompts-jobs.yaml`); the judgements
            are read from `judge` (`judgement_lines`).
        path: The Markdown file to write; its folder must exist.

    Raises:
        ValueError: If the judge prompt holds no numbered judgement.
    """
    lines = [
        "# Blind rating sheet",
        "",
        "Review each Slovene text and the entity mentions annotated in it. You do not know who wrote or "
        "annotated it. Fill in the fields after each colon. Leave a whole text blank to skip it.",
        "",
        "## The five judgements",
        "",
        *judgement_lines(prompts),
        "",
        f"Answer the mention questions with y or n. Write each error as `- <phrase> {SHEET_SEPARATOR} <correction>` "
        f"and each missed entity as `- <text> {SHEET_SEPARATOR} <label>`, one per line.",
    ]
    for domain, label_specs in labels.items():
        lines += ["", f"## Entity types: {domain}", "", label_lines(label_specs)]
    for unit in sorted(sample, key=lambda unit: unit.unit_id):
        lines += ["", *_sheet_section(unit)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_rating_sheet(path: Path) -> dict[str, dict[str, Any] | None]:
    """Read a filled rating sheet back into the judges' rating shape.

    A unit whose fields are all blank is unrated. A unit with any field filled
    must have both ratings and every mention answered.

    Args:
        path: The sheet written by `write_rating_sheet`.

    Returns:
        Unit id to its rating, or `None` for an unrated unit, in sheet order.

    Raises:
        ValueError: If a value is off its scale or a filled unit leaves a field blank, naming the unit and the value.
    """
    sections: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        heading = UNIT_HEADING.match(line)
        if heading:
            current = sections.setdefault(heading.group(1), [])
        elif current is not None:
            current.append(line)
    return {unit_id: _read_section(unit_id, section) for unit_id, section in sections.items()}


def agreement(
    ratings_a: Mapping[str, Mapping[str, Any] | None], ratings_b: Mapping[str, Mapping[str, Any] | None], quantity: str
) -> Agreement:
    """Measure how two raters agree on one quantity over the units both rated.

    Error counts are binned 0, 1, 2 and 3 or more. `label_correct` pairs the
    two raters' answers on each mention of a unit both rated.

    Args:
        ratings_a: Unit id to the first rater's rating, `None` where unrated.
        ratings_b: Unit id to the second rater's rating, `None` where unrated.
        quantity: One of `naturalness`, `realism`, `errors`, `label_correct`.

    Returns:
        The number of pairs, raw agreement, agreement within one point and weighted kappa.

    Raises:
        ValueError: If `quantity` is not one of `QUANTITIES`.
    """
    if quantity not in QUANTITIES:
        raise ValueError(f"unknown rated quantity {quantity!r}; known: {QUANTITIES}")

    # pair the two raters' values over the units both rated
    pairs: list[tuple[int, int]] = []
    for unit_id in sorted(set(ratings_a) & set(ratings_b)):
        rating_a, rating_b = ratings_a[unit_id], ratings_b[unit_id]
        if rating_a is not None and rating_b is not None:
            pairs += _paired_values(rating_a, rating_b, quantity)
    if not pairs:
        return Agreement(quantity, 0, None, None, None)

    # agreement shares, then kappa with linear weights
    n = len(pairs)
    values_a = [a for a, _ in pairs]
    values_b = [b for _, b in pairs]
    raw = sum(a == b for a, b in pairs) / n
    within_one = sum(abs(a - b) <= 1 for a, b in pairs) / n if quantity in SCALES else None
    kappa = None
    if len(set(values_a)) > 1 and len(set(values_b)) > 1:
        observed = sum(abs(a - b) for a, b in pairs) / n
        chance = sum(abs(a - b) for a in values_a for b in values_b) / n**2
        kappa = 1 - observed / chance

    return Agreement(quantity, n, raw, within_one, kappa)


# ===============================
# Helpers
# ===============================


def _unit_id(entry: KeyEntry) -> str:
    """Hash the maker, job and item id into the id a rater sees."""
    source = f"{entry.maker}\n{entry.job}\n{entry.item_id}"
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:UNIT_ID_CHARS]


def _is_int(value: Any) -> bool:
    """Return True for an integer that is not a boolean."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_text(value: Any) -> bool:
    """Return True for a non-empty string."""
    return isinstance(value, str) and bool(value)


def _all_shaped(items: Any, keys: frozenset[str], valid: Callable[[dict[str, Any]], bool]) -> bool:
    """Return True when `items` is a list of objects with exactly `keys` that each pass `valid`."""
    return isinstance(items, list) and all(
        isinstance(item, dict) and set(item) == keys and valid(item) for item in items
    )


def _rate_unit(
    judge: str,
    spec: Mapping[str, Any],
    unit: RatingUnit,
    label_specs: Mapping[str, Mapping[str, Any]],
    prompts: Mapping[str, Mapping[str, str]],
    backend: ChatBackend,
) -> dict[str, Any]:
    """Call the judge once for one unit and build its stored rating."""
    messages = build_judge_messages(unit, label_specs, prompts)
    rating: dict[str, Any] = {
        "judge": judge,
        "unit_id": unit.unit_id,
        "prompt_hash": prompt_hash(messages),
        "raw": None,
        "rating": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "cost_reported": False,
        "seconds": 0.0,
        "error": None,
    }

    # call once; a failed call is stored, not retried
    start = time.perf_counter()
    try:
        result = backend.chat(messages, judge_schema(label_specs, len(unit.mentions)))
    except Exception as error:
        return {**rating, "seconds": time.perf_counter() - start, "error": f"{type(error).__name__}: {error}"}

    # count an unreported cost at list prices
    read = result.input_tokens + result.cache_read_tokens + result.cache_write_tokens
    cost = result.cost_usd if result.cost_usd is not None else price_usd(spec, read, result.output_tokens)
    return {
        **rating,
        "raw": result.text,
        "rating": parse_rating(result.text, label_specs, len(unit.mentions)),
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "cost_usd": cost,
        "cost_reported": result.cost_usd is not None,
        "seconds": result.seconds,
    }


def _sheet_section(unit: RatingUnit) -> list[str]:
    """Write one unit's section of the rating sheet, with its fields blank."""
    lines = [f"## Unit {unit.unit_id}", "", f"Domain: {unit.domain}", ""]
    lines += [f"> {line}" if line else ">" for line in unit.text.splitlines()]
    lines += ["", "Mentions:", ""]
    if not unit.mentions:
        lines.append("(none)")
    for number, (text, label) in enumerate(unit.mentions, 1):
        lines += [f'{number}. "{text}" — {label}', "   label correct (y/n):", "   span correct (y/n):"]
    lines += ["", "Errors:", "", "- ", "", "Naturalness (1-5):", "", "Realism (1-5):", "", "Missed:", "", "- "]
    return lines


def _read_section(unit_id: str, section: list[str]) -> dict[str, Any] | None:
    """Read one unit's section of a filled sheet; `None` when every field is blank."""
    fields: dict[str, str] = {}
    answers: dict[int, dict[str, str]] = {}
    listed: dict[str, list[tuple[str, str]]] = {"Errors": [], "Missed": []}
    number = 0
    block = ""

    # collect the raw values, field by field
    for line in section:
        if line.startswith(">"):
            continue
        if line.rstrip(":") in listed:
            block = line.rstrip(":")
            continue
        mention = MENTION_LINE.match(line)
        field = FIELD_LINE.match(line)
        if mention:
            number = int(mention.group(1))
            answers[number] = {}
        elif field and field.group(1) in ("label correct", "span correct"):
            answers[number][field.group(1)] = field.group(2).strip()
        elif field:
            fields[field.group(1)] = field.group(2).strip()
        elif block and line.startswith("-") and line[1:].strip():
            phrase, separator, correction = line[1:].strip().partition(SHEET_SEPARATOR)
            if not separator:
                raise ValueError(f"unit {unit_id}: {block.lower()} line {line!r} has no '{SHEET_SEPARATOR}'")
            listed[block].append((phrase.strip(), correction.strip()))

    # a unit with nothing filled is unrated
    filled = [*fields.values(), *(value for answer in answers.values() for value in answer.values())]
    if not any(filled) and not any(listed.values()):
        return None

    return {
        "errors": [{"phrase": phrase, "correction": correction} for phrase, correction in listed["Errors"]],
        "naturalness": _sheet_scale(unit_id, "naturalness", fields.get("Naturalness", "")),
        "realism": _sheet_scale(unit_id, "realism", fields.get("Realism", "")),
        "mentions": [
            {
                "number": number,
                "label_correct": _sheet_yes_no(unit_id, f"mention {number} label", answer.get("label correct", "")),
                "span_correct": _sheet_yes_no(unit_id, f"mention {number} span", answer.get("span correct", "")),
            }
            for number, answer in sorted(answers.items())
        ],
        "missed": [{"text": text, "label": label} for text, label in listed["Missed"]],
    }


def _sheet_scale(unit_id: str, name: str, value: str) -> int:
    """Read a 1 to 5 rating from the sheet, raising on a blank or off-scale value."""
    if not value.isdigit() or not SCALE_MIN <= int(value) <= SCALE_MAX:
        raise ValueError(f"unit {unit_id}: {name} {value!r} is not an integer from {SCALE_MIN} to {SCALE_MAX}")
    return int(value)


def _sheet_yes_no(unit_id: str, name: str, value: str) -> bool:
    """Read a y/n answer from the sheet, raising on a blank or other value."""
    if value.lower() in YES:
        return True
    if value.lower() in NO:
        return False
    raise ValueError(f"unit {unit_id}: {name} {value!r} is not y or n")


def _paired_values(rating_a: Mapping[str, Any], rating_b: Mapping[str, Any], quantity: str) -> list[tuple[int, int]]:
    """Pair two raters' values of one quantity on one unit."""
    if quantity in SCALES:
        return [(rating_a[quantity], rating_b[quantity])]
    if quantity == "errors":
        return [(min(len(rating_a["errors"]), ERROR_BIN_TOP), min(len(rating_b["errors"]), ERROR_BIN_TOP))]
    labels_a = {mention["number"]: mention["label_correct"] for mention in rating_a["mentions"]}
    labels_b = {mention["number"]: mention["label_correct"] for mention in rating_b["mentions"]}
    return [(int(labels_a[number]), int(labels_b[number])) for number in sorted(labels_a.keys() & labels_b.keys())]


def _read_lines(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file; no rows when the file does not exist yet."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _append_line(path: Path, row: Mapping[str, Any]) -> None:
    """Append one row as a JSON line, so a stop loses no stored call."""
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
