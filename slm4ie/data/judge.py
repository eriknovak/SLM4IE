"""Label documents with a language-model judge driven from the command line.

The judge scores a sample of documents against a rubric so that a pipeline's
own decisions can be measured against something other than intuition. It runs
outside the session, as a plain subprocess per batch, which is what makes the
run bounded and resumable: batches already written to the destination are never
sent again, and `max_batches` stops a run before it costs more than intended.

Two backends run the same rubric. `api` sends the work to the Batch API: half
price, answers within hours rather than seconds, and the reply shape is enforced
by a structured-output schema. `cli` shells out to a command-line tool once per
group, which needs no API key and is the way to try a rubric change quickly.

The rubric is a prompt file holding a `{{DOCUMENTS}}` placeholder. Each group
substitutes a JSON array of `{"id", "text"}` objects there and asks for a JSON
array of verdicts in return. The judge sees ids and text only — never which
stage handled a document or what that stage decided — so its labels stay
independent of the decisions they are used to score.

Every verdict is validated before it is written: an id the judge was never
sent, a repeated id, or a field outside the rubric's values drops the whole
group. A judge that simply skips a document is treated more gently — the
verdicts that came back are kept and the skipped documents are judged again on
the next run, which is why a run is safe to repeat until nothing is left.
"""

import json
import logging
import random
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

#: Placeholder in the rubric file that the batch's documents replace.
DOCUMENTS_PLACEHOLDER: str = "{{DOCUMENTS}}"

#: What a document is, as a shape of text.
TEXT_TYPES: Tuple[str, ...] = ("prose", "boilerplate", "list", "code", "garbage")

#: Domain vocabulary, matching the dataset catalog's `domain` tags so a judged
#: label can be compared against the source's declared domain.
DOMAINS: Tuple[str, ...] = (
    "medical",
    "scientific",
    "legal",
    "news",
    "parliamentary",
    "academic",
    "wiki",
    "forum",
    "blog",
    "student",
    "web",
    "finance",
    "other",
)

#: Backends a judge run can use: a local CLI subprocess, or the Batch API,
#: which costs half as much and answers within hours rather than seconds.
BACKENDS: Tuple[str, str] = ("cli", "api")

#: Schema handed to the API as a structured output, so the reply is guaranteed
#: to parse and to carry only the rubric's values. The CLI backend cannot
#: enforce it and is checked by `validate_batch` alone.
VERDICT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "language": {"type": "string"},
                    "text_type": {"type": "string", "enum": list(TEXT_TYPES)},
                    "adult_or_spam": {"type": "boolean"},
                    "coherence": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                    "machine_translated": {"type": "boolean"},
                    "pii": {"type": "boolean"},
                    "domain": {"type": "string", "enum": list(DOMAINS)},
                    "note": {"type": "string"},
                },
                "required": [
                    "id",
                    "language",
                    "text_type",
                    "adult_or_spam",
                    "coherence",
                    "machine_translated",
                    "pii",
                    "domain",
                    "note",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["verdicts"],
    "additionalProperties": False,
}

#: Fields a verdict must carry, with the check each one has to pass.
_REQUIRED_FIELDS: Dict[str, Any] = {
    "language": lambda value: isinstance(value, str) and bool(re.fullmatch(r"[a-z]{2}|mixed", value)),
    "text_type": lambda value: value in TEXT_TYPES,
    "adult_or_spam": lambda value: isinstance(value, bool),
    "coherence": lambda value: isinstance(value, int) and 1 <= value <= 5,
    "machine_translated": lambda value: isinstance(value, bool),
    "pii": lambda value: isinstance(value, bool),
    "domain": lambda value: value in DOMAINS,
}


def read_documents(source: Path) -> List[Dict[str, Any]]:
    """Read the sampled documents to judge.

    Args:
        source: JSONL file with one document per line, carrying `id` and `text`.

    Returns:
        The documents, in file order.

    Raises:
        ValueError: If a line carries no `id` or no `text`.
    """
    documents = []
    # split("\n"), not splitlines(): document text carries \u2028 and friends,
    # which splitlines() would treat as line breaks and tear a JSON row in half.
    for number, line in enumerate(source.read_text(encoding="utf-8").split("\n"), start=1):
        if not line.strip():
            continue
        document = json.loads(line)
        if not document.get("id") or not document.get("text"):
            raise ValueError(f"{source}:{number} has no id or no text")
        documents.append(document)
    return documents


def judged_ids(destination: Path) -> Set[str]:
    """Read the ids already judged in an earlier run.

    Args:
        destination: JSONL file of verdicts; may not exist yet.

    Returns:
        The set of ids present in the file, empty when it does not exist.
    """
    if not destination.is_file():
        return set()
    ids = set()
    for line in destination.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            ids.add(json.loads(line)["id"])
    return ids


def draw_calibration_set(documents: Sequence[Dict[str, Any]], size: int, seed: int = 20260916) -> List[Dict[str, Any]]:
    """Choose the documents a person labels by hand, spread across the strata.

    The judge's labels only become evidence once their agreement with a human
    is known, and agreement measured on one corner of the sample says nothing
    about the rest. So the calibration set is taken round-robin across the
    cells the sample was drawn into, which spreads it over every source, stage
    and decision rather than concentrating it where documents are plentiful.

    Args:
        documents: Sample rows, each carrying the `cells` it was drawn into.
        size: How many documents to choose; fewer if the sample is smaller.
        seed: Seed fixing both the order of the cells and the pick within one.

    Returns:
        The chosen rows, ordered by id.
    """
    strata: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for document in documents:
        cell = sorted(document["cells"], key=lambda c: (c["stage"], c["decision"]))[0]
        strata.setdefault((document["dataset"], cell["stage"], cell["decision"]), []).append(document)

    rng = random.Random(seed)
    order = sorted(strata)
    rng.shuffle(order)
    for key in order:
        rng.shuffle(strata[key])

    chosen: List[Dict[str, Any]] = []
    while len(chosen) < size and any(strata[key] for key in order):
        for key in order:
            if not strata[key]:
                continue
            chosen.append(strata[key].pop())
            if len(chosen) == size:
                break
    return sorted(chosen, key=lambda document: document["id"])


def _batches(documents: Sequence[Dict[str, Any]], size: int) -> List[List[Dict[str, Any]]]:
    """Split documents into fixed-size batches.

    Args:
        documents: Documents to judge.
        size: Documents per batch.

    Returns:
        Batches in document order; the last one may be short.
    """
    return [list(documents[start : start + size]) for start in range(0, len(documents), size)]


def render_prompt(rubric: str, batch: Sequence[Dict[str, Any]], max_chars: int) -> str:
    """Fill the rubric's placeholder with one batch of documents.

    Args:
        rubric: The rubric prompt, containing `{{DOCUMENTS}}`.
        batch: Documents to judge in this call.
        max_chars: Characters of each document's text to send.

    Returns:
        The prompt to hand the judge.

    Raises:
        ValueError: If the rubric has no `{{DOCUMENTS}}` placeholder.
    """
    if DOCUMENTS_PLACEHOLDER not in rubric:
        raise ValueError(f"the rubric has no {DOCUMENTS_PLACEHOLDER} placeholder")
    payload = [{"id": document["id"], "text": document["text"][:max_chars]} for document in batch]
    return rubric.replace(DOCUMENTS_PLACEHOLDER, json.dumps(payload, ensure_ascii=False, indent=2))


def parse_verdicts(response: str) -> List[Dict[str, Any]]:
    """Pull the JSON array of verdicts out of a judge's reply.

    Args:
        response: Everything the judge wrote.

    Returns:
        The decoded verdicts.

    Raises:
        ValueError: If the reply holds no JSON array, or it is not a list of
            objects.
    """
    text = response.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    # Structured outputs return `{"verdicts": [...]}`; a prompted reply returns
    # the bare array, sometimes wrapped in prose.
    if text.startswith("{"):
        decoded = json.loads(text)
        verdicts = decoded.get("verdicts") if isinstance(decoded, dict) else None
        if verdicts is None:
            raise ValueError("the judge's reply is an object without a `verdicts` array")
    else:
        start, end = text.find("["), text.rfind("]")
        if start == -1 or end <= start:
            raise ValueError(f"no JSON array in the judge's reply: {response[:200]!r}")
        verdicts = json.loads(text[start : end + 1])
    if not isinstance(verdicts, list) or not all(isinstance(item, dict) for item in verdicts):
        raise ValueError("the judge's reply is not a list of objects")
    return verdicts


def validate_batch(verdicts: Sequence[Dict[str, Any]], batch: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Check a batch's verdicts against the rubric's vocabulary.

    Args:
        verdicts: What the judge returned.
        batch: The documents that were sent.

    Returns:
        The verdicts, each narrowed to the rubric's fields plus `id` and `note`.
        A judge that skipped some documents yields fewer rows than were sent;
        the rest are judged again on the next run.

    Raises:
        ValueError: If the judge returned an id that was never sent or the
            same id twice, or a field is missing or outside the rubric's values.
    """
    expected = {document["id"] for document in batch}
    returned = [str(verdict.get("id")) for verdict in verdicts]
    unknown = sorted(set(returned) - expected)
    if unknown:
        raise ValueError(f"the judge returned ids that were not sent: {unknown}")
    if len(set(returned)) != len(returned):
        raise ValueError("the judge returned the same id twice")
    # A short reply is kept, not discarded: the verdicts that came back are
    # sound, and the documents left out are simply judged again on the next run.
    missing = sorted(expected - set(returned))
    if missing:
        logger.warning("the judge skipped %d of %d documents: %s", len(missing), len(expected), missing[:3])

    checked = []
    for verdict in verdicts:
        for field, is_valid in _REQUIRED_FIELDS.items():
            if field not in verdict:
                raise ValueError(f"{verdict['id']}: no {field}")
            if not is_valid(verdict[field]):
                raise ValueError(f"{verdict['id']}: {field}={verdict[field]!r} is not an allowed value")
        row = {"id": str(verdict["id"]), **{field: verdict[field] for field in _REQUIRED_FIELDS}}
        note = verdict.get("note")
        row["note"] = str(note) if note else ""
        checked.append(row)
    return checked


def ask_judge(prompt: str, command: str, model: str, timeout: int) -> str:
    """Run one judging call as a subprocess.

    Args:
        prompt: The rendered prompt, passed on standard input.
        command: The judge executable, e.g. `claude`.
        model: Model name handed to the executable.
        timeout: Seconds to wait before giving up on the call.

    Returns:
        The judge's reply.

    Raises:
        RuntimeError: If the executable exits non-zero.
        subprocess.TimeoutExpired: If it outruns *timeout*.
    """
    completed = subprocess.run(
        [command, "-p", "--model", model],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{command} exited {completed.returncode}: {completed.stderr.strip()[:300]}")
    return completed.stdout


def _write_verdicts(destination: Path, verdicts: Sequence[Dict[str, Any]]) -> None:
    """Append verdicts to the output file.

    Args:
        destination: JSONL file of verdicts.
        verdicts: Checked verdicts to append.
    """
    with destination.open("a", encoding="utf-8") as fh:
        for verdict in verdicts:
            fh.write(json.dumps(verdict, ensure_ascii=False) + "\n")


def _request_params(model: str, batch: Sequence[Dict[str, Any]], prompt: str) -> Dict[str, Any]:
    """Build the Messages API parameters for one group of documents.

    Args:
        model: Model to judge with.
        batch: Documents in this group, used to size the reply.
        prompt: The rendered prompt.

    Returns:
        Keyword arguments for a Messages API request.
    """
    params: Dict[str, Any] = {
        "model": model,
        # ~120 tokens covers one verdict; the floor carries the shortest group.
        "max_tokens": max(1024, 120 * len(batch)),
        "messages": [{"role": "user", "content": prompt}],
        "output_config": {"format": {"type": "json_schema", "schema": VERDICT_SCHEMA}},
    }
    # Labelling against a fixed rubric needs no reasoning, and thinking bills as
    # output. Haiku does not think unless asked, so it takes no `thinking` key.
    if not model.startswith("claude-haiku"):
        params["thinking"] = {"type": "disabled"}
    return params


def submit_batch(
    client: Any, model: str, groups: Sequence[Sequence[Dict[str, Any]]], rubric: str, max_chars: int
) -> str:
    """Send every group to the Batch API as one batch.

    Args:
        client: An `anthropic.Anthropic` client.
        model: Model to judge with.
        groups: Document groups, one per request.
        rubric: The rubric prompt.
        max_chars: Characters of each document's text to send.

    Returns:
        The batch's id.
    """
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    requests = [
        Request(
            custom_id=f"group-{index:05d}",
            params=MessageCreateParamsNonStreaming(
                **_request_params(model, group, render_prompt(rubric, group, max_chars))
            ),
        )
        for index, group in enumerate(groups)
    ]
    batch = client.messages.batches.create(requests=requests)
    logger.info("submitted batch %s with %d request(s)", batch.id, len(requests))
    return str(batch.id)


def collect_batch(
    client: Any,
    batch_id: str,
    groups: Dict[str, List[Dict[str, Any]]],
    destination: Path,
    poll_seconds: int,
) -> Dict[str, int]:
    """Wait for a batch to end, then write every verdict it returned.

    Args:
        client: An `anthropic.Anthropic` client.
        batch_id: The batch to collect.
        groups: The documents sent, keyed by the request's `custom_id`.
        destination: JSONL file of verdicts.
        poll_seconds: Seconds between status checks.

    Returns:
        Counts keyed `judged` and `failed`.
    """
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            break
        logger.info(
            "batch %s: %s, %d still processing", batch_id, batch.processing_status, batch.request_counts.processing
        )
        time.sleep(poll_seconds)

    counts = {"judged": 0, "failed": 0}
    # Results arrive in any order, so every group is found by its custom_id.
    for result in client.messages.batches.results(batch_id):
        group = groups.get(result.custom_id, [])
        if result.result.type != "succeeded":
            logger.warning("%s: %s; will be judged again on the next run", result.custom_id, result.result.type)
            counts["failed"] += len(group)
            continue
        reply = next((block.text for block in result.result.message.content if block.type == "text"), "")
        try:
            verdicts = validate_batch(parse_verdicts(reply), group)
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning("%s failed validation, will be judged again: %s", result.custom_id, exc)
            counts["failed"] += len(group)
            continue
        _write_verdicts(destination, verdicts)
        counts["judged"] += len(verdicts)
        counts["failed"] += len(group) - len(verdicts)
    return counts


def _judge_with_cli(
    groups: Sequence[Sequence[Dict[str, Any]]],
    rubric: str,
    destination: Path,
    model: str,
    concurrency: int,
    max_chars: int,
    command: str,
    timeout: int,
) -> Dict[str, int]:
    """Judge every group through the command-line tool.

    Args:
        groups: Document groups, one per call.
        rubric: The rubric prompt.
        destination: JSONL file of verdicts.
        model: Model to judge with.
        concurrency: Calls in flight at once.
        max_chars: Characters of each document's text to send.
        command: The judge executable.
        timeout: Seconds allowed per call.

    Returns:
        Counts keyed `judged` and `failed`.
    """
    lock = threading.Lock()
    counts = {"judged": 0, "failed": 0}

    def run_group(index_and_group: Tuple[int, Sequence[Dict[str, Any]]]) -> None:
        """Judge one group and append its verdicts.

        Args:
            index_and_group: The group's position and its documents.
        """
        index, group = index_and_group
        try:
            reply = ask_judge(render_prompt(rubric, group, max_chars), command, model, timeout)
            verdicts = validate_batch(parse_verdicts(reply), group)
        except (RuntimeError, ValueError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
            logger.warning("group %d of %d failed, will be judged again: %s", index + 1, len(groups), exc)
            with lock:
                counts["failed"] += len(group)
            return
        with lock:
            _write_verdicts(destination, verdicts)
            counts["judged"] += len(verdicts)
            counts["failed"] += len(group) - len(verdicts)
            logger.info("group %d of %d: %d verdicts", index + 1, len(groups), len(verdicts))

    work: Iterable[Tuple[int, Sequence[Dict[str, Any]]]] = enumerate(groups)
    if concurrency > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            list(pool.map(run_group, work))
    else:
        for item in work:
            run_group(item)
    return counts


def judge_documents(
    source: Path,
    destination: Path,
    rubric_path: Path,
    backend: str = "api",
    model: str = "claude-sonnet-5",
    batch_size: int = 10,
    concurrency: int = 4,
    max_batches: Optional[int] = None,
    max_chars: int = 2000,
    command: str = "claude",
    timeout: int = 600,
    poll_seconds: int = 60,
    client: Optional[Any] = None,
) -> Dict[str, int]:
    """Judge a sample of documents, appending one verdict per line.

    Documents already present in *destination* are skipped, so an interrupted
    run continues where it stopped and a finished run is a no-op.

    Args:
        source: JSONL file of sampled documents.
        destination: JSONL file of verdicts; appended to, created if absent.
        rubric_path: Prompt file holding the rubric and `{{DOCUMENTS}}`.
        backend: `api` for the Batch API (half price, answers within hours) or
            `cli` for one subprocess per group.
        model: Model the judge should use.
        batch_size: Documents per request.
        concurrency: Calls in flight at once; the `cli` backend only.
        max_batches: Stop after this many requests, or None for all of them.
        max_chars: Characters of each document's text to send.
        command: The judge executable; the `cli` backend only.
        timeout: Seconds allowed per call; the `cli` backend only.
        poll_seconds: Seconds between batch status checks; the `api` backend only.
        client: An `anthropic.Anthropic` to use instead of building one, so a
            caller can supply a configured or fake client.

    Returns:
        Counts keyed `documents`, `skipped`, `batches`, `judged` and `failed`.

    Raises:
        ValueError: If *backend* is not one of `BACKENDS`.
    """
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; expected one of {', '.join(BACKENDS)}")

    rubric = rubric_path.read_text(encoding="utf-8")
    documents = read_documents(source)
    done = judged_ids(destination)
    pending = [document for document in documents if document["id"] not in done]
    groups = _batches(pending, batch_size)
    if max_batches is not None:
        groups = groups[:max_batches]

    logger.info(
        "judging %d of %d documents in %d request(s) of %d, via %s, with %s",
        sum(len(group) for group in groups),
        len(documents),
        len(groups),
        batch_size,
        backend,
        model,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    counts = {"documents": len(documents), "skipped": len(done), "batches": len(groups), "judged": 0, "failed": 0}

    if groups:
        if backend == "cli":
            outcome = _judge_with_cli(groups, rubric, destination, model, concurrency, max_chars, command, timeout)
        else:
            if client is None:
                import anthropic

                client = anthropic.Anthropic()
            batch_id = submit_batch(client, model, groups, rubric, max_chars)
            keyed = {f"group-{index:05d}": list(group) for index, group in enumerate(groups)}
            outcome = collect_batch(client, batch_id, keyed, destination, poll_seconds)
        counts.update(outcome)

    logger.info(
        "judged %d, failed %d, already done %d, of %d documents in %s",
        counts["judged"],
        counts["failed"],
        counts["skipped"],
        counts["documents"],
        destination,
    )
    return counts
