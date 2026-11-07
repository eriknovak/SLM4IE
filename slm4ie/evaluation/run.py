"""Run one model over one split of a NER task entry: answer every window, keep the answers, score, log.

A run reads the split's documents, cuts each into sentence windows and asks the
model once per window (`slm4ie.evaluation.prompts`). Every answer is appended to
an answers file the moment it returns, one JSON line per window holding the
answer text, the window's offsets, the token counts and the wall time, so an
interrupted run resumes where it stopped and `analysis.py` rescores a run
without calling the model again.

The answers are then scored: each window's mentions are aligned to spans of the
window text and shifted by the window's start into document offsets
(`slm4ie.evaluation.scoring`). The run logs one MLflow run with the scores, the
miss rate, the tokens, the speed and, for a model in this process, the peak GPU
memory; the answers file is its artifact.
"""

import json
import logging
import re
import subprocess
import tempfile
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

from tqdm import tqdm

from slm4ie.data.tasks.config import load_tasks_config, resolve_output_dir
from slm4ie.data.tasks.splits import iter_jsonl, outputs_for_splits
from slm4ie.evaluation.backends import MAX_NEW_TOKENS, ChatBackend, get_backend
from slm4ie.evaluation.prompts import (
    DEFAULT_EXAMPLES,
    MAX_WINDOW_CHARS,
    SYSTEM_PROMPT,
    USER_PROMPT,
    Window,
    answer_schema,
    build_messages,
    load_label_descriptions,
    parse_answer,
    sentence_windows,
)
from slm4ie.evaluation.scoring import Counts, DocumentScore, align_mentions, f1_interval, macro_f1, score_document
from slm4ie.utils import mlflow as ml
from slm4ie.utils.config import load_yaml
from slm4ie.utils.versioning import config_hash, corpus_digest

# ===============================
# Logging
# ===============================

logger = logging.getLogger(__name__)

# ===============================
# Defaults
# ===============================

TIMEOUT_ERROR = "timeout"

# characters a file name keeps from a model name; the rest become dashes
UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")

# ===============================
# Run settings
# ===============================


@dataclass(frozen=True)
class BenchmarkRun:
    """The settings of one run: one model on one split of one task registry entry.

    Attributes:
        backend: Chat backend name, one of `available_backends()`.
        model: The model in the backend's own naming.
        entry: Task registry entry key, such as `ner/suk`.
        split: Split name, such as `val` or `test`.
        tasks_config: Path to the task registry, `configs/data/tasks.yaml`.
        labels_config: Path to the label descriptions, `configs/evaluation/ner-labels.yaml`.
        answers_dir: Directory the answers file is written to.
        examples: Label example variant, `unseen` or `common`.
        backend_options: Backend-specific settings passed to `get_backend`.
        window_budget: Stop after the first document that brings the window count to this; all when `None`.
        resume: Keep the answers already in the answers file and ask only for the missing windows.
        force: Discard an existing answers file and start over.
        experiment: MLflow experiment name; no tracking when `None`.
        tags: Extra MLflow tags, such as `tier`, `branch`, `base_commit` and `type`.
    """

    backend: str
    model: str
    entry: str
    split: str
    tasks_config: Path
    labels_config: Path
    answers_dir: Path
    examples: str = DEFAULT_EXAMPLES
    backend_options: Mapping[str, Any] = field(default_factory=dict)
    window_budget: int | None = None
    resume: bool = False
    force: bool = False
    experiment: str | None = None
    tags: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SplitScore:
    """The scores of one run's answers on one split.

    Attributes:
        documents: Document id to its counts, for every document the run answered.
        windows: Windows answered.
        missed: Windows whose answer failed to parse or timed out.
        unaligned: Returned mentions with no free occurrence in their window.
    """

    documents: dict[str, DocumentScore]
    windows: int
    missed: int
    unaligned: int


# ===============================
# Entry point
# ===============================


def run_benchmark(run: BenchmarkRun) -> dict[str, float]:
    """Answer, score and log one model on one split.

    Args:
        run: The run settings.

    Returns:
        The metrics logged for the run.

    Raises:
        KeyError: If the entry is not in the task registry or has no such split.
        FileExistsError: If the answers file exists and neither `resume` nor `force` is set.
    """
    # load the split and its label descriptions
    tasks = load_tasks_config(run.tasks_config)
    entries = {f"{e.task}/{e.dataset}": e for e in tasks.entries}
    if run.entry not in entries:
        raise KeyError(f"entry '{run.entry}' is not in {run.tasks_config}")
    entry = entries[run.entry]
    output_dir = resolve_output_dir(entry, tasks.roots)
    split_paths = outputs_for_splits(output_dir, entry.splits)
    if run.split not in split_paths:
        raise KeyError(f"entry '{run.entry}' has no split '{run.split}'; it has {sorted(split_paths)}")
    documents = list(iter_jsonl(split_paths[run.split]))
    labels = [str(label) for label in entry.labels or []]
    descriptions = load_label_descriptions(run.labels_config, run.entry, run.examples)

    # prepare the answers file
    answers_path = Path(run.answers_dir) / answers_file_name(run, descriptions)
    if answers_path.exists() and not (run.resume or run.force):
        raise FileExistsError(f"answers file {answers_path} exists; pass resume to continue it or force to restart")
    if answers_path.exists() and run.force:
        answers_path.unlink()

    backend = get_backend(run.backend, run.model, **run.backend_options)
    tracking = run.experiment is not None and ml.ensure_experiment(run.experiment)
    tags = {
        "language": entry.language,
        "entry": run.entry,
        "split": run.split,
        "backend": run.backend,
        "model": run.model,
        "examples": run.examples,
        **run.tags,
    }
    run_name = f"{run.model}-{entry.dataset}-{run.split}"
    with ml.mlflow_run(run_name, enabled=tracking, tags=tags):
        ml.log_params(_run_params(run, backend), enabled=tracking)
        ml.log_dataset_input(
            f"tasks/{run.entry}", corpus_digest(output_dir), str(output_dir), context="eval", enabled=tracking
        )

        # answer every window, then score the answers
        backend.reset_peak_gpu_memory()
        answers, reused = answer_split(backend, documents, descriptions, labels, answers_path, run.window_budget)
        peak = backend.peak_gpu_memory_bytes()
        score = score_split(documents, answers, labels)
        metrics = split_metrics(answers, score)
        metrics["reused_windows"] = float(reused)
        # a resumed run generated only part of the windows in this process
        if peak is not None and reused == 0:
            metrics["peak_gpu_memory_bytes"] = float(peak)

        # log the metrics, the prompt and the answers
        ml.log_metrics(metrics, enabled=tracking)
        with tempfile.TemporaryDirectory() as tmp:
            prompt_path = Path(tmp) / "prompt.json"
            prompt = {
                "system": SYSTEM_PROMPT,
                "user_template": USER_PROMPT,
                "descriptions": descriptions,
                "schema": answer_schema(labels),
            }
            prompt_path.write_text(json.dumps(prompt, ensure_ascii=False, indent=2), encoding="utf8")
            ml.log_artifact(prompt_path, enabled=tracking)
        ml.log_artifact(answers_path, enabled=tracking)

    logger.info(
        "run=%s documents=%d windows=%d strict_f1=%.4f miss_rate=%.4f",
        run_name,
        len(score.documents),
        score.windows,
        metrics["strict_f1"],
        metrics["miss_rate"],
    )
    return metrics


def load_model_spec(config_path: Path, key: str, backend: str) -> tuple[str, dict[str, Any]]:
    """Read one model's name and options for one backend from the model list.

    Args:
        config_path: Path to the model list, such as the experiment's `configs/benchmark-models.yaml`.
        key: The model key under `models`.
        backend: The backend whose block to read, such as `vllm` or `transformers`.

    Returns:
        The model in that backend's naming and its backend options.

    Raises:
        KeyError: If the list has no such model, or the model has no block for `backend`.
    """
    models = load_yaml(config_path)["models"]
    if key not in models:
        raise KeyError(f"model '{key}' is not in {config_path}; it lists {sorted(models)}")
    if backend not in models[key]:
        raise KeyError(f"model '{key}' of {config_path} has no '{backend}' block")
    block = models[key][backend]
    return str(block["model"]), dict(block.get("options") or {})


def answers_file_name(run: BenchmarkRun, descriptions: Mapping[str, str]) -> str:
    """Name the answers file of a run by the settings that change the answers.

    The prompt hash covers the prompt templates, the label descriptions, the window
    size and the answer cap, so a resumed run never reuses answers to another prompt.

    Args:
        run: The run settings.
        descriptions: The label descriptions the prompt carries.

    Returns:
        `<backend>-<model>-<task>-<dataset>-<split>-<examples>-<prompt hash>[-<options hash>].jsonl`,
        the model made file-safe.
    """
    task, dataset = run.entry.split("/", 1)
    prompt = {
        "system": SYSTEM_PROMPT,
        "user": USER_PROMPT,
        "descriptions": dict(descriptions),
        "max_window_chars": MAX_WINDOW_CHARS,
        "max_new_tokens": MAX_NEW_TOKENS,
    }
    parts = [run.backend, UNSAFE_NAME_CHARS.sub("-", run.model), task, dataset, run.split, run.examples]
    parts.append(config_hash(prompt)[-8:])
    if run.backend_options:
        parts.append(config_hash(dict(run.backend_options))[-8:])
    return "-".join(parts) + ".jsonl"


# ===============================
# Answering
# ===============================


def answer_split(
    backend: ChatBackend,
    documents: Sequence[Mapping[str, Any]],
    descriptions: Mapping[str, str],
    labels: Sequence[str],
    answers_path: Path,
    window_budget: int | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Ask the model once per window of every document and append each answer to the answers file.

    Windows already in the answers file are kept, not asked again. The first
    window asked in this call is asked twice and the first reply discarded, so
    model loading and warm-up stay out of the timed answers. A call that times
    out is kept as a missed window with the error `timeout`.

    Args:
        backend: The model to ask.
        documents: The split's documents, each with `id` and `text`.
        descriptions: Label to description, as `load_label_descriptions` returns it.
        labels: The entry's labels, for the answer schema.
        answers_path: The JSON-lines file answers are read from and appended to.
        window_budget: Stop after the first document that brings the window count to this; all when `None`.

    Returns:
        The answers in document and window order, and how many were read from the file.
    """
    schema = answer_schema(labels)
    windows = _budget_windows(documents, window_budget)
    existing = {(a["id"], a["window"]): a for a in _read_answers(answers_path)}
    total = sum(len(doc_windows) for _, doc_windows in windows)

    answers, reused, warmed_up = [], 0, False
    answers_path.parent.mkdir(parents=True, exist_ok=True)
    with answers_path.open("a", encoding="utf8") as out, tqdm(total=total, desc="Answering windows") as bar:
        for doc_id, doc_windows in windows:
            for index, window in enumerate(doc_windows):
                # keep an answer the file already holds
                if (doc_id, index) in existing:
                    answers.append(existing[(doc_id, index)])
                    reused += 1
                    bar.update()
                    continue

                messages = build_messages(window.text, descriptions)
                if not warmed_up:
                    _ask(backend, messages, schema)
                    warmed_up = True

                # ask, then append the answer at once so a crash loses no answer
                answer = {"id": doc_id, "window": index, "start": window.start, "end": window.start + len(window.text)}
                answer.update(_ask(backend, messages, schema))
                out.write(json.dumps(answer, ensure_ascii=False) + "\n")
                out.flush()
                answers.append(answer)
                bar.update()
    return answers, reused


# ===============================
# Scoring
# ===============================


def score_split(
    documents: Iterable[Mapping[str, Any]],
    answers: Iterable[Mapping[str, Any]],
    labels: Sequence[str],
) -> SplitScore:
    """Score a run's answers against the gold spans of the documents it answered.

    Each window's answer is parsed and its mentions aligned to the window text;
    the spans shift by the window's start into document offsets. A missed
    window, off the schema or timed out, gives no mentions. A document with no
    answer is left out.

    Args:
        documents: The split's documents, each with `id`, `text` and gold `spans`.
        answers: Answers as `answer_split` writes them, each with `id`, `start`, `end` and `answer`.
        labels: The entry's labels; an answer with any other label is a miss.

    Returns:
        The counts per answered document and the window, miss and unaligned totals.
    """
    by_document = defaultdict(list)
    for answer in answers:
        by_document[answer["id"]].append(answer)

    scores, windows, missed, unaligned_total = {}, 0, 0, 0
    for document in documents:
        doc_answers = by_document.get(document["id"])
        if not doc_answers:
            continue

        # align each window's mentions and shift them into document offsets
        predicted, unaligned = [], []
        for answer in doc_answers:
            windows += 1
            mentions = None if answer.get("error") else parse_answer(answer["answer"], labels)
            if mentions is None:
                missed += 1
                continue
            alignment = align_mentions(document["text"][answer["start"] : answer["end"]], mentions)
            predicted += [
                {**s, "start": s["start"] + answer["start"], "end": s["end"] + answer["start"]} for s in alignment.spans
            ]
            unaligned += alignment.unaligned

        scores[document["id"]] = score_document(predicted, document["spans"], unaligned)
        unaligned_total += len(unaligned)
    return SplitScore(documents=scores, windows=windows, missed=missed, unaligned=unaligned_total)


def split_metrics(answers: Sequence[Mapping[str, Any]], score: SplitScore) -> dict[str, float]:
    """Turn a run's answers and scores into the metrics it logs.

    Args:
        answers: Answers as `answer_split` writes them.
        score: Their scores, as `score_split` returns them.

    Returns:
        F1 under the three match rules with the strict interval, strict F1 per label, macro F1,
        the miss rate, token counts, list-price cost where reported, and speed.
    """
    document_scores = list(score.documents.values())
    strict = sum((s.strict for s in document_scores), Counts())
    interval = f1_interval([s.strict for s in document_scores]) if document_scores else None
    metrics = {
        "documents": float(len(document_scores)),
        "windows": float(score.windows),
        "missed_windows": float(score.missed),
        "miss_rate": score.missed / score.windows if score.windows else 0.0,
        "timeouts": float(sum(1 for a in answers if a.get("error") == TIMEOUT_ERROR)),
        "unaligned_mentions": float(score.unaligned),
        "strict_f1": strict.f1,
        "strict_f1_low": interval.low if interval else 0.0,
        "strict_f1_high": interval.high if interval else 0.0,
        "strict_precision": strict.precision,
        "strict_recall": strict.recall,
        "relaxed_f1": sum((s.relaxed for s in document_scores), Counts()).f1,
        "partial_f1": sum((s.partial for s in document_scores), Counts()).f1,
        "macro_f1": macro_f1(document_scores),
    }

    # strict F1 per label
    by_label = defaultdict(Counts)
    for document_score in document_scores:
        for label, counts in document_score.by_label.items():
            by_label[label] += counts
    metrics.update({f"strict_f1_{label}": counts.f1 for label, counts in sorted(by_label.items())})

    # tokens and list-price cost
    for key in ("input_tokens", "output_tokens", "cache_write_tokens", "cache_read_tokens"):
        metrics[key] = float(sum(a[key] for a in answers))
    costs = [a["cost_usd"] for a in answers if a.get("cost_usd") is not None]
    if costs:
        metrics["cost_usd"] = float(sum(costs))

    # speed over the model's own time
    seconds = sum(a["seconds"] for a in answers)
    metrics["seconds"] = float(seconds)
    if answers:
        metrics["seconds_per_window_median"] = float(median(a["seconds"] for a in answers))
    if seconds > 0:
        metrics["windows_per_second"] = len(answers) / seconds
        metrics["documents_per_second"] = len(document_scores) / seconds
        metrics["output_tokens_per_second"] = metrics["output_tokens"] / seconds
    return metrics


# ===============================
# Helpers
# ===============================


def _budget_windows(
    documents: Sequence[Mapping[str, Any]], window_budget: int | None
) -> list[tuple[str, list[Window]]]:
    """Cut the documents into windows, keeping whole documents until the budget is reached."""
    windows, count = [], 0
    for document in documents:
        if window_budget is not None and count >= window_budget:
            break
        doc_windows = sentence_windows(document["text"])
        windows.append((document["id"], doc_windows))
        count += len(doc_windows)
    return windows


def _ask(backend: ChatBackend, messages: list[dict[str, str]], schema: dict[str, Any]) -> dict[str, Any]:
    """Ask one window and return the answer fields of its answers-file line."""
    start = time.perf_counter()
    try:
        result = backend.chat(messages, schema)
    except subprocess.TimeoutExpired:
        logger.warning("model=%s window timed out after %.0f s", backend.model, time.perf_counter() - start)
        return {
            "answer": "",
            "error": TIMEOUT_ERROR,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_write_tokens": 0,
            "cache_read_tokens": 0,
            "cost_usd": None,
            "seconds": time.perf_counter() - start,
        }
    return {
        "answer": result.text,
        "error": None,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "cache_write_tokens": result.cache_write_tokens,
        "cache_read_tokens": result.cache_read_tokens,
        "cost_usd": result.cost_usd,
        "seconds": result.seconds,
    }


def _read_answers(path: Path) -> list[dict[str, Any]]:
    """Read the answers file, or nothing when it does not exist yet."""
    if not path.exists():
        return []
    with path.open(encoding="utf8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _run_params(run: BenchmarkRun, backend: ChatBackend) -> dict[str, Any]:
    """Collect the run's MLflow parameters."""
    params = {
        "backend": run.backend,
        "model": run.model,
        "entry": run.entry,
        "split": run.split,
        "examples": run.examples,
        "window_budget": run.window_budget,
        # the Claude CLI takes no seed (D3)
        "seed": None if run.backend == "claude" else backend.seed,
        "max_window_chars": MAX_WINDOW_CHARS,
        "max_new_tokens": MAX_NEW_TOKENS,
    }
    params.update({f"option.{key}": value for key, value in run.backend_options.items()})
    return params
