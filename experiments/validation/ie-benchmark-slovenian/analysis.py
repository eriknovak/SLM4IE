r"""Table the four Slovene NER test sets and score the round's models on them.

`data` reads the validation and test splits of the four NER task registry
entries the record's `## Datasets` names (D9) and writes, per test set, the
summary, the fields and the statistics tables behind its row: documents and
characters per split, and spans, their share and their mean length per split
and label.

`scores` reads the finished tracked runs of the MLflow experiment, rescores
each run from its answers artifact without calling a model (M5, M3), and writes:

- the reference table: every model on every split, strict F1 with its interval
  beside the secondary metrics, tokens, cost and speed (Q1);
- the picks: each model's mean validation strict F1 and worst validation miss
  rate, and the best admitted model of each arm (D1, D6);
- C1: the picked local model minus the picked Claude model on each test set,
  with its paired interval, its reading against the margin (D2), and the
  verdict the record's rule gives;
- the memory table and the figure of mean test strict F1 against peak GPU
  memory: the weights and KV-cache memory per token of each model's vllm memory
  run, with the KV cache of the longest sequence its answers hold (M5, D7).

Run it with:

    uv run --group analysis --group evaluation python experiments/validation/ie-benchmark-slovenian/analysis.py data
    ~/.claude/skills/exp/scripts/tracking-env.sh -- uv run --group analysis --group evaluation \
        python experiments/validation/ie-benchmark-slovenian/analysis.py scores
"""

import argparse
import csv
import gzip
import json
import logging
import os
import statistics
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datachart.charts import ScatterChart
from datachart.constants import FIG_SIZE, SHOW_GRID
from mlflow.tracking import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from report_figures import save_figure  # noqa: E402  — path set above so the shared helper resolves

from slm4ie.data.tasks.config import load_tasks_config  # noqa: E402
from slm4ie.evaluation.memory import VllmMemory  # noqa: E402
from slm4ie.evaluation.run import SplitScore, score_split, split_metrics  # noqa: E402
from slm4ie.evaluation.scoring import Interval, paired_difference_interval  # noqa: E402
from slm4ie.utils.config import load_yaml  # noqa: E402

# ===============================
# Logging
# ===============================

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ===============================
# Defaults
# ===============================

HERE = Path(__file__).resolve().parent
TABLES_DIR = HERE / "tables"
FIGURES_DIR = HERE / "figures"
TASKS_DIR = HERE.parents[2] / "data" / "tasks"
MODELS_CONFIG = HERE / "configs" / "benchmark-models.yaml"
TASKS_CONFIG = HERE.parents[2] / "configs" / "data" / "tasks.yaml"
SPLITS = ("val", "test")

EXPERIMENT = "slm4ie/validation/ie-benchmark-slovenian"
# the label example variant C1 reads; `common` runs only measure the hint (D4)
CLAUSE_EXAMPLES = "unseen"
# the full runs log no window budget
NO_BUDGET = "None"
# strict F1 points within which local and Claude read as equivalent (D2)
MARGIN = 0.05
# highest validation miss rate a model may have and still be its arm's best (D6)
MISS_RATE_CEILING = 0.10
BYTES_PER_GIB = 2**30
DIGITS = 4

# table name -> task registry entry, in the order the record lists them
TEST_SETS = {
    "ssj500k": "ner/ssj500k",
    "suk": "ner/suk",
    "janes-tag": "ner/janes_tag",
    "slavicner": "ner/slavicner",
}


# ===============================
# Helper functions
# ===============================


def read_split(entry: str, split: str) -> list[dict[str, Any]]:
    """Read one split of a task registry entry.

    Args:
        entry: Task registry key, e.g. `ner/suk`.
        split: Split name.

    Returns:
        The split's examples, each with `id`, `text` and `spans`.

    Raises:
        FileNotFoundError: If the split was not built.
    """
    path = TASKS_DIR / entry / f"{split}.jsonl.gz"
    if not path.exists():
        raise FileNotFoundError(f"split {path} is missing; build it with `prepare_datasets.py tasks {entry}`")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def count_nested(spans: list[dict[str, Any]]) -> int:
    """Count spans that start inside an earlier span.

    Args:
        spans: One document's spans.

    Returns:
        The number of spans starting before an earlier span has ended.
    """
    nested = 0
    furthest_end = -1
    for start, end in sorted((s["start"], s["end"]) for s in spans):
        nested += start < furthest_end
        furthest_end = max(furthest_end, end)
    return nested


def summary_rows(splits: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Build the summary table of one test set.

    Args:
        splits: Split name -> examples.

    Returns:
        `statistic`, `value`, `note` rows.
    """
    rows: list[dict[str, Any]] = []
    for split, examples in splits.items():
        lengths = [len(e["text"]) for e in examples]
        spans = [s for e in examples for s in e["spans"]]
        rows += [
            {
                "statistic": f"{split}_documents",
                "value": len(examples),
                "note": "held-out documents in this split (D11)",
            },
            {"statistic": f"{split}_spans", "value": len(spans), "note": "gold entity mentions"},
            {"statistic": f"{split}_median_document_chars", "value": statistics.median_low(lengths), "note": ""},
            {"statistic": f"{split}_max_document_chars", "value": max(lengths), "note": "longest prompt input"},
            {
                "statistic": f"{split}_documents_without_spans",
                "value": sum(1 for e in examples if not e["spans"]),
                "note": "a model scores only false positives here",
            },
            {
                "statistic": f"{split}_nested_spans",
                "value": sum(count_nested(e["spans"]) for e in examples),
                "note": "spans starting inside another span",
            },
        ]
    return rows


def fields_rows(splits: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Build the fields table of one test set.

    Args:
        splits: Split name -> examples.

    Returns:
        `field`, `type`, `distinct`, `values` rows.
    """
    examples = [e for split in splits.values() for e in split]
    labels = sorted({s["label"] for e in examples for s in e["spans"]})
    return [
        {"field": "id", "type": "string", "distinct": len({e["id"] for e in examples}), "values": "<source>:<doc_id>"},
        {"field": "text", "type": "string", "distinct": len({e["text"] for e in examples}), "values": "document text"},
        {"field": "spans.start", "type": "integer", "distinct": "", "values": "character offset"},
        {"field": "spans.end", "type": "integer", "distinct": "", "values": "character offset, exclusive"},
        {"field": "spans.label", "type": "string", "distinct": len(labels), "values": " ".join(labels)},
    ]


def statistics_rows(splits: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Build the per split and label statistics table of one test set.

    Args:
        splits: Split name -> examples.

    Returns:
        One row per split and label, most frequent label first within a split.
    """
    rows: list[dict[str, Any]] = []
    for split, examples in splits.items():
        spans = [s for e in examples for s in e["spans"]]
        counts = Counter(s["label"] for s in spans)
        for label, count in counts.most_common():
            lengths = [s["end"] - s["start"] for s in spans if s["label"] == label]
            rows.append(
                {
                    "split": split,
                    "label": label,
                    "spans": count,
                    "share_pct": 100 * count / len(spans),
                    "mean_span_chars": statistics.mean(lengths),
                    "documents_with_label": sum(1 for e in examples if any(s["label"] == label for s in e["spans"])),
                    "documents": len(examples),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows as a CSV whose header holds every row's keys, in first-seen order.

    Args:
        path: Destination file.
        rows: Rows; a key a row lacks is written empty.
    """
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


# ===============================
# Scores helpers
# ===============================


@dataclass(frozen=True)
class ScoredRun:
    """One finished tracked run, rescored from its answers.

    Attributes:
        run_id: The MLflow run id.
        model_key: The model's key in the model list.
        arm: `claude` or `local`.
        entry: Task registry entry, such as `ner/suk`.
        split: `val` or `test`.
        score: The rescored counts per document.
        metrics: The metrics `split_metrics` computes from the answers.
        longest_sequence_tokens: The most input plus output tokens of one window.
    """

    run_id: str
    model_key: str
    arm: str
    entry: str
    split: str
    score: SplitScore
    metrics: dict[str, float]
    longest_sequence_tokens: int


def model_index(config_path: Path) -> dict[tuple[str, str], tuple[str, str]]:
    """Map each (backend, backend model name) of the model list to its model key and arm.

    Args:
        config_path: The model list.

    Returns:
        `(backend, model)` -> `(model key, arm)`.
    """
    index = {}
    for key, spec in load_yaml(config_path)["models"].items():
        for backend in ("claude", "vllm", "transformers"):
            if backend in spec:
                index[(backend, str(spec[backend]["model"]))] = (key, spec["arm"])
    return index


def finished_tracked_runs(client: MlflowClient) -> list[Any]:
    """Return the finished tracked runs of the experiment, newest first.

    Args:
        client: An MLflow client on the tracking server.

    Returns:
        The runs.

    Raises:
        RuntimeError: If the experiment does not exist.
    """
    experiment = client.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment {EXPERIMENT!r} does not exist")
    return client.search_runs(
        [experiment.experiment_id],
        filter_string="tags.tier = 'tracked' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=10_000,
    )


def read_answers(client: MlflowClient, run_id: str) -> list[dict[str, Any]]:
    """Download a run's answers artifact and read its lines.

    Args:
        client: An MLflow client on the tracking server.
        run_id: The run.

    Returns:
        The answers, one per window.

    Raises:
        RuntimeError: If the run holds no single `.jsonl` artifact.
    """
    names = [a.path for a in client.list_artifacts(run_id) if a.path.endswith(".jsonl")]
    if len(names) != 1:
        raise RuntimeError(f"run {run_id} holds {len(names)} answers files, expected one: {names}")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(client.download_artifacts(run_id, names[0], tmp))
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]


def score_runs(client: MlflowClient, runs: list[Any], index: dict[tuple[str, str], tuple[str, str]]) -> list[ScoredRun]:
    """Rescore the newest full run of each model, entry and split from its answers.

    Args:
        client: An MLflow client on the tracking server.
        runs: Finished tracked runs, newest first.
        index: The model index from `model_index`.

    Returns:
        One scored run per model, entry and split.

    Raises:
        KeyError: If a run's backend and model are not in the model list.
    """
    labels = {
        f"{e.task}/{e.dataset}": [str(label) for label in e.labels or []]
        for e in load_tasks_config(TASKS_CONFIG).entries
    }
    documents: dict[tuple[str, str], list[dict[str, Any]]] = {}
    scored: dict[tuple[str, str, str], ScoredRun] = {}
    for run in runs:
        params, tags = run.data.params, run.data.tags
        if params.get("window_budget") != NO_BUDGET or tags.get("examples") != CLAUSE_EXAMPLES:
            continue
        backend_model = (params["backend"], params["model"])
        if backend_model not in index:
            raise KeyError(f"run {run.info.run_id} ran {backend_model}, which {MODELS_CONFIG.name} does not list")
        model_key, arm = index[backend_model]

        # keep the newest run of each model, entry and split
        cell = (model_key, params["entry"], params["split"])
        if cell in scored:
            logger.info("run=%s skipped, run=%s is newer for %s", run.info.run_id, scored[cell].run_id, cell)
            continue

        # rescore from the answers, against the gold of the same split
        split_key = (params["entry"], params["split"])
        if split_key not in documents:
            documents[split_key] = read_split(*split_key)
        answers = read_answers(client, run.info.run_id)
        score = score_split(documents[split_key], answers, labels[params["entry"]])
        longest = max((a["input_tokens"] + a["output_tokens"] for a in answers), default=0)
        scored[cell] = ScoredRun(
            run.info.run_id,
            model_key,
            arm,
            params["entry"],
            params["split"],
            score,
            split_metrics(answers, score),
            longest,
        )
    return list(scored.values())


def peak_memory_gib(runs: list[Any], scored: list[ScoredRun]) -> dict[str, float]:
    """Compute each local model's peak GPU memory from its newest vllm memory run.

    A memory run logs the weights and the KV-cache memory per token that the
    vllm-service reported starting the model. The peak adds the KV cache of the
    longest sequence, input plus output tokens, of the model's scored windows.

    Args:
        runs: Finished tracked runs, newest first.
        scored: The scored runs.

    Returns:
        Model key -> peak GPU memory in GiB, for each model with a memory run and a scored window.
    """
    longest: dict[str, int] = {}
    for run in scored:
        longest[run.model_key] = max(longest.get(run.model_key, 0), run.longest_sequence_tokens)
    memory = {}
    for run in runs:
        metrics, tags = run.data.metrics, run.data.tags
        model_key = tags.get("model_key")
        if "weight_bytes" not in metrics or model_key not in longest or model_key in memory:
            continue
        reading = VllmMemory(int(metrics["weight_bytes"]), metrics["kv_bytes_per_token"])
        memory[model_key] = reading.peak_bytes(longest[model_key]) / BYTES_PER_GIB
    return memory


def reference_rows(scored: list[ScoredRun]) -> list[dict[str, Any]]:
    """Build the reference table: every model on every split (Q1).

    Args:
        scored: The scored runs.

    Returns:
        One row per model, entry and split, ordered by arm, model, entry and split.
    """
    keys = (
        "strict_f1", "strict_f1_low", "strict_f1_high", "relaxed_f1", "partial_f1", "macro_f1", "miss_rate",
        "documents", "windows", "input_tokens", "cache_write_tokens", "cache_read_tokens", "output_tokens",
        "cost_usd", "documents_per_second",
    )  # fmt: skip
    rows = []
    for run in sorted(scored, key=lambda r: (r.arm, r.model_key, r.entry, r.split)):
        row = {"arm": run.arm, "model": run.model_key, "entry": run.entry, "split": run.split}
        row.update({key: _rounded(run.metrics.get(key)) for key in keys})
        row["run_id"] = run.run_id
        rows.append(row)
    return rows


def pick_rows(scored: list[ScoredRun], entries: list[str]) -> list[dict[str, Any]]:
    """Rank each arm's models on validation and mark the best admitted one (D1, D6).

    A model is admitted when it has a validation run on every entry and its miss
    rate is at most `MISS_RATE_CEILING` on each. The best admitted model of an
    arm has the highest strict F1 averaged over the validation splits.

    Args:
        scored: The scored runs.
        entries: The task registry entries every model must cover.

    Returns:
        One row per model: arm, mean validation strict F1, worst validation miss rate, admitted, picked.
    """
    by_model: dict[tuple[str, str], dict[str, ScoredRun]] = {}
    for run in scored:
        if run.split == "val":
            by_model.setdefault((run.arm, run.model_key), {})[run.entry] = run

    rows = []
    for (arm, model_key), runs in sorted(by_model.items()):
        complete = set(runs) == set(entries)
        worst_miss = max(r.metrics["miss_rate"] for r in runs.values())
        rows.append(
            {
                "arm": arm,
                "model": model_key,
                "val_entries": len(runs),
                "val_strict_f1_mean": statistics.mean(r.metrics["strict_f1"] for r in runs.values()),
                "val_miss_rate_max": worst_miss,
                "admitted": complete and worst_miss <= MISS_RATE_CEILING,
            }
        )

    # mark the best admitted model of each arm
    for arm in {row["arm"] for row in rows}:
        admitted = [row for row in rows if row["arm"] == arm and row["admitted"]]
        best = max(admitted, key=lambda row: row["val_strict_f1_mean"], default=None)
        for row in rows:
            if row["arm"] == arm:
                row["picked"] = row is best
    return [{**row, **{k: _rounded(row[k]) for k in ("val_strict_f1_mean", "val_miss_rate_max")}} for row in rows]


def read_clause(interval: Interval, margin: float = MARGIN) -> str:
    """Read C1 from the interval of local minus Claude strict F1.

    Args:
        interval: The paired interval of the difference.
        margin: The equivalence margin.

    Returns:
        `better` above +margin, `worse` below -margin, `equivalent` inside it, else `undecided`.
    """
    if interval.low > margin:
        return "better"
    if interval.high < -margin:
        return "worse"
    if -margin <= interval.low and interval.high <= margin:
        return "equivalent"
    return "undecided"


def verdict(readings: list[str]) -> str:
    """Apply the record's verdict rule to the C1 readings of the four test sets.

    Args:
        readings: One C1 reading per test set.

    Returns:
        `adopted`, `inconclusive`, `rejected` or `partial`; `incomplete` while a test set has no reading.
    """
    if len(readings) != len(TEST_SETS) or "missing" in readings:
        return "incomplete"
    holds = sum(1 for r in readings if r in ("better", "equivalent"))
    worse = readings.count("worse")
    if holds == len(readings):
        return "adopted"
    if worse == 0:
        return "inconclusive"
    if holds == 0:
        return "rejected"
    return "partial"


def c1_rows(scored: list[ScoredRun], picks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare the picked local model with the picked Claude model on each test set (C1).

    Args:
        scored: The scored runs.
        picks: The rows of `pick_rows`.

    Returns:
        One row per test set with both F1 values, the paired difference and its reading,
        then a `verdict` row.
    """
    picked = {row["arm"]: row["model"] for row in picks if row.get("picked")}
    tests = {(r.model_key, r.entry): r for r in scored if r.split == "test"}

    rows, readings = [], []
    for entry in TEST_SETS.values():
        local = tests.get((picked.get("local"), entry))
        claude = tests.get((picked.get("claude"), entry))
        if local is None or claude is None:
            rows.append(
                {"entry": entry, "local": picked.get("local"), "claude": picked.get("claude"), "reading": "missing"}
            )
            readings.append("missing")
            continue

        # pair the two models on the same test documents
        difference = paired_difference_interval(
            {doc: s.strict for doc, s in local.score.documents.items()},
            {doc: s.strict for doc, s in claude.score.documents.items()},
        )
        reading = read_clause(difference)
        readings.append(reading)
        rows.append(
            {
                "entry": entry,
                "local": local.model_key,
                "claude": claude.model_key,
                "local_strict_f1": _rounded(local.metrics["strict_f1"]),
                "claude_strict_f1": _rounded(claude.metrics["strict_f1"]),
                "difference": _rounded(difference.estimate),
                "difference_low": _rounded(difference.low),
                "difference_high": _rounded(difference.high),
                "reading": reading,
            }
        )
    rows.append({"entry": "verdict", "reading": verdict(readings)})
    return rows


def memory_rows(scored: list[ScoredRun], memory: dict[str, float]) -> list[dict[str, Any]]:
    """Pair each local model's peak GPU memory with its mean test strict F1.

    Args:
        scored: The scored runs.
        memory: Model key -> peak GPU memory in GiB.

    Returns:
        One row per local model with a memory reading, lowest memory first.
    """
    rows = []
    for model_key, gib in sorted(memory.items(), key=lambda item: item[1]):
        tests = [r.metrics["strict_f1"] for r in scored if r.model_key == model_key and r.split == "test"]
        rows.append(
            {
                "model": model_key,
                "peak_gpu_memory_gib": _rounded(gib),
                "test_entries": len(tests),
                "test_strict_f1_mean": _rounded(statistics.mean(tests)) if tests else None,
            }
        )
    return rows


def memory_figure(rows: list[dict[str, Any]], claude_f1: float | None) -> Any:
    """Draw each local model's mean test strict F1 against its peak GPU memory.

    Args:
        rows: The rows of `memory_rows` that carry a test F1.
        claude_f1: The picked Claude model's mean test strict F1, drawn as a line; none when absent.

    Returns:
        The datachart figure.
    """
    points = [{"x": r["peak_gpu_memory_gib"], "y": r["test_strict_f1_mean"], "label": r["model"]} for r in rows]
    return ScatterChart(
        points,
        annotation="label",
        hlines={"y": claude_f1, "label": "best Claude model"} if claude_f1 is not None else None,
        title="Strict span F1 against peak GPU memory",
        xlabel="Peak GPU memory: weights and the longest sequence's KV cache (GiB)",
        ylabel="Strict span F1, mean of four test sets",
        figsize=FIG_SIZE.FULL_MEDIUM,
        show_grid=SHOW_GRID.BOTH,
        show_legend=claude_f1 is not None,
    )


def _rounded(value: Any) -> Any:
    """Round a float to `DIGITS` places; pass anything else through."""
    return round(value, DIGITS) if isinstance(value, float) else value


# ===============================
# Main function
# ===============================


def build_data() -> None:
    """Write the summary, fields and statistics tables of every test set."""
    for name, entry in TEST_SETS.items():
        splits = {split: read_split(entry, split) for split in SPLITS}
        write_csv(TABLES_DIR / f"dataset-{name}-summary.csv", summary_rows(splits))
        write_csv(TABLES_DIR / f"dataset-{name}-fields.csv", fields_rows(splits))
        write_csv(TABLES_DIR / f"dataset-{name}-statistics.csv", statistics_rows(splits))


def build_scores() -> None:
    """Rescore the tracked runs and write the reference, picks, C1 and memory tables and the figure.

    Raises:
        RuntimeError: If `MLFLOW_TRACKING_URI` is unset, or no full tracked run exists yet.
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        raise RuntimeError("MLFLOW_TRACKING_URI is unset: run `scores` through tracking-env.sh")

    # rescore every full tracked run from its answers
    client = MlflowClient(uri)
    runs = finished_tracked_runs(client)
    index = model_index(MODELS_CONFIG)
    scored = score_runs(client, runs, index)
    if not scored:
        raise RuntimeError(f"no finished tracked run with examples={CLAUSE_EXAMPLES} in {EXPERIMENT}")

    # write the reference table, the picks and the C1 readings
    picks = pick_rows(scored, list(TEST_SETS.values()))
    c1 = c1_rows(scored, picks)
    write_csv(TABLES_DIR / "benchmark-models-reference.csv", reference_rows(scored))
    write_csv(TABLES_DIR / "benchmark-models-picks.csv", picks)
    write_csv(TABLES_DIR / "benchmark-models-c1.csv", c1)
    logger.info("C1 verdict=%s", c1[-1]["reading"])

    # write the memory table and draw F1 against memory
    memory = memory_rows(scored, peak_memory_gib(runs, scored))
    if memory:
        write_csv(TABLES_DIR / "benchmark-models-memory.csv", memory)
        claude = next((row["model"] for row in picks if row["arm"] == "claude" and row.get("picked")), None)
        claude_tests = [r.metrics["strict_f1"] for r in scored if r.model_key == claude and r.split == "test"]
        # draw only models scored on all four test sets, as the y label says
        drawn = [row for row in memory if row["test_entries"] == len(TEST_SETS)]
        if drawn:
            FIGURES_DIR.mkdir(parents=True, exist_ok=True)
            save_figure(
                lambda: memory_figure(drawn, statistics.mean(claude_tests) if claude_tests else None),
                FIGURES_DIR / "benchmark-models-strict-f1-by-memory.svg",
            )


def get_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Table the test sets and score the models of ie-benchmark-slovenian")
    parser.add_argument(
        "command",
        choices=["data", "scores"],
        help="data: write the dataset statistics tables; scores: rescore the tracked runs into tables and a figure",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Run the chosen command.

    Args:
        args: Parsed command-line arguments.
    """
    if args.command == "data":
        build_data()
    elif args.command == "scores":
        build_scores()


if __name__ == "__main__":
    main(get_args())
