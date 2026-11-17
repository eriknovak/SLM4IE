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
  run, with the KV cache of the longest sequence its answers hold (M5, D7);
- mean test strict F1 of every local model scored on all four test sets against
  its parameter count and served precision, and against its test throughput
  pooled over the four test runs, beside whether vllm ran it in eager mode (Q1);
  the Claude models' pooled throughput sits apart, as sequential CLI calls; a
  figure of each, the throughput figure drawing the Claude models as points;
- the record's standard tables: the C1 reading per test set and the two
  picked arms with the verdict's adoption;
- the tables read beside the clauses: strict F1 per label, Claude's
  list-price cost and tokens, strict F1 per SUK subcorpus, the local-minus-
  Claude difference on SlavicNER beside the download-gated sets, and the
  answers cut off at the token cap apart from the miss rate;
- C1 read again on only the windows both picked models answered in format, a
  sensitivity reading beside C1;
- the audit counts the critique checks: test documents a tuning split also
  holds, the thinnest cells, seeds per configuration and the runs read.

Every table goes through `save_table` and every figure through `save_figure`,
both from the shared `create_assets.py`, which check the rules of labflow's
`tables.md` and `figures.md` before writing.

Run it with:

    uv run --group analysis --group evaluation python experiments/validation/ie-benchmark-slovenian/analysis.py data
    ~/.claude/skills/exp/scripts/tracking-env.sh -- uv run --group analysis --group evaluation \
        python experiments/validation/ie-benchmark-slovenian/analysis.py scores
"""

import argparse
import gzip
import hashlib
import json
import logging
import os
import re
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datachart.charts import BarChart, ScatterChart
from datachart.constants import AXIS_SCALE, BAR_MODE, LEGEND_LOCATION, LINE_MARKER, LINE_STYLE, ORIENTATION, SHOW_GRID
from datachart.config import config
from mlflow.tracking import MlflowClient

from slm4ie.data.tasks.config import load_tasks_config
from slm4ie.evaluation.backends.base import MAX_NEW_TOKENS
from slm4ie.evaluation.memory import VllmMemory
from slm4ie.evaluation.prompts import parse_answer
from slm4ie.evaluation.run import SplitScore, score_split, split_metrics
from slm4ie.evaluation.scoring import Counts, Interval, f1_interval, paired_difference_interval
from slm4ie.utils.config import load_yaml

# find the shared report helpers in experiments/tools/, or in experiments/ itself
EXPERIMENTS = next(p for p in Path(__file__).resolve().parents if p.name == "experiments")
TOOLS = EXPERIMENTS / "tools"
sys.path.insert(0, str(TOOLS if (TOOLS / "create_assets.py").is_file() else EXPERIMENTS))
from create_assets import ONE_COLUMN, Claim, save_figure, save_table  # noqa: E402  — path set above

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
# the splits a model or a pick may have seen before a test split is read
TUNING_SPLITS = ("train", "val")

EXPERIMENT = "slm4ie/validation/ie-benchmark-slovenian"
BRANCH = "exp/ie-benchmark-slovenian"
# the commit round R1 was pre-registered at; every tracked run carries it
BASE_COMMIT = "ac2e506"
RUN_TYPES = ("benchmark", "memory")
# the label example variant C1 reads; `common` runs only measure the hint (D4)
CLAUSE_EXAMPLES = "unseen"
# the full runs log no window budget
NO_BUDGET = "None"
# strict F1 points within which local and Claude read as equivalent (D2)
MARGIN = 0.05
# highest validation miss rate a model may have and still be its arm's best (D6)
MISS_RATE_CEILING = 0.10
BYTES_PER_GIB = 2**30
# the runs of a model, entry and split C1 reads, each decoded once with no seed sweep (D3)
SEEDS_PER_CONFIGURATION = 1

# table name -> task registry entry, in the order the record lists them
TEST_SETS = {
    "ssj500k": "ner/ssj500k",
    "suk": "ner/suk",
    "janes-tag": "ner/janes_tag",
    "slavicner": "ner/slavicner",
}
# the one test set public on the open web; the others sit behind a CLARIN.SI download (D9)
PUBLIC_TEST_SET = "slavicner"
CLAUSE = "C1"
DIAGNOSTIC_CLAUSE = "Q1"
# a SUK document id names its subcorpus after the source prefix: `suk:senticoref12`
SUK_SUBCORPUS = re.compile(r"^suk:([a-z]+)")
DOCUMENTS_PER_COST = 1_000
# the arm whose backend takes no output cap; its output tokens include reasoning (M2)
UNCAPPED_ARM = "claude"
# the metrics that count documents, windows or tokens, written as integers
COUNT_METRICS = (
    "documents", "windows", "input_tokens", "cache_write_tokens", "cache_read_tokens", "output_tokens",
)  # fmt: skip
MEMORY_FIGURE = "benchmark-models-strict-f1-by-memory"
# each of these tables is also the data table of the figure with its stem
PARAMETERS_TABLE = "benchmark-models-strict-f1-by-parameters"
SPEED_TABLE = "benchmark-models-strict-f1-by-speed"
CLAUDE_SPEED_TABLE = "benchmark-models-claude-speed-by-model"
# the facts each local model of the model list carries, and the precisions it may be served at
LOCAL_MODEL_FIELDS = ("parameters_billion", "served_precision", "vllm_eager")
SERVED_PRECISIONS = ("bf16", "fp8", "4-bit")
# how the Claude arm answers: one headless Claude Code call per window, one after another (D8)
CLAUDE_SERVING = "sequential calls through the Claude Code CLI"
C1_FIGURE = "benchmark-models-c1-difference-by-test-set"
DATASET_FIGURE = "dataset-test-sets-spans-by-label"
# each test set as the record names it
TEST_SET_TITLES = {"ssj500k": "ssj500k", "suk": "SUK", "janes-tag": "Janes-Tag", "slavicner": "SlavicNER"}
# the C1 figure's axis runs in strict F1 and its ticks read in points of F1
C1_TICKS = (-0.15, -0.10, -0.05, 0.0, 0.05)
F1_POINTS = 100
# rows of the C1 figure: the download-gated sets from the top, a gap, then the public set
C1_GATED_TOP = 4.0
C1_PUBLIC_ROW = 0.5
# how far above its point, in rows, each C1 point's label starts
C1_LABEL_RISE = 0.2
# the typographic minus the figure labels print
MINUS = "−"
# the F1 axis of the Q1 scatter figures, as fractions, runs from here up to the ceiling of F1
F1_TICKS = (0.2, 0.4, 0.6, 0.8, 1.0)
# how far above the Claude line, in axes height, its name starts
REFERENCE_NAME_RISE = 0.025
# names of the muted models: a mid grey at 5.3:1 against white, over the muted markers
MUTED_NAME_COLOR = "#6B6B6B"
# the speed figure's log axis: test windows per second, from 0.027 to 1.37
SPEED_TICKS = (0.03, 0.1, 0.3, 1.0)
SPEED_LIMITS = (0.02, 2.0)
# the figures' background colour, matplotlib's white, as the theme sets none
PLATE_COLOR = "#FFFFFF"
# a label past the palette's colours is drawn hatched in ink on the plate colour instead
LABEL_HATCH = {"plot_bar_color": PLATE_COLOR, "plot_bar_hatch": "////"}
# the other labels' segments are edged in the plate colour, so a thin gap parts them
LABEL_GAP = {"plot_bar_edge_color": PLATE_COLOR}
# the picked Claude model's grey-blue: its reference line, and the Claude points of the speed figure
CLAUDE_COLOR = "#1F77B4"
# the speed figure's key to its squares, in the empty lower right of the plot
EAGER_NOTE = "squares: served in a slower mode to fit the GPU"


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
        {"field": "id", "type": "text", "distinct": len({e["id"] for e in examples}), "values": None},
        {"field": "text", "type": "text", "distinct": len({e["text"] for e in examples}), "values": None},
        {"field": "spans.start", "type": "numeric", "distinct": None, "values": None},
        {"field": "spans.end", "type": "numeric", "distinct": None, "values": None},
        {"field": "spans.label", "type": "categorical", "distinct": len(labels), "values": "; ".join(labels)},
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


def write_table(name: str, rows: list[dict[str, Any]]) -> None:
    """Write rows to `tables/<name>.csv` through `save_table`, the header holding every row's keys.

    Args:
        name: The table's name stem.
        rows: Rows; a key a row lacks is written empty.

    Raises:
        TableCheckError: If the table breaks a rule of tables.md; nothing is written.
    """
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    save_table(rows, TABLES_DIR / f"{name}.csv", columns=columns)


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
        cut_off_windows: Windows whose answer reached `MAX_NEW_TOKENS` output tokens.
        cut_off_misses: Cut-off windows that also count as missed.
    """

    run_id: str
    model_key: str
    arm: str
    entry: str
    split: str
    score: SplitScore
    metrics: dict[str, float]
    longest_sequence_tokens: int
    cut_off_windows: int
    cut_off_misses: int


@dataclass(frozen=True)
class LocalModel:
    """The facts the model list records about one local model.

    Attributes:
        parameters_billion: Total parameters stored in the weights, in billions.
        served_precision: The precision the vllm-service serves the weights at: `bf16`, `fp8` or `4-bit`.
        vllm_eager: Whether the vllm-service runs the model in eager mode with a 4,096-token context.
    """

    parameters_billion: float
    served_precision: str
    vllm_eager: bool


@dataclass(frozen=True)
class C1Reading:
    """C1 on one test set: the picked local model against the picked Claude model.

    Attributes:
        test_set: The test set's table name, such as `suk`.
        local_model: The picked local model's key; none when no model is picked.
        claude_model: The picked Claude model's key; none when no model is picked.
        local: The picked local model's test run; none when it is missing.
        claude: The picked Claude model's test run; none when it is missing.
        difference: Local minus Claude strict F1 with its paired interval; none when a run is missing.
        reading: `better`, `worse`, `equivalent`, `undecided`, or `missing`.
    """

    test_set: str
    local_model: str | None
    claude_model: str | None
    local: ScoredRun | None
    claude: ScoredRun | None
    difference: Interval | None
    reading: str


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
    """Return the finished tracked benchmark and memory runs of round R1, newest first.

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
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=(
            f"tags.tier = 'tracked' and tags.branch = '{BRANCH}' and tags.base_commit = '{BASE_COMMIT}' "
            "and attributes.status = 'FINISHED'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=10_000,
    )
    return [run for run in runs if run.data.tags.get("type") in RUN_TYPES]


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
    labels = entry_labels()
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
            *cut_off_counts(answers, labels[params["entry"]]),
        )
    return list(scored.values())


def entry_labels() -> dict[str, list[str]]:
    """Read each task registry entry's labels.

    Returns:
        Entry key, such as `ner/suk`, -> its labels.
    """
    return {
        f"{e.task}/{e.dataset}": [str(label) for label in e.labels or []]
        for e in load_tasks_config(TASKS_CONFIG).entries
    }


def cut_off_counts(answers: list[dict[str, Any]], labels: list[str]) -> tuple[int, int]:
    """Count the windows whose answer reached the token cap, and how many of them were missed.

    A miss is judged as `score_split` judges it: an error, or an answer off the schema.

    Args:
        answers: One run's answers, one per window.
        labels: The entry's labels.

    Returns:
        The cut-off windows and the cut-off windows that were missed.
    """
    cut_off = [a for a in answers if a["output_tokens"] >= MAX_NEW_TOKENS]
    missed = sum(1 for a in cut_off if not _is_answered(a, labels))
    return len(cut_off), missed


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
        row.update({key: _count(key, run.metrics.get(key)) for key in keys})
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
    return rows


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


def c1_readings(scored: list[ScoredRun], picks: list[dict[str, Any]]) -> list[C1Reading]:
    """Compare the picked local model with the picked Claude model on each test set (C1).

    Args:
        scored: The scored runs.
        picks: The rows of `pick_rows`.

    Returns:
        One reading per test set, in the order of `TEST_SETS`.
    """
    picked = {row["arm"]: row["model"] for row in picks if row.get("picked")}
    tests = {(r.model_key, r.entry): r for r in scored if r.split == "test"}

    readings = []
    for name, entry in TEST_SETS.items():
        local_key, claude_key = picked.get("local"), picked.get("claude")
        local = tests.get((local_key, entry)) if local_key else None
        claude = tests.get((claude_key, entry)) if claude_key else None
        if local is None or claude is None:
            readings.append(C1Reading(name, local_key, claude_key, local, claude, None, "missing"))
            continue

        # pair the two models on the same test documents
        difference = paired_difference_interval(
            {doc: s.strict for doc, s in local.score.documents.items()},
            {doc: s.strict for doc, s in claude.score.documents.items()},
        )
        readings.append(
            C1Reading(name, local.model_key, claude.model_key, local, claude, difference, read_clause(difference))
        )
    return readings


def c1_rows(readings: list[C1Reading]) -> list[dict[str, Any]]:
    """Build the C1 table: both F1 values, the paired difference and its reading per test set.

    Args:
        readings: The readings of `c1_readings`.

    Returns:
        One row per test set, then a `verdict` row.
    """
    rows = []
    for c1 in readings:
        row = {"entry": TEST_SETS[c1.test_set], "local": c1.local_model, "claude": c1.claude_model}
        if c1.difference is not None and c1.local and c1.claude:
            row |= {
                "local_strict_f1": c1.local.metrics["strict_f1"],
                "claude_strict_f1": c1.claude.metrics["strict_f1"],
                "difference": c1.difference.estimate,
                "difference_low": c1.difference.low,
                "difference_high": c1.difference.high,
            }
        rows.append(row | {"reading": c1.reading})
    rows.append({"entry": "verdict", "reading": verdict([c1.reading for c1 in readings])})
    return rows


def clause_rows(readings: list[C1Reading]) -> list[dict[str, Any]]:
    """Build the record's clause readings: one C1 row per test set with a reading.

    Q1 is measured, not tested, so it takes no row. A test set without both
    picked runs has no interval to report and is left out.

    Args:
        readings: The readings of `c1_readings`.

    Returns:
        Rows in the columns of the standard `clause-readings.csv`.
    """
    return [
        {
            "clause": CLAUSE,
            "qualifier": c1.test_set,
            "value": c1.difference.estimate,
            "ci_low": c1.difference.low,
            "ci_high": c1.difference.high,
            "bar": None,
            "margin": MARGIN,
            "outcome": c1.reading,
        }
        for c1 in readings
        if c1.difference is not None
    ]


def verdict_arm_rows(readings: list[C1Reading], picks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build the verdict table: the picked local model as the arm, the picked Claude model as the baseline.

    The local arm is adopted only when the verdict rule reads `adopted`.

    Args:
        readings: The readings of `c1_readings`.
        picks: The rows of `pick_rows`.

    Returns:
        One row per arm with its validation and per-test-set strict F1; the local row
        also counts the test sets per C1 outcome.
    """
    outcome = verdict([c1.reading for c1 in readings])
    rows = []
    for arm, role in (("local", "arm"), ("claude", "baseline")):
        pick = next((row for row in picks if row["arm"] == arm and row.get("picked")), {})
        runs = {c1.test_set: c1.local if arm == "local" else c1.claude for c1 in readings}
        row = {
            "arm": arm,
            "clause": CLAUSE,
            "role": role,
            "adopted": "yes" if arm == "local" and outcome == "adopted" else "no",
            "model": pick.get("model"),
            "val_strict_f1_mean": pick.get("val_strict_f1_mean"),
        }
        for name, run in runs.items():
            row[f"{name.replace('-', '_')}_strict_f1"] = run.metrics["strict_f1"] if run else None
        for word in ("better", "equivalent", "undecided", "worse"):
            row[f"test_sets_{word}"] = sum(1 for c1 in readings if c1.reading == word) if arm == "local" else None
        rows.append(row)
    return rows


def gap_rows(readings: list[C1Reading]) -> list[dict[str, Any]]:
    """Set the C1 difference on the public SlavicNER beside the download-gated test sets.

    Args:
        readings: The readings of `c1_readings`.

    Returns:
        One row per download-gated test set, their mean, then the public test set.
    """
    read = [c1 for c1 in readings if c1.difference is not None]
    gated = [c1 for c1 in read if c1.test_set != PUBLIC_TEST_SET]
    public = [c1 for c1 in read if c1.test_set == PUBLIC_TEST_SET]
    f1 = {
        c1.test_set: (c1.local.metrics["strict_f1"], c1.claude.metrics["strict_f1"])
        for c1 in read
        if c1.local and c1.claude
    }

    rows = [_gap_row(c1.test_set, "download-gated", *f1[c1.test_set], c1.difference) for c1 in gated]
    if gated:
        local_mean = statistics.mean(f1[c1.test_set][0] for c1 in gated)
        claude_mean = statistics.mean(f1[c1.test_set][1] for c1 in gated)
        # the mean of the gated differences has no paired interval of its own
        rows.append(_gap_row("download-gated mean", "download-gated", local_mean, claude_mean, None))
    rows += [_gap_row(c1.test_set, "public", *f1[c1.test_set], c1.difference) for c1 in public]
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
                "variant": model_key,
                "peak_gpu_memory_gib": gib,
                "test_entries": len(tests),
                "test_strict_f1_mean": statistics.mean(tests) if tests else None,
            }
        )
    return rows


def local_models(config_path: Path) -> dict[str, LocalModel]:
    """Read each local model's parameter count, served precision and eager mode from the model list.

    Args:
        config_path: The model list.

    Returns:
        Local model key -> its facts.

    Raises:
        KeyError: If a local model lacks one of `LOCAL_MODEL_FIELDS`.
        ValueError: If a local model's `served_precision` is not one of `SERVED_PRECISIONS`.
    """
    models = {}
    for key, spec in load_yaml(config_path)["models"].items():
        if spec["arm"] != "local":
            continue
        missing = [field for field in LOCAL_MODEL_FIELDS if field not in spec]
        if missing:
            raise KeyError(f"local model '{key}' of {config_path.name} has no {missing}")
        if spec["served_precision"] not in SERVED_PRECISIONS:
            raise ValueError(
                f"local model '{key}' of {config_path.name} has served_precision {spec['served_precision']!r}, "
                f"expected one of {SERVED_PRECISIONS}"
            )
        models[key] = LocalModel(float(spec["parameters_billion"]), spec["served_precision"], bool(spec["vllm_eager"]))
    return models


def full_test_models(scored: list[ScoredRun], arm: str) -> list[str]:
    """List an arm's models that have a test run on every test set.

    Args:
        scored: The scored runs.
        arm: `claude` or `local`.

    Returns:
        The model keys, sorted.
    """
    entries: dict[str, set[str]] = defaultdict(set)
    for run in scored:
        if run.arm == arm and run.split == "test":
            entries[run.model_key].add(run.entry)
    return sorted(key for key, seen in entries.items() if seen >= set(TEST_SETS.values()))


def mean_test_strict_f1(scored: list[ScoredRun], model_key: str) -> float:
    """Average a model's strict F1 over its test runs, each test set weighted equally.

    Args:
        scored: The scored runs.
        model_key: The model.

    Returns:
        The mean test strict F1.
    """
    return statistics.mean(r.metrics["strict_f1"] for r in scored if r.model_key == model_key and r.split == "test")


def pooled_test_speed(scored: list[ScoredRun], model_key: str) -> dict[str, Any]:
    """Pool a model's test runs into one throughput: total windows and documents over total seconds.

    The seconds are the summed wall time of the model's calls, one window at a time.

    Args:
        scored: The scored runs.
        model_key: The model.

    Returns:
        `windows`, `duration_s`, `windows_per_second` and `documents_per_second`.
    """
    tests = [r for r in scored if r.model_key == model_key and r.split == "test"]
    windows = sum(int(r.metrics["windows"]) for r in tests)
    documents = sum(int(r.metrics["documents"]) for r in tests)
    seconds = sum(r.metrics["seconds"] for r in tests)
    return {
        "windows": windows,
        "duration_s": seconds,
        "windows_per_second": windows / seconds if seconds else None,
        "documents_per_second": documents / seconds if seconds else None,
    }


def parameter_rows(scored: list[ScoredRun], models: dict[str, LocalModel]) -> list[dict[str, Any]]:
    """Pair each fully tested local model's parameter count and served precision with its mean test strict F1.

    Args:
        scored: The scored runs.
        models: The local models of `local_models`.

    Returns:
        One row per local model scored on every test set, fewest parameters first.
    """
    rows = [
        {
            "variant": key,
            "parameters_billion": models[key].parameters_billion,
            "served_precision": models[key].served_precision,
            "test_strict_f1_mean": mean_test_strict_f1(scored, key),
        }
        for key in full_test_models(scored, "local")
    ]
    return sorted(rows, key=lambda row: row["parameters_billion"])


def speed_rows(scored: list[ScoredRun], models: dict[str, LocalModel]) -> list[dict[str, Any]]:
    """Pair each fully tested local model's pooled test throughput with its mean test strict F1.

    Args:
        scored: The scored runs.
        models: The local models of `local_models`.

    Returns:
        One row per local model scored on every test set, slowest first.
    """
    rows = []
    for key in full_test_models(scored, "local"):
        speed = pooled_test_speed(scored, key)
        rows.append(
            {
                "variant": key,
                "windows_per_second": speed["windows_per_second"],
                "documents_per_second": speed["documents_per_second"],
                "vllm_eager_mode": "yes" if models[key].vllm_eager else "no",
                "test_strict_f1_mean": mean_test_strict_f1(scored, key),
                "windows": speed["windows"],
                "duration_s": speed["duration_s"],
            }
        )
    return sorted(rows, key=lambda row: row["windows_per_second"])


def claude_speed_rows(scored: list[ScoredRun]) -> list[dict[str, Any]]:
    """Pool each fully tested Claude model's test throughput, marked with how it was called.

    Args:
        scored: The scored runs.

    Returns:
        One row per Claude model scored on every test set, slowest first.
    """
    rows = []
    for key in full_test_models(scored, "claude"):
        speed = pooled_test_speed(scored, key)
        rows.append(
            {
                "variant": key,
                "windows_per_second": speed["windows_per_second"],
                "documents_per_second": speed["documents_per_second"],
                "serving": CLAUDE_SERVING,
                "windows": speed["windows"],
                "duration_s": speed["duration_s"],
            }
        )
    return sorted(rows, key=lambda row: row["windows_per_second"])


def _f1_point(row: dict[str, Any], x_key: str) -> dict[str, Any]:
    """Place one model of a Q1 scatter figure: its `x_key` value against its mean test strict F1."""
    return {"x": row[x_key], "y": row["test_strict_f1_mean"], "label": row["variant"]}


def _f1_axis() -> dict[str, Any]:
    """Give the chart keywords every Q1 scatter figure shares: the F1 axis in points, the size and the grid."""
    return {
        "ylabel": "Strict span F1, mean of four test sets (points)",
        "yticks": list(F1_TICKS),
        "yticklabels": [f"{F1_POINTS * t:.0f}" for t in F1_TICKS],
        "ymin": F1_TICKS[0],
        "ymax": F1_TICKS[-1],
        "figsize": ONE_COLUMN,
        "show_grid": SHOW_GRID.BOTH,
    }


def _grey_muted_names(figure: Any, names: set[str]) -> None:
    """Draw the names of muted points in a mid grey: datachart draws them in its marker grey, too light to read."""
    for text in (text for ax in figure.axes for text in ax.texts if text.get_text() in names):
        text.set_color(MUTED_NAME_COLOR)


def local_f1_figure(
    rows: list[dict[str, Any]],
    x_key: str,
    picked: str | None,
    claude_model: str | None,
    claude_f1: float | None,
    title: str,
    xlabel: str,
) -> Any:
    """Draw each local model's mean test strict F1 against one of its measures (Q1).

    The picked local model is its own highlighted series, the other local
    models a muted one with their names in a readable mid grey, and the
    picked Claude model a horizontal line named at its right end. The axis
    holds F1 as a fraction and its ticks read in points, as the C1 figure's do.

    Args:
        rows: One row per local model, with `model`, `test_strict_f1_mean` and `x_key`.
        x_key: The column drawn on the x-axis.
        picked: The picked local model's key; none when no local model is picked.
        claude_model: The picked Claude model's key, which names the line.
        claude_f1: The picked Claude model's mean test strict F1, drawn as a line; none when absent.
        title: The figure title.
        xlabel: The x-axis label, with its unit.

    Returns:
        The datachart figure.
    """
    points = {r["variant"]: _f1_point(r, x_key) for r in rows}
    series = [[points[picked]]] if picked in points else []
    subtitles, emphasis = ([picked], ["highlight"]) if series else ([], [])
    others = [point for model, point in points.items() if model != picked]
    if others:
        series.append(others)
        subtitles.append("other local models")
        emphasis.append("background")

    # name the Claude line at its right end, just above it, instead of in a legend
    low, high = F1_TICKS[0], F1_TICKS[-1]
    claude_name = None
    if claude_f1 is not None:
        claude_name = {
            "text": f"{claude_model or 'Claude'}, {F1_POINTS * claude_f1:.1f}",
            "x": 0.99,
            "y": (claude_f1 - low) / (high - low) + REFERENCE_NAME_RISE,
            "coords": "axes",
            "style": {"plot_text_halign": "right", "plot_text_valign": "bottom"},
        }

    figure = ScatterChart(
        series,
        subtitle=subtitles,
        emphasis=emphasis,
        annotation="label",
        hlines={"y": claude_f1, "style": {"plot_hline_color": CLAUDE_COLOR}} if claude_f1 is not None else None,
        texts=claude_name,
        title=title,
        xlabel=xlabel,
        show_legend=False,
        **_f1_axis(),
    )
    _grey_muted_names(figure, {point["label"] for point in others})
    return figure


def speed_figure(
    rows: list[dict[str, Any]],
    claude_rows: list[dict[str, Any]],
    picked: str | None,
    claude_model: str | None,
) -> Any:
    """Draw each model's mean test strict F1 against its pooled test throughput on a log axis (Q1).

    The picked local model is highlighted and the other local models muted,
    those vllm runs in eager mode as squares, which a note in the lower right
    explains. The Claude models are their own series of triangles in the
    grey-blue of the other Q1 figures' Claude line, each named, the picked
    one with its mean F1 as that line's name gives it.

    Args:
        rows: The rows of `speed_rows`.
        claude_rows: The rows of `claude_speed_rows`, each with its `test_strict_f1_mean`.
        picked: The picked local model's key; none when no local model is picked.
        claude_model: The picked Claude model's key; none when no Claude model is picked.

    Returns:
        The datachart figure.
    """
    points = {r["variant"]: _f1_point(r, "windows_per_second") for r in rows}
    eager = {r["variant"] for r in rows if r["vllm_eager_mode"] == "yes"}
    claude_points = [_f1_point(r, "windows_per_second") for r in claude_rows]
    for point in claude_points:
        if point["label"] == claude_model:
            point["label"] = f"{claude_model}, {F1_POINTS * point['y']:.1f}"
    groups = [
        ([points[picked]] if picked in points else [], picked, "highlight", {}),
        (claude_points, "Claude models, one command-line call per window", None,
         {"plot_scatter_marker": LINE_MARKER.TRIANGLE, "plot_scatter_color": CLAUDE_COLOR}),
        ([p for m, p in points.items() if m != picked and m not in eager], "other local models", "background", {}),
        ([p for m, p in points.items() if m != picked and m in eager], "local models, vllm eager mode",
         "background", {"plot_scatter_marker": LINE_MARKER.SQUARE}),
    ]  # fmt: skip
    groups = [group for group in groups if group[0]]

    # explain the squares in the empty lower right instead of in a legend
    note = None
    if eager:
        note = {
            "text": EAGER_NOTE,
            "x": 0.99,
            "y": 0.03,
            "coords": "axes",
            "style": {"plot_text_halign": "right", "plot_text_valign": "bottom"},
        }

    figure = ScatterChart(
        [group[0] for group in groups],
        subtitle=[group[1] for group in groups],
        emphasis=[group[2] for group in groups],
        style=[group[3] for group in groups],
        annotation="label",
        texts=note,
        title="Strict F1 by throughput",
        xlabel="Test throughput, one window at a time (windows per second, log scale)",
        scalex=AXIS_SCALE.LOG,
        xticks=list(SPEED_TICKS),
        xticklabels=[f"{t:g}" for t in SPEED_TICKS],
        xmin=SPEED_LIMITS[0],
        xmax=SPEED_LIMITS[1],
        show_legend=False,
        **_f1_axis(),
    )
    _grey_muted_names(figure, {p["label"] for m, p in points.items() if m != picked})
    return figure


def c1_figure_rows(readings: list[C1Reading]) -> list[dict[str, Any]]:
    """Build the data table of the C1 figure: the paired difference per test set and its access.

    Args:
        readings: The readings of `c1_readings`.

    Returns:
        One row per test set with a reading, in the order of `TEST_SETS`.
    """
    return [
        {
            "test_set": c1.test_set,
            "access": "public" if c1.test_set == PUBLIC_TEST_SET else "download-gated",
            "local": c1.local_model,
            "claude": c1.claude_model,
            "difference": c1.difference.estimate,
            "difference_low": c1.difference.low,
            "difference_high": c1.difference.high,
            "reading": c1.reading,
        }
        for c1 in readings
        if c1.difference is not None
    ]


def c1_figure(rows: list[dict[str, Any]], levels: dict[str, tuple[float, float]]) -> Any:
    """Draw local minus Claude strict F1 per test set against the equivalence margin (C1).

    Each test set is a point with its 95% paired-bootstrap interval. The
    download-gated sets sit on top, sorted by the difference; the public
    SlavicNER sits apart below them in its own colour. Both models' strict F1
    stand at the right end of each row, so the levels behind each difference
    read beside it.

    Args:
        rows: The rows of `c1_figure_rows`.
        levels: Test set -> the local and the Claude model's strict F1, from the C1 table.

    Returns:
        The datachart figure.
    """
    # every row is the same picked pair, so the first row names both models
    pair = f"{rows[0]['local']} minus {rows[0]['claude']}"
    gated = sorted((r for r in rows if r["access"] == "download-gated"), key=lambda r: -r["difference"])
    public = [r for r in rows if r["access"] == "public"]
    gated_rows = [(r, C1_GATED_TOP - i) for i, r in enumerate(gated)]
    public_rows = [(r, C1_PUBLIC_ROW) for r in public]
    placed = gated_rows + public_rows

    # each point carries its interval as distances below and above the estimate
    def points(series: list[tuple[dict[str, Any], float]]) -> list[dict[str, Any]]:
        return [
            {
                "x": r["difference"],
                "y": y,
                "xerr": [r["difference"] - r["difference_low"], r["difference_high"] - r["difference"]],
            }
            for r, y in series
        ]

    # each point carries its difference in points and its reading just above it
    readings = [
        {
            "text": f"{F1_POINTS * r['difference']:+.1f}, {r['reading']}".replace("-", MINUS),
            "x": r["difference"],
            "y": y + C1_LABEL_RISE,
            "style": {"plot_text_halign": "center", "plot_text_valign": "bottom"},
        }
        for r, y in placed
    ]

    margin_line = {"plot_vline_style": LINE_STYLE.DASHED, "plot_vline_alpha": 0.6}
    capsize = {"plot_scatter_error_capsize": 3}
    figure = ScatterChart(
        [points(gated_rows), points(public_rows)],
        subtitle=["download-gated", "public"],
        show_yerr=False,
        # the public point takes its own shape, so colour is not the only signal
        style=[capsize, {**capsize, "plot_scatter_marker": LINE_MARKER.DIAMOND}],
        texts=readings,
        vspans={"xmin": -MARGIN, "xmax": MARGIN, "label": "equivalence margin"},
        vlines=[
            {"x": 0.0, "style": {"plot_vline_style": LINE_STYLE.SOLID}},
            {"x": -MARGIN, "style": margin_line},
            {"x": MARGIN, "style": margin_line},
        ],
        title="Local minus Claude",
        xlabel=f"{pair}, strict span F1 (points)",
        ylabel="Test set",
        xticks=list(C1_TICKS),
        xticklabels=[f"{F1_POINTS * t:+.0f}" if t else "0" for t in C1_TICKS],
        yticks=[y for _, y in placed],
        yticklabels=[TEST_SET_TITLES[r["test_set"]] for r, _ in placed],
        xmin=C1_TICKS[0] - MARGIN / 2,
        xmax=C1_TICKS[-1] + MARGIN / 2,
        ymin=C1_PUBLIC_ROW - 0.6,
        ymax=C1_GATED_TOP + 0.8,
        figsize=ONE_COLUMN,
        show_grid=SHOW_GRID.X,
        show_legend=True,
        # left of the margin and below the lowest download-gated row nothing is drawn
        legend={"location": LEGEND_LOCATION.LOWER_LEFT},
    )

    # both models' strict F1 as a column of tick labels on the right edge, one per row, in the theme's sizes
    ax = figure.axes[0]
    right = ax.twinx()
    right.set_ylim(ax.get_ylim())
    right.set_yticks(
        [y for _, y in placed],
        [
            f"{F1_POINTS * levels[r['test_set']][0]:.1f} vs {F1_POINTS * levels[r['test_set']][1]:.1f}"
            for r, _ in placed
        ],
        fontsize=config["axes_ticks_label_size"],
        color=ax.get_yticklabels()[0].get_color(),
    )
    right.tick_params(axis="y", length=0)
    right.set_ylabel(
        f"{rows[0]['local']} vs {rows[0]['claude']} (points)",
        fontsize=config["font_ylabel_size"],
        color=ax.yaxis.label.get_color(),
    )
    for spine in right.spines.values():
        spine.set_visible(False)
    return figure


def dataset_figure(test_rows: dict[str, list[dict[str, Any]]]) -> Any:
    """Draw each test set's mix of labels: one stacked bar per set, each label's share of its gold test spans.

    Every label keeps one colour across the rows and stacks in one order,
    most spans over the four test sets first. Each row name carries the
    set's total spans. Segments are parted by thin gaps in the plate colour.
    A label past the palette's colours is hatched and keeps its ink outline.

    Args:
        test_rows: Test set name -> the test split's rows of `statistics_rows`.

    Returns:
        The datachart figure.
    """
    # horizontal bars draw the first category at the bottom, so the first test set goes last
    names = {
        name: f"{TEST_SET_TITLES[name]} ({sum(r['spans'] for r in rows):,} spans)"
        for name, rows in reversed(test_rows.items())
    }
    totals: Counter[str] = Counter()
    for rows in test_rows.values():
        totals.update({r["label"]: r["spans"] for r in rows})
    labels = [label for label, _ in totals.most_common()]
    shares = {(name, r["label"]): r["share_pct"] for name, rows in test_rows.items() for r in rows}

    palette = len(config["color_general_multiple"])
    return BarChart(
        [[{"label": names[name], "y": shares.get((name, label), 0.0)} for name in names] for label in labels],
        subtitle=labels,
        style=[LABEL_GAP if i < palette else LABEL_HATCH for i in range(len(labels))],
        bar_mode=BAR_MODE.STACK,
        orientation=ORIENTATION.HORIZONTAL,
        title="Test spans by label",
        xlabel="Share of the test split's gold spans (%)",
        ylabel="Test set",
        xmin=0,
        xmax=100,
        figsize=ONE_COLUMN,
        show_grid=SHOW_GRID.X,
        show_legend=True,
        legend={"location": LEGEND_LOCATION.OUTSIDE_RIGHT},
    )


def label_rows(scored: list[ScoredRun]) -> list[dict[str, Any]]:
    """Pool each test run's strict counts per label.

    Args:
        scored: The scored runs.

    Returns:
        One row per model, test set and label, ordered by test set and label, highest F1 first.
    """
    names = {entry: name for name, entry in TEST_SETS.items()}
    rows = []
    for run in (r for r in scored if r.split == "test"):
        pooled: dict[str, Counts] = defaultdict(Counts)
        for document in run.score.documents.values():
            for label, counts in document.by_label.items():
                pooled[label] += counts
        rows += [
            {
                "model": run.model_key,
                "arm": run.arm,
                "test_set": names[run.entry],
                "label": label,
                "strict_f1": counts.f1,
                "strict_precision": counts.precision,
                "strict_recall": counts.recall,
                "gold_spans": counts.gold,
                "predicted_spans": counts.predicted,
            }
            for label, counts in pooled.items()
        ]
    order = list(TEST_SETS)
    return sorted(rows, key=lambda r: (order.index(r["test_set"]), r["label"], -r["strict_f1"]))


def claude_cost_rows(scored: list[ScoredRun]) -> list[dict[str, Any]]:
    """Tabulate each Claude model's tokens and list-price cost on each test set (D8).

    Args:
        scored: The scored runs.

    Returns:
        One row per Claude model and test set, cheapest per document first within a test set.
    """
    names = {entry: name for name, entry in TEST_SETS.items()}
    rows = []
    for run in (r for r in scored if r.arm == "claude" and r.split == "test"):
        m = run.metrics
        documents = int(m["documents"])
        prompt = int(m["input_tokens"] + m["cache_write_tokens"] + m["cache_read_tokens"])
        cost = m.get("cost_usd")
        rows.append(
            {
                "model": run.model_key,
                "test_set": names[run.entry],
                "documents": documents,
                "prompt_tokens": prompt,
                "fresh_input_tokens": int(m["input_tokens"]),
                "cache_write_tokens": int(m["cache_write_tokens"]),
                "cache_read_tokens": int(m["cache_read_tokens"]),
                "output_tokens": int(m["output_tokens"]),
                "prompt_tokens_per_document": prompt / documents,
                "output_tokens_per_document": m["output_tokens"] / documents,
                "cost_usd": cost,
                "cost_per_1000_documents_usd": cost / documents * DOCUMENTS_PER_COST if cost is not None else None,
            }
        )
    order = list(TEST_SETS)
    return sorted(rows, key=lambda r: (order.index(r["test_set"]), r["cost_per_1000_documents_usd"] or 0.0))


def subcorpus_rows(scored: list[ScoredRun]) -> list[dict[str, Any]]:
    """Score each model's SUK test run per subcorpus, the subcorpus read from the document id.

    Args:
        scored: The scored runs.

    Returns:
        One row per model and subcorpus with strict F1 and its interval, highest F1 first within a subcorpus.

    Raises:
        ValueError: If a SUK document id names no subcorpus.
    """
    rows = []
    for run in (r for r in scored if r.entry == TEST_SETS["suk"] and r.split == "test"):
        groups: dict[str, list[Counts]] = defaultdict(list)
        for doc_id, document in run.score.documents.items():
            match = SUK_SUBCORPUS.match(doc_id)
            if match is None:
                raise ValueError(f"SUK document id {doc_id!r} names no subcorpus")
            groups[match.group(1)].append(document.strict)
        for subcorpus, counts in sorted(groups.items()):
            interval = f1_interval(counts)
            rows.append(
                {
                    "model": run.model_key,
                    "arm": run.arm,
                    "subcorpus": subcorpus,
                    "documents": len(counts),
                    "gold_spans": sum(c.gold for c in counts),
                    "strict_f1": interval.estimate,
                    "strict_f1_low": interval.low,
                    "strict_f1_high": interval.high,
                }
            )
    return sorted(rows, key=lambda r: (r["subcorpus"], -r["strict_f1"]))


def cut_off_rows(scored: list[ScoredRun]) -> list[dict[str, Any]]:
    """Set each run's answers cut off at the token cap beside its miss rate.

    The Claude backend takes no token cap and counts its reasoning as output,
    so its cut-off cells stay empty.

    Args:
        scored: The scored runs.

    Returns:
        One row per model, entry and split, the highest cut-off rate first.
    """
    names = {entry: name for name, entry in TEST_SETS.items()}
    rows = []
    for run in scored:
        windows, missed = run.score.windows, run.score.missed
        capped = run.arm != UNCAPPED_ARM
        cut_off, cut_off_missed = (run.cut_off_windows, run.cut_off_misses) if capped else (None, None)
        rows.append(
            {
                "model": run.model_key,
                "arm": run.arm,
                "test_set": names[run.entry],
                "split": run.split,
                "windows": windows,
                "cut_off_windows": cut_off,
                "cut_off_rate": cut_off / windows if capped and windows else None,
                "missed_windows": missed,
                "miss_rate": missed / windows if windows else None,
                "missed_cut_off_windows": cut_off_missed,
                "misses_cut_off_share": cut_off_missed / missed if capped and missed else None,
                "run_id": run.run_id,
            }
        )
    return sorted(rows, key=lambda r: (-(r["cut_off_rate"] or 0.0), r["model"], r["test_set"], r["split"]))


def answered_c1_rows(client: MlflowClient, readings: list[C1Reading]) -> list[dict[str, Any]]:
    """Read C1 again on only the windows both picked models answered in format.

    A window either model missed is dropped from both, with the gold spans that
    start inside it; the rest is scored and bootstrapped as C1 is. It is a
    sensitivity reading beside C1, not a replacement.

    Args:
        client: An MLflow client on the tracking server.
        readings: The readings of `c1_readings`.

    Returns:
        One row per test set with both picked runs.
    """
    labels = entry_labels()
    rows = []
    for c1 in (c for c in readings if c.difference is not None):
        entry = TEST_SETS[c1.test_set]
        local = read_answers(client, c1.local.run_id)
        claude = read_answers(client, c1.claude.run_id)

        # keep the windows both models answered in format
        windows = {(a["id"], a["window"]) for a in local + claude}
        kept = _answered_windows(local, labels[entry]) & _answered_windows(claude, labels[entry])
        dropped = defaultdict(set)
        for a in local + claude:
            if (a["id"], a["window"]) not in kept:
                dropped[a["id"]].add((a["start"], a["end"]))

        # drop the gold spans that start inside a dropped window
        documents = [
            {**d, "spans": [s for s in d["spans"] if not _in_windows(s["start"], dropped[d["id"]])]}
            for d in read_split(entry, "test")
        ]
        local_score = score_split(documents, [a for a in local if (a["id"], a["window"]) in kept], labels[entry])
        claude_score = score_split(documents, [a for a in claude if (a["id"], a["window"]) in kept], labels[entry])

        # pair the two models on the same kept documents
        local_counts = {doc: s.strict for doc, s in local_score.documents.items()}
        claude_counts = {doc: s.strict for doc, s in claude_score.documents.items()}
        difference = paired_difference_interval(local_counts, claude_counts)
        rows.append(
            {
                "test_set": c1.test_set,
                "windows_kept": len(kept),
                "windows_dropped": len(windows - kept),
                "local_strict_f1": sum(local_counts.values(), Counts()).f1,
                "claude_strict_f1": sum(claude_counts.values(), Counts()).f1,
                "difference": difference.estimate,
                "difference_low": difference.low,
                "difference_high": difference.high,
                "reading": read_clause(difference),
            }
        )
    return rows


def scratch_run_count(client: MlflowClient) -> int:
    """Count the experiment's scratch runs, which no table reads.

    Args:
        client: An MLflow client on the tracking server.

    Returns:
        The scratch runs of the experiment, any status.
    """
    experiment = client.get_experiment_by_name(EXPERIMENT)
    runs = client.search_runs([experiment.experiment_id], filter_string="tags.tier = 'scratch'", max_results=10_000)
    return len(runs)


def shared_documents() -> dict[str, int]:
    """Count each test set's documents whose id or normalised text sits in a tuning split.

    The tuning splits are the train and validation splits of all four test
    sets, since the picks read every validation split (D1).

    Returns:
        Test set name -> its test documents found in a tuning split.
    """
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    for entry in TEST_SETS.values():
        for split in TUNING_SPLITS:
            if not (TASKS_DIR / entry / f"{split}.jsonl.gz").exists():
                continue
            for example in read_split(entry, split):
                seen_ids.add(example["id"])
                seen_texts.add(_text_hash(example["text"]))
    return {
        name: sum(1 for e in read_split(entry, "test") if e["id"] in seen_ids or _text_hash(e["text"]) in seen_texts)
        for name, entry in TEST_SETS.items()
    }


def audit_rows(
    runs: list[Any],
    scored: list[ScoredRun],
    labels: list[dict[str, Any]],
    subcorpora: list[dict[str, Any]],
    scratch_runs: int,
) -> list[dict[str, Any]]:
    """Count what the critique checks for leakage, thin cells and seeds, one row per check.

    The five checks of tables.md come first, in its order, then the runs set
    aside for a newer rerun; the thin-cell counts by window, gold span, label
    and SUK subcorpus follow.

    Args:
        runs: The finished tracked runs the pass read.
        scored: The scored runs.
        labels: The rows of `label_rows`.
        subcorpora: The rows of `subcorpus_rows`.
        scratch_runs: The experiment's scratch runs.

    Returns:
        `check`, `count`, `expected`, `note` rows.
    """
    names = {entry: name for name, entry in TEST_SETS.items()}
    tests = [r for r in scored if r.split == "test"]

    # test documents a tuning split also holds
    shared = shared_documents()
    shared_note = ", ".join(f"{name} {count}" for name, count in shared.items() if count)

    # the fewest documents, windows and gold spans of a model on a test set
    documents = _thinnest([(names[r.entry], r.model_key, len(r.score.documents)) for r in tests])
    windows = _thinnest([(names[r.entry], r.model_key, r.score.windows) for r in tests])
    gold = _thinnest(
        [(names[r.entry], r.model_key, sum(d.strict.gold for d in r.score.documents.values())) for r in tests]
    )
    label = _thinnest([(f"{r['label']} on {r['test_set']}", r["model"], r["gold_spans"]) for r in labels])
    subcorpus = _thinnest([(r["subcorpus"], r["model"], r["documents"]) for r in subcorpora])

    # distinct seeds per model, entry and split among the full benchmark runs
    full = [
        r for r in runs if r.data.tags.get("type") == "benchmark" and r.data.params.get("window_budget") == NO_BUDGET
    ]
    seeds: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for run in full:
        params = run.data.params
        seeds[(run.data.tags.get("model_key"), params["entry"], params["split"])].add(params.get("seed"))
    fewest_seeds = min(len(found) for found in seeds.values())

    # the runs read, by kind; a full run C1 reads but did not score had a newer rerun
    benchmark = sum(1 for run in runs if run.data.tags.get("type") == "benchmark")
    memory = len(runs) - benchmark
    superseded = sum(1 for run in full if run.data.tags.get("examples") == CLAUSE_EXAMPLES) - len(scored)

    return [
        {"check": "rows shared between splits", "count": sum(shared.values()), "expected": 0, "note": shared_note},
        {"check": "fewest rows per cell", "count": documents[0], "expected": None, "note": documents[1]},
        {
            "check": "fewest seeds per configuration",
            "count": fewest_seeds,
            "expected": SEEDS_PER_CONFIGURATION,
            "note": "" if fewest_seeds >= SEEDS_PER_CONFIGURATION else "a model, entry and split without a run",
        },
        {
            "check": "tracked runs",
            "count": len(runs),
            "expected": None,
            "note": f"{benchmark} benchmark runs, {len(scored)} of them scored; {memory} memory runs",
        },
        {
            "check": "scratch runs",
            "count": 0,
            "expected": None,
            "note": f"the pass reads tracked runs only; the experiment holds {scratch_runs} scratch runs",
        },
        {
            "check": "runs set aside for a newer rerun",
            "count": superseded,
            "expected": None,
            "note": f"full examples={CLAUSE_EXAMPLES} runs whose model, entry and split has a newer run, which is scored",
        },
        {"check": "fewest windows per cell", "count": windows[0], "expected": None, "note": windows[1]},
        {"check": "fewest gold spans per cell", "count": gold[0], "expected": None, "note": gold[1]},
        {"check": "fewest gold spans per label cell", "count": label[0], "expected": None, "note": label[1]},
        {
            "check": "fewest documents per suk subcorpus cell",
            "count": subcorpus[0],
            "expected": None,
            "note": subcorpus[1],
        },
    ]


def _thinnest(cells: list[tuple[str, str, int]]) -> tuple[int, str]:
    """Find the smallest cell size and name the groups that reach it.

    Args:
        cells: `(group, model, size)` per cell, the group a test set, label or subcorpus.

    Returns:
        The smallest size, and a note naming each group that reaches it with its share of models.
    """
    smallest = min(size for _, _, size in cells)
    models = Counter(group for group, _, _ in cells)
    hits = Counter(group for group, _, size in cells if size == smallest)
    return smallest, "; ".join(f"{group}, {count} of {models[group]} models" for group, count in hits.items())


def _text_hash(text: str) -> str:
    """Hash a text after lowercasing it and collapsing its whitespace."""
    return hashlib.sha256(" ".join(text.lower().split()).encode("utf-8")).hexdigest()


def _is_answered(answer: dict[str, Any], labels: list[str]) -> bool:
    """Tell whether a window's answer parsed, as `score_split` judges it."""
    return not answer.get("error") and parse_answer(answer["answer"], labels) is not None


def _answered_windows(answers: list[dict[str, Any]], labels: list[str]) -> set[tuple[str, int]]:
    """Return the `(document id, window)` keys of the windows answered in format."""
    return {(a["id"], a["window"]) for a in answers if _is_answered(a, labels)}


def _in_windows(offset: int, windows: set[tuple[int, int]]) -> bool:
    """Tell whether a document offset falls inside one of a document's `(start, end)` windows."""
    return any(start <= offset < end for start, end in windows)


def _gap_row(test_set: str, access: str, local: float, claude: float, difference: Interval | None) -> dict[str, Any]:
    """Build one row of the gap table; the difference without an interval is local minus Claude."""
    return {
        "test_set": test_set,
        "access": access,
        "local_strict_f1": local,
        "claude_strict_f1": claude,
        "difference": difference.estimate if difference else local - claude,
        "difference_low": difference.low if difference else None,
        "difference_high": difference.high if difference else None,
    }


def _count(key: str, value: Any) -> Any:
    """Write a count metric as an integer; pass any other metric and a missing one through."""
    return int(value) if key in COUNT_METRICS and value is not None else value


# ===============================
# Main function
# ===============================


def build_data() -> None:
    """Write the summary, fields and statistics tables of every test set, and draw its spans per label."""
    test_rows = {}
    for name, entry in TEST_SETS.items():
        splits = {split: read_split(entry, split) for split in SPLITS}
        stats = statistics_rows(splits)
        write_table(f"dataset-{name}-summary", summary_rows(splits))
        write_table(f"dataset-{name}-fields", fields_rows(splits))
        write_table(f"dataset-{name}-statistics", stats)
        test_rows[name] = [row for row in stats if row["split"] == "test"]

    # a dataset figure carries no claim
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_figure(lambda: dataset_figure(test_rows), FIGURES_DIR / f"{DATASET_FIGURE}.svg")


def build_scores() -> None:
    """Rescore the tracked runs and write the score, clause and read-beside tables and the figure.

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

    # write the reference table, the picks, the C1 readings and the verdict
    picks = pick_rows(scored, list(TEST_SETS.values()))
    readings = c1_readings(scored, picks)
    c1 = c1_rows(readings)
    write_table("benchmark-models-strict-f1-by-model-and-split", reference_rows(scored))
    write_table("benchmark-models-picks-by-model", picks)
    write_table("benchmark-models-c1-by-test-set", c1)
    write_table("clause-readings", clause_rows(readings))
    write_table("verdict-by-arm", verdict_arm_rows(readings, picks))
    logger.info("C1 verdict=%s", c1[-1]["reading"])

    # draw the C1 decision figure from its own data table
    figure_rows = c1_figure_rows(readings)
    if figure_rows:
        write_table(C1_FIGURE, figure_rows)
        # the claim names no variant: every point is the same picked pair, and the table has no variant column
        claim = Claim(CLAUSE, "local minus Claude strict span F1", threshold=-MARGIN)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        # both models' strict F1 per test set come from the C1 table
        names = {entry: name for name, entry in TEST_SETS.items()}
        levels = {
            names[row["entry"]]: (row["local_strict_f1"], row["claude_strict_f1"]) for row in c1 if "difference" in row
        }
        save_figure(lambda: c1_figure(figure_rows, levels), FIGURES_DIR / f"{C1_FIGURE}.svg", claim=claim)

    # write the tables read beside the clauses
    labels = label_rows(scored)
    subcorpora = subcorpus_rows(scored)
    write_table("benchmark-models-strict-f1-by-label", labels)
    write_table("benchmark-models-claude-cost-by-test-set", claude_cost_rows(scored))
    write_table("benchmark-models-strict-f1-by-suk-subcorpus", subcorpora)
    write_table("benchmark-models-local-minus-claude-by-test-set", gap_rows(readings))
    write_table("benchmark-models-cut-off-answers-by-split", cut_off_rows(scored))
    write_table("benchmark-models-c1-on-answered-windows-by-test-set", answered_c1_rows(client, readings))

    # write the counts the critique checks
    write_table("audit-count-by-check", audit_rows(runs, scored, labels, subcorpora, scratch_run_count(client)))

    # write mean test F1 against model size and against pooled test throughput
    models = local_models(MODELS_CONFIG)
    parameters = parameter_rows(scored, models)
    speeds = speed_rows(scored, models)
    claude_speeds = claude_speed_rows(scored)
    write_table(PARAMETERS_TABLE, parameters)
    write_table(SPEED_TABLE, speeds)
    write_table(CLAUDE_SPEED_TABLE, claude_speeds)

    # the picked pair and the picked Claude model's mean test F1, the reference of the Q1 figures
    picked = {row["arm"]: row["model"] for row in picks if row.get("picked")}
    local, claude = picked.get("local"), picked.get("claude")
    claude_tests = [r.metrics["strict_f1"] for r in scored if r.model_key == claude and r.split == "test"]
    claude_f1 = statistics.mean(claude_tests) if claude_tests else None
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # write the memory table and draw F1 against memory
    memory = memory_rows(scored, peak_memory_gib(runs, scored))
    if memory:
        write_table(MEMORY_FIGURE, memory)
        # draw only models scored on all four test sets, as the y label says
        drawn = [row for row in memory if row["test_entries"] == len(TEST_SETS)]
        if drawn:
            # the picked Claude model has no GPU memory, so it is the claim's baseline line
            claim = Claim(
                DIAGNOSTIC_CLAUSE,
                "test strict span F1",
                arms=(local,) if any(row["variant"] == local for row in drawn) else (),
                threshold=claude_f1,
            )
            save_figure(
                lambda: local_f1_figure(
                    drawn, "peak_gpu_memory_gib", local, claude, claude_f1,
                    "Strict F1 by peak memory", "Peak GPU memory (GiB)",
                ),
                FIGURES_DIR / f"{MEMORY_FIGURE}.svg",
                claim=claim,
            )  # fmt: skip

    # draw F1 against parameters, the picked Claude model again as the baseline line
    if parameters:
        claim = Claim(
            DIAGNOSTIC_CLAUSE,
            "test strict span F1",
            arms=(local,) if any(row["variant"] == local for row in parameters) else (),
            threshold=claude_f1,
        )
        save_figure(
            lambda: local_f1_figure(
                parameters, "parameters_billion", local, claude, claude_f1,
                "Strict F1 by parameters", "Total parameters (billions)",
            ),
            FIGURES_DIR / f"{PARAMETERS_TABLE}.svg",
            claim=claim,
        )  # fmt: skip

    # draw F1 against throughput; Claude has a throughput, so its models are points, not a line
    if speeds:
        claude_points = [
            row | {"test_strict_f1_mean": mean_test_strict_f1(scored, row["variant"])} for row in claude_speeds
        ]
        claim = Claim(
            DIAGNOSTIC_CLAUSE,
            "test strict span F1",
            arms=(local,) if any(row["variant"] == local for row in speeds) else (),
        )
        save_figure(
            lambda: speed_figure(speeds, claude_points, local, claude),
            FIGURES_DIR / f"{SPEED_TABLE}.svg",
            claim=claim,
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
