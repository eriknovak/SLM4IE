"""Tests for slm4ie.evaluation.run."""

import gzip
import json
import subprocess
from pathlib import Path

import pytest

from slm4ie.evaluation.backends import ChatBackend, ChatResult, base
from slm4ie.evaluation.run import (
    TIMEOUT_ERROR,
    BenchmarkRun,
    answer_split,
    answers_file_name,
    load_model_spec,
    run_benchmark,
    score_split,
    split_metrics,
)
from slm4ie.utils.config import load_yaml

# ===============================
# Test Data
# ===============================

LABELS = ["PER", "LOC"]
DESCRIPTIONS = {"PER": "A person.", "LOC": "A place."}

# two sentences of about 1,500 characters each, so the document takes two windows
FILLER = "x" * 1_480
LONG_TEXT = f"Janez je šel. {FILLER}.\nV Kamnik je šel Janez. {FILLER}."
SECOND_LINE = LONG_TEXT.index("V Kamnik")
DOCUMENTS = [
    {
        "id": "d1",
        "text": LONG_TEXT,
        "spans": [
            {"start": 0, "end": 5, "label": "PER"},
            {"start": SECOND_LINE + 2, "end": SECOND_LINE + 8, "label": "LOC"},
            {"start": SECOND_LINE + 16, "end": SECOND_LINE + 21, "label": "PER"},
        ],
    },
    {"id": "d2", "text": "Marija živi v Celju.", "spans": [{"start": 0, "end": 6, "label": "PER"}]},
]

LABELS_YAML = """
label_sets:
  test:
    entries: [ner/toy]
    labels:
      PER: {description: A person., examples: {unseen: [Ana], common: [Janez]}}
      LOC: {description: A place., examples: {unseen: [Bled], common: [Ljubljana]}}
"""


def _answer(mentions):
    return json.dumps({"mentions": [{"text": t, "label": lbl} for t, lbl in mentions]})


def _gold_answer(text):
    """Answer every name of the toy documents that appears in the window."""
    names = [("Janez", "PER"), ("Kamnik", "LOC"), ("Marija", "PER")]
    found = sorted((text.find(n), n, lbl) for n, lbl in names if n in text)
    return _answer([(n, lbl) for _, n, lbl in found])


class ScriptedBackend(ChatBackend):
    """A backend that answers each window with a function of its text and counts its calls."""

    def __init__(self, model="scripted", reply=_gold_answer, seed=0):
        """Store the reply function and start the call count."""
        super().__init__(model, seed)
        self.reply = reply
        self.calls = 0

    def _complete(self, messages, schema):
        self.calls += 1
        window = messages[1]["content"].rsplit("Text:\n", 1)[1]
        return ChatResult(self.reply(window), 100, 10, 0.0, cache_read_tokens=5, cost_usd=0.01)


class TimeoutBackend(ChatBackend):
    """A backend whose every call times out, as a stuck Claude CLI does."""

    def _complete(self, messages, schema):
        raise subprocess.TimeoutExpired("claude", 600)


@pytest.fixture(autouse=True)
def scripted_backend_registered(monkeypatch):
    """Register the scripted backend for `run_benchmark`, only for the test."""
    monkeypatch.setitem(base._BACKEND_REGISTRY, "scripted-test", ScriptedBackend)


def _write_toy_entry(tmp_path):
    tasks_dir = tmp_path / "tasks" / "ner" / "toy"
    tasks_dir.mkdir(parents=True)
    with gzip.open(tasks_dir / "val.jsonl.gz", "wt", encoding="utf8") as f:
        for document in DOCUMENTS:
            f.write(json.dumps(document, ensure_ascii=False) + "\n")
    tasks_yaml = tmp_path / "tasks.yaml"
    tasks_yaml.write_text(
        f"roots: {{extracted: {tmp_path}/extracted, raw: {tmp_path}/raw, tasks: {tmp_path}/tasks}}\n"
        "converters: {ner: spans}\n"
        "entries:\n"
        "  ner/toy:\n"
        "    role: held_out\n"
        "    source: {kind: extracted, keys: [toy]}\n"
        "    splits: {val: val.jsonl.gz}\n"
        "    labels: [PER, LOC]\n"
        "    suite: null\n"
        "    language: sl\n"
        "    license: cc-by-4.0\n",
        encoding="utf8",
    )
    labels_yaml = tmp_path / "ner-labels.yaml"
    labels_yaml.write_text(LABELS_YAML, encoding="utf8")
    return tasks_yaml, labels_yaml


def _run(tmp_path, **overrides):
    tasks_yaml, labels_yaml = _write_toy_entry(tmp_path)
    settings = {
        "backend": "scripted-test",
        "model": "org/model",
        "entry": "ner/toy",
        "split": "val",
        "tasks_config": tasks_yaml,
        "labels_config": labels_yaml,
        "answers_dir": tmp_path / "answers",
    }
    return BenchmarkRun(**{**settings, **overrides})


# ===============================
# answer_split
# ===============================


def test_answer_split_writes_one_line_per_window_with_offsets(tmp_path):
    """Every window gets one answers-file line carrying its document offsets."""
    path = tmp_path / "answers.jsonl"
    answers, reused = answer_split(ScriptedBackend(), DOCUMENTS, DESCRIPTIONS, LABELS, path)

    lines = [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]
    assert [(a["id"], a["window"]) for a in lines] == [("d1", 0), ("d1", 1), ("d2", 0)]
    assert lines[1]["start"] == DOCUMENTS[0]["text"].index("V Kamnik")
    assert lines[0]["end"] == DOCUMENTS[0]["text"].index("\n")
    assert answers == lines
    assert reused == 0


def test_answer_split_discards_one_warm_up_reply(tmp_path):
    """The first window asked is asked twice, so warm-up stays out of the timed answers."""
    backend = ScriptedBackend()
    answer_split(backend, DOCUMENTS, DESCRIPTIONS, LABELS, tmp_path / "answers.jsonl")

    assert backend.calls == 4


def test_answer_split_resumes_without_asking_answered_windows(tmp_path):
    """A second call over the same file asks only for windows the file lacks."""
    path = tmp_path / "answers.jsonl"
    answer_split(ScriptedBackend(), DOCUMENTS[:1], DESCRIPTIONS, LABELS, path)
    backend = ScriptedBackend()
    answers, reused = answer_split(backend, DOCUMENTS, DESCRIPTIONS, LABELS, path)

    assert reused == 2
    assert backend.calls == 2
    assert [(a["id"], a["window"]) for a in answers] == [("d1", 0), ("d1", 1), ("d2", 0)]


def test_answer_split_keeps_a_timeout_as_a_missed_window(tmp_path):
    """A timed-out call is kept with the timeout error and no tokens, and the run goes on."""
    answers, _ = answer_split(TimeoutBackend("opus"), DOCUMENTS[1:], DESCRIPTIONS, LABELS, tmp_path / "a.jsonl")

    assert answers[0]["error"] == TIMEOUT_ERROR
    assert answers[0]["answer"] == ""
    assert answers[0]["input_tokens"] == 0


def test_answer_split_stops_after_the_document_that_reaches_the_budget(tmp_path):
    """A window budget keeps whole documents, stopping once the count reaches it."""
    answers, _ = answer_split(ScriptedBackend(), DOCUMENTS, DESCRIPTIONS, LABELS, tmp_path / "a.jsonl", 1)

    assert [(a["id"], a["window"]) for a in answers] == [("d1", 0), ("d1", 1)]


# ===============================
# score_split
# ===============================


def test_score_split_shifts_window_spans_into_document_offsets(tmp_path):
    """A perfect answer in a later window scores strict F1 1.0 on document offsets."""
    answers, _ = answer_split(ScriptedBackend(), DOCUMENTS, DESCRIPTIONS, LABELS, tmp_path / "a.jsonl")
    score = score_split(DOCUMENTS, answers, LABELS)

    assert score.documents["d1"].strict.f1 == 1.0
    assert score.documents["d2"].strict.f1 == 1.0
    assert (score.windows, score.missed, score.unaligned) == (3, 0, 0)


def test_score_split_counts_an_off_schema_answer_as_a_missed_window(tmp_path):
    """An answer the schema rejects is a miss and gives no mentions."""
    backend = ScriptedBackend(reply=lambda text: _answer([("Marija", "ORG")]))
    answers, _ = answer_split(backend, DOCUMENTS[1:], DESCRIPTIONS, LABELS, tmp_path / "a.jsonl")
    score = score_split(DOCUMENTS, answers, LABELS)

    assert score.missed == 1
    assert score.documents["d2"].strict.predicted == 0


def test_score_split_counts_a_mention_absent_from_its_window_as_unaligned(tmp_path):
    """A mention whose text is not in its window counts as an unaligned false positive."""
    backend = ScriptedBackend(reply=lambda text: _answer([("Maribor", "LOC")]))
    answers, _ = answer_split(backend, DOCUMENTS[1:], DESCRIPTIONS, LABELS, tmp_path / "a.jsonl")
    score = score_split(DOCUMENTS, answers, LABELS)

    assert score.unaligned == 1
    assert score.documents["d2"].strict.predicted == 1


def test_score_split_leaves_out_documents_without_answers(tmp_path):
    """A document past the window budget is not scored."""
    answers, _ = answer_split(ScriptedBackend(), DOCUMENTS, DESCRIPTIONS, LABELS, tmp_path / "a.jsonl", 1)

    assert set(score_split(DOCUMENTS, answers, LABELS).documents) == {"d1"}


# ===============================
# split_metrics
# ===============================


def test_split_metrics_report_miss_rate_tokens_cost_and_speed(tmp_path):
    """The metrics carry the miss rate per window, token sums, cost and speed."""
    answers, _ = answer_split(ScriptedBackend(), DOCUMENTS, DESCRIPTIONS, LABELS, tmp_path / "a.jsonl")
    answers[2] = {**answers[2], "answer": "not json"}
    metrics = split_metrics(answers, score_split(DOCUMENTS, answers, LABELS))

    assert metrics["miss_rate"] == pytest.approx(1 / 3)
    assert metrics["input_tokens"] == 300
    assert metrics["cache_read_tokens"] == 15
    assert metrics["cost_usd"] == pytest.approx(0.03)
    assert metrics["strict_f1_LOC"] == 1.0
    assert "windows_per_second" in metrics


def test_split_metrics_leave_out_cost_when_no_backend_reports_one(tmp_path):
    """A local model reports no cost, so no cost metric is logged."""
    answers, _ = answer_split(ScriptedBackend(), DOCUMENTS[1:], DESCRIPTIONS, LABELS, tmp_path / "a.jsonl")
    answers = [{**a, "cost_usd": None} for a in answers]

    assert "cost_usd" not in split_metrics(answers, score_split(DOCUMENTS, answers, LABELS))


# ===============================
# run_benchmark
# ===============================


def test_run_benchmark_scores_a_split_without_tracking(tmp_path):
    """A run with no experiment answers and scores the split, writing only the answers file."""
    run = _run(tmp_path)
    metrics = run_benchmark(run)

    assert metrics["strict_f1"] == 1.0
    assert metrics["documents"] == 2
    assert len(list(run.answers_dir.glob("*.jsonl"))) == 1


def test_run_benchmark_refuses_an_existing_answers_file_without_resume_or_force(tmp_path):
    """A second run over the same answers file needs resume or force."""
    run = _run(tmp_path)
    run_benchmark(run)

    with pytest.raises(FileExistsError, match="resume"):
        run_benchmark(run)


def test_answers_file_name_is_file_safe_and_names_the_options():
    """The answers file name makes the model file-safe and hashes backend options in."""
    run = BenchmarkRun("transformers", "cjvt/GaMS-9B", "ner/suk", "val", None, None, None)
    with_options = BenchmarkRun("transformers", "cjvt/GaMS-9B", "ner/suk", "val", None, None, None, "unseen", {"q": 1})
    descriptions = {"PER": "a person"}

    assert answers_file_name(run, descriptions).startswith("transformers-cjvt-GaMS-9B-ner-suk-val-unseen-")
    assert answers_file_name(with_options, descriptions) != answers_file_name(run, descriptions)


def test_answers_file_name_changes_with_the_label_descriptions():
    """A run with other label descriptions writes another answers file, so resume never mixes prompts."""
    run = BenchmarkRun("vllm", "gams-9b", "ner/suk", "val", None, None, None)

    assert answers_file_name(run, {"PER": "a person"}) != answers_file_name(run, {"PER": "a human being"})


# ===============================
# load_model_spec
# ===============================

MODELS_CONFIG = (
    Path(__file__).parents[2] / "experiments/validation/ie-benchmark-slovenian/configs/benchmark-models.yaml"
)


def test_load_model_spec_reads_the_backend_block_with_its_options():
    """A local model's transformers block gives its HuggingFace id and its quantisation."""
    assert load_model_spec(MODELS_CONFIG, "gams-9b", "transformers") == (
        "cjvt/GaMS-9B-Instruct",
        {"quantization": "8bit", "fold_system": True},
    )


def test_every_listed_model_runs_on_its_arm_backends():
    """Claude models carry a claude block; local models carry a vllm and a transformers block."""
    models = load_yaml(MODELS_CONFIG)["models"]
    backends = {"claude": ["claude"], "local": ["vllm", "transformers"]}

    for key, spec in models.items():
        for backend in backends[spec["arm"]]:
            assert load_model_spec(MODELS_CONFIG, key, backend)[0]


def test_load_model_spec_names_a_missing_backend_block():
    """Asking a Claude model for a vllm block fails with the model and backend named."""
    with pytest.raises(KeyError, match="claude-opus-5-5.*vllm"):
        load_model_spec(MODELS_CONFIG, "claude-opus-5-5", "vllm")
