"""Tests for slm4ie.silver.tracking."""

import hashlib
import json
from contextlib import contextmanager

import pytest

from slm4ie.silver import tracking
from slm4ie.silver.make import prompt_hash
from slm4ie.silver.tracking import call_metrics, file_hashes, log_round_runs, template_hashes

# ===============================
# Test Data
# ===============================

SAMPLING = {"seed": 13, "temperature": 0.0, "reasoning_effort": "low", "max_tokens": 500}
SPEC = {"model": "toy/alpha", "family": "family-one"}
TAGS = {"tier": "tracked", "round": "R1", "branch": "exp/toy", "base_commit": "abc123"}


def row(cost, error=None, rating=None):
    """Build one stored call with 100 input and 20 output tokens."""
    return {"cost_usd": cost, "error": error, "input_tokens": 100, "output_tokens": 20, "rating": rating}


class FakeTracker:
    """Stand in for the MLflow helpers: hold runs by their `run_key` tag and record what each logged."""

    def __init__(self):
        """Start with no runs."""
        self.runs = {}
        self.deleted = []

    def find_run_by_tag(self, experiment, key, value):
        """Return the id of the run whose tag matches, or None."""
        return next((run_id for run_id, run in self.runs.items() if run["tags"].get(key) == value), None)

    def delete_run(self, run_id):
        """Forget a run and note its id."""
        self.deleted.append(run_id)
        del self.runs[run_id]

    @contextmanager
    def mlflow_run(self, name, tags):
        """Open a run under a fresh id."""
        run_id = f"run-{len(self.runs) + len(self.deleted)}"
        self.runs[run_id] = {"name": name, "tags": tags, "params": {}}
        self.active = run_id
        yield

    def log_params(self, params):
        """Record the parameters of the open run."""
        self.runs[self.active]["params"] = params


@pytest.fixture
def tracker(monkeypatch):
    """Route the MLflow helpers of `tracking` to a fake that writes nothing."""
    fake = FakeTracker()
    monkeypatch.setattr(tracking.ml, "ensure_experiment", lambda experiment: True)
    monkeypatch.setattr(tracking.ml, "find_run_by_tag", fake.find_run_by_tag)
    monkeypatch.setattr(tracking.ml, "delete_run", fake.delete_run)
    monkeypatch.setattr(tracking.ml, "mlflow_run", fake.mlflow_run)
    monkeypatch.setattr(tracking.ml, "log_params", fake.log_params)
    monkeypatch.setattr(tracking.ml, "log_metrics", lambda metrics: None)
    monkeypatch.setattr(tracking.ml, "log_artifact", lambda path: None)
    return fake


def log(records_dir, tags, input_hashes=None):
    """Log the maker run of `alpha` from the records in `records_dir`."""
    return log_round_runs("toy", "maker", {"alpha": SPEC}, records_dir, SAMPLING, {}, input_hashes or {}, tags)


@pytest.fixture
def records_dir(tmp_path):
    """Write one stored call of the maker `alpha`."""
    (tmp_path / "alpha.jsonl").write_text(json.dumps(row(0.01)) + "\n", encoding="utf-8")
    return tmp_path


# ===============================
# call_metrics
# ===============================


def test_call_metrics_of_a_maker_sum_cost_and_failures():
    """Two calls costing 0.01 and 0.03, one of them failed, average 0.02 per example."""
    metrics = call_metrics([row(0.01), row(0.03, error="boom")], "maker")

    assert metrics == {
        "calls": 2.0,
        "failed_calls": 1.0,
        "input_tokens": 200.0,
        "output_tokens": 40.0,
        "cost_usd": pytest.approx(0.04),
        "cost_per_example_usd": pytest.approx(0.02),
    }


def test_call_metrics_of_a_judge_count_dropped_ratings():
    """A judge call with no error and no rating is dropped; a failed call is not."""
    metrics = call_metrics([row(0.01, rating={"naturalness": 4}), row(0.01), row(0.01, error="boom")], "judge")

    assert (metrics["dropped_ratings"], metrics["cost_per_rating_usd"]) == (1.0, pytest.approx(0.01))


# ===============================
# template_hashes and file_hashes
# ===============================


def test_template_hashes_hash_the_unfilled_system_and_user_turns():
    """Each named prompt is hashed as its system and user templates, the way a record's prompt is."""
    prompts = {"annotate": {"system": "S.", "user": "U {text}"}, "write": {"system": "W.", "user": "V"}}

    hashes = template_hashes(prompts, ["annotate"])

    expected = prompt_hash([{"role": "system", "content": "S."}, {"role": "user", "content": "U {text}"}])
    assert hashes == {"prompt_hash_annotate": expected}


def test_file_hashes_name_each_file_by_its_sha256(tmp_path):
    """A file is hashed by the first 12 hex characters of the sha256 of its bytes."""
    path = tmp_path / "passages.jsonl"
    path.write_bytes(b'{"passage_id": "p1"}\n')

    assert file_hashes({"passages": path}) == {"passages_sha256": hashlib.sha256(path.read_bytes()).hexdigest()[:12]}


# ===============================
# log_round_runs
# ===============================


def test_log_round_runs_keys_and_names_the_run_by_tier_and_round(tracker, records_dir):
    """A tracked run of round 1 is keyed `maker/alpha/tracked/R1` and named `alpha-R1`."""
    log(records_dir, TAGS)

    (run,) = tracker.runs.values()
    assert (run["tags"]["run_key"], run["name"]) == ("maker/alpha/tracked/R1", "alpha-R1")


def test_log_round_runs_scratch_run_leaves_the_tracked_run(tracker, records_dir):
    """A scratch run of the same model and round gets its own key and replaces no tracked run."""
    log(records_dir, TAGS)

    log(records_dir, {**TAGS, "tier": "scratch"})

    keys = sorted(run["tags"]["run_key"] for run in tracker.runs.values())
    assert (keys, tracker.deleted) == (["maker/alpha/scratch/R1", "maker/alpha/tracked/R1"], [])


def test_log_round_runs_replaces_the_run_of_the_same_key(tracker, records_dir):
    """Logging the same tier and round again leaves one run of the model."""
    log(records_dir, TAGS)

    log(records_dir, TAGS)

    assert (len(tracker.runs), tracker.deleted) == (1, ["run-0"])


def test_log_round_runs_logs_the_input_hashes_as_parameters(tracker, records_dir):
    """The hashes of the input files passed in are logged beside the model's settings."""
    log(records_dir, TAGS, {"passages_sha256": "aaa", "briefs_sha256": "bbb", "labels_sha256": "ccc"})

    (run,) = tracker.runs.values()
    assert {key: run["params"][key] for key in ("passages_sha256", "briefs_sha256", "labels_sha256")} == {
        "passages_sha256": "aaa",
        "briefs_sha256": "bbb",
        "labels_sha256": "ccc",
    }
