"""Tests for scripts/screen_silver_llms.py, run in-process on tiny configs and data under tmp_path."""

import gzip
import hashlib
import json
from pathlib import Path

import pytest
import yaml

import scripts.screen_silver_llms as screen
from slm4ie.evaluation.backends import ChatBackend, ChatResult

# ===============================
# Test Data
# ===============================

PASSAGE = "Bolnik je prejel paracetamol zaradi glavobola."
WRITTEN = "Bolnica je vzela paracetamol proti glavobolu."
PRICES = {"usd_per_million_input": 1.0, "usd_per_million_output": 2.0}

MODELS = {
    "limits": {"round_usd": 5, "experiment_usd": 10},
    "sampling": {"seed": 13, "temperature": 0.0, "reasoning_effort": "low", "max_tokens": 500},
    "models": {
        "maker-alpha": {"display_name": "Alpha", "model": "toy/alpha", "family": "family-one", **PRICES},
        "maker-beta": {"display_name": "Beta", "model": "toy/beta", "family": "family-two", **PRICES},
    },
    "judges": {"judge-gamma": {"display_name": "Gamma", "model": "toy/gamma", "family": "family-three", **PRICES}},
}
LABELS = {
    "domains": {
        "medical": {
            "source": "toy",
            "labels": {
                "Drug": {"description": "A medicine.", "examples": ["aspirin"]},
                "Condition": {"description": "A disease.", "examples": ["gripa"]},
            },
        }
    }
}
PROMPTS = {
    "annotate": {"system": "Annotate.", "user": "Labels:\n{labels}\nText:\n{text}"},
    "write": {"system": "Write.", "user": "Write {text_type} on {topic}, {words} words, with {required_labels}."},
    "judge": {
        "system": "Judge.",
        "user": "Labels:\n{labels}\nText:\n{text}\nMentions:\n{mentions}\n1. naturalness: from 1 to 5.",
    },
}
BRIEFS = {
    "briefs": [
        {
            "brief_id": "med-01",
            "passage_id": "toy:doc-1#0",
            "domain": "medical",
            "text_type": "a clinical note",
            "topic": "a headache treated with a pain medicine",
            "words": 6,
            "labels": ["Drug"],
        }
    ]
}
TRACKED = ["--tag", "tier=tracked", "--tag", "branch=exp/toy", "--tag", "base_commit=abc123"]
PASSAGE_ROW = {
    "set": "medical-theses",
    "domain": "medical",
    "native": True,
    "passage_id": "toy:doc-1#0",
    "doc_id": "toy:doc-1",
    "group": "",
    "text": PASSAGE,
    "words": 6,
}

# a body-prose paragraph of 14 words; eight of them make a 112-word passage
PARAGRAPH = "Ta odstavek opisuje metodo in rezultate raziskave na vzorcu podatkov iz leta dva tisoč."
PROSE = "\n".join([PARAGRAPH] * 8)
SOURCES = {
    "seed": 13,
    "corpus_tier": "pretrain/06",
    "sets": {
        "theses": {
            "domain": "medical",
            "native": True,
            "source": "oss",
            "count": 1,
            "wrapped": False,
            "doctypes": ["Diplomsko delo"],
            "udc_prefix": ["61"],
        },
        "medicines": {
            "domain": "medical",
            "native": False,
            "source": "med",
            "count": 2,
            "wrapped": False,
            "metadata_equals": {"source": "CBZ"},
            "group_field": "file_name",
        },
    },
}
THESES = [
    {"id": "oss:kept", "text": PROSE, "metadata": {"doctype": "Diplomsko delo", "udc": "616.1"}},
    {"id": "oss:wrong-doctype", "text": PROSE, "metadata": {"doctype": "Seminarska naloga", "udc": "616.1"}},
    {"id": "oss:wrong-udc", "text": PROSE, "metadata": {"doctype": "Diplomsko delo", "udc": "331.1"}},
]
MEDICINES = [
    {"id": "med:1", "text": PROSE, "metadata": {"source": "CBZ", "file_name": "1.0_SmPC_Aspirin 500 mg.txt"}},
    {"id": "med:2", "text": PROSE, "metadata": {"source": "CBZ", "file_name": "2.0_SmPC_ASPIRIN forte.txt"}},
    {"id": "med:3", "text": PROSE, "metadata": {"source": "CBZ", "file_name": "3.0_SmPC_Lekadol 500 mg.txt"}},
    {"id": "med:4", "text": PROSE, "metadata": {"source": "NIJZ", "file_name": "4.0_SmPC_Nalgesin.txt"}},
]


class FakeBackend(ChatBackend):
    """A backend that answers each job with a fixed valid reply and reports a fixed cost."""

    def _complete(self, messages, schema):
        properties = schema["properties"]
        if "naturalness" in properties:
            count = properties["mentions"]["minItems"]
            mentions = [{"number": n, "label_correct": True, "span_correct": True} for n in range(1, count + 1)]
            reply = {"errors": [], "naturalness": 4, "realism": 4, "mentions": mentions, "missed": []}
        elif "text" in properties:
            reply = {"text": WRITTEN, "mentions": [{"text": "paracetamol", "label": "Drug"}]}
        else:
            reply = {"mentions": [{"text": "paracetamol", "label": "Drug"}]}
        return ChatResult(json.dumps(reply), 100, 20, 0.0, cost_usd=0.001)


def write_yaml(path, data):
    """Write a mapping as YAML."""
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")


def write_shard(path, documents):
    """Write corpus documents as one gzipped JSONL shard."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.writelines(json.dumps(document) + "\n" for document in documents)


@pytest.fixture
def layout(tmp_path):
    """Write the configs and one drawn passage; return the shared options pointing at them."""
    configs = tmp_path / "configs"
    configs.mkdir()
    write_yaml(configs / "screen-models.yaml", MODELS)
    write_yaml(configs / "labels-domains.yaml", LABELS)
    write_yaml(configs / "prompts-jobs.yaml", PROMPTS)
    write_yaml(configs / "briefs-passages.yaml", BRIEFS)
    write_yaml(configs / "passages-sources.yaml", SOURCES)
    passages = tmp_path / "data" / "final" / "passages-round1" / "passages.jsonl"
    passages.parent.mkdir(parents=True)
    passages.write_text(json.dumps(PASSAGE_ROW) + "\n", encoding="utf-8")
    return ["--configs-dir", str(configs), "--data-dir", str(tmp_path / "data")]


@pytest.fixture
def no_backend(monkeypatch):
    """Make building any backend fail the test."""

    def refuse(sampling):
        raise AssertionError("a backend was built")

    monkeypatch.setattr(screen, "openrouter_factory", refuse)


@pytest.fixture
def fake_backend(monkeypatch, tmp_path):
    """Answer every call with the fake backend and track to a SQLite store under tmp_path."""
    pytest.importorskip("mlflow")
    monkeypatch.setattr(screen, "openrouter_factory", lambda sampling: lambda name, spec: FakeBackend(spec["model"]))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    # the artifact root `./mlruns` resolves against the working directory
    monkeypatch.chdir(tmp_path)


def read_jsonl(path):
    """Read a JSONL file's rows."""
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def logged_runs():
    """Return the runs MLflow holds in the experiment of the script."""
    import mlflow

    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(screen.DEFAULT_EXPERIMENT)
    return client.search_runs([experiment.experiment_id])


# ===============================
# make
# ===============================


def test_make_dry_run_prints_the_projection_and_builds_no_backend(layout, no_backend, monkeypatch, capsys):
    """A dry run prints calls and list-price spend per model and in total, with no backend and no tracking server."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    assert screen.main(["make", "--dry-run", *layout]) == 0

    out = capsys.readouterr().out
    # two items per model at 1000 input and 750 output tokens: 2 x 0.0025 USD each
    assert "maker-alpha" in out and "maker-beta" in out
    assert "total" in out and "4" in out and "0.0100" in out


def test_make_stores_one_record_per_call_and_logs_one_run_per_variant(layout, fake_backend, tmp_path):
    """Each model answers both items once, and MLflow holds one run per model."""
    assert screen.main(["make", *layout]) == 0

    examples = tmp_path / "data" / "interim" / "examples-round1-scratch"
    assert [len(read_jsonl(examples / f"{variant}.jsonl")) for variant in ("maker-alpha", "maker-beta")] == [2, 2]
    assert sorted(run.data.tags["variant"] for run in logged_runs()) == ["maker-alpha", "maker-beta"]


def test_make_refuses_a_tracked_run_without_base_commit(layout, no_backend):
    """A tracked run that names its branch but not its base commit stops before any call, naming the tag."""
    with pytest.raises(SystemExit, match="base_commit"):
        screen.main(["make", *layout, "--tag", "tier=tracked", "--tag", "branch=exp/toy"])


def test_scratch_make_keeps_apart_from_tracked_make(layout, fake_backend, tmp_path):
    """A scratch run after a tracked one writes its own folder and leaves the tracked runs, tagged R1, in place."""
    screen.main(["make", *layout, *TRACKED])

    assert screen.main(["make", *layout]) == 0

    interim = tmp_path / "data" / "interim"
    rows = [
        len(read_jsonl(interim / folder / "maker-alpha.jsonl"))
        for folder in ("examples-round1", "examples-round1-scratch")
    ]
    assert rows == [2, 2]
    tiers = sorted((run.data.tags["tier"], run.data.tags["round"]) for run in logged_runs())
    assert tiers == [("scratch", "R1"), ("scratch", "R1"), ("tracked", "R1"), ("tracked", "R1")]


def test_make_dry_run_counts_only_the_spend_of_its_own_tier(layout, no_backend, tmp_path, capsys):
    """3 USD stored by scratch calls count against a scratch run and not against a tracked one."""
    scratch = tmp_path / "data" / "interim" / "examples-round1-scratch" / "maker-alpha.jsonl"
    scratch.parent.mkdir(parents=True)
    scratch.write_text(json.dumps({"cost_usd": 3.0}) + "\n", encoding="utf-8")

    screen.main(["make", "--dry-run", *layout])
    screen.main(["make", "--dry-run", *layout, *TRACKED])

    scratch_out, tracked_out = capsys.readouterr().out.split("projected spend")[1:]
    assert "already recorded 3.00 USD" in scratch_out and "already recorded 0.00 USD" in tracked_out


def test_make_logs_the_hashes_of_its_input_files(layout, fake_backend, tmp_path):
    """Every maker run names the passages, briefs and labels it read by the sha256 of each file."""
    screen.main(["make", *layout])

    passages = tmp_path / "data" / "final" / "passages-round1" / "passages.jsonl"
    params = logged_runs()[0].data.params
    assert params["passages_sha256"] == hashlib.sha256(passages.read_bytes()).hexdigest()[:12]
    assert {"briefs_sha256", "labels_sha256"} <= set(params)


# ===============================
# check
# ===============================


def test_check_writes_one_flag_row_per_example(layout, fake_backend, tmp_path):
    """Every stored example gets one row of rule-check flags."""
    screen.main(["make", *layout])

    assert screen.main(["check", *layout]) == 0

    rows = read_jsonl(tmp_path / "data" / "interim" / "checks-round1-scratch" / "checks.jsonl")
    assert len(rows) == 4 and all(row["passes"] for row in rows)


# ===============================
# rate
# ===============================


def test_rate_dry_run_builds_no_backend(layout, no_backend, capsys):
    """A rating dry run prints the judges' projected spend and calls no judge."""
    assert screen.main(["rate", "--dry-run", *layout]) == 0

    assert "judge-gamma" in capsys.readouterr().out


# ===============================
# sheet
# ===============================


def test_sheet_hides_the_maker(layout, fake_backend, tmp_path):
    """The person's sheet names no model, family or real-text marker."""
    screen.main(["make", *layout])
    screen.main(["rate", *layout])

    assert screen.main(["sheet", "--size", "3", *layout]) == 0

    sheet = (tmp_path / "data" / "interim" / "ratings-round1-scratch" / "person-sheet.md").read_text(encoding="utf-8")
    hidden = ("maker-alpha", "maker-beta", "family-one", "family-two", "toy/alpha", "real-text")
    assert "## Unit " in sheet and not [word for word in hidden if word in sheet]


# ===============================
# preview
# ===============================


def test_preview_follows_each_passage_with_its_brief(layout, tmp_path):
    """The preview page shows the real passage, then the brief derived from it."""
    assert screen.main(["preview", *layout]) == 0

    page = (tmp_path / "data" / "interim" / "preview-round1.md").read_text(encoding="utf-8")
    assert page.index(PASSAGE) < page.index("med-01") < page.index("a headache treated with a pain medicine")


# ===============================
# draw
# ===============================


def test_draw_applies_the_document_filters(layout, tmp_path):
    """Only documents of the listed doctype and UDC class, or source, are drawn, one per medicine."""
    root = tmp_path / "root"
    write_shard(root / "pretrain" / "06" / "oss" / "00000.jsonl.gz", THESES)
    write_shard(root / "pretrain" / "06" / "med" / "00000.jsonl.gz", MEDICINES)
    data = tmp_path / "drawn"

    assert screen.main(["draw", *layout, "--data-dir", str(data), "--data-root", str(root)]) == 0

    rows = read_jsonl(data / "final" / "passages-round1" / "passages.jsonl")
    assert [row["doc_id"] for row in rows if row["set"] == "theses"] == ["oss:kept"]
    assert sorted(row["group"] for row in rows if row["set"] == "medicines") == ["aspirin", "lekadol"]
