"""Tests for scripts/finetune_encoders.py."""

import logging

import pytest

import scripts.finetune_encoders as finetune_encoders

# ===============================
# Test Data
# ===============================

ENCODERS = ["org/tiny-encoder", "/models/local-encoder"]
SPLIT_FILES = ("train.jsonl.gz", "val.jsonl.gz", "test.jsonl.gz")
TASKS_REGISTRY = """\
roots: {{ extracted: {root}/extracted, tasks: {root}/tasks, raw: {root}/raw }}
converters: {{ ner: spans, pos: pos }}
entries:
  ner/tiny:
    role: finetune_and_eval
    source: {{ kind: extracted, keys: [tiny_ner] }}
    splits: {{ train: train.jsonl.gz, val: val.jsonl.gz, test: test.jsonl.gz }}
    labels: [PER, LOC]
    suite: null
    language: sl
    license: cc-by-4.0
  pos/unbuilt:
    role: finetune_only
    source: {{ kind: extracted, keys: [tiny_pos] }}
    splits: {{ train: train.jsonl.gz, val: val.jsonl.gz, test: test.jsonl.gz }}
    labels: [NOUN, VERB]
    suite: null
    language: sl
    license: cc-by-4.0
"""
SHARED_CONFIG = """\
tasks_config: {tasks_config}
encoders: []
defaults:
  epochs: 1
  batch_size: 2
  eval_batch_size: 2
  gradient_accumulation: 1
  max_length: 32
  learning_rates: [1.0e-5, 2.0e-5]
  seeds: [0]
  weight_decay: 0.0
  warmup_ratio: 0.0
  max_grad_norm: 1.0
  precision: fp32
families: {{}}
"""

# ===============================
# Fixtures
# ===============================


@pytest.fixture
def finetune_config(tmp_path, monkeypatch):
    """Write a registry where ner/tiny has its splits and pos/unbuilt has none; return the shared config."""
    tasks_config = tmp_path / "tasks.yaml"
    tasks_config.write_text(TASKS_REGISTRY.format(root=tmp_path), encoding="utf-8")
    config = tmp_path / "finetune.yaml"
    config.write_text(SHARED_CONFIG.format(tasks_config=tasks_config), encoding="utf-8")
    (tmp_path / "tasks" / "ner" / "tiny").mkdir(parents=True)
    for name in SPLIT_FILES:
        (tmp_path / "tasks" / "ner" / "tiny" / name).touch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").as_uri())
    return config


@pytest.fixture
def trained(monkeypatch):
    """Replace training with a stub and return the list of runs it was asked to train."""
    calls = []
    monkeypatch.setattr(finetune_encoders, "track_finetune", lambda *args, **kwargs: calls.append(args))
    return calls


# ===============================
# run --dry-run
# ===============================


def test_dry_run_prints_every_combination_skips_unbuilt_entries_and_writes_nothing(
    finetune_config, trained, tmp_path, capsys, caplog
):
    """A dry run over --all lists encoder x built entry x learning rate x seed, warns of the unbuilt entry."""
    before = sorted(tmp_path.rglob("*"))
    encoder_args = [arg for encoder in ENCODERS for arg in ("--encoder", encoder)]

    with caplog.at_level(logging.WARNING):
        code = finetune_encoders.main(
            ["run", "--all", *encoder_args, "--config", str(finetune_config), "--experiment", "e", "--dry-run"]
        )

    lines = capsys.readouterr().out.splitlines()
    assert code == 0
    assert lines == [
        *(
            f"{encoder}-ner-tiny-lr{lr}-seed0"
            for encoder in ("tiny-encoder", "local-encoder")
            for lr in ("1e-05", "2e-05")
        ),
        "4 planned runs",
    ]
    assert "skipped=1 entries=['pos/unbuilt']" in caplog.text
    assert trained == []
    assert sorted(tmp_path.rglob("*")) == before


def test_a_named_entry_with_missing_split_files_fails_before_training(finetune_config, trained, caplog):
    """Naming an unbuilt entry exits with an error that names it, and nothing trains."""
    with caplog.at_level(logging.ERROR):
        code = finetune_encoders.main(
            ["run", "pos/unbuilt", "--encoder", ENCODERS[0], "--config", str(finetune_config), "--dry-run"]
        )

    assert code == finetune_encoders.ERROR_EXIT_CODE
    assert "entry 'pos/unbuilt' has split files missing on disk" in caplog.text
    assert trained == []
