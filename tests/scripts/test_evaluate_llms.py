"""Tests for scripts/evaluate_llms.py argument parsing."""

import pytest

import scripts.evaluate_llms as evaluate

# ===============================
# Test Data
# ===============================

REQUIRED = ["run", "--backend", "vllm", "--model", "m", "--entry", "ner/suk", "--split", "val", "--answers-dir", "a"]

# ===============================
# run
# ===============================


def test_run_defaults_to_unseen_examples_and_every_window():
    """A run names no example variant or budget unless asked."""
    args = evaluate.parse_args(REQUIRED)

    assert args.examples == "unseen"
    assert args.window_budget is None


def test_resume_and_force_are_exclusive():
    """A run cannot both continue and discard its answers file."""
    with pytest.raises(SystemExit):
        evaluate.parse_args([*REQUIRED, "--resume", "--force"])


def test_option_values_are_read_as_yaml():
    """Backend options keep their YAML types, so false is a bool and 8bit a string."""
    args = evaluate.parse_args([*REQUIRED, "--option", "enable_thinking=false", "--option", "quantization=8bit"])

    assert evaluate._parse_pairs(args.option, "--option") == {"enable_thinking": False, "quantization": "8bit"}


def test_a_pair_without_equals_sign_is_rejected():
    """An option that is not KEY=VALUE stops the script."""
    with pytest.raises(SystemExit):
        evaluate._parse_pairs(["quantization"], "--option")


def test_model_and_model_key_are_exclusive():
    """A run names its model either directly or by key, never both."""
    with pytest.raises(SystemExit):
        evaluate.parse_args([*REQUIRED, "--model-key", "gams-9b"])


# ===============================
# memory
# ===============================


def test_memory_takes_several_model_keys():
    """One memory call reads several models, in the order named."""
    args = evaluate.parse_args(["memory", "--model-key", "gams-9b", "--model-key", "gemma4-12b"])

    assert args.model_key == ["gams-9b", "gemma4-12b"]


def test_memory_needs_a_model_key():
    """A memory call without a model reads nothing and stops."""
    with pytest.raises(SystemExit):
        evaluate.parse_args(["memory"])


def test_a_tracked_call_without_a_tracking_server_stops(monkeypatch):
    """--experiment without MLFLOW_TRACKING_URI exits with an error before any model loads."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    assert evaluate.main(["memory", "--model-key", "gams-9b", "--experiment", "e"]) == evaluate.ERROR_EXIT_CODE
