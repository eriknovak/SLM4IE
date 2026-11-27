"""Tests for slm4ie.finetune.search."""

import dataclasses

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
optuna = pytest.importorskip("optuna")
mlflow = pytest.importorskip("mlflow")

from slm4ie.finetune.config import FamilySettings, SearchConfig, SearchRange  # noqa: E402
from slm4ie.finetune.search import search_finetune, study_name  # noqa: E402

# ===============================
# Test Data
# ===============================

EXPERIMENT = "slm4ie/test-search"
LABELS = ["entailment", "contradiction"]
GRID = SearchConfig(
    sampler="grid", trials=2, seed=0, pruning=False, space={"learning_rate": (1e-3, 1e-2), "warmup_ratio": (0.0,)}
)
TPE = SearchConfig(
    sampler="tpe", trials=3, seed=0, pruning=True, space={"learning_rate": SearchRange(low=1e-4, high=1e-2, log=True)}
)

# ===============================
# Fixtures
# ===============================


@pytest.fixture
def family(settings) -> FamilySettings:
    """Return the tiny settings as a family with one learning rate and two seeds."""
    rest = {k: v for k, v in dataclasses.asdict(settings).items() if k not in ("learning_rate", "seed")}
    return FamilySettings(learning_rates=(settings.learning_rate,), seeds=(0, 1), settings=rest)


# ===============================
# Tests
# ===============================


def test_grid_search_runs_every_combination_and_scores_the_best_on_test_per_seed(
    tiny_encoder, roots, family, fixture_entry
):
    """A two-point grid runs two trials, keeps the best on val and scores it on test once per seed."""
    result = search_finetune(
        tiny_encoder, fixture_entry("nli", LABELS), family, GRID, roots, storage=None, experiment=None, tags={}
    )

    assert result.best_settings.learning_rate in (1e-3, 1e-2)
    assert 0.0 <= result.best_value <= 1.0
    assert [0.0 <= r.scores["macro_f1"] <= 1.0 for r in result.finals] == [True, True]


def test_resumed_search_runs_no_trial_its_budget_has_used(tiny_encoder, roots, family, fixture_entry, tmp_path):
    """A second call on the same stored study adds no trial and returns the same best settings."""
    storage = f"sqlite:///{tmp_path / 'optuna.db'}"
    entry = fixture_entry("nli", LABELS)
    first = search_finetune(tiny_encoder, entry, family, TPE, roots, storage=storage, experiment=None, tags={})

    second = search_finetune(tiny_encoder, entry, family, TPE, roots, storage=storage, experiment=None, tags={})

    study = optuna.load_study(study_name=first.study_name, storage=storage)
    assert len(study.trials) == TPE.trials
    assert second.best_settings == first.best_settings


def test_resumed_tpe_search_draws_new_settings_rather_than_the_stored_ones(
    tiny_encoder, roots, family, fixture_entry, tmp_path
):
    """Raising the budget of a stored TPE study adds trials whose learning rates differ from the stored ones."""
    storage = f"sqlite:///{tmp_path / 'optuna.db'}"
    entry = fixture_entry("nli", LABELS)
    first = search_finetune(tiny_encoder, entry, family, TPE, roots, storage=storage, experiment=None, tags={})

    search_finetune(
        tiny_encoder,
        entry,
        family,
        dataclasses.replace(TPE, trials=6),
        roots,
        storage=storage,
        experiment=None,
        tags={},
    )

    rates = [t.params["learning_rate"] for t in optuna.load_study(study_name=first.study_name, storage=storage).trials]
    assert len(rates) == len(set(rates)) == 6


def test_tpe_range_of_a_float_setting_draws_floats_from_integer_bounds(tiny_encoder, roots, family, fixture_entry):
    """A max_grad_norm range written with whole-number bounds still draws a non-integer value."""
    search = SearchConfig(
        sampler="tpe", trials=1, seed=0, pruning=False, space={"max_grad_norm": SearchRange(low=1, high=5, log=False)}
    )

    result = search_finetune(
        tiny_encoder, fixture_entry("nli", LABELS), family, search, roots, storage=None, experiment=None, tags={}
    )

    assert not float(result.best_settings.max_grad_norm).is_integer()


def test_tracked_search_logs_a_parent_run_with_a_child_per_trial_and_per_final_seed(
    tiny_encoder, roots, family, fixture_entry, tmp_path, monkeypatch
):
    """The parent run names its best child run and the final test mean; each trial child has a state tag."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.chdir(tmp_path)

    search_finetune(
        tiny_encoder,
        fixture_entry("nli", LABELS),
        family,
        GRID,
        roots,
        storage=None,
        experiment=EXPERIMENT,
        tags={"tier": "scratch"},
        device="cpu",
    )

    client = mlflow.MlflowClient()
    runs = client.search_runs([client.get_experiment_by_name(EXPERIMENT).experiment_id])
    [parent] = [r for r in runs if r.data.tags.get("run_type") == "search"]
    children = [r for r in runs if r.data.tags.get("mlflow.parentRunId") == parent.info.run_id]
    trials = [r for r in children if r.data.tags.get("run_type") == "trial"]
    finals = [r for r in children if r.data.tags.get("stage") == "final"]
    assert (len(trials), len(finals)) == (2, 2)
    assert {r.data.tags["trial_state"] for r in trials} == {"COMPLETE"}
    assert parent.data.tags["best_child_run_id"] in {r.info.run_id for r in trials}
    assert 0.0 <= parent.data.metrics["final_test_macro_f1_mean"] <= 1.0


def test_tracked_search_without_a_tier_tag_fails_before_training(tiny_encoder, roots, family, fixture_entry):
    """Tracking needs a tier tag, checked before any trial trains."""
    with pytest.raises(ValueError, match="needs a 'tier' tag"):
        search_finetune(
            tiny_encoder, fixture_entry("nli", LABELS), family, GRID, roots, storage=None, experiment="e", tags={}
        )


def test_study_name_separates_configs_and_encoders_but_not_trial_budgets(family, fixture_entry):
    """Another space or another encoder path names another study; a larger trial budget names the same one."""
    entry = fixture_entry("nli", LABELS)
    other_space = dataclasses.replace(GRID, space={"learning_rate": (1e-3,)})

    name = study_name("runs/a/final", entry, family, TPE)

    assert name.startswith("final-nli-tiny-tpe-")
    assert name == study_name("runs/a/final", entry, family, dataclasses.replace(TPE, trials=50))
    assert name != study_name("runs/b/final", entry, family, TPE)
    assert study_name("runs/a/final", entry, family, GRID) != study_name("runs/a/final", entry, family, other_space)
