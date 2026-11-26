"""Tests for the ModernBERT pilot's spike count and schedule bounds in its analysis.py."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "programmes"
    / "slovene-encoders"
    / "modernbert-pilot-slovenian"
    / "analysis.py"
)


def _load() -> ModuleType:
    """Import the experiment's analysis.py, whose folder name is no package name.

    Returns:
        The loaded module.
    """
    spec = importlib.util.spec_from_file_location("modernbert_pilot_analysis", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


analysis = _load()


def _bin(batch: int, grad_ratio: float = 1.2, loss_excess: float = 0.05) -> dict:
    """Build one training-table row with the given maximum-to-mean gradient ratio and loss excess.

    Args:
        batch: The bin's last batch.
        grad_ratio: The bin's largest gradient-norm reading over its mean.
        loss_excess: The bin's largest batch loss minus its mean.

    Returns:
        A row as `read_rows` returns it.
    """
    return {
        "batch": float(batch),
        "tokens_b": batch / 200,
        "train_loss": 1.5,
        "train_loss_max": 1.5 + loss_excess,
        "grad_norm": 0.3,
        "grad_norm_max": 0.3 * grad_ratio,
    }


def test_one_planted_gradient_spike_counts_once() -> None:
    """A series with one bin above three times its mean counts exactly one gradient spike."""
    rows = analysis.spike_rows([_bin(500), _bin(1000, grad_ratio=4.0), _bin(1500)])
    assert analysis.count_spikes(rows) == {"grad_spikes": 1, "loss_spikes": 0}


def test_loss_excess_at_the_threshold_is_no_spike() -> None:
    """A loss maximum exactly 0.5 above its mean does not cross the threshold."""
    rows = analysis.spike_rows([_bin(500, loss_excess=0.5), _bin(1000, loss_excess=0.51)])
    assert analysis.count_spikes(rows)["loss_spikes"] == 1


def test_schedule_bounds_read_the_config_and_manifest() -> None:
    """The decay starts t_decay before the end, and the fourth pass after three passes of training tokens."""
    config = {"max_duration": "66_836_000_000tok", "scheduler": {"t_decay": "6_700_000_000tok"}}
    manifest = {"totals": {"train": {"tokens": 16_709_182_973}}}
    bounds = analysis.schedule_bounds(config, manifest)
    assert bounds["end"] == pytest.approx(66.836)
    assert bounds["decay"] == pytest.approx(60.136)
    assert bounds["fourth_pass"] == pytest.approx(50.128, abs=1e-3)
