"""Tests for slm4ie.evaluation.memory."""

import pytest

from slm4ie.evaluation.memory import VllmMemory, parse_vllm_memory

# ===============================
# Test Data
# ===============================

GIB = 2**30

# the three lines as vllm 0.31 writes them while starting Qwen3.5 9B
STARTUP_LOG = """\
(EngineCore pid=247) INFO 11-06 22:57:42 [model_runner.py:407] Model loading took 10.36 GiB memory and 14.9 seconds
(EngineCore pid=247) INFO 11-06 23:00:23 [gpu_worker.py:692] Available KV cache memory: 1.68 GiB
(EngineCore pid=247) INFO 11-06 23:00:23 [kv_cache_utils.py:2464] GPU KV cache size: 38,725 tokens, Maximum concurrency
"""

# ===============================
# parse_vllm_memory
# ===============================


def test_reads_weights_and_kv_memory_per_token():
    """The weights come from the loading line, the per-token KV memory from the two cache lines."""
    memory = parse_vllm_memory(STARTUP_LOG)

    assert memory.weight_bytes == int(10.36 * GIB)
    assert memory.kv_bytes_per_token == pytest.approx(1.68 * GIB / 38725)


def test_reads_the_latest_start_of_a_log_with_two():
    """A container started twice reports the second start."""
    restarted = STARTUP_LOG + STARTUP_LOG.replace("10.36", "12.77").replace("38,725", "7,479")

    memory = parse_vllm_memory(restarted)

    assert memory.weight_bytes == int(12.77 * GIB)
    assert memory.kv_bytes_per_token == pytest.approx(1.68 * GIB / 7479)


def test_a_log_without_the_loading_line_is_rejected():
    """A log missing one line names the line it missed."""
    with pytest.raises(ValueError, match="Model loading"):
        parse_vllm_memory(STARTUP_LOG.splitlines()[1])


def test_an_empty_kv_cache_is_rejected():
    """A KV cache of zero tokens has no per-token memory."""
    with pytest.raises(ValueError, match="0.0 tokens"):
        parse_vllm_memory(STARTUP_LOG.replace("38,725", "0"))


# ===============================
# VllmMemory
# ===============================


def test_peak_adds_one_sequence_of_kv_cache_to_the_weights():
    """The peak is the weights plus the KV cache of the sequence."""
    memory = VllmMemory(weight_bytes=1000, kv_bytes_per_token=2.5)

    assert memory.peak_bytes(400) == 2000
