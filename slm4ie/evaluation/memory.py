"""Read a local model's GPU memory from the startup log of its vllm-service container.

vllm logs, while it starts a model, how much GPU memory the weights took and how
large the KV cache it then allocated is, in GiB and in tokens. Those two lines give
the weight memory and the KV-cache memory per token for the model exactly as the
benchmark serves it: the same checkpoint, quantisation and KV-cache dtype.
`analysis.py` adds the KV cache the longest sequence of the model's answers needs
to the weights to get its peak memory (D7).
"""

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from slm4ie.evaluation.backends import get_backend
from slm4ie.evaluation.run import load_model_spec
from slm4ie.utils import mlflow as ml

# ===============================
# Logging
# ===============================

logger = logging.getLogger(__name__)

# ===============================
# Defaults
# ===============================

BYTES_PER_GIB = 2**30
MODEL_LOADING = re.compile(r"Model loading took ([\d.]+) GiB")
KV_CACHE_MEMORY = re.compile(r"Available KV cache memory: ([\d.]+) GiB")
KV_CACHE_TOKENS = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
# the vllm-service names each model's container after its served name
CONTAINER_NAME = "vllm-{model}"
# one short request makes the vllm-service start the model's container
LOAD_MESSAGES = [{"role": "user", "content": "Reply with an empty JSON object."}]
LOAD_SCHEMA = {"type": "object"}

# ===============================
# Main class
# ===============================


@dataclass(frozen=True)
class VllmMemory:
    """The GPU memory of one model as vllm serves it.

    Attributes:
        weight_bytes: GPU memory the loaded weights take.
        kv_bytes_per_token: KV-cache memory per token of context.
    """

    weight_bytes: int
    kv_bytes_per_token: float

    def peak_bytes(self, sequence_tokens: int) -> float:
        """Return the weights plus the KV cache of one sequence.

        Args:
            sequence_tokens: Input plus output tokens of the sequence.

        Returns:
            The memory in bytes.
        """
        return self.weight_bytes + self.kv_bytes_per_token * sequence_tokens


def parse_vllm_memory(log: str) -> VllmMemory:
    """Read the weight memory and the KV-cache memory per token from a vllm startup log.

    The last match of each line counts, so a log of several starts reads the latest.

    Args:
        log: The container's log text.

    Returns:
        The model's memory.

    Raises:
        ValueError: If the log lacks one of the three lines, or reports an empty KV cache.
    """
    values = {}
    for name, pattern in (("weights", MODEL_LOADING), ("kv_gib", KV_CACHE_MEMORY), ("kv_tokens", KV_CACHE_TOKENS)):
        matches = pattern.findall(log)
        if not matches:
            raise ValueError(f"vllm log has no line matching {pattern.pattern!r}")
        values[name] = float(matches[-1].replace(",", ""))
    if values["kv_tokens"] <= 0:
        raise ValueError(f"vllm log reports a KV cache of {values['kv_tokens']} tokens")
    return VllmMemory(
        weight_bytes=int(values["weights"] * BYTES_PER_GIB),
        kv_bytes_per_token=values["kv_gib"] * BYTES_PER_GIB / values["kv_tokens"],
    )


def read_container_log(container: str) -> str:
    """Return a running container's log, standard output and error together.

    Args:
        container: The container name.

    Returns:
        The log text.

    Raises:
        subprocess.CalledProcessError: If `docker logs` fails, for example when no such container runs.
    """
    result = subprocess.run(["docker", "logs", container], capture_output=True, text=True, check=True)
    return result.stdout + result.stderr


def measure_vllm_memory(
    model_key: str,
    models_config: Path,
    experiment: str | None = None,
    tags: dict[str, str] | None = None,
) -> VllmMemory:
    """Start a model on the vllm-service, read its memory from the container log, and log it.

    Args:
        model_key: The model's key in the model list.
        models_config: The model list.
        experiment: MLflow experiment to log one run to; no tracking when `None`.
        tags: Extra MLflow tags, such as `tier`, `branch`, `base_commit` and `type`.

    Returns:
        The model's memory.

    Raises:
        KeyError: If the model list has no vllm block for `model_key`.
        ValueError: If the container log lacks a startup line or reports an empty KV cache.
        subprocess.CalledProcessError: If `docker logs` fails, for example when no such container runs.
    """
    # start the container with one request, then read its startup log
    model, options = load_model_spec(models_config, model_key, "vllm")
    get_backend("vllm", model, **options).chat(LOAD_MESSAGES, LOAD_SCHEMA)
    container = CONTAINER_NAME.format(model=model)
    memory = parse_vllm_memory(read_container_log(container))

    # log the reading
    tracking = experiment is not None and ml.ensure_experiment(experiment)
    run_tags = {"backend": "vllm", "model": model, "model_key": model_key, **(tags or {})}
    with ml.mlflow_run(f"{model_key}-memory", enabled=tracking, tags=run_tags):
        ml.log_params({"backend": "vllm", "model": model, "container": container}, enabled=tracking)
        ml.log_metrics(
            {"weight_bytes": float(memory.weight_bytes), "kv_bytes_per_token": memory.kv_bytes_per_token},
            enabled=tracking,
        )
    logger.info(
        "model=%s weights=%.2f GiB kv_per_token=%.0f bytes",
        model_key,
        memory.weight_bytes / BYTES_PER_GIB,
        memory.kv_bytes_per_token,
    )
    return memory
