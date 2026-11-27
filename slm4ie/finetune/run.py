"""Fine-tune one pretrained encoder on one task registry entry and score it.

`finetune` trains the encoder with its family's task head on the entry's train
split, scores val after every epoch and keeps the weights of the best epoch,
then scores test with them. A hyperparameter search instead skips test, so a
trial never sees it, and watches the val scores of each epoch through a
callback that may stop the run. It returns the test scores with the costs of the
run: the parameter count, the peak GPU memory of training and of inference,
and the wall time of fine-tuning. It logs nothing to MLflow; a caller wraps
it in a tracked run.

Peak GPU memory is `torch.cuda.max_memory_allocated` since the start of each
phase, weights included, as the evaluation package's transformers backend
measures it; a CPU run has none.

`unwritten_splits` and `missing_splits` tell a caller, before training, whether
an entry's role writes the train, val and test splits and whether their files
are on disk.
"""

import logging
import math
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerFast, get_linear_schedule_with_warmup, set_seed

from slm4ie.data.tasks.config import TaskEntry, TasksRoots, resolve_output_dir
from slm4ie.data.tasks.run import target_splits
from slm4ie.data.tasks.splits import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL, iter_jsonl, outputs_for_splits
from slm4ie.finetune.config import FinetuneSettings
from slm4ie.finetune.heads import TaskHead, get_head
from slm4ie.finetune.heads.base import Feature

# ===============================
# Logging
# ===============================

logger = logging.getLogger(__name__)

# ===============================
# Constants
# ===============================

# the splits a fine-tuning run reads: train to fit, val to pick the epoch, test to score
FINETUNE_SPLITS = (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)

# ===============================
# Result
# ===============================


@dataclass(frozen=True)
class FinetuneResult:
    """The scores and costs of one fine-tuning run.

    Attributes:
        scores: Test scores of the best epoch's weights, by name; empty when test was skipped.
        val_scores: Val scores of the best epoch, by name.
        best_epoch: The epoch whose weights were kept, counted from 1.
        parameters: Parameters of the model, task head included.
        train_peak_gpu_memory_bytes: Peak GPU memory of training; `None` on a CPU.
        inference_peak_gpu_memory_bytes: Peak GPU memory of scoring test; `None` on a CPU
            or when test was skipped.
        train_seconds: Wall time of training, val scoring included.
    """

    scores: dict[str, float]
    val_scores: dict[str, float]
    best_epoch: int
    parameters: int
    train_peak_gpu_memory_bytes: int | None
    inference_peak_gpu_memory_bytes: int | None
    train_seconds: float


# ===============================
# Entry point
# ===============================


def finetune(
    entry: TaskEntry,
    encoder: str,
    settings: FinetuneSettings,
    roots: TasksRoots,
    *,
    device: str | None = None,
    score_test: bool = True,
    on_epoch: Callable[[int, Mapping[str, float]], None] | None = None,
) -> FinetuneResult:
    """Fine-tune an encoder on an entry, keep the best epoch on val and score test.

    Args:
        entry: The task registry entry; its role must keep a train split.
        encoder: The encoder, by HuggingFace name or local path.
        settings: The run's settings.
        roots: The task registry roots; the splits are read under `roots.tasks`.
        device: The torch device to train on; the first GPU when one is visible, else the CPU.
        score_test: Score test with the best epoch's weights; a search trial turns it off.
        on_epoch: Called after each epoch with the epoch, counted from 1, and its val
            scores; an exception it raises, such as a pruned trial's, stops the run.

    Returns:
        The test scores and the costs of the run.

    Raises:
        ValueError: If the entry writes no train, val or test split, naming the
            entry and its role, or an example's label is not among its labels.
        KeyError: If no task head is registered for the entry's family.
        FileNotFoundError: If a split file is missing.
    """
    if unwritten := unwritten_splits(entry):
        raise ValueError(
            f"entry '{entry.key}' with role '{entry.role}' writes no {unwritten} split; it cannot be fine-tuned"
        )
    if not entry.labels:
        raise ValueError(f"entry '{entry.key}' declares no labels; fine-tuning needs its label set")

    # read the splits
    head = get_head(entry.task)
    paths = outputs_for_splits(resolve_output_dir(entry, roots), entry.splits)
    examples = {split: list(iter_jsonl(paths[split])) for split in FINETUNE_SPLITS}

    # load the tokenizer and the model, then encode every split
    set_seed(settings.seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = _load_tokenizer(encoder)
    if tokenizer.pad_token_id is None:
        raise ValueError(f"tokenizer of '{encoder}' has no padding token")
    network = head.load_model(encoder, entry.labels).to(device)
    parameters = sum(p.numel() for p in network.parameters())
    features = {
        split: head.encode(split_examples, tokenizer, entry.labels, settings.max_length)
        for split, split_examples in examples.items()
    }

    # train, keeping the weights of the best epoch on val
    _reset_peak_memory(device)
    started = time.perf_counter()
    best_state, best_epoch, val_scores = _train(
        network, head, features, examples, entry.labels, tokenizer, settings, on_epoch
    )
    train_seconds = time.perf_counter() - started
    train_peak = _peak_memory(device)

    # score test with the best weights, unless a search trial skips it
    scores, inference_peak = {}, None
    if score_test:
        network.load_state_dict(best_state)
        _reset_peak_memory(device)
        logits = _predict(network, head, features[SPLIT_TEST], tokenizer.pad_token_id, settings)
        scores = head.score(examples[SPLIT_TEST], features[SPLIT_TEST], logits, entry.labels)
        inference_peak = _peak_memory(device)
    logger.info("entry=%s encoder=%s best_epoch=%d val=%s test=%s", entry.key, encoder, best_epoch, val_scores, scores)
    return FinetuneResult(
        scores=scores,
        val_scores=val_scores,
        best_epoch=best_epoch,
        parameters=parameters,
        train_peak_gpu_memory_bytes=train_peak,
        inference_peak_gpu_memory_bytes=inference_peak,
        train_seconds=train_seconds,
    )


# ===============================
# Split checks
# ===============================


def unwritten_splits(entry: TaskEntry) -> list[str]:
    """Return the fine-tuning splits an entry's role does not write.

    Args:
        entry: The task registry entry.

    Returns:
        The names among train, val and test that the entry writes no file for;
        empty when the entry can be fine-tuned.
    """
    written = target_splits(entry)
    return [split for split in FINETUNE_SPLITS if split not in written]


def missing_splits(entry: TaskEntry, roots: TasksRoots) -> list[Path]:
    """Return the train, val and test split files of an entry that are not on disk.

    Args:
        entry: A task registry entry that writes every fine-tuning split.
        roots: The task registry roots; the splits are read under `roots.tasks`.

    Returns:
        The missing split files, in train, val, test order.
    """
    splits = {split: entry.splits[split] for split in FINETUNE_SPLITS}
    paths = outputs_for_splits(resolve_output_dir(entry, roots), splits)
    return [path for path in paths.values() if not path.is_file()]


# ===============================
# Helpers
# ===============================


def _load_tokenizer(encoder: str) -> Any:
    """Load an encoder's tokenizer, or its `tokenizer.json` alone when the mapped class fails to build."""
    try:
        return AutoTokenizer.from_pretrained(encoder)
    except ValueError as exc:
        # transformers 5 maps SloBERTa's BPE tokenizer.json to CamembertTokenizer
        logger.warning("encoder=%s tokenizer failed to load (%s); loading its tokenizer.json directly", encoder, exc)
        return PreTrainedTokenizerFast.from_pretrained(encoder)


def _train(
    network: torch.nn.Module,
    head: TaskHead,
    features: Mapping[str, list[Feature]],
    examples: Mapping[str, list[dict[str, Any]]],
    labels: Sequence[Any],
    tokenizer: Any,
    settings: FinetuneSettings,
    on_epoch: Callable[[int, Mapping[str, float]], None] | None,
) -> tuple[dict[str, torch.Tensor], int, dict[str, float]]:
    """Train for every epoch, score val after each and pass the scores to `on_epoch`.

    Returns:
        The best epoch's weights on the CPU, the epoch counted from 1, and its val scores.
    """
    # build the optimizer and the warmup schedule
    train = features[SPLIT_TRAIN]
    steps_per_epoch = math.ceil(math.ceil(len(train) / settings.batch_size) / settings.gradient_accumulation)
    total_steps = steps_per_epoch * settings.epochs
    decay = [p for p in network.parameters() if p.ndim >= 2]
    no_decay = [p for p in network.parameters() if p.ndim < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": settings.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=settings.learning_rate,
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, int(settings.warmup_ratio * total_steps), total_steps)
    generator = torch.Generator().manual_seed(settings.seed)

    best_state, best_epoch, best_scores = {}, 0, {}
    for epoch in range(1, settings.epochs + 1):
        # take one pass over the shuffled train split
        network.train()
        order = torch.randperm(len(train), generator=generator).tolist()
        batches = list(_batches([train[i] for i in order], settings.batch_size))
        for index, batch in enumerate(tqdm(batches, desc=f"epoch {epoch}", leave=False)):
            inputs = {k: v.to(network.device) for k, v in head.collate(batch, tokenizer.pad_token_id).items()}
            with _autocast(network.device, settings.precision):
                loss = network(**inputs).loss / settings.gradient_accumulation
            loss.backward()

            # step at every accumulation boundary and after the last batch
            if (index + 1) % settings.gradient_accumulation == 0 or index + 1 == len(batches):
                torch.nn.utils.clip_grad_norm_(network.parameters(), settings.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # score val and keep the weights of a new best
        logits = _predict(network, head, features[SPLIT_VAL], tokenizer.pad_token_id, settings)
        scores = head.score(examples[SPLIT_VAL], features[SPLIT_VAL], logits, labels)
        logger.info("epoch=%d val=%s", epoch, scores)
        if not best_state or scores[head.primary_metric] > best_scores[head.primary_metric]:
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in network.state_dict().items()}
            best_epoch, best_scores = epoch, scores
        if on_epoch is not None:
            on_epoch(epoch, scores)
    return best_state, best_epoch, best_scores


def _predict(
    network: torch.nn.Module, head: TaskHead, features: list[Feature], pad_id: int, settings: FinetuneSettings
) -> list[np.ndarray]:
    """Return the model's logits for every feature, in order, as float32 arrays."""
    network.eval()
    logits = []
    with torch.no_grad():
        for batch in _batches(features, settings.eval_batch_size):
            inputs = {k: v.to(network.device) for k, v in head.collate(batch, pad_id).items() if k != "labels"}
            with _autocast(network.device, settings.precision):
                output = network(**inputs).logits
            logits.extend(output.float().cpu().numpy())
    return logits


def _batches(items: list[Feature], size: int) -> Iterator[list[Feature]]:
    """Yield consecutive batches of at most `size` items."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _autocast(device: torch.device, precision: str) -> AbstractContextManager:
    """Return bfloat16 autocast on a GPU when the precision asks for it, else a no-op context."""
    return torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda" and precision == "bf16"
    )


def _reset_peak_memory(device: torch.device) -> None:
    """Start a new peak-memory measurement from the memory in use now; a no-op off the GPU."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory(device: torch.device) -> int | None:
    """Return the peak GPU memory since the last reset, or `None` off the GPU."""
    return int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
