"""Score the exported model on every validation sample and bootstrap its masked-token loss.

Loads the HuggingFace export of the pilot run with transformers' `ModernBertForMaskedLM`, masks
15% of the tokens of each validation sample (the masking rate of the training evaluation, fixed
seed), and writes each sample's masked-token count and summed cross-entropy to
tables/validation-sample-loss.csv. It then resamples the samples with replacement and writes the
token-weighted mean loss with its 95% percentile interval to tables/bootstrap-final-loss.csv
(the noise band of Q1). Run it with the ModernSloBERT environment, which holds the model, the
validation shards and the streaming reader.

Args:
    --model         Exported HuggingFace model directory (ModernSloBERT models/<run_name>/hf).
    --validation    Folder of validation shards (ModernSloBERT data/corpora/<dataset>/<tokenizer>/validation).
    --resamples     Bootstrap resamples (default: 1000).
    --batch-size    Samples per forward pass (default: 64).

Example:
    cd ~/code/ModernSloBERT && CUDA_VISIBLE_DEVICES=2 .venv/bin/python \
        ~/code/SLM4IE/experiments/programmes/slovene-encoders/modernbert-pilot-slovenian/bootstrap_loss.py \
        --model models/modernbert-base_slm4ie-pretrain_bpe-32000/hf \
        --validation data/corpora/slm4ie-pretrain/bpe-32000/validation

"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from streaming import LocalDataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, ModernBertForMaskedLM, PreTrainedTokenizerBase

# ===============================
# Constants
# ===============================

RECORD_DIR = Path(__file__).resolve().parent
MASKING_RATE = 0.15  # the evaluation masking rate of the run, eval_loader.dataset.mlm_probability
IGNORE_INDEX = -100  # the label the HuggingFace collator gives unmasked tokens
INTERVAL = (2.5, 97.5)  # percentiles of the 95% interval
SEED = 17

# ===============================
# Helper functions
# ===============================


def score_samples(
    model: ModernBertForMaskedLM,
    tokenizer: PreTrainedTokenizerBase,
    dataset: LocalDataset,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """Score every validation sequence with one fixed mask draw each.

    Args:
        model: The exported masked language model, in evaluation mode.
        tokenizer: Its tokenizer, which the collator masks with.
        dataset: The validation sequences.
        batch_size: Sequences per forward pass.
        device: Where the model runs.

    Returns:
        One row per sequence: its masked-token count and its summed masked-token cross-entropy.
    """
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=MASKING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=IGNORE_INDEX)
    rows = []
    for start in range(0, len(dataset), batch_size):
        samples = [
            {"input_ids": dataset[i]["input_ids"].astype(np.int64).tolist()}
            for i in range(start, min(start + batch_size, len(dataset)))
        ]
        batch = {key: value.to(device) for key, value in collator(samples).items()}
        with torch.no_grad():
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        losses = loss_fn(logits.transpose(1, 2).float(), batch["labels"])
        masked = batch["labels"] != IGNORE_INDEX
        rows.append(torch.stack([masked.sum(dim=1).float(), (losses * masked).sum(dim=1)], dim=1).cpu().numpy())
    return np.concatenate(rows)


def bootstrap(per_sample: np.ndarray, resamples: int, seed: int) -> tuple[float, float, float]:
    """Resample the sequences with replacement and take the token-weighted mean loss of each draw.

    Args:
        per_sample: The output of `score_samples`.
        resamples: How many draws.
        seed: The seed of the draws.

    Returns:
        The mean loss over all sequences and the 2.5th and 97.5th percentiles of the draws.
    """
    rng = np.random.default_rng(seed)
    counts, sums = per_sample[:, 0], per_sample[:, 1]
    draws = rng.integers(0, len(per_sample), size=(resamples, len(per_sample)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return (
        float(sums.sum() / counts.sum()),
        float(np.percentile(means, INTERVAL[0])),
        float(np.percentile(means, INTERVAL[1])),
    )


# ===============================
# Main function
# ===============================


def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, type=Path, help="Exported HuggingFace model directory")
    parser.add_argument("--validation", required=True, type=Path, help="Folder of validation shards")
    parser.add_argument("--resamples", type=int, default=1000, help="Bootstrap resamples (default: 1000)")
    parser.add_argument("--batch-size", type=int, default=64, help="Samples per forward pass (default: 64)")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Score the validation split with the exported model and write both tables.

    Args:
        args: The parsed arguments.

    Raises:
        FileNotFoundError: If the model or the validation shards are missing.
    """
    if not (args.model / "config.json").is_file():
        raise FileNotFoundError(f"no HuggingFace model in {args.model}")
    if not args.validation.is_dir():
        raise FileNotFoundError(f"validation shards missing: {args.validation}")

    # load the model, the tokenizer and the validation samples
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = ModernBertForMaskedLM.from_pretrained(args.model, attn_implementation="sdpa").to(device).eval()
    dataset = LocalDataset(local=str(args.validation))

    # score every sample, then bootstrap over samples
    per_sample = score_samples(model, tokenizer, dataset, args.batch_size, device)
    loss, low, high = bootstrap(per_sample, args.resamples, SEED)

    # write both tables
    tables = RECORD_DIR / "tables"
    tables.mkdir(exist_ok=True)
    with open(tables / "validation-sample-loss.csv", "w", newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample", "masked_tokens", "loss_sum"])
        writer.writerows([i, int(c), f"{s:.4f}"] for i, (c, s) in enumerate(per_sample))
    with open(tables / "bootstrap-final-loss.csv", "w", newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow(["samples", "masked_tokens", "loss", "ci_low", "ci_high", "resamples", "seed"])
        writer.writerow(
            [
                len(per_sample),
                int(per_sample[:, 0].sum()),
                f"{loss:.4f}",
                f"{low:.4f}",
                f"{high:.4f}",
                args.resamples,
                SEED,
            ]
        )
    print(f"final validation loss {loss:.4f}, 95% interval [{low:.4f}, {high:.4f}] over {len(per_sample)} samples")


if __name__ == "__main__":
    main(get_args())
