"""Generate the worked-example tables of the encoder-transformers topic.

The tables show two parts of the lineage that a reader cannot check by eye:
how DeBERTa turns a distance between two tokens into a row of its relative
position table, and which layers of ModernBERT see the whole text. Both are
computed with the rules the released code applies, ported line for line from
the HuggingFace `transformers` implementations, so the project does not need
PyTorch to regenerate them.

    uv run python experiments/reference/encoder-transformers/examples.py
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List

TABLES = Path(__file__).resolve().parent / "tables"

#: DeBERTa V1: the largest relative distance kept apart; `max_position_embeddings` in the config.
V1_MAX_DISTANCE = 512
#: DeBERTa V2 and V3: `position_buckets` in the config, rows per side of the relative table.
V2_BUCKETS = 256
#: DeBERTa V2 and V3: the largest distance the log scale is stretched over, `max_position_embeddings`.
V2_MAX_DISTANCE = 512
#: Distances between a query token and a key token shown in the bucket table.
DISTANCES: List[int] = [0, 1, 2, 64, 127, 128, 129, 130, 150, 200, 256, 300, 400, 511, 512, 600, 1000]
#: ModernBERT-base: `num_hidden_layers`, `global_attn_every_n_layers`, `local_attention`, the two RoPE bases.
MODERNBERT_LAYERS = 22
GLOBAL_EVERY = 3
LOCAL_WINDOW = 128
GLOBAL_ROPE_THETA = 160_000
LOCAL_ROPE_THETA = 10_000
#: The sequence length the visible-token column is computed for.
SEQUENCE_LENGTH = 8192


def v1_relative_index(distance: int, k: int = V1_MAX_DISTANCE) -> int:
    """Index of the relative position row in DeBERTa V1, equation 3 of arXiv:2006.03654.

    Args:
        distance: Query position minus key position.
        k: The maximum relative distance.

    Returns:
        int: A row index in `[0, 2k)`; distances beyond `k` share the two outer rows.
    """
    if distance <= -k:
        return 0
    if distance >= k:
        return 2 * k - 1
    return distance + k


def v2_log_bucket(distance: int, bucket_size: int = V2_BUCKETS, max_position: int = V2_MAX_DISTANCE) -> int:
    """Bucket of a relative distance in DeBERTa V2, as `make_log_bucket_position` computes it.

    Distances up to half the bucket count keep their own value; longer ones
    are placed on a log scale that reaches the last bucket at `max_position - 1`.

    Args:
        distance: Query position minus key position.
        bucket_size: Rows per side of the relative position table.
        max_position: The distance the log scale is stretched over.

    Returns:
        int: A signed bucket in `(-bucket_size, bucket_size)`.
    """
    mid = bucket_size // 2
    if abs(distance) <= mid:
        return distance
    log_pos = math.ceil(math.log(abs(distance) / mid) / math.log((max_position - 1) / mid) * (mid - 1)) + mid
    return int(math.copysign(log_pos, distance))


def v2_table_row(distance: int, bucket_size: int = V2_BUCKETS, max_position: int = V2_MAX_DISTANCE) -> int:
    """Row of the relative position table read for a distance, `c2p_pos` in the released code.

    Args:
        distance: Query position minus key position.
        bucket_size: Rows per side of the relative position table.
        max_position: The distance the log scale is stretched over.

    Returns:
        int: A row index in `[0, 2 * bucket_size)`.
    """
    return min(max(v2_log_bucket(distance, bucket_size, max_position) + bucket_size, 0), 2 * bucket_size - 1)


def write_table(name: str, rows: List[Dict[str, object]]) -> None:
    """Write one CSV table under `tables/`.

    Args:
        name: File stem.
        rows: Dictionaries sharing one key order.
    """
    TABLES.mkdir(exist_ok=True)
    with (TABLES / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def deberta_relative_buckets() -> None:
    """Table: how a distance between two tokens becomes a row of the relative position table, V1 against V2."""
    rows = []
    for distance in DISTANCES:
        rows.append(
            {
                "distance (query minus key)": distance,
                "V1 row, k = 512": v1_relative_index(distance),
                "V2 bucket": v2_log_bucket(distance),
                "V2 row, 256 buckets": v2_table_row(distance),
                "V2 distances below 512 sharing the row": sum(
                    1 for d in range(V2_MAX_DISTANCE) if v2_table_row(d) == v2_table_row(distance)
                ),
            }
        )
    write_table("deberta-v2-relative-buckets", rows)


def modernbert_attention_layout() -> None:
    """Table: for every ModernBERT-base layer, its attention span and RoPE base."""
    rows = []
    for layer in range(MODERNBERT_LAYERS):
        is_global = layer % GLOBAL_EVERY == 0
        rows.append(
            {
                "layer": layer + 1,
                "attention": "global" if is_global else "local",
                "RoPE base": GLOBAL_ROPE_THETA if is_global else LOCAL_ROPE_THETA,
                f"tokens a middle token sees at length {SEQUENCE_LENGTH}": SEQUENCE_LENGTH
                if is_global
                else LOCAL_WINDOW + 1,
            }
        )
    write_table("modernbert-attention-layout", rows)


if __name__ == "__main__":
    deberta_relative_buckets()
    modernbert_attention_layout()
    print(f"tables written to {TABLES}")
