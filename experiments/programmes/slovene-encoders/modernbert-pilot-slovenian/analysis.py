"""Tables and figures of the ModernBERT pilot, from the run tables imported from ModernSloBERT.

The run tables `tables/run-eval.csv`, `tables/run-train.csv` and `tables/run-system.csv` were
exported from the local MLflow store of run 0e0b096c by ModernSloBERT's
`scripts/report_training.py` (D1); this script never reads MLflow. `tables` derives the data table
of each figure and counts spikes by the thresholds of Q2. `data` tables the corpus from its
conversion manifest, copied from ModernSloBERT without local paths. `figures` draws both figures
from the tables alone.

Run it with:

    uv run --group analysis python experiments/programmes/slovene-encoders/modernbert-pilot-slovenian/analysis.py data
    uv run --group analysis python experiments/programmes/slovene-encoders/modernbert-pilot-slovenian/analysis.py tables
    uv run --group analysis python experiments/programmes/slovene-encoders/modernbert-pilot-slovenian/analysis.py figures

"""

import argparse
import csv
import json
import sys
from pathlib import Path

EXPERIMENTS = next(p for p in Path(__file__).resolve().parents if p.name == "experiments")
TOOLS = EXPERIMENTS / "tools"
sys.path.insert(0, str(TOOLS if (TOOLS / "create_assets.py").is_file() else EXPERIMENTS))

from create_assets import FULL_WIDTH, Claim, save_figure, save_table
from datachart.charts import LineChart
from datachart.constants import SHOW_GRID
from datachart.utils import Grid

# ===============================
# Constants
# ===============================

RECORD_DIR = Path(__file__).resolve().parent
TABLES = RECORD_DIR / "tables"
FIGURES = RECORD_DIR / "figures"
MANIFEST = RECORD_DIR / "configs" / "corpus-manifest.json"
DECAY_START_B = 60.14  # tokens seen (billions) when the learning rate starts to fall (D3)
GRAD_SPIKE_RATIO = 3.0  # a 10-batch gradient-norm reading above 3x its bin mean is a spike (Q2)
LOSS_SPIKE_EXCESS = 0.5  # a batch's training loss 0.5 above its bin mean is a spike (Q2)

# ===============================
# Helper functions
# ===============================


def read_rows(name: str) -> list[dict[str, float]]:
    """Return a table under tables/ as rows of floats, empty cells as None."""
    path = TABLES / name
    if not path.is_file():
        raise FileNotFoundError(f"table missing: {path}")
    with open(path, encoding="utf8") as f:
        return [{key: float(value) if value else None for key, value in row.items()} for row in csv.DictReader(f)]


def spike_rows(train: list[dict[str, float]]) -> list[dict[str, object]]:
    """Return one row per 500-batch bin with its spike measures and whether each threshold is crossed."""
    rows = []
    for row in train:
        ratio = row["grad_norm_max"] / row["grad_norm"]
        excess = row["train_loss_max"] - row["train_loss"]
        rows.append(
            {
                "batch": int(row["batch"]),
                "tokens_b": round(row["tokens_b"], 2),
                "grad_norm_ratio": round(ratio, 3),
                "loss_excess": round(excess, 3),
                "grad_spike": int(ratio > GRAD_SPIKE_RATIO),
                "loss_spike": int(excess > LOSS_SPIKE_EXCESS),
            }
        )
    return rows


def dataset_tables(manifest: dict) -> None:
    """Write the summary, fields and per-source statistics tables of the corpus."""
    totals = manifest["totals"]
    save_table(
        [
            {"statistic": "documents", "value": totals["train"]["documents"], "note": "training split"},
            {"statistic": "sources", "value": len(manifest["counts"]), "note": ""},
            {"statistic": "tokens", "value": totals["train"]["tokens"], "note": "training split"},
            {"statistic": "samples", "value": totals["train"]["samples"], "note": "1,024-token sequences"},
            {
                "statistic": "validation documents",
                "value": totals["validation"]["documents"],
                "note": "held out by document",
            },
            {"statistic": "validation tokens", "value": totals["validation"]["tokens"], "note": ""},
        ],
        TABLES / "dataset-slm4ie-pretrain-summary.csv",
    )
    save_table(
        [
            {
                "field": "input_ids",
                "type": "numeric",
                "distinct": "",
                "values": "",
            },
            {
                "field": "source",
                "type": "categorical",
                "distinct": len(manifest["counts"]),
                "values": ", ".join(sorted(manifest["counts"])[:5]),
            },
            {"field": "split", "type": "categorical", "distinct": 2, "values": "train, validation"},
        ],
        TABLES / "dataset-slm4ie-pretrain-fields.csv",
    )
    rows = [
        {
            "source": source,
            "documents": counts["train"]["documents"],
            "tokens": counts["train"]["tokens"],
            "validation_documents": counts["validation"]["documents"],
            "validation_tokens": counts["validation"]["tokens"],
        }
        for source, counts in manifest["counts"].items()
    ]
    save_table(sorted(rows, key=lambda row: -row["documents"]), TABLES / "dataset-slm4ie-pretrain-statistics.csv")


def curve(rows: list[dict[str, float]], column: str, scale: float = 1.0) -> list[dict[str, float]]:
    """Return the points of one line, tokens seen on x."""
    return [{"x": row["tokens_b"], "y": row[column] * scale} for row in rows if row[column] is not None]


def decay_span() -> dict:
    """Return the shaded span of the learning-rate decay."""
    return {"xmin": DECAY_START_B, "xmax": 67.0, "label": "learning-rate decay"}


def validation_figure(evals: list[dict[str, float]]) -> Grid:
    """Draw validation loss and masked-token accuracy against tokens seen, the first evaluation left out."""
    shown = evals[1:]
    charts = [
        LineChart(
            curve(shown, "loss"),
            title="Validation loss",
            xlabel="Tokens seen (billions)",
            ylabel="Cross-entropy (nats)",
            show_grid=SHOW_GRID.Y,
            vspans=decay_span(),
        ),
        LineChart(
            curve(shown, "accuracy", 100),
            title="Masked-token accuracy",
            xlabel="Tokens seen (billions)",
            ylabel="Accuracy (%)",
            show_grid=SHOW_GRID.Y,
            vspans=decay_span(),
        ),
    ]
    return Grid(charts, max_cols=2, title="Validation loss and accuracy", figsize=FULL_WIDTH)


def dynamics_figure(train: list[dict[str, float]]) -> Grid:
    """Draw training loss and gradient norm, bin mean and maximum, against tokens seen; warmup bins left out of the loss."""
    after_warmup = [row for row in train if row["tokens_b"] > 4]
    charts = [
        LineChart(
            [curve(after_warmup, "train_loss"), curve(after_warmup, "train_loss_max")],
            subtitle=["bin mean", "bin maximum"],
            title="Training loss, 30% masked",
            xlabel="Tokens seen (billions)",
            ylabel="Cross-entropy (nats)",
            show_grid=SHOW_GRID.Y,
            vspans=decay_span(),
        ),
        LineChart(
            [curve(train, "grad_norm"), curve(train, "grad_norm_max")],
            subtitle=["bin mean", "bin maximum"],
            title="Gradient norm",
            xlabel="Tokens seen (billions)",
            ylabel="L2 norm (unitless)",
            show_grid=SHOW_GRID.Y,
            vspans=decay_span(),
            show_legend=True,
        ),
    ]
    return Grid(charts, max_cols=2, title="Training dynamics per 500 batches", figsize=FULL_WIDTH)


# ===============================
# Main function
# ===============================


def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "step",
        choices=["data", "tables", "figures"],
        help="data: table the corpus; tables: derive the figure tables; figures: draw them",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    if args.step == "data":
        if not MANIFEST.is_file():
            raise FileNotFoundError(f"corpus manifest missing: {MANIFEST}")
        dataset_tables(json.loads(MANIFEST.read_text(encoding="utf8")))
        return

    if args.step == "tables":
        evals, train = read_rows("run-eval.csv"), read_rows("run-train.csv")
        save_table(
            [
                {"batch": int(r["batch"]), "tokens_b": r["tokens_b"], "loss": r["loss"], "accuracy": r["accuracy"]}
                for r in evals
            ],
            TABLES / "pilot-validation-loss-by-tokens.csv",
        )
        save_table(
            [{"batch": int(r["batch"]), **{k: r[k] for k in r if k != "batch"}} for r in train],
            TABLES / "pilot-training-dynamics-by-tokens.csv",
        )
        save_table(spike_rows(train), TABLES / "spike-count-by-bin.csv")
        return

    FIGURES.mkdir(exist_ok=True)
    evals = read_rows("pilot-validation-loss-by-tokens.csv")
    train = read_rows("pilot-training-dynamics-by-tokens.csv")
    save_figure(
        validation_figure(evals),
        FIGURES / "pilot-validation-loss-by-tokens.svg",
        claim=Claim("Q1", "masked-token cross-entropy"),
        width="full-width",
    )
    save_figure(
        dynamics_figure(train),
        FIGURES / "pilot-training-dynamics-by-tokens.svg",
        claim=Claim("Q2", "gradient norm"),
        width="full-width",
    )


if __name__ == "__main__":
    main(get_args())
