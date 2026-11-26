"""Tables and figures of the ModernBERT pilot, from the run tables imported from ModernSloBERT.

The run tables `tables/run-eval.csv`, `tables/run-train.csv` and `tables/run-system.csv` were
exported from the local MLflow store of run 0e0b096c by ModernSloBERT's
`scripts/report_training.py` (D1); this script never reads MLflow. `data` tables the corpus from
its conversion manifest and the training settings from the resolved config, both copied from
ModernSloBERT without local paths. `tables` derives the data table of each figure. `spikes` counts
spikes per 500-batch bin by the thresholds of Q2 and prints the totals. `figures` draws the three
figures from the tables alone; the close-up also needs `tables/bootstrap-final-loss.csv`, which
`bootstrap_loss.py` writes.

Run it with:

    uv run --group analysis python experiments/programmes/slovene-encoders/modernbert-pilot-slovenian/analysis.py data
    uv run --group analysis python experiments/programmes/slovene-encoders/modernbert-pilot-slovenian/analysis.py tables
    uv run --group analysis python experiments/programmes/slovene-encoders/modernbert-pilot-slovenian/analysis.py spikes
    uv run --group analysis python experiments/programmes/slovene-encoders/modernbert-pilot-slovenian/analysis.py figures

"""

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

EXPERIMENTS = next(p for p in Path(__file__).resolve().parents if p.name == "experiments")
TOOLS = EXPERIMENTS / "tools"
sys.path.insert(0, str(TOOLS if (TOOLS / "create_assets.py").is_file() else EXPERIMENTS))

from create_assets import FULL_WIDTH, ONE_COLUMN, Claim, save_figure, save_table  # noqa: E402
from datachart.charts import LineChart  # noqa: E402
from datachart.constants import SHOW_GRID  # noqa: E402
from datachart.utils import Grid  # noqa: E402

# ===============================
# Constants
# ===============================

RECORD_DIR = Path(__file__).resolve().parent
TABLES = RECORD_DIR / "tables"
FIGURES = RECORD_DIR / "figures"
MANIFEST = RECORD_DIR / "configs" / "corpus-manifest.json"
TRAIN_CONFIG = RECORD_DIR / "configs" / "pretrain-modernbert-base.yaml"
TRAINING_KEYS = (
    "max_duration",
    "optimizer.lr",
    "global_train_batch_size",
    "mlm_probability",
    "scheduler.t_warmup",
    "scheduler.t_decay",
    "eval_interval",
    "eval_loader.dataset.mlm_probability",
)
ZOOM_FROM_B = 45.0  # the close-up starts here, before the fourth pass
WARMUP_END_B = 4.0  # loss bins before this many billion tokens are left out: they fall from 10.5
PASSES = 4
GRAD_SPIKE_RATIO = 3.0  # a 10-batch gradient-norm reading above 3x its bin mean is a spike (Q2)
LOSS_SPIKE_EXCESS = 0.5  # a batch's training loss 0.5 above its bin mean is a spike (Q2)

# ===============================
# Helper functions
# ===============================


def read_rows(name: str) -> list[dict[str, float | None]]:
    """Read a table under tables/ as rows of floats.

    Args:
        name: The table's file name.

    Returns:
        One mapping per row, column to value, an empty cell as None.

    Raises:
        FileNotFoundError: If the table does not exist.
    """
    path = TABLES / name
    if not path.is_file():
        raise FileNotFoundError(f"table missing: {path}")
    with open(path, encoding="utf8") as f:
        return [{key: float(value) if value else None for key, value in row.items()} for row in csv.DictReader(f)]


def spike_rows(train: list[dict[str, float | None]]) -> list[dict[str, object]]:
    """Flag the 500-batch bins that cross a spike threshold of Q2.

    Args:
        train: Rows of the training table, each with the bin's mean and maximum gradient norm and loss.

    Returns:
        One row per bin with the maximum-to-mean gradient ratio, the loss excess and a 0/1 flag for each.
    """
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


def count_spikes(rows: list[dict[str, object]]) -> dict[str, int]:
    """Count the bins flagged by each spike threshold.

    Args:
        rows: The rows of `spike_rows`.

    Returns:
        The number of bins with a gradient spike and with a loss spike.
    """
    return {
        "grad_spikes": sum(int(r["grad_spike"]) for r in rows),
        "loss_spikes": sum(int(r["loss_spike"]) for r in rows),
    }


def schedule_bounds(config: dict, manifest: dict) -> dict[str, float]:
    """Read where the fourth pass, the decay and training start and end, in billions of tokens.

    Args:
        config: The run's resolved training config.
        manifest: The corpus conversion manifest.

    Returns:
        The tokens seen at the fourth pass's start, the decay's start and the end of training.
    """
    end = int(str(config["max_duration"]).removesuffix("tok")) / 1e9
    decay = int(str(config["scheduler"]["t_decay"]).removesuffix("tok")) / 1e9
    pass_tokens = manifest["totals"]["train"]["tokens"] / 1e9
    return {"fourth_pass": (PASSES - 1) * pass_tokens, "decay": end - decay, "end": end}


def training_settings(config: dict) -> list[dict[str, object]]:
    """Read the training settings the record cites from the run's resolved config.

    Args:
        config: The run's resolved training config.

    Returns:
        One row per setting, its dotted key and its value as written.
    """
    rows = []
    for key in TRAINING_KEYS:
        value = config
        for part in key.split("."):
            value = value[part]
        rows.append({"key": key, "value": str(value)})
    return rows


def dataset_tables(manifest: dict) -> None:
    """Write the summary, fields and per-source statistics tables of the corpus.

    Args:
        manifest: The corpus conversion manifest.
    """
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


def curve(rows: list[dict[str, float | None]], column: str, scale: float = 1.0) -> list[dict[str, float]]:
    """Turn one column into the points of a line, tokens seen on x.

    Args:
        rows: Table rows with a `tokens_b` column.
        column: The column drawn on y.
        scale: Factor applied to every y value.

    Returns:
        One point per row whose value is not empty.
    """
    return [{"x": row["tokens_b"], "y": row[column] * scale} for row in rows if row[column] is not None]


def decay_span(bounds: dict[str, float]) -> dict:
    """Build the shaded span of the learning-rate decay.

    Args:
        bounds: The output of `schedule_bounds`.

    Returns:
        A datachart vertical span from the decay's start to the end of training.
    """
    return {"xmin": bounds["decay"], "xmax": bounds["end"], "label": "learning-rate decay"}


def validation_figure(evals: list[dict[str, float | None]], bounds: dict[str, float]) -> Grid:
    """Draw validation loss and masked-token accuracy against tokens seen, the first evaluation left out.

    Args:
        evals: Rows of the validation table.
        bounds: The output of `schedule_bounds`.

    Returns:
        The two-panel figure.
    """
    shown = evals[1:]
    charts = [
        LineChart(
            curve(shown, "loss"),
            title="Validation loss",
            xlabel="Tokens seen (billions)",
            ylabel="Cross-entropy (nats)",
            show_grid=SHOW_GRID.Y,
            vspans=decay_span(bounds),
        ),
        LineChart(
            curve(shown, "accuracy", 100),
            title="Masked-token accuracy",
            xlabel="Tokens seen (billions)",
            ylabel="Accuracy (%)",
            show_grid=SHOW_GRID.Y,
            vspans=decay_span(bounds),
        ),
    ]
    return Grid(charts, max_cols=2, title="Validation loss and accuracy", figsize=FULL_WIDTH, show_legend=True)


def zoom_figure(zoom: list[dict[str, float | None]], bounds: dict[str, float]) -> LineChart:
    """Draw the close-up of validation loss with the bootstrap half-width as error bars.

    Args:
        zoom: Rows of the close-up table, each with its loss and half-width.
        bounds: The output of `schedule_bounds`.

    Returns:
        The one-panel figure, the fourth pass and the decay shaded.
    """
    points = [{"x": r["tokens_b"], "y": r["loss"], "yerr": r["half_width"]} for r in zoom]
    return LineChart(
        points,
        title="Validation loss, last passes",
        xlabel="Tokens seen (billions)",
        ylabel="Cross-entropy (nats)",
        show_grid=SHOW_GRID.Y,
        show_yerr=True,
        show_legend=True,
        vspans=[
            {
                "xmin": bounds["fourth_pass"],
                "xmax": bounds["decay"],
                "label": "fourth pass, constant learning rate",
                "style": {"plot_vspan_color": "#e8c37a", "plot_vspan_alpha": 0.25},
            },
            decay_span(bounds),
        ],
        figsize=ONE_COLUMN,
    )


def dynamics_figure(train: list[dict[str, float | None]], bounds: dict[str, float]) -> Grid:
    """Draw training loss and gradient norm per bin, mean and maximum, against tokens seen.

    Args:
        train: Rows of the training table.
        bounds: The output of `schedule_bounds`.

    Returns:
        The two-panel figure; the loss panel leaves out the warmup bins.
    """
    after_warmup = [row for row in train if row["tokens_b"] > WARMUP_END_B]
    charts = [
        LineChart(
            [curve(after_warmup, "train_loss"), curve(after_warmup, "train_loss_max")],
            subtitle=["bin mean", "bin maximum"],
            title="Training loss after warmup",
            xlabel="Tokens seen (billions)",
            ylabel="Cross-entropy (nats)",
            show_grid=SHOW_GRID.Y,
            vspans=decay_span(bounds),
        ),
        LineChart(
            [curve(train, "grad_norm"), curve(train, "grad_norm_max"), curve(train, "grad_norm", GRAD_SPIKE_RATIO)],
            subtitle=["bin mean", "bin maximum", f"spike threshold, {GRAD_SPIKE_RATIO:g} x bin mean"],
            style=[None, None, {"plot_line_style": "--", "plot_line_color": "#888888"}],
            scaley="log",
            title="Gradient norm",
            xlabel="Tokens seen (billions)",
            ylabel="L2 norm (unitless)",
            show_grid=SHOW_GRID.Y,
            vspans=decay_span(bounds),
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
        choices=["data", "tables", "spikes", "figures"],
        help="data: table the corpus; tables: derive the figure tables; spikes: count spikes; figures: draw them",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Run one step of the analysis.

    Args:
        args: The parsed arguments, naming the step.

    Raises:
        FileNotFoundError: If the corpus manifest or the training config is missing.
    """
    for path in (MANIFEST, TRAIN_CONFIG):
        if not path.is_file():
            raise FileNotFoundError(f"input missing: {path}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf8"))
    config = yaml.safe_load(TRAIN_CONFIG.read_text(encoding="utf8"))

    if args.step == "data":
        dataset_tables(manifest)
        save_table(training_settings(config), TABLES / "training-value-by-setting.csv")
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
        bootstrap_row = read_rows("bootstrap-final-loss.csv")[0]
        half_width = round((bootstrap_row["ci_high"] - bootstrap_row["ci_low"]) / 2, 4)
        save_table(
            [
                {"batch": int(r["batch"]), "tokens_b": r["tokens_b"], "loss": r["loss"], "half_width": half_width}
                for r in evals
                if r["tokens_b"] >= ZOOM_FROM_B
            ],
            TABLES / "pilot-validation-loss-zoom-by-tokens.csv",
        )
        return

    if args.step == "spikes":
        rows = spike_rows(read_rows("run-train.csv"))
        save_table(rows, TABLES / "spike-count-by-bin.csv")
        print(count_spikes(rows))
        return

    # draw the figures from the tables alone
    FIGURES.mkdir(exist_ok=True)
    bounds = schedule_bounds(config, manifest)
    save_figure(
        validation_figure(read_rows("pilot-validation-loss-by-tokens.csv"), bounds),
        FIGURES / "pilot-validation-loss-by-tokens.svg",
        claim=Claim("Q1", "masked-token cross-entropy"),
        width="full-width",
    )
    save_figure(
        zoom_figure(read_rows("pilot-validation-loss-zoom-by-tokens.csv"), bounds),
        FIGURES / "pilot-validation-loss-zoom-by-tokens.svg",
        claim=Claim("Q1", "masked-token cross-entropy"),
    )
    save_figure(
        dynamics_figure(read_rows("pilot-training-dynamics-by-tokens.csv"), bounds),
        FIGURES / "pilot-training-dynamics-by-tokens.svg",
        claim=Claim("Q2", "gradient norm"),
        width="full-width",
    )


if __name__ == "__main__":
    main(get_args())
