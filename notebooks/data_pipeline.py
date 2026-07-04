import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import plotly.express as px
    import plotly.graph_objects as go

    from slm4ie.utils import mlflow_report as mr
    from slm4ie.viz import register_theme

    register_theme()  # apply the SLM4IE plotly look to every figure below
    return go, mr, px


@app.cell
def _(mo):
    mo.md(
        r"""
        # SLM4IE data pipeline — end-to-end report

        Read live from the MLflow store (`$MLFLOW_TRACKING_URI`, default
        `http://localhost:5555`). Follows the corpus as it flows through the
        three tracked producers:

        `extract` → `pretrain` (8-stage curation) → `tasks`.

        Each producer upserts one run per build keyed by a content digest, so
        this report reflects the **latest** build of each. Run the pipelines
        with `--mlflow` to populate the experiments below.
        """
    )
    return


@app.cell
def _(mo, mr):
    _summary = []
    for _label, _experiment in (
        ("extract", mr.EXTRACT_EXPERIMENT),
        ("pretrain", mr.PRETRAIN_EXPERIMENT),
        ("tasks", mr.TASKS_EXPERIMENT),
    ):
        _table = mr.experiment_run_table(_experiment)
        _summary.append({"stage": _label, "experiment": _experiment, "runs": len(_table)})

    _rows = "\n".join(f"| {_s['stage']} | `{_s['experiment']}` | {_s['runs']} |" for _s in _summary)
    mo.md(
        "## Tracked experiments\n\n"
        "| stage | experiment | runs |\n|---|---|---|\n" + _rows
    )
    return


@app.cell
def _(go, mo, mr):
    _funnel = mr.pretrain_funnel()
    if _funnel.empty:
        _out = mo.md("*No `pretrain` build tracked yet — run `to_pretrain.py --all --mlflow`.*")
    else:
        _fig = go.Figure(
            go.Bar(x=_funnel["stage"], y=_funnel["docs_remaining"], text=_funnel["docs_remaining"])
        )
        _fig.update_layout(
            title="Per-stage survival funnel (documents remaining)",
            xaxis_title="stage",
            yaxis_title="documents",
        )
        _out = _fig
    _out
    return


@app.cell
def _(mo, mr, px):
    _shares = mr.mixture_shares()
    if _shares.empty:
        _out = mo.md("*No per-dataset mixture shares yet — needs a `pretrain` build with stats.*")
    else:
        _fig = px.bar(
            _shares,
            x="dataset",
            y="word_share",
            title="Post-dedup corpus mixture — word share per dataset",
        )
        _out = _fig
    _out
    return


@app.cell
def _(mo, mr):
    _risk = mr.contamination_risk()
    if _risk.empty:
        _out = mo.md(
            "## Pretrain ↔ eval contamination\n\n"
            "*No overlap detected between task sources and the pretrain corpus "
            "(or the experiments are empty).*"
        )
    else:
        _table = mo.ui.table(_risk, selection=None)
        _out = mo.vstack(
            [
                mo.md(
                    "## Pretrain ↔ eval contamination\n\n"
                    "Task datasets whose sources also feed the pretraining corpus "
                    "(leakage risk — worst for `held_out`):"
                ),
                _table,
            ]
        )
    _out
    return


if __name__ == "__main__":
    app.run()
