import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # Tokenizer comparison (Slovenian)

        Exploratory companion to the tokenizer sweep trained by
        `scripts/tokenizers/train.py` and scored by `scripts/tokenizers/analyze.py`.
        Every plot is interactive (hover for details, drag to zoom, click legend
        entries to toggle series).

        - **Part A — Segmentation.** Pick an example (a curated lexicon form, a
          sampled dataset sentence, or your own text) and see how each selected
          tokenizer splits it. Tokens render as colored, character-aligned blocks;
          the gold **morpheme** boundaries from the Sloleks-derived lexicon are
          overlaid as dashed lines, and a token whose span matches a morpheme
          exactly is outlined in green. Hover a block for its piece, span, and
          morpheme match.
        - **Part B — Metrics.** How vocab size and algorithm move each of the six
          metrics, from the aggregated `report.json`.

        Morphological gold is *inflectional silver gold* — read the morph metrics
        as relative comparators, not absolute morphology.
        """
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import json
    import math
    import random
    import re
    from pathlib import Path

    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    import slm4ie.tokenizers.backends  # noqa: F401  (registers backends on import)
    from slm4ie.tokenizers.corpus import iter_sample_cache
    from slm4ie.tokenizers.morphology import load_lexicon
    from slm4ie.tokenizers.registry import get_tokenizer
    from slm4ie.utils.config import load_tokenizer_config

    return (
        Path,
        get_tokenizer,
        go,
        iter_sample_cache,
        json,
        load_lexicon,
        load_tokenizer_config,
        make_subplots,
        math,
        px,
        random,
        re,
    )


@app.cell
def _(Path, load_tokenizer_config):
    def _find_repo_root(start):
        for candidate in [start, *start.parents]:
            if (candidate / "pyproject.toml").exists():
                return candidate
        return start

    REPO_ROOT = _find_repo_root(Path.cwd())
    CONFIG_PATH = REPO_ROOT / "configs" / "tokenizers" / "tokenizers.yaml"

    cfg = load_tokenizer_config(CONFIG_PATH)
    OUTPUT_ROOT = cfg.output_root
    REPORT_JSON = cfg.report_dir / "report.json"
    LEXICON_PATH = cfg.lexicon_path
    EVAL_SAMPLE_PATH = cfg.eval_sample_path
    TOKENIZERS = list(cfg.tokenizers)
    VOCAB_SIZES = list(cfg.vocab_sizes)
    return (
        CONFIG_PATH,
        EVAL_SAMPLE_PATH,
        LEXICON_PATH,
        OUTPUT_ROOT,
        REPORT_JSON,
        TOKENIZERS,
        VOCAB_SIZES,
    )


@app.cell
def _(CONFIG_PATH, OUTPUT_ROOT, TOKENIZERS, VOCAB_SIZES, mo):
    ALL_RUN_KEYS = [f"{name}-{vocab}" for name in TOKENIZERS for vocab in VOCAB_SIZES]
    AVAILABLE_RUN_KEYS = [key for key in ALL_RUN_KEYS if (OUTPUT_ROOT / key / "metadata.json").exists()]
    _missing = [key for key in ALL_RUN_KEYS if key not in AVAILABLE_RUN_KEYS]
    mo.md(
        f"**Config:** `{CONFIG_PATH}`  \n"
        f"**Artifacts:** `{OUTPUT_ROOT}`  \n"
        f"**Runs found:** {len(AVAILABLE_RUN_KEYS)} / {len(ALL_RUN_KEYS)}"
        + (f"  \n**Missing:** {', '.join(_missing)}" if _missing else "")
    )
    return (AVAILABLE_RUN_KEYS,)


@app.cell
def _(TOKENIZERS, px):
    _palette = px.colors.qualitative.Plotly
    TOKENIZER_COLORS = {name: _palette[i % len(_palette)] for i, name in enumerate(TOKENIZERS)}
    return (TOKENIZER_COLORS,)


@app.cell
def _(LEXICON_PATH, load_lexicon):
    LEXICON = load_lexicon(LEXICON_PATH)
    return (LEXICON,)


@app.cell
def _(OUTPUT_ROOT, get_tokenizer, json):
    _tok_cache = {}

    def load_run(run_key):
        if run_key not in _tok_cache:
            run_dir = OUTPUT_ROOT / run_key
            name = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))["name"]
            _tok_cache[run_key] = get_tokenizer(name).load(run_dir)
        return _tok_cache[run_key]

    return (load_run,)


@app.cell
def _(EVAL_SAMPLE_PATH, LEXICON, iter_sample_cache, random):
    # Curated examples are drawn straight from the lexicon so the gold-morpheme
    # overlay is guaranteed. The Sloleks silver-gold is inflectional, so forms
    # carry at most a stem+suffix split; keep the 2-morpheme forms (the only ones
    # with an internal boundary) whose stem and suffix are both non-trivial.
    _candidates = [
        seg.form
        for seg in LEXICON.by_form.values()
        if len(seg.morphemes) == 2 and 6 <= len(seg.form) <= 14 and min(len(m) for m in seg.morphemes) >= 2
    ]
    _rng = random.Random(13)
    CURATED = sorted(_rng.sample(_candidates, min(12, len(_candidates)))) if _candidates else ["najlepšega"]

    # Dataset examples: sentence-length lines sampled from the held-out eval set.
    _lines = []
    for _i, _text in enumerate(iter_sample_cache(EVAL_SAMPLE_PATH)):
        if _i >= 4000:
            break
        _stripped = _text.strip()
        if 30 <= len(_stripped) <= 160 and " " in _stripped:
            _lines.append(_stripped)
    DATASET_EXAMPLES = sorted(_rng.sample(_lines, min(8, len(_lines)))) if _lines else []
    return CURATED, DATASET_EXAMPLES


@app.cell
def _(LEXICON, re):
    _WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def _word_spans(text):
        return [(match.group(0), match.start(), match.end()) for match in _WORD_RE.finditer(text)]

    def gold_segments(text):
        """Gold morpheme spans `(piece, start, end)` and boundary offsets.

        The span shape matches a tokenizer's `encode_offsets` output so the GOLD
        row renders through the same drawing path as the tokenizer rows.
        """
        segments = []
        boundaries = set()
        for word, start, _end in _word_spans(text):
            entry = LEXICON.by_form.get(word) or LEXICON.by_form.get(word.lower())
            if entry is None:
                continue
            cursor = start
            for morpheme in entry.morphemes:
                segments.append((morpheme, cursor, cursor + len(morpheme)))
                cursor += len(morpheme)
            for offset in entry.boundaries():
                boundaries.add(start + offset)
        return segments, boundaries

    return (gold_segments,)


@app.cell
def _(TOKENIZER_COLORS, gold_segments, go, load_run):
    _GOLD_HEX = "#D9BF73"

    def _rgba(hex_color, alpha):
        raw = hex_color.lstrip("#")
        red, green, blue = (int(raw[i : i + 2], 16) for i in (0, 2, 4))
        return f"rgba({red},{green},{blue},{alpha})"

    def _add_token_block(fig, label, base_hex, idx, piece, start, end, surface, on_morpheme, row_y, row_h):
        fig.add_trace(
            go.Scatter(
                x=[start, end, end, start, start],
                y=[row_y, row_y, row_y + row_h, row_y + row_h, row_y],
                fill="toself",
                mode="lines",
                fillcolor=_rgba(base_hex, 0.85 if idx % 2 == 0 else 0.5),
                line=dict(
                    color="#2e8b2e" if on_morpheme else "rgba(255,255,255,0.9)", width=2.5 if on_morpheme else 1.0
                ),
                hoveron="fills",
                hoverinfo="text",
                hovertext=(
                    f"<b>{label}</b><br>piece: {piece}<br>surface: '{surface}'"
                    f"<br>span: {start}–{end} (len {end - start})"
                    f"<br>morpheme match: {'✓' if on_morpheme else '✗'}"
                ),
                showlegend=False,
            )
        )

    def plot_segmentation(text, run_keys):
        """Build an interactive character-aligned segmentation grid for `text`."""
        if not text or not run_keys:
            return None
        n_chars = len(text)
        gold_segs, gold_bnds = gold_segments(text)

        rows = []  # (label, base_hex, spans) bottom-to-top
        for run_key in reversed(run_keys):
            name = run_key.rpartition("-")[0]
            spans = load_run(run_key).encode_offsets(text)
            rows.append((run_key, TOKENIZER_COLORS.get(name, "#888888"), spans))
        if gold_segs:
            rows.append(("GOLD (morphemes)", _GOLD_HEX, gold_segs))

        row_h = 0.72
        fig = go.Figure()
        annotations = []
        for row_idx, (label, base_hex, spans) in enumerate(rows):
            for idx, (piece, start, end) in enumerate(spans):
                if end <= start:
                    continue
                on_morpheme = (start == 0 or start in gold_bnds) and (end == n_chars or end in gold_bnds)
                surface = text[start:end]
                _add_token_block(fig, label, base_hex, idx, piece, start, end, surface, on_morpheme, row_idx, row_h)
                annotations.append(
                    dict(x=(start + end) / 2, y=row_idx + row_h / 2, text=surface, showarrow=False, font=dict(size=13))
                )

        shapes = [
            dict(
                type="line",
                x0=b,
                x1=b,
                y0=0,
                y1=len(rows),
                line=dict(color="rgba(60,60,60,0.55)", width=1, dash="dash"),
            )
            for b in gold_bnds
        ]
        annotations.append(
            dict(
                x=0,
                y=len(rows) + 0.2,
                text="green outline = token span matches a gold morpheme; dashed = morpheme boundary",
                showarrow=False,
                font=dict(size=11, color="#555"),
                xanchor="left",
            )
        )
        fig.update_layout(
            shapes=shapes,
            annotations=annotations,
            xaxis=dict(range=[0, n_chars], visible=False),
            yaxis=dict(
                tickmode="array",
                tickvals=[i + row_h / 2 for i in range(len(rows))],
                ticktext=[label for label, _, _ in rows],
                range=[-0.1, len(rows) + 0.6],
            ),
            height=46 * len(rows) + 130,
            margin=dict(l=150, r=20, t=20, b=20),
            plot_bgcolor="white",
            hovermode="closest",
        )
        return fig

    return (plot_segmentation,)


@app.cell
def _(mo):
    mo.md(r"""## Part A — Segmentation""")
    return


@app.cell
def _(AVAILABLE_RUN_KEYS, CURATED, DATASET_EXAMPLES, VOCAB_SIZES, mo):
    _default_vocab = 32000 if 32000 in VOCAB_SIZES else VOCAB_SIZES[len(VOCAB_SIZES) // 2]
    _default_runs = [k for k in AVAILABLE_RUN_KEYS if k.endswith(f"-{_default_vocab}")] or AVAILABLE_RUN_KEYS
    _options = CURATED + DATASET_EXAMPLES
    _first = _options[0] if _options else "najlepšega"

    preset_select = mo.ui.dropdown(options=_options or [_first], value=_first, label="Preset")
    free_text = mo.ui.text(value="", placeholder="type any Slovenian text…", label="Free text", full_width=True)
    run_select = mo.ui.multiselect(options=AVAILABLE_RUN_KEYS, value=_default_runs, label="Tokenizers (name-vocab)")
    mo.vstack([preset_select, free_text, run_select])
    return free_text, preset_select, run_select


@app.cell
def _(free_text, preset_select):
    example_text = free_text.value.strip() or preset_select.value
    return (example_text,)


@app.cell
def _(example_text, mo, plot_segmentation, run_select):
    _fig = plot_segmentation(example_text, run_select.value)
    _fig if _fig is not None else mo.md("*Pick an example and at least one tokenizer.*")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Part B — Metrics vs vocab size & algorithm

        Read from the aggregated `report.json`. Arrows in titles mark the better
        direction (↑ higher-is-better, ↓ lower-is-better).
        """
    )
    return


@app.cell
def _(REPORT_JSON, json):
    _payload = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    RESULTS = _payload["results"]
    DIRECTIONS = _payload["directions"]
    METRIC_COLUMNS = [
        "fertility",
        "tokens_per_byte",
        "chars_per_token",
        "renyi_efficiency",
        "morph_score_f1",
        "morph_edit_distance",
        "morph_consistency",
    ]
    _ARROW = {"higher": "↑", "lower": "↓"}

    def metric_title(metric):
        return f"{metric} {_ARROW.get(DIRECTIONS.get(metric, ''), '')}".strip()

    def series_by_tokenizer(metric):
        series = {}
        for record in RESULTS:
            series.setdefault(record["tokenizer"], []).append((record["vocab_size"], record[metric]))
        for points in series.values():
            points.sort()
        return series

    return DIRECTIONS, METRIC_COLUMNS, RESULTS, metric_title, series_by_tokenizer


@app.cell
def _(METRIC_COLUMNS, TOKENIZER_COLORS, go, make_subplots, math, metric_title, series_by_tokenizer):
    _ncols = 3
    _nrows = math.ceil(len(METRIC_COLUMNS) / _ncols)
    _vocabs = sorted({v for _pts in series_by_tokenizer(METRIC_COLUMNS[0]).values() for v, _ in _pts})
    _fig = make_subplots(
        rows=_nrows,
        cols=_ncols,
        subplot_titles=[metric_title(m) for m in METRIC_COLUMNS],
        vertical_spacing=0.12,
        horizontal_spacing=0.07,
    )
    for _idx, _metric in enumerate(METRIC_COLUMNS):
        _row = _idx // _ncols + 1
        _col = _idx % _ncols + 1
        for _tname, _points in series_by_tokenizer(_metric).items():
            _fig.add_trace(
                go.Scatter(
                    x=[v for v, _ in _points],
                    y=[y for _, y in _points],
                    mode="lines+markers",
                    name=_tname,
                    legendgroup=_tname,
                    showlegend=_idx == 0,
                    line=dict(color=TOKENIZER_COLORS.get(_tname)),
                    marker=dict(color=TOKENIZER_COLORS.get(_tname)),
                    hovertemplate=f"{_tname}<br>vocab=%{{x}}<br>{_metric}=%{{y:.4f}}<extra></extra>",
                ),
                row=_row,
                col=_col,
            )
    _fig.update_xaxes(tickvals=_vocabs, title_text="vocab")
    _fig.update_layout(
        height=300 * _nrows,
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=-0.07, xanchor="center", x=0.5),
        margin=dict(t=40, l=40, r=20, b=60),
    )
    _fig
    return


@app.cell
def _(METRIC_COLUMNS, TOKENIZERS, mo):
    metric_pick = mo.ui.dropdown(options=METRIC_COLUMNS, value="morph_score_f1", label="Metric")
    tok_pick = mo.ui.multiselect(options=TOKENIZERS, value=TOKENIZERS, label="Tokenizers")
    chart_kind = mo.ui.radio(options=["line", "bar"], value="line", label="Chart")
    mo.hstack([metric_pick, chart_kind, tok_pick], justify="start", gap=2)
    return chart_kind, metric_pick, tok_pick


@app.cell
def _(TOKENIZER_COLORS, chart_kind, go, metric_pick, metric_title, series_by_tokenizer, tok_pick):
    _metric = metric_pick.value
    _data = series_by_tokenizer(_metric)
    _toks = [t for t in tok_pick.value if t in _data]
    _vocabs = sorted({v for _t in _toks for v, _ in _data[_t]})
    _fig = go.Figure()
    for _t in _toks:
        _lookup = dict(_data[_t])
        _ys = [_lookup.get(v) for v in _vocabs]
        _hover = f"{_t}<br>vocab=%{{x}}<br>{_metric}=%{{y:.4f}}<extra></extra>"
        if chart_kind.value == "line":
            _fig.add_trace(
                go.Scatter(
                    x=_vocabs,
                    y=_ys,
                    mode="lines+markers",
                    name=_t,
                    line=dict(color=TOKENIZER_COLORS.get(_t)),
                    marker=dict(color=TOKENIZER_COLORS.get(_t)),
                    hovertemplate=_hover,
                )
            )
        else:
            _fig.add_trace(
                go.Bar(x=_vocabs, y=_ys, name=_t, marker_color=TOKENIZER_COLORS.get(_t), hovertemplate=_hover)
            )
    _fig.update_layout(
        title=metric_title(_metric),
        xaxis=dict(title="vocab size", tickvals=_vocabs),
        barmode="group",
        height=480,
        hovermode="closest",
    )
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Cost ↔ quality tradeoff

        Each point is one run. Pick a compression/cost metric for x and a
        morphology/quality metric for y; the Pareto-optimal runs (best achievable
        given both metrics' directions) are connected.
        """
    )
    return


@app.cell
def _(METRIC_COLUMNS, mo):
    x_metric = mo.ui.dropdown(options=METRIC_COLUMNS, value="fertility", label="x (cost)")
    y_metric = mo.ui.dropdown(options=METRIC_COLUMNS, value="morph_score_f1", label="y (quality)")
    mo.hstack([x_metric, y_metric], justify="start", gap=2)
    return x_metric, y_metric


@app.cell
def _(DIRECTIONS, RESULTS, TOKENIZER_COLORS, go, metric_title, x_metric, y_metric):
    _xm = x_metric.value
    _ym = y_metric.value

    def _better(a, b, direction):
        return a < b if direction == "lower" else a > b

    def _pareto(points):
        xdir = DIRECTIONS.get(_xm, "higher")
        ydir = DIRECTIONS.get(_ym, "higher")
        frontier = []
        for p in points:
            dominated = False
            for q in points:
                if q is p:
                    continue
                x_ge = _better(q[0], p[0], xdir) or q[0] == p[0]
                y_ge = _better(q[1], p[1], ydir) or q[1] == p[1]
                strict = _better(q[0], p[0], xdir) or _better(q[1], p[1], ydir)
                if x_ge and y_ge and strict:
                    dominated = True
                    break
            if not dominated:
                frontier.append(p)
        return sorted(frontier)

    _by_tokenizer = {}
    for _r in RESULTS:
        _by_tokenizer.setdefault(_r["tokenizer"], []).append(_r)

    _fig = go.Figure()
    for _tname, _recs in _by_tokenizer.items():
        _fig.add_trace(
            go.Scatter(
                x=[r[_xm] for r in _recs],
                y=[r[_ym] for r in _recs],
                mode="markers+text",
                name=_tname,
                marker=dict(color=TOKENIZER_COLORS.get(_tname), size=12, line=dict(color="white", width=1)),
                text=[f"{r['vocab_size'] // 1000}k" for r in _recs],
                textposition="top center",
                textfont=dict(size=9),
                customdata=[[r["tokenizer"], r["vocab_size"]] for r in _recs],
                hovertemplate=(
                    f"%{{customdata[0]}} @ %{{customdata[1]}}<br>{_xm}=%{{x:.4f}}<br>{_ym}=%{{y:.4f}}<extra></extra>"
                ),
            )
        )

    _front = _pareto([(r[_xm], r[_ym]) for r in RESULTS])
    if len(_front) >= 2:
        _fig.add_trace(
            go.Scatter(
                x=[p[0] for p in _front],
                y=[p[1] for p in _front],
                mode="lines",
                name="Pareto frontier",
                line=dict(color="rgba(60,60,60,0.7)", dash="dash"),
                hoverinfo="skip",
            )
        )
    _fig.update_layout(
        title="Cost ↔ quality (Pareto frontier dashed)",
        xaxis_title=metric_title(_xm),
        yaxis_title=metric_title(_ym),
        height=560,
        hovermode="closest",
    )
    _fig
    return


if __name__ == "__main__":
    app.run()
