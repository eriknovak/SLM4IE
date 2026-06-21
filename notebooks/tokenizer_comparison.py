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

        - **Part A — Segmentation.** Pick an example (a curated lexicon form, a
          sampled dataset sentence, or your own text) and see how each selected
          tokenizer splits it. Tokens render as colored, character-aligned blocks;
          the gold **morpheme** boundaries from the Sloleks-derived lexicon are
          overlaid as dashed lines, and a token whose span matches a morpheme
          exactly is outlined in green.
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

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle

    import slm4ie.tokenizers.backends  # noqa: F401  (registers backends on import)
    from slm4ie.tokenizers.corpus import iter_sample_cache
    from slm4ie.tokenizers.morphology import load_lexicon
    from slm4ie.tokenizers.registry import get_tokenizer
    from slm4ie.utils.config import load_tokenizer_config

    return (
        Path,
        Rectangle,
        get_tokenizer,
        iter_sample_cache,
        json,
        load_lexicon,
        load_tokenizer_config,
        math,
        np,
        plt,
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
def _(TOKENIZERS, plt):
    _cmap = plt.get_cmap("tab10")
    TOKENIZER_COLORS = {name: _cmap(i % 10) for i, name in enumerate(TOKENIZERS)}
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
def _(Rectangle, TOKENIZER_COLORS, gold_segments, load_run, plt):
    _GOLD_COLOR = (0.85, 0.75, 0.45)

    def _shade(rgba, alt):
        # Two alternating alpha levels make adjacent token blocks distinguishable.
        return (rgba[0], rgba[1], rgba[2], 0.85 if alt else 0.5)

    def _draw_row(ax, y, spans, text, base_rgba, gold_bnds, n_chars, row_h):
        for idx, (_piece, start, end) in enumerate(spans):
            if end <= start:
                continue
            on_morpheme = (start == 0 or start in gold_bnds) and (end == n_chars or end in gold_bnds)
            ax.add_patch(
                Rectangle(
                    (start, y),
                    end - start,
                    row_h,
                    facecolor=_shade(base_rgba, idx % 2 == 0),
                    edgecolor="#2e8b2e" if on_morpheme else "white",
                    linewidth=2.0 if on_morpheme else 0.8,
                )
            )
            ax.text((start + end) / 2, y + row_h / 2, text[start:end], ha="center", va="center", fontsize=10)

    def plot_segmentation(text, run_keys):
        """Render a character-aligned segmentation grid for `text`."""
        if not text or not run_keys:
            return None
        n_chars = len(text)
        gold_segs, gold_bnds = gold_segments(text)

        rows = []  # (label, base_rgba, spans) bottom-to-top
        for run_key in reversed(run_keys):
            name = run_key.rpartition("-")[0]
            spans = load_run(run_key).encode_offsets(text)
            rows.append((run_key, TOKENIZER_COLORS.get(name, (0.4, 0.4, 0.4, 1.0)), spans))
        if gold_segs:
            rows.append(("GOLD (morphemes)", _GOLD_COLOR, gold_segs))

        row_h = 0.78
        fig, ax = plt.subplots(figsize=(max(7.0, n_chars * 0.34), 0.55 * len(rows) + 0.7))
        for row_idx, (label, base_rgba, spans) in enumerate(rows):
            _draw_row(ax, row_idx, spans, text, base_rgba, gold_bnds, n_chars, row_h)
        for boundary in gold_bnds:
            ax.axvline(boundary, linestyle="--", color="0.35", linewidth=0.9, zorder=0)

        ax.set_xlim(0, n_chars)
        ax.set_ylim(0, len(rows))
        ax.set_yticks([i + row_h / 2 for i in range(len(rows))])
        ax.set_yticklabels([label for label, _, _ in rows])
        ax.set_xticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title("green outline = token span matches a gold morpheme", fontsize=9, loc="left")
        fig.tight_layout()
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
def _(METRIC_COLUMNS, TOKENIZER_COLORS, math, metric_title, plt, series_by_tokenizer):
    _ncols = 3
    _nrows = math.ceil(len(METRIC_COLUMNS) / _ncols)
    _fig, _axes = plt.subplots(_nrows, _ncols, figsize=(13, 3.1 * _nrows))
    _flat = _axes.flat
    _handles, _labels = [], []
    for _ax, _metric in zip(_flat, METRIC_COLUMNS):
        _series = series_by_tokenizer(_metric)
        _ticks = sorted({v for _points in _series.values() for v, _ in _points})
        for _tname, _points in _series.items():
            _xs = [v for v, _ in _points]
            _ys = [y for _, y in _points]
            (_line,) = _ax.plot(_xs, _ys, marker="o", color=TOKENIZER_COLORS.get(_tname), label=_tname)
            if _tname not in _labels:
                _handles.append(_line)
                _labels.append(_tname)
        _ax.set_title(metric_title(_metric), fontsize=10)
        _ax.set_xlabel("vocab size")
        _ax.set_xticks(_ticks)
        _ax.grid(True, alpha=0.3)
    for _ax in list(_flat):
        _ax.set_visible(False)
    _fig.legend(_handles, _labels, loc="lower center", ncol=len(_labels), bbox_to_anchor=(0.5, -0.02))
    _fig.suptitle("All metrics vs vocab size", y=1.0)
    _fig.tight_layout()
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
def _(TOKENIZER_COLORS, chart_kind, metric_pick, metric_title, np, plt, series_by_tokenizer, tok_pick):
    _metric = metric_pick.value
    _data = series_by_tokenizer(_metric)
    _toks = [t for t in tok_pick.value if t in _data]
    _fig, _ax = plt.subplots(figsize=(8.5, 5))
    if chart_kind.value == "line":
        for _t in _toks:
            _xs = [v for v, _ in _data[_t]]
            _ys = [y for _, y in _data[_t]]
            _ax.plot(_xs, _ys, marker="o", color=TOKENIZER_COLORS.get(_t), label=_t)
        _ax.set_xlabel("vocab size")
    else:
        _vocabs = sorted({v for _t in _toks for v, _ in _data[_t]})
        _x = np.arange(len(_vocabs))
        _width = 0.8 / max(1, len(_toks))
        for _i, _t in enumerate(_toks):
            _lookup = dict(_data[_t])
            _ys = [_lookup.get(v, float("nan")) for v in _vocabs]
            _ax.bar(_x + _i * _width, _ys, _width, color=TOKENIZER_COLORS.get(_t), label=_t)
        _ax.set_xticks(_x + (len(_toks) - 1) * _width / 2)
        _ax.set_xticklabels(_vocabs)
        _ax.set_xlabel("vocab size")
    _ax.set_title(metric_title(_metric))
    _ax.grid(True, axis="y", alpha=0.3)
    _ax.legend()
    _fig.tight_layout()
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
def _(DIRECTIONS, RESULTS, TOKENIZER_COLORS, metric_title, plt, x_metric, y_metric):
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

    _points = [(r[_xm], r[_ym], r["tokenizer"], r["vocab_size"]) for r in RESULTS]
    _fig, _ax = plt.subplots(figsize=(8.5, 6))
    _seen = set()
    for _x, _y, _tname, _vocab in _points:
        _ax.scatter(
            _x,
            _y,
            color=TOKENIZER_COLORS.get(_tname),
            s=70,
            edgecolor="white",
            linewidth=0.6,
            label=_tname if _tname not in _seen else None,
            zorder=3,
        )
        _seen.add(_tname)
        _ax.annotate(f"{_vocab // 1000}k", (_x, _y), fontsize=7, xytext=(4, 3), textcoords="offset points")

    _front = _pareto([(p[0], p[1]) for p in _points])
    if len(_front) >= 2:
        _ax.plot([p[0] for p in _front], [p[1] for p in _front], color="0.3", linestyle="--", linewidth=1.2, zorder=2)

    _ax.set_xlabel(metric_title(_xm))
    _ax.set_ylabel(metric_title(_ym))
    _ax.set_title("Pareto frontier (dashed) across all runs")
    _ax.grid(True, alpha=0.3)
    _ax.legend(title="tokenizer", fontsize=8)
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
