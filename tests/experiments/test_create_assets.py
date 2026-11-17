"""The project theme: datachart starts in it, and it is colour-blind safe on the report plate."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("datachart_themes.slm4ie")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))
import create_assets  # noqa: E402  — path set above so the shared helper resolves


def test_session_starts_in_the_project_theme() -> None:
    """The installed default theme is slm4ie, so every chart draws in its palette."""
    from datachart.config import config
    from datachart_themes.slm4ie import PALETTE

    assert config.theme == create_assets.THEME
    assert config["color_general_multiple"] == PALETTE


def test_series_palette_passes_the_gate() -> None:
    """Every pair of series stays apart for colour-blind readers on the report plate.

    The soft fills sit under 3:1 against the plate by design; the theme's ink
    outline on every filled mark carries that contrast instead.
    """
    from datachart.themes import score_palette
    from datachart_themes.slm4ie import PALETTE

    score = score_palette(PALETTE, "#fbfcfd")
    assert score.verdict == "pass", str(score)


# a row-wide shape, 1.88:1, which the record page sets in one column
ROW_FIGSIZE = (6.4, 3.4)


def bar_chart(figsize: tuple[float, float] = ROW_FIGSIZE):
    """A small bar chart that passes every figure rule in the project theme."""
    from datachart.charts import BarChart

    return BarChart(
        [{"label": "a", "y": 1}, {"label": "b", "y": 2}],
        title="Documents by label",
        xlabel="label",
        ylabel="documents (count)",
        figsize=figsize,
    )


def test_save_figure_takes_a_figure_or_a_builder(tmp_path: Path) -> None:
    """Both call shapes write a date-free SVG in the report's font stack."""
    figures = tmp_path / "figures"
    figures.mkdir()
    for name, figure in (("built-documents-by-label.svg", bar_chart()), ("builder-documents-by-label.svg", bar_chart)):
        create_assets.save_figure(figure, figures / name)
        text = (figures / name).read_text(encoding="utf-8")
        assert "<dc:date>" not in text
        assert "Instrument Sans" in text


def test_save_figure_writes_the_same_bytes_twice(tmp_path: Path) -> None:
    """Two saves of one figure are identical, so a rerun leaves an unchanged file untouched in git."""
    first = tmp_path / "first" / "figures" / "documents-by-label.svg"
    second = tmp_path / "second" / "figures" / "documents-by-label.svg"
    for path in (first, second):
        path.parent.mkdir(parents=True)
        create_assets.save_figure(bar_chart, path)
    assert first.read_bytes() == second.read_bytes()


def test_save_figure_stamps_the_rules_it_ran(tmp_path: Path) -> None:
    """A passing SVG carries the check stamp the report builder reads."""
    path = tmp_path / "figures" / "documents-by-label.svg"
    path.parent.mkdir()
    create_assets.save_figure(bar_chart, path)
    stamp = create_assets.read_stamp(path)
    assert stamp is not None and "FIG-004" in stamp["rules"]


def test_save_figure_refuses_a_breach_and_writes_nothing(tmp_path: Path) -> None:
    """A figure with no title and a misnamed file raises, and no file appears."""
    from datachart.charts import BarChart

    path = tmp_path / "plot.svg"
    with pytest.raises(create_assets.FigureCheckError, match="FIG-009"):
        create_assets.save_figure(lambda: BarChart([{"label": "a", "y": 1}]), path)
    assert not path.exists()


def test_save_figure_reads_font_sizes_from_the_project_theme(tmp_path: Path) -> None:
    """The theme's own text sizes pass the font-size rule; an override does not."""

    path = tmp_path / "figures" / "documents-by-label.svg"
    path.parent.mkdir()

    def oversized():
        from matplotlib.text import Text

        figure = bar_chart()
        for text in figure.findobj(Text):
            if text.get_text() == "Documents by label":
                text.set_fontsize(30)
        return figure

    with pytest.raises(create_assets.FigureCheckError, match="FIG-004 title at 30 pt"):
        create_assets.save_figure(oversized, path)


# ===============================
# The column from the shape
# ===============================


@pytest.mark.parametrize("width", ["one-column", "two-column", "full-width"])
def test_narrow_figure_is_checked_at_two_column_whatever_width_says(tmp_path: Path, width: str) -> None:
    """A figure under 1.8:1 sits in the half column, so its text falls under the floor there."""
    path = tmp_path / "figures" / "documents-by-label.svg"
    path.parent.mkdir()
    with pytest.raises(create_assets.FigureCheckError, match="FIG-005 .* two-column figure"):
        create_assets.save_figure(lambda: bar_chart((6.4, 4.8)), path, width=width)


def test_narrow_figure_drawn_half_width_passes_and_stamps_two_column(tmp_path: Path) -> None:
    """A narrow figure drawn for the half column is written and stamped two-column."""
    path = tmp_path / "figures" / "documents-by-label.svg"
    path.parent.mkdir()
    create_assets.save_figure(lambda: bar_chart((3.2, 2.4)), path)
    assert create_assets.read_stamp(path)["width"] == "two-column"


def test_wide_figure_passed_two_column_is_checked_at_one_column(tmp_path: Path) -> None:
    """The page sets a figure 1.8:1 or wider in the whole row, never the half column."""
    path = tmp_path / "figures" / "documents-by-label.svg"
    path.parent.mkdir()
    create_assets.save_figure(bar_chart, path, width="two-column")
    assert create_assets.read_stamp(path)["width"] == "one-column"


def test_wide_figure_past_one_column_passes_only_at_full_width(tmp_path: Path) -> None:
    """A figure too wide for one column is refused by default and written when passed full-width."""
    path = tmp_path / "figures" / "documents-by-label.svg"
    path.parent.mkdir()
    with pytest.raises(create_assets.FigureCheckError, match="FIG-005"):
        create_assets.save_figure(lambda: bar_chart((8.4, 4.0)), path)
    create_assets.save_figure(lambda: bar_chart((8.4, 4.0)), path, width="full-width")
    assert create_assets.read_stamp(path)["width"] == "full-width"


@pytest.mark.parametrize(
    ("size_name", "width", "column"),
    [
        ("ONE_COLUMN", "one-column", "one-column"),
        ("HALF_SQUARE", "one-column", "two-column"),
        ("HALF_TALL", "one-column", "two-column"),
        ("FULL_WIDTH", "full-width", "full-width"),
    ],
)
def test_named_size_passes_in_its_column(tmp_path: Path, size_name: str, width: str, column: str) -> None:
    """Each named figure size clears the floor in the project theme, in the column its shape picks."""
    path = tmp_path / "figures" / "documents-by-label.svg"
    path.parent.mkdir()
    figsize = getattr(create_assets, size_name)
    create_assets.save_figure(lambda: bar_chart(figsize), path, width=width)
    assert create_assets.read_stamp(path)["width"] == column


def test_axis_naming_a_coefficient_needs_no_unit(tmp_path: Path) -> None:
    """A rank correlation on a -1 to 1 axis passes FIG-008 without a unit in parentheses."""
    from datachart.charts import BarChart

    path = tmp_path / "figures" / "rho-by-pair.svg"
    path.parent.mkdir()
    create_assets.save_figure(
        lambda: BarChart(
            [{"label": "a", "y": 0.4}, {"label": "b", "y": 0.9}],
            title="Agreement by pair",
            xlabel="pair",
            ylabel="Spearman rho",
            figsize=ROW_FIGSIZE,
        ),
        path,
    )
    assert "FIG-008" in create_assets.read_stamp(path)["rules"]


def test_check_version_matches_the_report_checker() -> None:
    """A figure saved here carries the version the report checker asks for (REC-165)."""
    # rules loads every check module in order, so it imports before figures
    import labflow_report.rules  # noqa: F401
    from labflow_report.checks import figures

    assert create_assets.CHECK_VERSION == figures.CHECK_VERSION


def test_half_width_takes_the_wider_side() -> None:
    """An asymmetric interval becomes the symmetric bar datachart draws, never narrower."""
    assert create_assets.half_width(0.5, 0.25, 1.0) == 0.5


def test_narrow_ratio_matches_the_record_page() -> None:
    """The floor check and the record page split narrow from wide at the same ratio."""
    from labflow_report import record

    assert create_assets.NARROW_FIGURE == record.NARROW_FIGURE


# ===============================
# Tables
# ===============================


def test_table_rules_travel_beside_create_assets() -> None:
    """The table rules live in table_rules.py and create_assets re-exports them."""
    import table_rules

    assert create_assets.save_table is table_rules.save_table
    assert create_assets.check_table is table_rules.check_table


def test_save_table_writes_a_passing_table(tmp_path: Path) -> None:
    """A table within the rules is written with full-precision floats and empty missing cells."""
    path = tmp_path / "tables" / "f1-by-arm.csv"
    path.parent.mkdir()
    create_assets.save_table([{"arm": "ours", "f1": 0.8125, "f1_low": None}], path)
    assert path.read_text(encoding="utf-8") == "arm,f1,f1_low\nours,0.8125,\n"


def test_save_table_refuses_a_breach_and_writes_nothing(tmp_path: Path) -> None:
    """A column that is not snake_case raises TableCheckError, and no file appears."""
    path = tmp_path / "tables" / "f1-by-arm.csv"
    path.parent.mkdir()
    with pytest.raises(create_assets.TableCheckError, match="TAB-010"):
        create_assets.save_table([{"arm": "ours", "F1 score": 0.8}], path)
    assert not path.exists()


def test_check_table_reads_a_written_table(tmp_path: Path) -> None:
    """A clause reading with an outcome word its clause type lacks is reported."""
    path = tmp_path / "tables" / "clause-readings.csv"
    path.parent.mkdir()
    path.write_text(
        "clause,qualifier,value,ci_low,ci_high,bar,margin,outcome\nQ1,,0.5,,,,,confirmed\n", encoding="utf-8"
    )
    assert [breach.rule for breach in create_assets.check_table(path)] == ["TAB-043"]
