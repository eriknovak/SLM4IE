"""Tests for scripts/prepare_datasets.py argument parsing."""

import pytest

import scripts.prepare_datasets as prepare


class TestSelection:
    """Every subcommand takes positional keys or `--all`, never both."""

    def test_positional_keys(self) -> None:
        """Positional keys land in the subcommand's selection attribute."""
        assert prepare.parse_args(["download", "kzb", "solar"]).datasets == ["kzb", "solar"]
        assert prepare.parse_args(["extract", "macocu_sl"]).datasets == ["macocu_sl"]
        assert prepare.parse_args(["tasks", "ner/ssj500k"]).entries == ["ner/ssj500k"]
        assert prepare.parse_args(["tokenization", "sloleks"]).datasets == ["sloleks"]

    def test_all_flag(self) -> None:
        """`--all` parses without positional keys for every subcommand."""
        for command in ("download", "extract", "tasks", "tokenization"):
            assert prepare.parse_args([command, "--all"]).all is True

    def test_bare_subcommand_errors(self) -> None:
        """A subcommand naming no work is rejected."""
        for command in ("download", "extract", "tasks", "tokenization"):
            with pytest.raises(SystemExit):
                prepare.parse_args([command])

    def test_keys_and_all_are_exclusive(self) -> None:
        """Positional keys cannot be combined with `--all`."""
        with pytest.raises(SystemExit):
            prepare.parse_args(["download", "kzb", "--all"])

    def test_a_subcommand_is_required(self) -> None:
        """A bare invocation names no step and is rejected."""
        with pytest.raises(SystemExit):
            prepare.parse_args([])


class TestTasksOptions:
    """The `tasks` subcommand carries the SuperGLUE variant for every family."""

    def test_variant_defaults_to_humant(self) -> None:
        """Without `--variant` the human-translated bundle is read."""
        assert prepare.parse_args(["tasks", "--all"]).variant == "humant"

    def test_variant_is_selectable(self) -> None:
        """`--variant googlemt` selects the machine-translated bundle."""
        assert prepare.parse_args(["tasks", "nli/cb", "--variant", "googlemt"]).variant == "googlemt"

    def test_unknown_variant_rejected(self) -> None:
        """An undeclared variant is rejected by argparse."""
        with pytest.raises(SystemExit):
            prepare.parse_args(["tasks", "--all", "--variant", "deepl"])


class TestDownloadRoleFilters:
    """The benchmark role filters are mutually exclusive."""

    def test_each_filter_parses(self) -> None:
        """Either role filter is accepted on its own."""
        assert prepare.parse_args(["download", "--all", "--only-benchmarks"]).only_benchmarks is True
        assert prepare.parse_args(["download", "--all", "--exclude-benchmarks"]).exclude_benchmarks is True

    def test_filters_cannot_be_combined(self) -> None:
        """Restricting to and excluding benchmarks at once is contradictory."""
        with pytest.raises(SystemExit):
            prepare.parse_args(["download", "--all", "--only-benchmarks", "--exclude-benchmarks"])


class TestDispatch:
    """`main` turns a handler's return code into the process exit code."""

    def test_exit_code_comes_from_the_handler(self, monkeypatch) -> None:
        """A failing subcommand exits with the code its handler returned."""
        monkeypatch.setattr(prepare, "_run_tasks", lambda _args: 2)
        monkeypatch.setattr("sys.argv", ["prepare_datasets.py", "tasks", "--all"])

        with pytest.raises(SystemExit) as exc:
            prepare.main()
        assert exc.value.code == 2
