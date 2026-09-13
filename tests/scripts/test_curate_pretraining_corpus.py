"""Tests for scripts/curate_pretraining_corpus.py argument parsing."""

import pytest

import scripts.curate_pretraining_corpus as curate_cli

#: Argument prefix every `run` invocation needs; `_load_yaml` is never called
#: here, so the path only has to satisfy the required flag.
RUN = ["run", "--config", "curation.yaml"]


class TestRunSelection:
    """`run` accepts either positional dataset keys or `--all`."""

    def test_accepts_single_key(self) -> None:
        """A single positional resolves into `args.datasets`."""
        args = curate_cli.parse_args([*RUN, "kzb"])
        assert args.datasets == ["kzb"]
        assert args.all is False

    def test_accepts_multiple_keys(self) -> None:
        """Multiple positionals are gathered into `args.datasets`."""
        args = curate_cli.parse_args([*RUN, "kzb", "solar"])
        assert args.datasets == ["kzb", "solar"]
        assert args.all is False

    def test_accepts_all_flag(self) -> None:
        """`--all` parses without any positional keys."""
        args = curate_cli.parse_args([*RUN, "--all"])
        assert args.all is True
        assert args.datasets == []

    def test_errors_when_nothing_selected(self) -> None:
        """Bare invocation fails: must pass datasets or `--all`."""
        with pytest.raises(SystemExit):
            curate_cli.parse_args(RUN)

    def test_errors_without_a_curation_config(self) -> None:
        """The curation config is an experiment's, so it must be explicit."""
        with pytest.raises(SystemExit):
            curate_cli.parse_args(["run", "--all"])

    def test_a_subcommand_is_required(self) -> None:
        """An invocation naming no subcommand is rejected."""
        with pytest.raises(SystemExit):
            curate_cli.parse_args(["--all"])


class TestStageSelection:
    """`--stage` picks one stage; corpus stages need the whole corpus."""

    def test_default_stage_is_all(self) -> None:
        """No `--stage` flag means run every stage in order."""
        assert curate_cli.parse_args([*RUN, "--all"]).stage == "all"

    def test_accepts_each_stage_name(self) -> None:
        """Every documented `--stage` value parses."""
        for name in (
            "convert",
            "language",
            "quality",
            "repetition",
            "exact_dedup",
            "sentence_dedup",
            "statistics",
            "all",
        ):
            assert curate_cli.parse_args([*RUN, "--all", "--stage", name]).stage == name

    def test_rejects_unknown_stage(self) -> None:
        """An unknown `--stage` value is rejected by argparse."""
        with pytest.raises(SystemExit):
            curate_cli.parse_args([*RUN, "--all", "--stage", "lang"])

    def test_corpus_stage_with_positional_keys_errors(self) -> None:
        """A corpus-wide stage cannot run on a subset of datasets."""
        with pytest.raises(SystemExit):
            curate_cli.parse_args([*RUN, "gigafida", "--stage", "exact_dedup"])

    def test_corpus_stage_with_all_is_ok(self) -> None:
        """`--all --stage statistics` parses fine."""
        args = curate_cli.parse_args([*RUN, "--all", "--stage", "statistics"])
        assert args.all is True
        assert args.stage == "statistics"

    def test_scoped_stage_with_positional_keys_ok(self) -> None:
        """`--stage quality` with positional keys is allowed (scoped stage)."""
        args = curate_cli.parse_args([*RUN, "gigafida", "--stage", "quality"])
        assert args.datasets == ["gigafida"]
        assert args.stage == "quality"


class TestWorkers:
    """Worker-count flag and its back-compat alias."""

    def test_default_workers_is_serial(self) -> None:
        """The default worker count is 1 (serial)."""
        assert curate_cli.parse_args([*RUN, "--all"]).workers == 1

    def test_max_workers_zero(self) -> None:
        """`--max-workers 0` is the cpu_default sentinel."""
        assert curate_cli.parse_args([*RUN, "--all", "--max-workers", "0"]).workers == 0

    def test_tasks_alias(self) -> None:
        """`--tasks N` is accepted as an alias for `--max-workers N`."""
        assert curate_cli.parse_args([*RUN, "--all", "--tasks", "4"]).workers == 4


class TestRecount:
    """`recount` is a standalone maintenance subcommand."""

    def test_recount_needs_no_target(self) -> None:
        """`recount` needs neither positional datasets nor `--all`."""
        args = curate_cli.parse_args(["recount", "--config", "curation.yaml"])
        assert args.command == "recount"

    def test_recount_rejects_targets(self) -> None:
        """`recount` rewrites no data, so it takes no dataset selection."""
        with pytest.raises(SystemExit):
            curate_cli.parse_args(["recount", "--config", "curation.yaml", "--all"])
        with pytest.raises(SystemExit):
            curate_cli.parse_args(["recount", "--config", "curation.yaml", "alfa"])

    def test_recount_requires_a_curation_config(self) -> None:
        """The curation config must be explicit here too."""
        with pytest.raises(SystemExit):
            curate_cli.parse_args(["recount"])
