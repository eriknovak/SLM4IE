"""Tests for scripts/pretokenize_corpus.py (the run and check subcommands)."""

from pathlib import Path

import pytest

import scripts.pretokenize_corpus as cli
from slm4ie.data.pretokenize import PretokenizeConfig, PretokenizeSummary
from slm4ie.data.pretokenize.check import CheckReport


class TestParseArgs:
    """The run and check subcommands select entries by key or with --all."""

    def test_run_takes_keys_force_and_workers(self):
        """`run` accepts entry keys, --force and --max-workers."""
        args = cli.parse_args(["run", "a/b-8", "--force", "--max-workers", "4"])
        assert (args.command, args.entries, args.force, args.max_workers) == ("run", ["a/b-8"], True, 4)

    def test_check_takes_all(self):
        """`check` accepts --all instead of keys."""
        assert cli.parse_args(["check", "--all"]).all is True

    def test_a_selection_is_required(self):
        """A subcommand with neither keys nor --all is rejected."""
        with pytest.raises(SystemExit):
            cli.parse_args(["run"])


def _stub_config(monkeypatch, keys: list[str]) -> None:
    """Make the CLI load a registry declaring `keys`.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        keys: Entry keys the stub registry declares.
    """
    entries = {key: type("Entry", (), {"output_dir": Path("/nowhere") / key})() for key in keys}
    monkeypatch.setattr(cli, "load_pretokenize_config", lambda _path: PretokenizeConfig(entries=entries))
    monkeypatch.setattr(cli, "configure_script_logging", lambda **_kwargs: None)


def test_run_passes_the_selection_and_reports_failures(monkeypatch):
    """`run --all` converts every declared entry and exits 2 when one failed."""
    _stub_config(monkeypatch, ["a/b-8", "c/d-8"])
    seen = {}

    def fake_run(config, keys, *, force, max_workers):
        seen.update(keys=keys, force=force)
        return PretokenizeSummary(written=["a/b-8"], failed={"c/d-8": ["00003"]})

    monkeypatch.setattr(cli, "run_pretokenize", fake_run)
    assert cli._run(cli.parse_args(["run", "--all"])) == cli.FAILED_EXIT_CODE
    assert seen == {"keys": ["a/b-8", "c/d-8"], "force": False}


def test_run_with_an_unknown_entry_exits_with_the_error_code(monkeypatch):
    """A KeyError from the run (an unknown entry) gives the error exit code."""
    _stub_config(monkeypatch, ["a/b-8"])

    def fake_run(config, keys, *, force, max_workers):
        raise KeyError("Unknown pretokenize entries ['x/y-8']")

    monkeypatch.setattr(cli, "run_pretokenize", fake_run)
    assert cli._run(cli.parse_args(["run", "x/y-8"])) == cli.ERROR_EXIT_CODE


def test_check_exits_2_when_an_output_has_problems(monkeypatch):
    """`check` exits 2 when any selected output has a problem, 0 when all pass."""
    _stub_config(monkeypatch, ["a/b-8", "c/d-8"])
    reports = {
        "a/b-8": CheckReport(output_dir=Path("a")),
        "c/d-8": CheckReport(output_dir=Path("c"), problems=["mds/train: tokens off"]),
    }
    monkeypatch.setattr(
        cli, "check_pretokenized", lambda output_dir: reports[f"{output_dir.parent.name}/{output_dir.name}"]
    )
    assert cli._check(cli.parse_args(["check", "a/b-8"])) == 0
    assert cli._check(cli.parse_args(["check", "--all"])) == cli.FAILED_EXIT_CODE


def test_check_with_an_unknown_entry_exits_with_the_error_code(monkeypatch):
    """`check` of an entry the registry does not declare gives the error exit code."""
    _stub_config(monkeypatch, ["a/b-8"])
    assert cli._check(cli.parse_args(["check", "x/y-8"])) == cli.ERROR_EXIT_CODE
