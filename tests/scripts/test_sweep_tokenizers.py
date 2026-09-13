"""Tests for scripts/sweep_tokenizers.py."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.sweep_tokenizers as sweep


class TestDescribe:
    """Tests for sweep_tokenizers._describe."""

    def test_reports_size_in_mib(self, tmp_path: Path):
        """An existing file is described with its size in MiB."""
        path = tmp_path / "sample.txt.gz"
        path.write_bytes(b"x" * (2 * 1024 * 1024))
        described = sweep._describe(path)
        assert str(path) in described
        assert "2.0 MiB" in described

    def test_falls_back_to_path_when_unstatable(self, tmp_path: Path):
        """A missing file degrades to just the path string."""
        path = tmp_path / "missing.gz"
        assert sweep._describe(path) == str(path)


class TestSelectionFlags:
    """Every run-scoped subcommand exposes the same one-or-all selection."""

    def test_subcommands_expose_selection_flags(self):
        """train/evaluate/export parse --tokenizer/--vocab-size/--all alike."""
        for command in ("train", "evaluate", "export"):
            args = sweep.parse_args([command, "--config", "sweep.yaml", "--tokenizer", "bpe", "--vocab-size", "16000"])
            assert args.command == command
            assert args.tokenizer == "bpe"
            assert args.vocab_size == 16000
            assert args.all is False
            assert sweep.parse_args([command, "--config", "sweep.yaml", "--all"]).all is True

    def test_subcommands_require_an_explicit_sweep_config(self):
        """The sweep config belongs to an experiment, so it has no default."""
        for command in ("sample", "train", "evaluate", "export"):
            with pytest.raises(SystemExit):
                sweep.parse_args([command, "--all"])

    def test_a_subcommand_is_required(self):
        """A bare invocation names no step and is rejected."""
        with pytest.raises(SystemExit):
            sweep.parse_args([])


class TestSample:
    """Tests for the `sample` subcommand wiring."""

    def _patch_common(self, monkeypatch, cfg, recorder):
        """Stub logging setup and capture the prepare_inputs call.

        Args:
            monkeypatch: pytest monkeypatch fixture.
            cfg: Object standing in for the loaded sweep config.
            recorder (dict): Mutated with the `force` value prepare_inputs saw.
        """
        monkeypatch.setattr(sweep, "configure_script_logging", lambda **_kwargs: None)

        def fake_prepare(passed_cfg, force=False):
            recorder["force"] = force
            recorder["cfg"] = passed_cfg
            return cfg.corpus_sample_path, cfg.lexicon_path

        monkeypatch.setattr(sweep, "prepare_inputs", fake_prepare)

    def test_materializes_and_passes_force(self, monkeypatch, tmp_path: Path):
        """`--force` is threaded through to prepare_inputs."""
        sample = tmp_path / "out" / "corpus_sample.txt.gz"
        sample.parent.mkdir(parents=True)
        sample.write_bytes(b"data")
        lexicon = tmp_path / "out" / "morph_lexicon.jsonl.gz"
        lexicon.write_bytes(b"lex")
        cfg = SimpleNamespace(output_root=tmp_path / "out", corpus_sample_path=sample, lexicon_path=lexicon)
        recorder: dict = {}
        self._patch_common(monkeypatch, cfg, recorder)

        args = sweep.parse_args(["sample", "--config", "sweep.yaml", "--force"])
        assert sweep._run_sample(cfg, args) == 0

        assert recorder["force"] is True
        assert recorder["cfg"] is cfg

    def test_fails_when_sample_absent(self, monkeypatch, tmp_path: Path):
        """A non-existent sample after prepare_inputs is a hard failure."""
        cfg = SimpleNamespace(
            output_root=tmp_path / "out",
            corpus_sample_path=tmp_path / "out" / "corpus_sample.txt.gz",
            lexicon_path=None,
        )
        recorder: dict = {}
        self._patch_common(monkeypatch, cfg, recorder)

        args = sweep.parse_args(["sample", "--config", "sweep.yaml"])
        assert sweep._run_sample(cfg, args) == 1


class TestMainExits:
    """`main` turns a handler's return code into the process exit code."""

    def test_exit_code_comes_from_the_handler(self, monkeypatch, tmp_path: Path):
        """A failing subcommand exits with the code its handler returned."""
        cfg = SimpleNamespace(output_root=tmp_path / "out")
        monkeypatch.setattr(sweep, "load_tokenizer_config", lambda _path: cfg)
        monkeypatch.setattr(sweep, "_run_export", lambda _cfg, _args: 2)
        monkeypatch.setattr(sys, "argv", ["sweep_tokenizers.py", "export", "--config", "sweep.yaml", "--all"])

        with pytest.raises(SystemExit) as exc:
            sweep.main()
        assert exc.value.code == 2
