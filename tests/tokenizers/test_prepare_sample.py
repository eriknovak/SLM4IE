"""Tests for scripts/tokenizers/prepare_sample.py."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2] / "scripts" / "tokenizers"),
)
import prepare_sample  # noqa: E402


class TestDescribe:
    """Tests for prepare_sample._describe."""

    def test_reports_size_in_mib(self, tmp_path: Path):
        """An existing file is described with its size in MiB."""
        path = tmp_path / "sample.txt.gz"
        path.write_bytes(b"x" * (2 * 1024 * 1024))
        described = prepare_sample._describe(path)
        assert str(path) in described
        assert "2.0 MiB" in described

    def test_falls_back_to_path_when_unstatable(self, tmp_path: Path):
        """A missing file degrades to just the path string."""
        path = tmp_path / "missing.gz"
        assert prepare_sample._describe(path) == str(path)


class TestMain:
    """Tests for the prepare_sample CLI wiring."""

    def _patch_common(self, monkeypatch, tmp_path: Path, cfg, recorder):
        """Stub config/root resolution and capture the prepare_inputs call.

        Args:
            monkeypatch: pytest monkeypatch fixture.
            tmp_path (Path): Temp directory used as the project root.
            cfg: Object returned in place of the loaded sweep config.
            recorder (dict): Mutated with the `force` value prepare_inputs saw.
        """
        monkeypatch.setattr(prepare_sample, "find_project_root", lambda: tmp_path)
        monkeypatch.setattr(prepare_sample, "load_tokenizer_config", lambda _path: cfg)
        monkeypatch.setattr(prepare_sample, "configure_script_logging", lambda **_kwargs: None)

        def fake_prepare(passed_cfg, force=False):
            recorder["force"] = force
            recorder["cfg"] = passed_cfg
            return cfg.corpus_sample_path, cfg.lexicon_path

        monkeypatch.setattr(prepare_sample, "prepare_inputs", fake_prepare)

    def test_materializes_and_passes_force(self, monkeypatch, tmp_path: Path):
        """`--force` is threaded through to prepare_inputs."""
        sample = tmp_path / "out" / "corpus_sample.txt.gz"
        sample.parent.mkdir(parents=True)
        sample.write_bytes(b"data")
        lexicon = tmp_path / "out" / "morph_lexicon.jsonl.gz"
        lexicon.write_bytes(b"lex")
        cfg = SimpleNamespace(
            output_root=tmp_path / "out",
            corpus_sample_path=sample,
            lexicon_path=lexicon,
        )
        recorder: dict = {}
        self._patch_common(monkeypatch, tmp_path, cfg, recorder)
        monkeypatch.setattr(sys, "argv", ["prepare_sample.py", "--force"])

        prepare_sample.main()

        assert recorder["force"] is True
        assert recorder["cfg"] is cfg

    def test_exits_when_sample_absent(self, monkeypatch, tmp_path: Path):
        """A non-existent sample after prepare_inputs is a hard failure."""
        cfg = SimpleNamespace(
            output_root=tmp_path / "out",
            corpus_sample_path=tmp_path / "out" / "corpus_sample.txt.gz",
            lexicon_path=None,
        )
        recorder: dict = {}
        self._patch_common(monkeypatch, tmp_path, cfg, recorder)
        monkeypatch.setattr(sys, "argv", ["prepare_sample.py"])

        with pytest.raises(SystemExit) as exc:
            prepare_sample.main()
        assert exc.value.code == 1
