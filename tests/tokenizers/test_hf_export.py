"""Tests for slm4ie/tokenizers/hf_export.py (HuggingFace export + offsets)."""

import dataclasses
import json
from pathlib import Path

import pytest

import slm4ie.tokenizers.backends  # noqa: F401  (registers backends)
from slm4ie.tokenizers.base import TrainContext
from slm4ie.tokenizers.config import ExportTarget
from slm4ie.tokenizers.hf_export import (
    ParityError,
    check_parity,
    export_sweep_runs,
    load_pretrained,
    read_parity_texts,
    save_pretrained_dir,
    scores_tie_in_float32,
    to_hf,
)
from slm4ie.tokenizers.morphology import MorphLexicon, _add_to_lexicon, segment_form
from slm4ie.tokenizers.registry import get_tokenizer

_LEXICON_ENTRIES = [
    ("hiša", "hiša", "Ncfsn"),
    ("hiše", "hiša", "Ncfsg"),
    ("hiši", "hiša", "Ncfsd"),
    ("hišami", "hiša", "Ncfpi"),
]

_CORPUS = [
    "hiša ob cesti",
    "hiše so velike in lepe",
    "mačka spi v hiši",
    "hišami gradijo mesto",
] * 60

_SPECIAL = ["<pad>", "<unk>", "<s>", "</s>", "<mask>"]
_TEXT = "hiše ob cesti"
# characters SentencePiece's NFKC normalizer rewrites: ellipsis, trademark, no-break space, double space
_NORMALIZED_TEXT = "hiša… ob cesti™  spi"
_FAST_BACKENDS = ["bpe", "charbpe", "wordpiece", "unigram", "morphbpe"]
_MODERNBERT = ExportTarget(
    name="modernbert",
    roles={
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "mask_token": "<mask>",
        "cls_token": "<s>",
        "sep_token": "</s>",
        "bos_token": "<s>",
        "eos_token": "</s>",
    },
    single="<s> $A </s>",
    pair="<s> $A </s> $B:1 </s>:1",
    model_max_length=8192,
)


def _lexicon() -> MorphLexicon:
    """Build the fixture morpheme lexicon.

    Returns:
        MorphLexicon: A lexicon over the fixture entries.
    """
    lexicon = MorphLexicon()
    for form, lemma, msd in _LEXICON_ENTRIES:
        seg = segment_form(form, lemma, msd)
        if seg is not None and seg.is_reliable:
            _add_to_lexicon(lexicon, seg)
    return lexicon


def _train_artifact(name: str, tmp_path: Path) -> Path:
    """Train a backend and save it under `tmp_path`.

    Args:
        name (str): Registry key.
        tmp_path (Path): Destination directory.

    Returns:
        Path: The artifact directory.
    """
    tokenizer = get_tokenizer(name)()
    tokenizer.train(_CORPUS, 160, config=TrainContext(special_tokens=_SPECIAL, lexicon=_lexicon()))
    tokenizer.save(tmp_path)
    return tmp_path


@pytest.mark.parametrize("name", _FAST_BACKENDS)
class TestFastExport:
    """Export of the five fast backends to PreTrainedTokenizerFast."""

    def test_offsets_present_and_aligned(self, name: str, tmp_path: Path):
        """The exported tokenizer yields offset mappings into the text."""
        hf = to_hf(_train_artifact(name, tmp_path))
        out = hf(_TEXT, return_offsets_mapping=True)
        assert out["input_ids"]
        offsets = out["offset_mapping"]
        assert len(offsets) == len(out["input_ids"])
        # Every span lies within the text.
        assert all(0 <= s <= e <= len(_TEXT) for s, e in offsets)

    def test_decode_runs(self, name: str, tmp_path: Path):
        """Decoding the ids returns the (non-empty) text content."""
        hf = to_hf(_train_artifact(name, tmp_path))
        ids = hf(_TEXT)["input_ids"]
        decoded = hf.decode(ids, skip_special_tokens=True)
        # All schemes recover the word characters (spacing may differ for
        # character-level BPE, which has no continuation marker).
        assert "hiš" in decoded and "cesti" in decoded

    def test_target_wraps_single_text_in_cls_and_sep(self, name: str, tmp_path: Path):
        """With the ModernBERT target a text is encoded as `<s> ... </s>`."""
        hf = to_hf(_train_artifact(name, tmp_path), _MODERNBERT)
        ids = hf(_TEXT)["input_ids"]
        assert ids[0] == hf.cls_token_id == hf.convert_tokens_to_ids("<s>")
        assert ids[-1] == hf.sep_token_id == hf.convert_tokens_to_ids("</s>")

    def test_target_wraps_pair_as_cls_a_sep_b_sep(self, name: str, tmp_path: Path):
        """With the ModernBERT target a pair is encoded as `<s> A </s> B </s>`."""
        hf = to_hf(_train_artifact(name, tmp_path), _MODERNBERT)
        body = hf(_TEXT, add_special_tokens=False)["input_ids"]
        assert hf(_TEXT, _TEXT)["input_ids"] == [hf.cls_token_id, *body, hf.sep_token_id, *body, hf.sep_token_id]

    def test_target_without_special_tokens_adds_no_markers(self, name: str, tmp_path: Path):
        """`add_special_tokens=False` gives the same ids as the export without a target."""
        artifact = _train_artifact(name, tmp_path)
        wrapped = to_hf(artifact, _MODERNBERT)(_TEXT, add_special_tokens=False)["input_ids"]
        assert wrapped == to_hf(artifact)(_TEXT, add_special_tokens=False)["input_ids"]


@pytest.mark.parametrize("name", _FAST_BACKENDS)
def test_target_export_matches_backend_ids(name: str, tmp_path: Path):
    """A target export reloaded from disk encodes the corpus as the trained backend does."""
    artifact = _train_artifact(name, tmp_path)
    written = save_pretrained_dir(artifact, target=_MODERNBERT)
    texts = [*dict.fromkeys(_CORPUS), _TEXT, "neznana beseda", _NORMALIZED_TEXT]
    assert check_parity(artifact, written, texts) == len(texts)


@pytest.mark.parametrize("name", _FAST_BACKENDS)
def test_plain_export_matches_backend_ids(name: str, tmp_path: Path):
    """An export without a target, reloaded from disk, encodes as the trained backend does."""
    artifact = _train_artifact(name, tmp_path / "artifact")
    written = save_pretrained_dir(artifact, out_dir=tmp_path / "plain")
    texts = [*dict.fromkeys(_CORPUS), _TEXT, _NORMALIZED_TEXT]
    assert check_parity(artifact, written, texts) == len(texts)


def test_parity_check_rejects_a_different_tokenizer(tmp_path: Path):
    """An export compared with another backend's artifact fails, naming the mismatch count."""
    bpe = _train_artifact("bpe", tmp_path / "bpe")
    wordpiece = _train_artifact("wordpiece", tmp_path / "wordpiece")
    written = save_pretrained_dir(bpe, target=_MODERNBERT)
    with pytest.raises(ParityError, match="encode differently"):
        check_parity(wordpiece, written, [_TEXT])


class TestParityTies:
    """The id check accepts Unigram splits that tie at float32 precision and nothing else."""

    @staticmethod
    def _export_with_backend_ids(name: str, tmp_path: Path, monkeypatch, reorder) -> tuple[Path, Path]:
        """Export `name` and make its backend return a reordering of the exported ids.

        Args:
            name: Registry key.
            tmp_path: Destination directory.
            monkeypatch: The pytest monkeypatch fixture.
            reorder: Maps the exported ids to the ids the backend returns.

        Returns:
            The artifact and the export directory.
        """
        from transformers import AutoTokenizer

        artifact = _train_artifact(name, tmp_path)
        written = save_pretrained_dir(artifact, target=_MODERNBERT)
        ids = AutoTokenizer.from_pretrained(str(written))(_TEXT, add_special_tokens=False)["input_ids"]
        monkeypatch.setattr(get_tokenizer(name), "encode_ids", lambda self, text: reorder(ids))
        return artifact, written

    def test_unigram_split_with_equal_score_passes(self, tmp_path: Path, monkeypatch):
        """A Unigram split of the same pieces in another order counts as a tie."""
        artifact, written = self._export_with_backend_ids("unigram", tmp_path, monkeypatch, lambda ids: ids[::-1])
        assert check_parity(artifact, written, [_TEXT]) == 1

    def test_unigram_split_of_other_length_fails(self, tmp_path: Path, monkeypatch):
        """A Unigram split with one token fewer is a mismatch, not a tie."""
        artifact, written = self._export_with_backend_ids("unigram", tmp_path, monkeypatch, lambda ids: ids[:-1])
        with pytest.raises(ParityError):
            check_parity(artifact, written, [_TEXT])

    def test_bpe_reordered_split_fails(self, tmp_path: Path, monkeypatch):
        """Outside Unigram any difference fails, since no piece scores define a tie."""
        artifact, written = self._export_with_backend_ids("bpe", tmp_path, monkeypatch, lambda ids: ids[::-1])
        with pytest.raises(ParityError):
            check_parity(artifact, written, [_TEXT])


@pytest.mark.parametrize(
    ("total", "other_total", "ties"),
    [
        (-69958.6178, -69958.6103, True),  # unigram-32000 line 324: gap below float32's 0.0078 step
        (-15248.6615, -15248.6606, True),  # unigram-32000 line 606: gap below float32's 0.002 step
        (-69958.6178, -69958.5678, False),  # a gap of 0.05 is six float32 steps
        (-31.901, -31.894, False),  # the same gap on a short text is far above the step
    ],
)
def test_scores_tie_in_float32(total: float, other_total: float, ties: bool):
    """Two path scores tie only when float32 summing cannot tell them apart."""
    assert scores_tie_in_float32(total, other_total) is ties


class TestTargetRoles:
    """A target sets the special-token roles and the maximum length."""

    def test_roles_and_length_follow_the_target(self, tmp_path: Path):
        """The ModernBERT target names `<s>` and `</s>` as cls/bos and sep/eos."""
        hf = to_hf(_train_artifact("bpe", tmp_path), _MODERNBERT)
        assert (hf.cls_token, hf.bos_token, hf.sep_token, hf.eos_token) == ("<s>", "<s>", "</s>", "</s>")
        assert (hf.pad_token, hf.unk_token, hf.mask_token) == ("<pad>", "<unk>", "<mask>")
        assert hf.model_max_length == 8192

    def test_bpe_special_tokens_keep_ids_zero_to_four(self, tmp_path: Path):
        """For BPE the export keeps the trained ids: pad 0, unk 1, cls 2, sep 3, mask 4."""
        hf = to_hf(_train_artifact("bpe", tmp_path), _MODERNBERT)
        ids = (hf.pad_token_id, hf.unk_token_id, hf.cls_token_id, hf.sep_token_id, hf.mask_token_id)
        assert ids == (0, 1, 2, 3, 4)

    def test_morphpiece_target_wraps_single_text(self, tmp_path: Path):
        """MorphPiece with the ModernBERT target wraps a text as `<s> ... </s>`."""
        hf = to_hf(_train_artifact("morphpiece", tmp_path), _MODERNBERT)
        body = hf(_TEXT, add_special_tokens=False)["input_ids"]
        assert hf(_TEXT)["input_ids"] == [hf.cls_token_id, *body, hf.sep_token_id]

    def test_morphpiece_target_wraps_pair_with_type_ids(self, tmp_path: Path):
        """MorphPiece wraps a pair as `<s> A </s> B </s>`, the second part with type id 1."""
        hf = to_hf(_train_artifact("morphpiece", tmp_path), _MODERNBERT)
        body = hf(_TEXT, add_special_tokens=False)["input_ids"]
        pair = hf(_TEXT, _TEXT)
        assert pair["input_ids"] == [hf.cls_token_id, *body, hf.sep_token_id, *body, hf.sep_token_id]
        assert pair["token_type_ids"] == [0] * (len(body) + 2) + [1] * (len(body) + 1)

    def test_morphpiece_roles_and_length_follow_the_target(self, tmp_path: Path):
        """MorphPiece takes the target's cls and sep tokens and its maximum length."""
        hf = to_hf(_train_artifact("morphpiece", tmp_path), _MODERNBERT)
        assert (hf.cls_token, hf.sep_token, hf.model_max_length) == ("<s>", "</s>", 8192)


class TestAutoTokenizer:
    """A fast export round-trips through AutoTokenizer.from_pretrained."""

    def test_save_and_reload(self, tmp_path: Path):
        """save_pretrained_dir produces an AutoTokenizer-loadable directory."""
        from transformers import AutoTokenizer

        artifact = _train_artifact("bpe", tmp_path)
        save_pretrained_dir(artifact)
        reloaded = AutoTokenizer.from_pretrained(str(artifact))
        assert reloaded(_TEXT)["input_ids"] == to_hf(artifact)(_TEXT)["input_ids"]

    def test_target_export_reloads_with_roles_and_template(self, tmp_path: Path):
        """A target export writes `hf-<target>/`, which reloads with the same roles and wrapping."""
        from transformers import AutoTokenizer

        artifact = _train_artifact("bpe", tmp_path)
        written = save_pretrained_dir(artifact, target=_MODERNBERT)
        reloaded = AutoTokenizer.from_pretrained(str(written))
        exported = to_hf(artifact, _MODERNBERT)
        assert written == artifact / "hf-modernbert"
        assert (reloaded.cls_token_id, reloaded.sep_token_id) == (exported.cls_token_id, exported.sep_token_id)
        assert reloaded(_TEXT)["input_ids"] == exported(_TEXT)["input_ids"]
        assert reloaded.model_max_length == 8192


class TestMorphPieceExport:
    """MorphPiece exports as a slow tokenizer that carries its own code."""

    def test_export_reloads_through_autotokenizer(self, tmp_path: Path):
        """The export reloads through AutoTokenizer with custom code allowed and wraps as the target says."""
        from transformers import AutoTokenizer

        artifact = _train_artifact("morphpiece", tmp_path)
        written = save_pretrained_dir(artifact, target=_MODERNBERT)
        reloaded = AutoTokenizer.from_pretrained(str(written), trust_remote_code=True)
        assert reloaded(_TEXT)["input_ids"] == to_hf(artifact, _MODERNBERT)(_TEXT)["input_ids"]

    def test_export_code_imports_nothing_from_this_package(self, tmp_path: Path):
        """The tokenizer code copied into the export imports nothing from slm4ie."""
        written = save_pretrained_dir(_train_artifact("morphpiece", tmp_path), target=_MODERNBERT)
        code = (written / "hf_morph_piece.py").read_text()
        assert "from slm4ie" not in code and "import slm4ie" not in code

    def test_export_has_no_fast_tokenizer_file(self, tmp_path: Path):
        """The export holds no `tokenizer.json`, so AutoTokenizer does not treat it as fast."""
        written = save_pretrained_dir(_train_artifact("morphpiece", tmp_path), target=_MODERNBERT)
        assert not (written / "tokenizer.json").exists()

    def test_export_matches_backend_ids(self, tmp_path: Path):
        """The reloaded MorphPiece export encodes the corpus as the trained backend does."""
        artifact = _train_artifact("morphpiece", tmp_path)
        written = save_pretrained_dir(artifact, target=_MODERNBERT)
        texts = [*dict.fromkeys(_CORPUS), _TEXT, "neznana beseda", _NORMALIZED_TEXT]
        assert check_parity(artifact, written, texts) == len(texts)

    def test_encode_with_offsets(self, tmp_path: Path):
        """encode_with_offsets returns aligned ids and spans."""
        hf = load_pretrained(_train_artifact("morphpiece", tmp_path))
        encoded = hf.encode_with_offsets(_TEXT)
        assert encoded["input_ids"]
        assert len(encoded["input_ids"]) == len(encoded["offset_mapping"])
        assert all(0 <= s <= e <= len(_TEXT) for s, e in encoded["offset_mapping"])

    def test_decode_recovers_known_word(self, tmp_path: Path):
        """Decoding recovers the known word's surface form."""
        hf = load_pretrained(_train_artifact("morphpiece", tmp_path))
        ids = hf.encode_with_offsets(_TEXT)["input_ids"]
        assert "hiše" in hf.decode(ids, skip_special_tokens=True)


class TestSweepExport:
    """export_sweep_runs exports each run per target, checked, skippable and summarised."""

    @staticmethod
    def _sweep(tmp_path: Path, names: list[str]) -> Path:
        """Train one artifact per backend under a sweep output root.

        Args:
            tmp_path: Test directory.
            names: Registry keys to train.

        Returns:
            Path: The output root holding `<name>-160/` per backend.
        """
        root = tmp_path / "tokenizers"
        for name in names:
            _train_artifact(name, root / f"{name}-160")
        return root

    @staticmethod
    def _export(root: Path, keys: list[str], force: bool = False):
        """Run the sweep export for the ModernBERT target with the fixture texts.

        Args:
            root: Sweep output root.
            keys: Run keys to export.
            force: Rewrite complete exports.

        Returns:
            The export outcome.
        """
        texts = [*dict.fromkeys(_CORPUS), _TEXT]
        return export_sweep_runs(root, keys, [_MODERNBERT], texts, root / "_reports" / "export-summary.json", force)

    def test_writes_target_dir_and_summary_record(self, tmp_path: Path):
        """An export lands in `hf-<target>/` and its summary record names the cls and sep ids."""
        root = self._sweep(tmp_path, ["bpe"])
        summary = self._export(root, ["bpe-160"])
        record = json.loads((root / "_reports" / "export-summary.json").read_text())["bpe-160/modernbert"]
        vocab = json.loads((root / "bpe-160" / "hf-modernbert" / "tokenizer.json").read_text())["model"]["vocab"]
        assert summary.exported == ["bpe-160/modernbert"]
        assert record["status"] == "exported"
        assert record["special_token_ids"]["cls_token"] == vocab["<s>"]
        assert record["special_token_ids"]["sep_token"] == vocab["</s>"]

    def test_complete_export_is_skipped_unless_forced(self, tmp_path: Path):
        """A second export skips a complete directory; `force` rewrites it."""
        root = self._sweep(tmp_path, ["bpe"])
        self._export(root, ["bpe-160"])
        assert self._export(root, ["bpe-160"]).skipped == ["bpe-160/modernbert"]
        assert self._export(root, ["bpe-160"], force=True).exported == ["bpe-160/modernbert"]

    def test_morphpiece_exports_with_the_other_backends(self, tmp_path: Path):
        """MorphPiece exports and is recorded like a fast backend, its skip working the same way."""
        root = self._sweep(tmp_path, ["bpe", "morphpiece"])
        summary = self._export(root, ["morphpiece-160", "bpe-160"])
        record = json.loads((root / "_reports" / "export-summary.json").read_text())["morphpiece-160/modernbert"]
        assert (summary.exported, summary.failed) == (["morphpiece-160/modernbert", "bpe-160/modernbert"], [])
        assert record["status"] == "exported"
        assert self._export(root, ["morphpiece-160"]).skipped == ["morphpiece-160/modernbert"]

    def test_failed_id_check_leaves_no_complete_export(self, tmp_path: Path, monkeypatch):
        """A run whose ids differ is reported failed and leaves neither the export nor its partial folder."""
        root = self._sweep(tmp_path, ["bpe"])
        monkeypatch.setattr(get_tokenizer("bpe"), "encode_ids", lambda self, text: [])
        summary = self._export(root, ["bpe-160"])
        assert summary.failed == ["bpe-160/modernbert"]
        assert not (root / "bpe-160" / "hf-modernbert").exists()
        assert not (root / "bpe-160" / "hf-modernbert.partial").exists()

    def test_export_missing_a_written_file_is_exported_again(self, tmp_path: Path):
        """A folder that lost a file its lineage lists is exported again instead of skipped."""
        root = self._sweep(tmp_path, ["bpe"])
        self._export(root, ["bpe-160"])
        (root / "bpe-160" / "hf-modernbert" / "tokenizer.json").unlink()
        assert self._export(root, ["bpe-160"]).exported == ["bpe-160/modernbert"]

    def test_reevaluated_run_is_skipped(self, tmp_path: Path):
        """Rewriting a run's evaluation metrics leaves its digest unchanged, so the export is skipped."""
        root = self._sweep(tmp_path, ["bpe"])
        self._export(root, ["bpe-160"])
        (root / "bpe-160" / "metrics.json").write_text('{"fertility": 1.5}')
        assert self._export(root, ["bpe-160"]).skipped == ["bpe-160/modernbert"]

    @staticmethod
    def _retrain(artifact_dir: Path) -> None:
        """Change an artifact file the way a retraining would, so its digest changes.

        Args:
            artifact_dir: The trained artifact directory.
        """
        meta_path = artifact_dir / "metadata.json"
        meta_path.write_text(json.dumps({**json.loads(meta_path.read_text()), "retrained": True}))

    @staticmethod
    def _record(root: Path, entry: str) -> dict:
        """Read one record of the export summary.

        Args:
            root: Sweep output root.
            entry: `<run key>/<target name>`.

        Returns:
            The summary record.
        """
        return json.loads((root / "_reports" / "export-summary.json").read_text())[entry]

    def test_record_digests_match_after_export(self, tmp_path: Path):
        """A fresh export records the same digest for its folder and the trained artifact."""
        root = self._sweep(tmp_path, ["bpe"])
        self._export(root, ["bpe-160"])
        record = self._record(root, "bpe-160/modernbert")
        assert record["export_digest"] == record["artifact_digest"]

    def test_retrained_artifact_is_exported_again_without_force(self, tmp_path: Path):
        """A changed artifact makes the next normal run export again instead of skipping."""
        root = self._sweep(tmp_path, ["bpe"])
        self._export(root, ["bpe-160"])
        self._retrain(root / "bpe-160")
        assert self._export(root, ["bpe-160"]).exported == ["bpe-160/modernbert"]

    def test_changed_target_is_exported_again_without_force(self, tmp_path: Path):
        """Changed target settings make the next normal run export again."""
        root = self._sweep(tmp_path, ["bpe"])
        self._export(root, ["bpe-160"])
        longer = dataclasses.replace(_MODERNBERT, model_max_length=1024)
        summary = export_sweep_runs(root, ["bpe-160"], [longer], [_TEXT], root / "_reports" / "export-summary.json")
        assert summary.exported == ["bpe-160/modernbert"]

    def test_failed_reexport_keeps_old_folder_marked_stale(self, tmp_path: Path, monkeypatch):
        """A failed re-export keeps the earlier folder, recorded as stale with differing digests."""
        root = self._sweep(tmp_path, ["bpe"])
        self._export(root, ["bpe-160"])
        self._retrain(root / "bpe-160")
        monkeypatch.setattr(get_tokenizer("bpe"), "encode_ids", lambda self, text: [])
        summary = self._export(root, ["bpe-160"])
        record = self._record(root, "bpe-160/modernbert")
        assert summary.failed == ["bpe-160/modernbert"]
        assert (root / "bpe-160" / "hf-modernbert" / "tokenizer.json").exists()
        assert record["status"] == "stale"
        assert record["export_digest"] != record["artifact_digest"]

    def test_skipped_export_without_record_is_recorded(self, tmp_path: Path):
        """A complete export missing from a deleted summary is listed again when skipped."""
        root = self._sweep(tmp_path, ["bpe"])
        self._export(root, ["bpe-160"])
        (root / "_reports" / "export-summary.json").unlink()
        self._export(root, ["bpe-160"])
        records = json.loads((root / "_reports" / "export-summary.json").read_text())
        assert records["bpe-160/modernbert"]["status"] == "exported"

    def test_unreadable_export_fails_alone(self, tmp_path: Path):
        """A complete export that cannot be loaded fails that run, and the other run still exports."""
        root = self._sweep(tmp_path, ["bpe", "wordpiece"])
        self._export(root, ["bpe-160"])
        (root / "bpe-160" / "hf-modernbert" / "tokenizer.json").write_text("not json")
        summary = self._export(root, ["bpe-160", "wordpiece-160"])
        assert (summary.failed, summary.exported) == (["bpe-160/modernbert"], ["wordpiece-160/modernbert"])
        assert self._record(root, "bpe-160/modernbert")["status"] == "failed"

    def test_summary_keeps_records_of_other_runs(self, tmp_path: Path):
        """Exporting one run keeps the summary records of runs exported earlier."""
        root = self._sweep(tmp_path, ["bpe", "wordpiece"])
        self._export(root, ["bpe-160"])
        self._export(root, ["wordpiece-160"])
        records = json.loads((root / "_reports" / "export-summary.json").read_text())
        assert set(records) == {"bpe-160/modernbert", "wordpiece-160/modernbert"}


class TestReadParityTexts:
    """read_parity_texts reads the first non-empty lines of the evaluation sample."""

    def test_reads_the_first_lines(self, tmp_path: Path):
        """It returns the first `n_lines` non-empty lines."""
        sample = tmp_path / "eval_sample.txt"
        sample.write_text("prva\n\ndruga\ntretja\n", encoding="utf-8")
        assert read_parity_texts(sample, 2) == ["prva", "druga"]

    def test_missing_sample_names_the_sample_step(self, tmp_path: Path):
        """A missing sample fails and points at the `sample` step."""
        with pytest.raises(FileNotFoundError, match="sample"):
            read_parity_texts(tmp_path / "eval_sample.txt", 10)

    def test_empty_sample_rejected(self, tmp_path: Path):
        """A sample with no text fails instead of checking nothing."""
        sample = tmp_path / "eval_sample.txt"
        sample.write_text("\n\n", encoding="utf-8")
        with pytest.raises(ValueError, match="fewer than"):
            read_parity_texts(sample, 10)

    def test_short_sample_rejected(self, tmp_path: Path):
        """A sample with fewer lines than asked fails instead of checking fewer texts."""
        sample = tmp_path / "eval_sample.txt"
        sample.write_text("prva\ndruga\n", encoding="utf-8")
        with pytest.raises(ValueError, match="2 text lines"):
            read_parity_texts(sample, 3)
