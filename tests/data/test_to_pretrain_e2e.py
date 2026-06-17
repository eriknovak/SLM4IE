"""End-to-end incremental curation on a tiny fixture corpus."""

import json
from pathlib import Path
from typing import List, Set

import pytest
import yaml

pytest.importorskip("datatrove")
pytest.importorskip("lingua")

from scripts.data.to_pretrain import _curate  # noqa: E402

# Several clearly-Slovenian sentences per dataset. Distinct topics keep
# the cross-dataset exact/sentence dedup from collapsing them, and the
# length clears the (loosened) Gopher word floors.
ALFA_DOCS: List[str] = [
    "Slovenščina je uradni jezik Republike Slovenije. "
    "Govori jo približno dva milijona ljudi. "
    "V Evropski uniji je eden od uradnih jezikov. "
    "Spada v skupino južnoslovanskih jezikov.",
    "Triglav je najvišja gora v Sloveniji in stoji v Julijskih Alpah. "
    "Mnogi planinci se vsako leto povzpnejo na njegov vrh. "
    "Pot je zahtevna in zahteva izkušnje ter dobro opremo.",
    "Ljubljana je glavno mesto Slovenije in leži ob reki Ljubljanici. "
    "V starem mestnem jedru stoji znameniti Tromostovje. "
    "Grad nad mestom ponuja lep razgled na okolico.",
]
BETA_DOCS: List[str] = [
    "Pravna pravila urejajo razmerja med posamezniki in državo. "
    "Sodišča razlagajo zakone v posameznih primerih. "
    "Ustavni sodniki varujejo temeljne pravice državljanov.",
    "Morje ob slovenski obali je del Jadranskega morja. "
    "Piran je staro pristaniško mesto z bogato zgodovino. "
    "Turisti radi obiščejo ozke ulice in obalno promenado.",
    "Kmetijstvo na podeželju se prilagaja podnebnim spremembam. "
    "Vinogradniki gojijo trto na sončnih pobočjih gričev. "
    "Pridelava hrane ostaja pomembna gospodarska panoga.",
]


def _write_extracted(input_dir: Path, key: str, texts: List[str]) -> None:
    """Write a minimal extracted <key>.jsonl for the convert stage.

    Args:
        input_dir: Directory that holds the extraction-tier JSONLs.
        key: Dataset key; the file is written as `<key>.jsonl`.
        texts: Per-document text bodies.
    """
    input_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                "text": t,
                "doc_id": f"{key}-{i}",
                "uid": f"{key}:{key}-{i}",
                "domain": "web",
                "source": key,
            },
            ensure_ascii=False,
        )
        for i, t in enumerate(texts)
    ]
    (input_dir / f"{key}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _dataset_dirs(stage_dir: Path) -> Set[str]:
    """Return dataset subfolder names that contain shards under stage_dir.

    Args:
        stage_dir: A scoped stage's output folder (e.g. `01_language/`).

    Returns:
        Names of immediate subfolders holding at least one `*.jsonl.gz`
        shard, or an empty set when the stage folder does not exist.
    """
    if not stage_dir.exists():
        return set()
    return {p.name for p in stage_dir.iterdir() if p.is_dir() and any(p.glob("*.jsonl.gz"))}


def _write_extract_config(path: Path) -> None:
    """Write an extract.yaml whose roster is exactly {alfa, beta}.

    Args:
        path: Destination path for the extract config.
    """
    cfg = {
        "input_dir": "unused",
        "output_dir": "unused",
        "datasets": {
            "alfa": {"extractor": "jsonl", "domain": "web"},
            "beta": {"extractor": "jsonl", "domain": "web"},
        },
    }
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def _write_pretrain_config(path: Path, input_dir: Path, output_dir: Path) -> None:
    """Write a permissive pretrain.yaml pointed at the test dirs.

    The thresholds are loosened so the tiny fixture docs survive language
    detection and the Gopher quality filter; the structure mirrors the
    repo's real `configs/data/pretrain.yaml`.

    Args:
        path: Destination path for the pretrain config.
        input_dir: Extraction-tier root the convert stage reads.
        output_dir: Pretrain-owned stage tree root.
    """
    cfg = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "stopwords": "sl",
        "convert": {
            "text_field": "text",
            "id_field": "doc_id",
            "metadata_fields": ["domain", "doc_id"],
            "include_annotations": False,
            "max_shard_bytes": 200000000,
        },
        "language": {
            "targets": ["sl"],
            "candidates": ["sl", "hr", "sr", "en", "de"],
            "minimum_relative_distance": 0.0,
            "mode": "filter",
            "low_accuracy": False,
            "max_chars": None,
        },
        "spam": {
            "languages": ["sl", "en"],
            "use_ldnoobw": False,
            "url_blocklist": True,
            "min_adult_hits": 2,
            "min_spam_hits": 2,
            "keep_fraction": 0.0,
            "default_language": "sl",
        },
        "quality": {
            "min_doc_words": 5,
            "max_doc_words": 1000000,
            "min_avg_word_length": 2,
            "max_avg_word_length": 15,
            "max_symbol_word_ratio": 0.3,
            "max_bullet_lines_ratio": 0.9,
            "max_ellipsis_lines_ratio": 0.9,
            "max_non_alpha_words_ratio": 0.5,
            "min_stop_words": 0,
        },
        "repetition": {},
        "exact_dedup": {
            "precision": 64,
            "hash_fc": "xxhash",
            "only_dedup_in_index": True,
        },
        "sentence_dedup": {
            "n_sentences": 3,
            "min_doc_words": 5,
            "min_num_sentences": 1,
            "split_sentences": True,
        },
        "stats": {"top_k_words": 200},
    }
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


@pytest.mark.slow
def test_subset_run_then_all_is_incremental(tmp_path: Path) -> None:
    """A subset run does scoped-only work; a later --all skips it and runs corpus stages.

    First runs the real datatrove pipeline for `alfa` alone and asserts
    only its scoped stages ran (no corpus dedup/stats). Then runs `--all`
    and asserts `alfa`'s scoped work is skipped, `beta` is processed
    through the scoped stages, and the corpus dedup/stats stages run
    across both datasets.
    """
    in_dir = tmp_path / "extracted"
    out_dir = tmp_path / "pretrain"
    _write_extracted(in_dir, "alfa", ALFA_DOCS)
    _write_extracted(in_dir, "beta", BETA_DOCS)

    extract_cfg = tmp_path / "extract.yaml"
    pretrain_cfg = tmp_path / "pretrain.yaml"
    _write_extract_config(extract_cfg)
    _write_pretrain_config(pretrain_cfg, in_dir, out_dir)

    # 1) Subset run for alfa: scoped stages only.
    _curate(
        datasets=["alfa"],
        run_all=False,
        stage="all",
        input_dir=in_dir,
        output_dir=out_dir,
        force=False,
        workers=1,
        pretrain_config=pretrain_cfg,
        extract_config=extract_cfg,
    )
    assert _dataset_dirs(out_dir / "01_language") == {"alfa"}
    assert _dataset_dirs(out_dir / "04_repetition") == {"alfa"}
    assert (out_dir / "03_quality" / "alfa" / ".complete").exists()
    # No corpus stage ran during the subset pass.
    assert not (out_dir / "05_1_dedup" / ".complete").exists()
    assert not (out_dir / "06_statistics" / ".complete").exists()

    # Capture alfa's per-dataset sentinel mtimes before the --all run so we
    # can prove they are NOT rewritten (i.e. alfa is skipped, not reprocessed).
    alfa_sentinels = [
        out_dir / "00_convert" / "alfa" / ".complete",
        out_dir / "01_language" / "alfa" / ".complete",
        out_dir / "02_spam" / "alfa" / ".complete",
        out_dir / "03_quality" / "alfa" / ".complete",
        out_dir / "04_repetition" / "alfa" / ".complete",
    ]
    for p in alfa_sentinels:
        assert p.exists(), f"Expected alfa sentinel after subset run: {p}"
    alfa_mtimes_before = {p: p.stat().st_mtime_ns for p in alfa_sentinels}

    # 2) --all: alfa scoped work skipped, beta processed, corpus stages run.
    _curate(
        datasets=[],
        run_all=True,
        stage="all",
        input_dir=in_dir,
        output_dir=out_dir,
        force=False,
        workers=1,
        pretrain_config=pretrain_cfg,
        extract_config=extract_cfg,
    )
    assert _dataset_dirs(out_dir / "04_repetition") == {"alfa", "beta"}
    assert (out_dir / "05_1_dedup" / ".complete").exists()
    assert (out_dir / "06_statistics" / ".complete").exists()
    # Prove alfa's scoped sentinels were NOT rewritten during --all (skip-proof).
    for p in alfa_sentinels:
        assert p.stat().st_mtime_ns == alfa_mtimes_before[p], (
            f"{p} was rewritten — alfa should have been skipped during --all"
        )
    # Beta's scoped work must now be present too.
    assert (out_dir / "04_repetition" / "beta" / ".complete").exists()


def _write_extract_config_with_ghost(path: Path) -> None:
    """Write an extract.yaml whose roster is {alfa, beta, ghost}.

    `ghost` is declared but the test writes no `extracted/ghost.jsonl`, so
    it produces no convert shards — mirroring a roster dataset that was
    never downloaded.

    Args:
        path: Destination path for the extract config.
    """
    cfg = {
        "input_dir": "unused",
        "output_dir": "unused",
        "datasets": {
            "alfa": {"extractor": "jsonl", "domain": "web"},
            "beta": {"extractor": "jsonl", "domain": "web"},
            "ghost": {"extractor": "jsonl", "domain": "web"},
        },
    }
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


@pytest.mark.slow
def test_override_reruns_only_target_dataset(tmp_path: Path) -> None:
    """Adding a per-dataset override re-runs only that dataset's scoped stage.

    Runs the full pipeline for {alfa, beta}, then adds a `beta` quality
    override and re-runs. `alfa`'s quality sentinel must be untouched
    (same hash, not rewritten); `beta`'s must change (re-run with the
    merged config).
    """
    from slm4ie.data.curate.sentinel import Sentinel, read_sentinel

    def sentinel(folder: Path) -> Sentinel:
        s = read_sentinel(folder)
        assert s is not None, f"expected a sentinel under {folder}"
        return s

    in_dir = tmp_path / "extracted"
    out_dir = tmp_path / "pretrain"
    _write_extracted(in_dir, "alfa", ALFA_DOCS)
    _write_extracted(in_dir, "beta", BETA_DOCS)

    extract_cfg = tmp_path / "extract.yaml"
    pretrain_cfg = tmp_path / "pretrain.yaml"
    _write_extract_config(extract_cfg)
    _write_pretrain_config(pretrain_cfg, in_dir, out_dir)

    def run() -> None:
        _curate(
            datasets=[],
            run_all=True,
            stage="all",
            input_dir=in_dir,
            output_dir=out_dir,
            force=False,
            workers=1,
            pretrain_config=pretrain_cfg,
            extract_config=extract_cfg,
        )

    # 1) Full run, no overrides.
    run()
    alfa_q = out_dir / "03_quality" / "alfa" / ".complete"
    beta_q = out_dir / "03_quality" / "beta" / ".complete"
    alfa_hash0 = sentinel(alfa_q.parent).config_hash
    beta_hash0 = sentinel(beta_q.parent).config_hash
    alfa_mtime0 = alfa_q.stat().st_mtime_ns
    beta_mtime0 = beta_q.stat().st_mtime_ns

    # 2) Add a beta-only quality override (changes the hash, keeps docs).
    cfg = yaml.safe_load(pretrain_cfg.read_text())
    cfg["overrides"] = {"beta": {"quality": {"max_ellipsis_lines_ratio": 0.8}}}
    pretrain_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    run()

    alfa_hash1 = sentinel(alfa_q.parent).config_hash
    beta_hash1 = sentinel(beta_q.parent).config_hash

    # alfa: untouched (same hash, sentinel not rewritten).
    assert alfa_hash1 == alfa_hash0
    assert alfa_q.stat().st_mtime_ns == alfa_mtime0, "alfa quality should not re-run"
    # beta: re-run with the merged override (hash changed, sentinel rewritten).
    assert beta_hash1 != beta_hash0
    assert beta_q.stat().st_mtime_ns != beta_mtime0, "beta quality should re-run"
    # The effective slice stored in beta's sentinel carries the override.
    assert sentinel(beta_q.parent).config_slice["max_ellipsis_lines_ratio"] == 0.8


@pytest.mark.slow
def test_all_skips_roster_dataset_with_no_input(tmp_path: Path) -> None:
    """--all tolerates a roster dataset that was never extracted.

    Regression: a dataset declared in extract.yaml but with no
    `extracted/<key>.jsonl` (e.g. not downloaded) produces no convert
    shards. The scoped stages must drop it instead of raising
    FileNotFoundError when building the per-dataset input view.
    """
    in_dir = tmp_path / "extracted"
    out_dir = tmp_path / "pretrain"
    _write_extracted(in_dir, "alfa", ALFA_DOCS)
    _write_extracted(in_dir, "beta", BETA_DOCS)
    # 'ghost' is in the roster but has NO extracted/<key>.jsonl.

    extract_cfg = tmp_path / "extract.yaml"
    pretrain_cfg = tmp_path / "pretrain.yaml"
    _write_extract_config_with_ghost(extract_cfg)
    _write_pretrain_config(pretrain_cfg, in_dir, out_dir)

    # Must complete without FileNotFoundError on 'ghost'.
    _curate(
        datasets=[],
        run_all=True,
        stage="all",
        input_dir=in_dir,
        output_dir=out_dir,
        force=False,
        workers=1,
        pretrain_config=pretrain_cfg,
        extract_config=extract_cfg,
    )
    # alfa + beta are processed; ghost has no shards at any scoped stage.
    assert _dataset_dirs(out_dir / "01_language") == {"alfa", "beta"}
    assert _dataset_dirs(out_dir / "04_repetition") == {"alfa", "beta"}
    # Corpus stages completed across the real datasets.
    assert (out_dir / "05_1_dedup" / ".complete").exists()
    assert (out_dir / "06_statistics" / ".complete").exists()
