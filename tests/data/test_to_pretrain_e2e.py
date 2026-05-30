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
    assert _dataset_dirs(out_dir / "03_repetition") == {"alfa"}
    assert (out_dir / "02_quality" / "alfa" / ".complete").exists()
    # No corpus stage ran during the subset pass.
    assert not (out_dir / "04_1_dedup" / ".complete").exists()
    assert not (out_dir / "05_statistics" / ".complete").exists()

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
    assert _dataset_dirs(out_dir / "03_repetition") == {"alfa", "beta"}
    assert (out_dir / "04_1_dedup" / ".complete").exists()
    assert (out_dir / "05_statistics" / ".complete").exists()
