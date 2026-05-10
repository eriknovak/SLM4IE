"""End-to-end smoke tests for the merged curation pipeline.

Builds two synthetic shards with one cross-shard verbatim duplicate and
one shared 3-sentence span, runs `build_curate_executors(...)` against
a tempdir, and asserts the final corpus contains the expected survivors
plus a `statistics/` folder with both an aggregate JSON and per-dataset
breakdowns.
"""

import gzip
import importlib.metadata  # noqa: F401  (datatrove workaround)
import importlib.util  # noqa: F401  (datatrove workaround)
import json
from pathlib import Path
from typing import List

import pytest

pytest.importorskip("datatrove")
pytest.importorskip("lingua")

from datatrove.pipeline.dedup import SentDedupConfig  # noqa: E402

from slm4ie.data.curate.pipeline import CuratePaths, build_curate_executors  # noqa: E402


SHARED_DOC = (
    "Slovenščina je uradni jezik Republike Slovenije. "
    "Govori jo približno dva milijona ljudi. "
    "V Evropski uniji je eden od uradnih jezikov. "
    "Spada v skupino južnoslovanskih jezikov. "
    "Razvijala se je iz praslovanščine pred več stoletji. "
    "Standardni jezik se uporablja v javnem govoru, šolstvu in medijih. "
    "Pogovorni jezik ima številne narečne različice po vsej državi. "
    "Pisni jezik temelji na latinici z dodatki za posebne glasove."
)
SHARED_SPAN = (
    "Akademske raziskave preučujejo družbene vzorce. "
    "Sociologi analizirajo gibanja prebivalstva. "
    "Lingvisti dokumentirajo razvoj jezika skozi čas."
)
A2_TEXT = (
    SHARED_SPAN
    + " Filozofi razmišljajo o naravi spoznanja. "
      "Zgodovinarji raziskujejo arhive in primarne vire. "
      "Znanstveniki sodelujejo v interdisciplinarnih projektih po vsem svetu. "
      "Rezultati se objavljajo v recenziranih revijah."
)
B2_TEXT = (
    "Pravna pravila urejajo razmerja med posamezniki in državo. "
    + SHARED_SPAN
    + " Sodišča razlagajo zakone v posameznih primerih. "
      "Ustavni sodniki varujejo temeljne pravice državljanov. "
      "Mednarodno pravo ureja odnose med državami v globalnem sistemu."
)


def _write_shard(path: Path, dataset: str, domain: str, docs: List[dict]) -> None:
    """Gzip-write a list of (id, text) docs as datatrove JSONL.

    *path* must point at a `<dataset>/<NNNNN>.jsonl.gz` file inside the
    new sharded layout; the parent folder is created if missing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for d in docs:
            line = {
                "text": d["text"],
                "id": d["id"],
                "dataset": dataset,
                "domain": domain,
            }
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")


@pytest.mark.slow
def test_final_corpus_drops_cross_dataset_duplicates(tmp_path: Path) -> None:
    """Two shards with one full-doc dup and one shared span produce 5 survivors."""
    input_folder = tmp_path / "datatrove"
    _write_shard(
        input_folder / "alpha" / "00000.jsonl.gz",
        dataset="alpha",
        domain="scientific",
        docs=[
            {"id": "alpha:1", "text": SHARED_DOC},
            {"id": "alpha:2", "text": A2_TEXT},
            {"id": "alpha:3", "text": "Solnce sveti nad gorami in dolinami slovenskih krajev. " * 8},
        ],
    )
    _write_shard(
        input_folder / "beta" / "00000.jsonl.gz",
        dataset="beta",
        domain="legal",
        docs=[
            {"id": "beta:1", "text": SHARED_DOC},  # whole-doc dup of alpha:1
            {"id": "beta:2", "text": B2_TEXT},     # shares 3-sentence span with alpha:2
            {"id": "beta:3", "text": "Pravna doktrina se razvija s časom in družbenimi spremembami. " * 8},
        ],
    )

    final_folder = tmp_path / "final"
    paths = CuratePaths(
        input_folder=input_folder,
        final_folder=final_folder,
        statistics_folder=final_folder / "statistics",
        scratch_folder=tmp_path / "scratch",
    )
    executors = build_curate_executors(
        paths,
        # Skip classla on the smoke test — it would download a model.
        compute_keywords=False,
        # Lower min_doc_words so our small fixtures aren't filtered out
        # after sentence dedup. Real corpus runs use the default 50.
        sentence_config=SentDedupConfig(
            n_sentences=3,
            min_doc_words=5,
            min_num_sentences=1,
            split_sentences=True,
        ),
        stopwords=set(),
    )
    executors[-1].run()

    survivors: List[str] = []
    survivor_dirs: set = set()
    for shard in sorted(final_folder.glob("**/*.jsonl.gz")):
        survivor_dirs.add(shard.parent.name)
        with gzip.open(shard, "rt", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                survivors.append(rec["id"])

    # 6 input docs minus 1 whole-document duplicate = 5 survivors.
    assert len(survivors) == 5
    assert "alpha:1" in survivors
    assert "beta:1" not in survivors  # whole-doc dup dropped
    # Final corpus is sharded as <final>/<dataset>/<rank>.jsonl.gz —
    # both datasets must show up as their own subfolders.
    assert survivor_dirs == {"alpha", "beta"}

    bundle = json.loads((final_folder / "statistics" / "aggregate.json").read_text(encoding="utf-8"))
    assert bundle["total_docs"] == 5
    assert "alpha" in bundle["by_dataset"]
    assert "beta" in bundle["by_dataset"]
    assert bundle["by_dataset"]["alpha"]["doc_count"] == 3
    assert bundle["by_dataset"]["beta"]["doc_count"] == 2

    per_dataset_dir = final_folder / "statistics" / "per_dataset"
    assert (per_dataset_dir / "alpha.json").exists()
    assert (per_dataset_dir / "beta.json").exists()
    alpha_slim = json.loads((per_dataset_dir / "alpha.json").read_text(encoding="utf-8"))
    assert alpha_slim["dataset"] == "alpha"
    assert alpha_slim["domain"] == "scientific"
    assert alpha_slim["doc_count"] == 3
