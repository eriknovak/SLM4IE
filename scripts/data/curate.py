"""Build the final SLM4IE pretraining corpus from the per-dataset shards.

`scripts/data/curate.py` is a thin CLI on top of **datatrove**: it builds
a single five-executor pipeline that fuses language detection,
cross-dataset exact and 3-sentence dedup, and corpus-statistics
aggregation. The output is the deduplicated training corpus under
`<output_dir>/final/<key>.jsonl.gz` plus a `final/statistics/` folder
with the aggregate JSON and per-dataset breakdowns.

All dedup intermediate state lives in a `tempfile.TemporaryDirectory`
that is cleaned up at the end of the run. Pass `--debug` to keep that
state under `<output_dir>/final/_dedup/` and route every dropped
duplicate to inspectable JSONL shards.

Examples:
    # Canonical: lang + dedup + stats on every dataset
    uv run python scripts/data/curate.py --all

    # Single dataset (still all three concerns; dedup operates within
    # that one shard only)
    uv run python scripts/data/curate.py kzb

    # Re-run only the stats stage on the existing final/ corpus
    uv run python scripts/data/curate.py --all --stage stats

    # Debug: keep dedup state and dropped-duplicate shards on disk
    uv run python scripts/data/curate.py --all --debug
"""

import argparse
import contextlib
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set

import yaml

from slm4ie.data.curate.pipeline import CuratePaths, build_curate_executors
from slm4ie.data.curate.stats import CorpusStats
from slm4ie.data.io_utils import find_project_root as _find_project_root

logger = logging.getLogger(__name__)

# Stage names accepted by the `--stage` flag, plus the default
# `"all"` sentinel.
VALID_STAGES = ("all", "lang", "dedup", "stats")


def list_datasets_from_config(config_path: Path) -> List[str]:
    """Return dataset keys declared in `extract.yaml`.

    Args:
        config_path: Path to the extraction YAML config.

    Returns:
        Dataset keys in declaration order.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open() as fh:
        cfg = yaml.safe_load(fh) or {}
    return list((cfg.get("datasets") or {}).keys())


def load_curate_config(config_path: Path) -> Dict[str, Any]:
    """Read and parse `configs/data/curate.yaml`.

    Args:
        config_path: Path to the curate YAML config.

    Returns:
        Parsed config dict with `language` / `dedup` / `stats` sections
        (empty dicts when missing).
    """
    if not config_path.exists():
        logger.warning("curate config not found at %s; using built-in defaults.", config_path)
        return {}
    with config_path.open() as fh:
        return yaml.safe_load(fh) or {}


def load_stopwords(path: Optional[Path]) -> Set[str]:
    """Load one stopword per non-empty, non-comment line.

    Args:
        path: File path, or `None` to skip loading.

    Returns:
        Lowercased set of stopword tokens.
    """
    if path is None or not path.exists():
        return set()
    out: Set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.add(line.lower())
    return out


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to `sys.argv`).

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Build the final SLM4IE pretraining corpus: lingua-py "
            "language tagging, cross-dataset deduplication, and corpus "
            "statistics — all in one fused datatrove pipeline."
        )
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "dataset",
        nargs="?",
        help=(
            "Dataset key (e.g. 'kzb'). Single-key invocations still run "
            "all three concerns; dedup operates within that one shard "
            "only. For the canonical training corpus, use --all so "
            "dedup is cross-dataset."
        ),
    )
    target.add_argument(
        "--all",
        action="store_true",
        help="Process every dataset declared in extract.yaml.",
    )
    parser.add_argument(
        "--stage",
        choices=VALID_STAGES,
        default="all",
        help=(
            "Run a single stage instead of the full pipeline. lang/dedup "
            "produce ephemeral output (warning issued); stats re-reads "
            "<output_dir>/final/ and refreshes the statistics folder."
        ),
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help=(
            "Override the processed-data root from extract.yaml. "
            "Inputs are read from <processed-dir>/datatrove and the "
            "training corpus is written under <processed-dir>/final/."
        ),
    )
    parser.add_argument(
        "--curate-config",
        type=Path,
        default=None,
        help="Path to curate.yaml (default: configs/data/curate.yaml).",
    )
    parser.add_argument(
        "--extract-config",
        type=Path,
        default=None,
        help="Path to extract.yaml (default: configs/data/extract.yaml).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the existing <output_dir>/final/ before running.",
    )
    parser.add_argument(
        "--no-keywords",
        action="store_true",
        help="Skip the classla-lemmatized TF-IDF keyword pass in the stats stage.",
    )
    debug = parser.add_mutually_exclusive_group()
    debug.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Keep all dedup intermediate state and dropped-duplicate "
            "shards under <output_dir>/final/_dedup/ instead of using "
            "an auto-cleaning tempdir."
        ),
    )
    debug.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help=(
            "Same as --debug but writes the inspectable state to a "
            "user-chosen path."
        ),
    )
    parser.add_argument(
        "--tasks",
        type=int,
        default=1,
        help=(
            "Parallel tasks for the per-shard executors (lang+sig, "
            "exact-filter+sent-sig). Stats stay single-worker. "
            "Default: 1."
        ),
    )
    return parser.parse_args(argv)


def _resolve_processed_dir(args: argparse.Namespace, project_root: Path) -> Path:
    """Compute the processed-data root, honoring CLI overrides.

    Args:
        args: Parsed CLI namespace.
        project_root: Project root directory.

    Returns:
        Filesystem path to the processed-data root (parent of
        `datatrove/` and `final/`).

    Raises:
        FileNotFoundError: If neither `--processed-dir` nor the extract
            config is available.
    """
    if args.processed_dir is not None:
        return args.processed_dir
    extract_config = args.extract_config or (project_root / "configs" / "data" / "extract.yaml")
    if not extract_config.exists():
        raise FileNotFoundError(
            f"Pass --processed-dir or place extract.yaml at {extract_config}"
        )
    with extract_config.open() as fh:
        cfg = yaml.safe_load(fh) or {}
    return Path(cfg.get("output_dir", "data/processed"))


@contextlib.contextmanager
def _scratch_root(args: argparse.Namespace, processed_dir: Path) -> Iterator[Path]:
    """Yield the folder used for dedup intermediate state.

    Args:
        args: Parsed CLI namespace.
        processed_dir: Resolved processed-data root.

    Yields:
        A folder path. Cleaned up automatically unless `--debug` or
        `--debug-dir` was passed, in which case the folder lives at
        a stable location and is preserved.
    """
    if args.debug_dir is not None:
        path = args.debug_dir
        path.mkdir(parents=True, exist_ok=True)
        logger.info("Debug mode: keeping intermediate state at %s", path)
        yield path
        return
    if args.debug:
        path = processed_dir / "final" / "_dedup"
        path.mkdir(parents=True, exist_ok=True)
        logger.info("Debug mode: keeping intermediate state at %s", path)
        yield path
        return
    with tempfile.TemporaryDirectory(prefix="slm4ie-curate-") as td:
        yield Path(td)


def _drop_existing_final(processed_dir: Path) -> None:
    """Remove the existing `final/` folder before re-running.

    Args:
        processed_dir: Resolved processed-data root.
    """
    target = processed_dir / "final"
    if target.exists():
        shutil.rmtree(target)
        logger.info("Removed existing %s for --force re-run", target)


def _filter_input_keys(processed_dir: Path, key: str) -> Path:
    """Materialize a folder containing only `<key>.jsonl.gz`.

    For single-dataset invocations we still feed a folder-glob to the
    pipeline's `JsonlReader`. We therefore symlink the one shard into
    a fresh tempdir so the rest of the pipeline doesn't have to
    special-case single-key reads.

    Args:
        processed_dir: Resolved processed-data root.
        key: Dataset key to keep.

    Returns:
        A folder containing exactly `<key>.jsonl.gz`. The caller is
        responsible for removing it once the pipeline finishes.

    Raises:
        FileNotFoundError: If the requested shard does not exist.
    """
    src = processed_dir / "datatrove" / f"{key}.jsonl.gz"
    if not src.exists():
        raise FileNotFoundError(f"No datatrove shard for dataset {key!r} at {src}")
    holder = Path(tempfile.mkdtemp(prefix=f"slm4ie-curate-{key}-"))
    target = holder / f"{key}.jsonl.gz"
    target.symlink_to(src.resolve())
    return holder


def _stats_only(processed_dir: Path, cfg: Dict[str, Any], no_keywords: bool) -> None:
    """Re-run just the stats stage against the existing `final/` corpus.

    Args:
        processed_dir: Resolved processed-data root.
        cfg: Parsed curate config.
        no_keywords: When True, skip the classla TF-IDF pass.
    """
    final_folder = processed_dir / "final"
    statistics_folder = final_folder / "statistics"
    if not any(final_folder.glob("*.jsonl.gz")):
        logger.error(
            "stats stage requires a deduplicated corpus in %s. Run the "
            "full pipeline first (drop --stage stats).",
            final_folder,
        )
        return

    statistics_folder.mkdir(parents=True, exist_ok=True)
    stats_cfg = cfg.get("stats") or {}
    project_root = _find_project_root()
    stopword_rel = stats_cfg.get("stopwords")
    stopwords = load_stopwords((project_root / stopword_rel) if stopword_rel else None)

    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.readers import JsonlReader

    executor = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(
                str(final_folder),
                glob_pattern="*.jsonl.gz",
                shuffle_files=False,
                recursive=False,
            ),
            CorpusStats(
                output_path=statistics_folder / "aggregate.json",
                per_dataset_dir=statistics_folder / "per_dataset",
                stopwords=stopwords,
                top_k_words=int(stats_cfg.get("top_k_words", 5_000)),
                top_k_ngrams=int(stats_cfg.get("top_k_ngrams", 5_000)),
                keyword_top_k=int(stats_cfg.get("keyword_top_k", 200)),
                compute_keywords=(not no_keywords) and bool(stats_cfg.get("compute_keywords", True)),
            ),
        ],
        tasks=1,
        workers=1,
        logging_dir=str(statistics_folder / "_logs"),
        skip_completed=False,
    )
    executor.run()
    logger.info("Refreshed stats under %s", statistics_folder)


def main() -> None:
    """Entry point for the `curate.py` CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    project_root = _find_project_root()
    extract_config = args.extract_config or (project_root / "configs" / "data" / "extract.yaml")
    curate_config = args.curate_config or (project_root / "configs" / "data" / "curate.yaml")
    cfg = load_curate_config(curate_config)
    processed_dir = _resolve_processed_dir(args, project_root)

    if args.stage == "stats":
        _stats_only(processed_dir, cfg, args.no_keywords)
        return

    if args.all:
        all_keys = list_datasets_from_config(extract_config)
        logger.info("Running on all %d datasets from %s", len(all_keys), extract_config)
        single_key_holder: Optional[Path] = None
        input_folder = processed_dir / "datatrove"
    else:
        logger.info("Running on single dataset: %s", args.dataset)
        single_key_holder = _filter_input_keys(processed_dir, args.dataset)
        input_folder = single_key_holder

    if args.force:
        _drop_existing_final(processed_dir)

    final_folder = processed_dir / "final"
    statistics_folder = final_folder / "statistics"

    lang_cfg = cfg.get("language") or {}
    dedup_cfg = cfg.get("dedup") or {}
    sentence_cfg = dedup_cfg.get("sentence") or {}
    stats_cfg = cfg.get("stats") or {}
    stopword_rel = stats_cfg.get("stopwords")
    stopwords = load_stopwords((project_root / stopword_rel) if stopword_rel else None)

    from datatrove.pipeline.dedup import SentDedupConfig

    sent_dedup = SentDedupConfig(
        n_sentences=int(sentence_cfg.get("n_sentences", 3)),
        min_doc_words=int(sentence_cfg.get("min_doc_words", 50)),
        min_num_sentences=int(sentence_cfg.get("min_num_sentences", 2)),
        split_sentences=bool(sentence_cfg.get("split_sentences", True)),
    )

    try:
        with _scratch_root(args, processed_dir) as scratch:
            paths = CuratePaths(
                input_folder=input_folder,
                final_folder=final_folder,
                statistics_folder=statistics_folder,
                scratch_folder=scratch,
                debug=bool(args.debug or args.debug_dir),
            )
            executors = build_curate_executors(
                paths,
                tasks=args.tasks,
                finder_workers=int(dedup_cfg.get("finder_workers", 1)),
                sentence_config=sent_dedup,
                target_language_iso2=lang_cfg.get("target", "sl"),
                candidate_languages=lang_cfg.get("candidates"),
                lang_threshold=float(lang_cfg.get("threshold", 0.5)),
                lang_mode=lang_cfg.get("mode", "tag"),
                stopwords=stopwords,
                top_k_words=int(stats_cfg.get("top_k_words", 5_000)),
                top_k_ngrams=int(stats_cfg.get("top_k_ngrams", 5_000)),
                keyword_top_k=int(stats_cfg.get("keyword_top_k", 200)),
                compute_keywords=(not args.no_keywords) and bool(stats_cfg.get("compute_keywords", True)),
            )

            if args.stage == "lang":
                logger.warning(
                    "--stage lang produces ephemeral output that is "
                    "discarded with the scratch directory. Use --debug "
                    "to inspect the lang-tagged shards."
                )
                executors[0].run()
            elif args.stage == "dedup":
                logger.warning(
                    "--stage dedup produces ephemeral output that is "
                    "discarded with the scratch directory. Use --debug "
                    "to inspect intermediate dedup state."
                )
                executors[3].run()
            else:
                executors[-1].run()
    finally:
        if single_key_holder is not None:
            shutil.rmtree(single_key_holder, ignore_errors=True)

    logger.info("Final corpus written to %s", final_folder)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
