"""Build the final SLM4IE pretraining corpus from the per-dataset shards.

`scripts/data/curate.py` is a thin CLI on top of **datatrove**: it builds
a single six-executor pipeline that fuses lingua-py language
filtering, Gopher-style quality and repetition heuristics, two
cumulative cross-dataset dedup passes (exact, 3-sentence), and
corpus-statistics aggregation. Input and output paths come from
`configs/data/curate.yaml` (`input_dir` and `output_dir`) or the
matching CLI overrides; the dataset key list still comes from
`configs/data/extract.yaml`. The output is the deduplicated training
corpus under `<output_dir>/<key>.jsonl.gz` plus a
`<output_dir>/statistics/` folder with the aggregate JSON and
per-dataset breakdowns.

All dedup intermediate state lives in a `tempfile.TemporaryDirectory`
that is cleaned up at the end of the run. Pass `--debug` to keep that
state under `<output_dir>/_dedup/` and route every dropped duplicate
to inspectable JSONL shards.

Examples:
    # Canonical: lang + dedup + stats on every dataset
    uv run python scripts/data/curate.py --all

    # Saturate the box: parallel lang/dedup/write across all cores
    uv run python scripts/data/curate.py --all --workers 0

    # Subset of datasets (still all three concerns; dedup operates
    # within the given subset only)
    uv run python scripts/data/curate.py kzb solar

    # Re-run only the stats stage on the existing output_dir corpus
    uv run python scripts/data/curate.py --all --stage stats

    # Debug: keep dedup state and dropped-duplicate shards on disk
    uv run python scripts/data/curate.py --all --debug
"""

import argparse
import contextlib
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import yaml

from slm4ie.data.curate.pipeline import CuratePaths, QualityConfig, build_curate_executors
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
        "datasets",
        nargs="*",
        default=[],
        help=(
            "Dataset keys (e.g. 'kzb' or 'kzb solar'). Subset invocations "
            "still run all three concerns; dedup operates within the "
            "given subset only — for the canonical training corpus, "
            "use --all so dedup is cross-dataset."
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
            "<output_dir> and refreshes the statistics folder."
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help=(
            "Override curate.yaml::input_dir. Folder of <key>.jsonl.gz "
            "datatrove shards to read from."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Override curate.yaml::output_dir. Folder where the "
            "deduplicated corpus and statistics/ are written."
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
        help="Overwrite the existing <output_dir> before running.",
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
            "shards under <output_dir>/_dedup/ instead of using an "
            "auto-cleaning tempdir."
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
        "--max-workers",
        "--tasks",
        dest="workers",
        type=int,
        default=0,
        help=(
            "Number of parallel CPU workers for the per-shard "
            "executors (lang+quality+exact-sig, exact-filter+sent-sig, "
            "sent-filter+write). Pass 0 to use "
            "every available core (`os.cpu_count()`). The dedup find "
            "stages and the corpus-stats stage stay single-worker by "
            "design. `--tasks` is accepted as a back-compat alias. "
            "Default: 0."
        ),
    )
    args = parser.parse_args(argv)
    # argparse's `required=True` on a mutex group accepts a positional
    # with `nargs="*"`, but treats an empty positional as "provided" —
    # so bare invocation can slip through. Validate the xor by hand.
    if args.all and args.datasets:
        parser.error("argument --all: not allowed with positional datasets")
    if not args.all and not args.datasets:
        parser.error("one of the arguments datasets --all is required")
    return args


def _resolve_worker_count(requested: int) -> int:
    """Translate the `--workers` CLI value into a concrete count.

    Args:
        requested: User-requested worker count. Any value below `1`
            selects `os.cpu_count()`, falling back to `1` if the OS
            does not report a CPU count.

    Returns:
        The number of parallel workers to use, always at least `1`.
    """
    if requested < 1:
        return os.cpu_count() or 1
    return requested


def _resolve_target_languages(lang_cfg: Dict[str, Any]) -> List[str]:
    """Normalize the lingua target-language list from the curate config.

    Accepts either the new `targets` (list of ISO 639-1 codes) or the
    deprecated singular `target` (a single code) and returns a non-empty
    list. A `DeprecationWarning`-style message is logged when the old
    key is used.

    Args:
        lang_cfg: The `language:` subsection of `curate.yaml`.

    Returns:
        List of lowercased ISO 639-1 codes. Defaults to `["sl"]` when
        neither key is present.

    Raises:
        ValueError: If `targets` is set to an empty list.
    """
    raw_targets = lang_cfg.get("targets")
    if raw_targets is None:
        raw_target = lang_cfg.get("target")
        if raw_target is not None:
            logger.warning(
                "curate.yaml::language.target (singular) is deprecated; "
                "use targets: [%s] instead.",
                raw_target,
            )
            raw_targets = [raw_target]
        else:
            raw_targets = ["sl"]
    if not raw_targets:
        raise ValueError("curate.yaml::language.targets must be a non-empty list")
    return [str(code).lower() for code in raw_targets]


def _resolve_curate_dirs(
    args: argparse.Namespace, cfg: Dict[str, Any]
) -> Tuple[Path, Path]:
    """Resolve the curation input and output folders.

    CLI flags `--input-dir` and `--output-dir` take precedence over the
    matching keys in `curate.yaml`. Both must end up resolved; otherwise
    the call fails with a message naming the missing source.

    Args:
        args: Parsed CLI namespace.
        cfg: Parsed curate config dict.

    Returns:
        Tuple `(input_dir, output_dir)`. `input_dir` is the folder of
        `<key>.jsonl.gz` datatrove shards. `output_dir` is the folder
        where the deduplicated corpus and `statistics/` are written.

    Raises:
        FileNotFoundError: If neither the CLI flag nor the YAML key is
            set for either side.
    """
    raw_input = args.input_dir if args.input_dir is not None else cfg.get("input_dir")
    raw_output = args.output_dir if args.output_dir is not None else cfg.get("output_dir")

    missing: List[str] = []
    if raw_input is None:
        missing.append("--input-dir or curate.yaml::input_dir")
    if raw_output is None:
        missing.append("--output-dir or curate.yaml::output_dir")
    if missing or raw_input is None or raw_output is None:
        raise FileNotFoundError(
            "Curation paths are not set. Provide: " + "; ".join(missing) + "."
        )

    return Path(raw_input), Path(raw_output)


@contextlib.contextmanager
def _scratch_root(args: argparse.Namespace, output_dir: Path) -> Iterator[Path]:
    """Yield the folder used for dedup intermediate state.

    Args:
        args: Parsed CLI namespace.
        output_dir: Resolved curation output folder.

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
        path = output_dir / "_dedup"
        path.mkdir(parents=True, exist_ok=True)
        logger.info("Debug mode: keeping intermediate state at %s", path)
        yield path
        return
    with tempfile.TemporaryDirectory(prefix="slm4ie-curate-") as td:
        yield Path(td)


def _drop_existing_output(output_dir: Path) -> None:
    """Remove the existing curation output folder before re-running.

    Args:
        output_dir: Resolved curation output folder.
    """
    if output_dir.exists():
        shutil.rmtree(output_dir)
        logger.info("Removed existing %s for --force re-run", output_dir)


def _filter_input_keys(input_dir: Path, keys: List[str]) -> Path:
    """Materialize a folder mirroring the requested `<key>/` subdirs with symlinks.

    For subset-of-datasets invocations we still feed a folder-glob to
    the pipeline's `JsonlReader`. We mirror each per-dataset shard
    folder via symlinks into a fresh tempdir so the rest of the
    pipeline doesn't have to special-case subset reads.

    Args:
        input_dir: Folder of datatrove `<key>/<NNNNN>.jsonl.gz` shards.
        keys: Dataset keys to keep.

    Returns:
        A folder containing `<key>/<NNNNN>.jsonl.gz` symlinks for each
        requested key. The caller is responsible for removing it once
        the pipeline finishes.

    Raises:
        FileNotFoundError: If any of the requested shard folders is
            missing or empty. The error message lists all such keys.
    """
    missing: List[str] = []
    for key in keys:
        src = input_dir / key
        if not src.is_dir() or not any(src.glob("*.jsonl.gz")):
            missing.append(key)
    if missing:
        raise FileNotFoundError(
            f"No datatrove shard folder(s) under {input_dir} for dataset(s): "
            + ", ".join(repr(k) for k in missing)
        )
    holder = Path(tempfile.mkdtemp(prefix="slm4ie-curate-subset-"))
    for key in keys:
        src = input_dir / key
        holder_key = holder / key
        holder_key.mkdir()
        for shard in src.glob("*.jsonl.gz"):
            (holder_key / shard.name).symlink_to(shard.resolve())
    return holder


def _stats_only(output_dir: Path, cfg: Dict[str, Any], no_keywords: bool) -> None:
    """Re-run just the stats stage against the existing curated corpus.

    Args:
        output_dir: Resolved curation output folder.
        cfg: Parsed curate config.
        no_keywords: When True, skip the classla TF-IDF pass.
    """
    statistics_folder = output_dir / "statistics"
    if not any(output_dir.glob("**/*.jsonl.gz")):
        logger.error(
            "stats stage requires a deduplicated corpus in %s. Run the "
            "full pipeline first (drop --stage stats).",
            output_dir,
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
                str(output_dir),
                glob_pattern="**/*.jsonl.gz",
                shuffle_files=False,
                recursive=True,
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


def _suppress_writer_rank_warning() -> None:
    """Drop datatrove's `${rank}`-template warning from the loguru sink.

    `JsonlWriter.__init__` warns when the output filename template lacks
    `${rank}`, on the grounds that parallel workers may overwrite each
    other. With our round-robin file sharding (datatrove's `get_shard`)
    and a 1:1 mapping from input file to dataset, two ranks never share
    a `${dataset}` value, so the warning is a false positive whether
    `--workers` is `1` or higher. Adding `${rank}` would either bury the
    documented `<output_dir>/<key>.jsonl.gz` final-corpus layout under
    per-rank subfolders or rename it to `<key>_00000.jsonl.gz`. Replace
    datatrove's default loguru sink with one that filters out this
    specific message.
    """
    from datatrove.utils.logging import DATATROVE_COLORIZE_LOGS
    from loguru import logger as loguru_logger

    loguru_logger.remove()
    loguru_logger.add(
        sys.stderr,
        colorize=DATATROVE_COLORIZE_LOGS,
        filter=lambda record: "does not include ${rank}" not in record["message"],
    )


def main() -> None:
    """Entry point for the `curate.py` CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _suppress_writer_rank_warning()
    args = parse_args()
    workers = _resolve_worker_count(args.workers)

    project_root = _find_project_root()
    extract_config = args.extract_config or (project_root / "configs" / "data" / "extract.yaml")
    curate_config = args.curate_config or (project_root / "configs" / "data" / "curate.yaml")
    cfg = load_curate_config(curate_config)
    input_dir, output_dir = _resolve_curate_dirs(args, cfg)

    if args.stage == "stats":
        _stats_only(output_dir, cfg, args.no_keywords)
        return

    if args.all:
        all_keys = list_datasets_from_config(extract_config)
        logger.info(
            "Running on all %d datasets from %s with %d worker(s)",
            len(all_keys),
            extract_config,
            workers,
        )
        subset_holder: Optional[Path] = None
        input_folder = input_dir
    else:
        logger.info(
            "Running on %d dataset(s): %s with %d worker(s)",
            len(args.datasets),
            ", ".join(args.datasets),
            workers,
        )
        subset_holder = _filter_input_keys(input_dir, args.datasets)
        input_folder = subset_holder

    if args.force:
        _drop_existing_output(output_dir)

    final_folder = output_dir
    statistics_folder = final_folder / "statistics"

    lang_cfg = cfg.get("language") or {}
    target_languages = _resolve_target_languages(lang_cfg)
    dedup_cfg = cfg.get("dedup") or {}
    sentence_cfg = dedup_cfg.get("sentence") or {}
    quality_cfg = cfg.get("quality") or {}
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
    quality_conf = QualityConfig(
        min_doc_words=int(quality_cfg.get("min_doc_words", 50)),
        max_doc_words=int(quality_cfg.get("max_doc_words", 100_000)),
        min_avg_word_length=int(quality_cfg.get("min_avg_word_length", 3)),
        max_avg_word_length=int(quality_cfg.get("max_avg_word_length", 10)),
        max_symbol_word_ratio=float(quality_cfg.get("max_symbol_word_ratio", 0.1)),
        max_bullet_lines_ratio=float(quality_cfg.get("max_bullet_lines_ratio", 0.9)),
        max_ellipsis_lines_ratio=float(quality_cfg.get("max_ellipsis_lines_ratio", 0.3)),
        max_non_alpha_words_ratio=float(quality_cfg.get("max_non_alpha_words_ratio", 0.8)),
        min_stop_words=int(quality_cfg.get("min_stop_words", 2)),
    )

    try:
        with _scratch_root(args, output_dir) as scratch:
            paths = CuratePaths(
                input_folder=input_folder,
                final_folder=final_folder,
                statistics_folder=statistics_folder,
                scratch_folder=scratch,
                debug=bool(args.debug or args.debug_dir),
            )
            executors = build_curate_executors(
                paths,
                tasks=workers,
                finder_workers=int(dedup_cfg.get("finder_workers", 1)),
                sentence_config=sent_dedup,
                quality_config=quality_conf,
                target_languages=target_languages,
                candidate_languages=lang_cfg.get("candidates"),
                lang_mode=lang_cfg.get("mode", "tag"),
                lang_minimum_relative_distance=float(
                    lang_cfg.get("minimum_relative_distance", 0.0)
                ),
                lang_low_accuracy=bool(lang_cfg.get("low_accuracy", False)),
                lang_max_chars=lang_cfg.get("max_chars"),
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
                    "to inspect the lang-tagged and quality-filtered shards."
                )
                executors[0].run()
            elif args.stage == "dedup":
                logger.warning(
                    "--stage dedup produces ephemeral output that is "
                    "discarded with the scratch directory. Use --debug "
                    "to inspect intermediate dedup state."
                )
                # Run through the sentence-find stage (executor index 3).
                # The downstream sentence-filter + write stage is the
                # final materialization step and lives at index 4.
                executors[3].run()
            else:
                executors[-1].run()
    finally:
        if subset_holder is not None:
            shutil.rmtree(subset_holder, ignore_errors=True)

    logger.info("Final corpus written to %s", final_folder)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
