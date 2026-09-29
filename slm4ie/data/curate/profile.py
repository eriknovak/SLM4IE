"""Describe the finished corpus without asking a judge anything.

Most of what a corpus is can be measured rather than judged: how many documents
each source contributed and how long they are, how much of the vocabulary is
repeated, how much of it a Slovene lexicon recognises, and how sure a language
identifier is that the text is Slovene at all. None of that needs a model's
opinion, so none of it inherits the judge's blind spots.

Four statistics are computed per source, over a sample of the corpus:

* length, as percentiles of characters and words, because a mean hides the
  short-document tail that the length floors act on;
* type-token ratio, at a fixed token budget per source — the measure falls as a
  sample grows, so sources compared at different budgets cannot be compared at
  all;
* out-of-vocabulary rate against Sloleks, which is a proxy for how much of the
  text is ordinary Slovene: names, foreign words, code and encoding damage all
  read as out of vocabulary;
* language-identification confidence, as the share of documents lingua calls
  Slovene and the confidence it assigns, recomputed here because the pipeline
  keeps the verdict but not the score.

The sample reads every `stride`-th document from the first shards of each
source, which follows file order rather than drawing uniformly. That is a bias
worth knowing about: it is fine for lexical statistics, which concentrate
quickly, and wrong for anything that varies along a source's file order.
"""

import gzip
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

#: A token, for the lexical statistics: a run of word characters.
_TOKEN = re.compile(r"\w+", re.UNICODE)

#: Percentiles reported for every length distribution.
PERCENTILES: Tuple[int, ...] = (5, 25, 50, 75, 95)

#: Tokens per source used for the type-token ratio, which shrinks as a sample
#: grows and so is only comparable at one fixed budget.
TTR_BUDGET: int = 100_000


def tokens(text: str) -> List[str]:
    """Split text into lowercased word tokens.

    Args:
        text: The document's text.

    Returns:
        The tokens, lowercased.
    """
    return [match.group(0).lower() for match in _TOKEN.finditer(text)]


def percentiles(values: Sequence[float], points: Sequence[int] = PERCENTILES) -> Dict[str, float]:
    """Return percentiles of a sample, by nearest rank.

    Args:
        values: The sample; may be unsorted.
        points: Percentiles to report.

    Returns:
        One entry per percentile, keyed `p5`, `p50` and so on.
    """
    if not values:
        return {f"p{point}": 0.0 for point in points}
    ordered = sorted(values)
    return {
        f"p{point}": float(ordered[min(len(ordered) - 1, int(round(point / 100 * (len(ordered) - 1))))])
        for point in points
    }


def type_token_ratio(token_list: Sequence[str], budget: int = TTR_BUDGET) -> Tuple[float, int]:
    """Return the type-token ratio over a fixed number of tokens.

    Args:
        token_list: Tokens in corpus order.
        budget: Tokens to measure; a shorter sample is measured whole.

    Returns:
        The ratio of distinct tokens to tokens, and the budget actually used.
    """
    window = token_list[:budget]
    return (len(set(window)) / len(window) if window else 0.0), len(window)


def sloleks_forms(path: Path) -> Set[str]:
    """Load every word form Sloleks lists.

    Args:
        path: The gzipped Sloleks JSONL written by `prepare_datasets.py
            tokenization`.

    Returns:
        Lowercased lemmas and inflected forms.
    """
    forms: Set[str] = set()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("lemma"):
                forms.add(entry["lemma"].lower())
            for form in entry.get("forms", ()):
                if form.get("form"):
                    forms.add(form["form"].lower())
    logger.info("Sloleks: %d forms from %s", len(forms), path)
    return forms


def oov_rate(token_list: Sequence[str], forms: Set[str]) -> Tuple[float, int]:
    """Return the share of alphabetic tokens Sloleks does not list.

    Numbers and mixed alphanumeric tokens are left out rather than counted as
    unknown words, since a lexicon is not expected to hold them. Names, foreign
    words, code and encoding damage do count, which is what makes the rate a
    proxy for how much of the text is ordinary Slovene.

    Args:
        token_list: Tokens to check.
        forms: The lexicon's forms, lowercased.

    Returns:
        The out-of-vocabulary share and the number of tokens it was taken over.
    """
    alphabetic = [token for token in token_list if token.isalpha()]
    if not alphabetic:
        return 0.0, 0
    unknown = sum(1 for token in alphabetic if token not in forms)
    return unknown / len(alphabetic), len(alphabetic)


def _detector(candidates: Sequence[str], low_accuracy: bool = True) -> Any:
    """Build a lingua detector over the candidate languages.

    Args:
        candidates: ISO 639-1 codes to consider.
        low_accuracy: Use lingua's trigram-only model, which is much faster.

    Returns:
        A lingua `LanguageDetector`.

    Raises:
        ValueError: If a candidate code is unknown to lingua.
    """
    from lingua import Language, LanguageDetectorBuilder

    by_code = {language.iso_code_639_1.name.lower(): language for language in Language.all()}
    try:
        languages = [by_code[code] for code in candidates]
    except KeyError as exc:
        raise ValueError(f"unknown lingua language code: {exc.args[0]!r}") from exc
    builder = LanguageDetectorBuilder.from_languages(*languages).with_preloaded_language_models()
    if low_accuracy:
        builder = builder.with_low_accuracy_mode()
    return builder.build()


def language_confidence(
    texts: Sequence[str], detector: Any, target: str = "sl", max_chars: int = 2000
) -> Dict[str, Any]:
    """Score how confidently a sample reads as the target language.

    Args:
        texts: Document texts.
        detector: A lingua detector from `_detector`.
        target: ISO 639-1 code the corpus is meant to be in.
        max_chars: Characters of each document handed to the detector.

    Returns:
        The share of documents predicted as *target*, percentiles of the
        confidence given to *target*, and the languages predicted instead.
    """
    scores: List[float] = []
    predicted: Counter = Counter()
    for text in texts:
        head = text[:max_chars]
        values = detector.compute_language_confidence_values(head)
        best = values[0] if values else None
        predicted[best.language.iso_code_639_1.name.lower() if best else "none"] += 1
        for value in values:
            if value.language.iso_code_639_1.name.lower() == target:
                scores.append(float(value.value))
                break
    return {
        "in_language_share": predicted.get(target, 0) / len(texts) if texts else 0.0,
        "confidence": percentiles(scores),
        "predicted": dict(predicted.most_common(5)),
    }


def sample_documents(source_dir: Path, per_source: int, stride: int = 10, max_shards: int = 3) -> List[Dict[str, Any]]:
    """Read a sample of one source's documents from the finished corpus.

    Args:
        source_dir: The source's folder inside the final stage.
        per_source: Documents to collect at most.
        stride: Keep every *stride*-th document, to spread the sample over a
            shard rather than taking its opening run.
        max_shards: Shards to read at most.

    Returns:
        The sampled documents.
    """
    shards = sorted(source_dir.glob("*.jsonl.gz"))[:max_shards]
    for step in (stride, 1) if stride > 1 else (1,):
        collected: List[Dict[str, Any]] = []
        for shard in shards:
            with gzip.open(shard, "rt", encoding="utf-8") as handle:
                for index, line in enumerate(handle):
                    if index % step or not line.strip():
                        continue
                    collected.append(json.loads(line))
                    if len(collected) >= per_source:
                        return collected
        # A small source cannot fill the sample at this stride, so it is read
        # again taking every document rather than being under-sampled.
        if len(collected) >= per_source // 2:
            return collected
    return collected


def profile_source(
    documents: Sequence[Dict[str, Any]], forms: Optional[Set[str]], detector: Optional[Any]
) -> Dict[str, Any]:
    """Compute every statistic for one source's sample.

    Args:
        documents: The source's sampled documents.
        forms: Sloleks forms, or None to skip the vocabulary rate.
        detector: A lingua detector, or None to skip language confidence.

    Returns:
        The source's statistics, ready to serialise.
    """
    texts = [document["text"] for document in documents]
    token_list = [token for text in texts for token in tokens(text)]
    ratio, budget = type_token_ratio(token_list)
    profile: Dict[str, Any] = {
        "documents_sampled": len(documents),
        "chars": percentiles([len(text) for text in texts]),
        "words": percentiles([len(tokens(text)) for text in texts]),
        "type_token_ratio": round(ratio, 4),
        "type_token_budget": budget,
        "duplicate_count_mean": (
            sum(document.get("metadata", {}).get("duplicate_count", 0) for document in documents) / len(documents)
            if documents
            else 0.0
        ),
    }
    if forms is not None:
        rate, checked = oov_rate(token_list, forms)
        profile["oov_rate"] = round(rate, 4)
        profile["oov_tokens_checked"] = checked
    if detector is not None:
        profile["language"] = language_confidence(texts, detector)
    return profile


def profile_corpus(
    final_dir: Path,
    sloleks_path: Optional[Path] = None,
    candidates: Sequence[str] = ("sl", "hr", "sr", "bs", "en", "de", "it", "hu"),
    per_source: int = 2000,
    stride: int = 10,
    sources: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Profile every source in the finished corpus.

    Args:
        final_dir: The final stage folder, one directory per source.
        sloleks_path: The Sloleks JSONL, or None to skip the vocabulary rate.
        candidates: Languages the identifier may choose between.
        per_source: Documents sampled per source.
        stride: Keep every *stride*-th document.
        sources: Restrict to these sources, or None for all of them.

    Returns:
        One profile per source, keyed by source name.
    """
    forms = sloleks_forms(sloleks_path) if sloleks_path else None
    detector = _detector(candidates)
    folders = [
        folder
        for folder in sorted(final_dir.iterdir())
        if folder.is_dir() and (sources is None or folder.name in sources)
    ]
    profiles: Dict[str, Dict[str, Any]] = {}
    for folder in folders:
        documents = sample_documents(folder, per_source=per_source, stride=stride)
        if not documents:
            logger.warning("%s: no documents in the finished corpus", folder.name)
            continue
        profiles[folder.name] = profile_source(documents, forms, detector)
        done = profiles[folder.name]
        logger.info(
            "%s: %d docs, median %d words, TTR %.3f, OOV %.1f%%, Slovene %.0f%%",
            folder.name,
            done["documents_sampled"],
            done["words"]["p50"],
            done["type_token_ratio"],
            100 * done.get("oov_rate", 0.0),
            100 * done["language"]["in_language_share"],
        )
    return profiles


def iter_stage_sentinels(pretrain_dir: Path, stages: Sequence[str]) -> Iterator[Tuple[str, str, int, int]]:
    """Yield the per-source document counts each scoped stage recorded.

    The stages stamp `records_in` and `records_out` into a `.complete` sentinel
    beside their output, so the survival funnel needs no corpus read.

    Args:
        pretrain_dir: The curation `output_dir`.
        stages: Stage folder names, in pipeline order.

    Yields:
        Stage, source, documents read, documents written.
    """
    for stage in stages:
        for path in sorted((pretrain_dir / stage).glob("*/.complete")):
            sentinel = json.loads(path.read_text(encoding="utf-8"))
            yield stage, path.parent.name, sentinel["records_in"], sentinel["records_out"]
