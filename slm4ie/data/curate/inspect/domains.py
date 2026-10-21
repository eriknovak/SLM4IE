"""Assign each document its own domain from one word list per domain.

The curated corpus labels every document with its source's domain, so a medical
page inside a web crawl counts as web. This module reads the text instead: each
domain has a hand-curated lexicon of marker words, a document is scored per
domain as the weighted number of marker hits per thousand words, and the
highest-scoring domain is assigned when it clears a threshold and leads the
runner-up by a margin. Otherwise the document keeps its source's label.

A lexicon entry is either a whole word (`dne`) or a prefix ending in `*`
(`zdravil*`), which matches every word that starts with it. Slovene inflects
heavily, so prefixes do most of the work. A word that matches several entries
of one domain counts once there, at the strongest of them.

The lexicons and the rule's settings live in the experiment's YAML config; see
`load_lexicons` for its shape.
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple, Union

from slm4ie.data.curate.inspect.judge import DOMAINS
from slm4ie.utils.config import load_yaml

#: A word, for scoring: a run of letters and digits, lowercased before matching.
_WORD = re.compile(r"[^\W_]+", re.UNICODE)

#: Scores are hits per this many words.
PER_WORDS: float = 1000.0

#: Entry marker that makes an entry a prefix.
PREFIX_MARK: str = "*"

#: Distinct words whose lookups are kept; a corpus repeats its vocabulary, so most lookups are hits.
_CACHE_SIZE: int = 1 << 20

DomainWeights = Tuple[Tuple[str, float], ...]


@dataclass(frozen=True)
class DecisionRule:
    """When the leading domain is assigned rather than the source's label.

    Attributes:
        threshold: Lowest leading score that assigns a domain.
        margin: Smallest lead over the runner-up that assigns a domain.
        min_words: Shortest text the rule is read on; a shorter one keeps its source label.
    """

    threshold: float
    margin: float
    min_words: int


# Identity equality: two loads of one config are two objects, and it keeps the word cache keyable.
@dataclass(frozen=True, eq=False)
class DomainLexicons:
    """The compiled lexicons and the decision rule they are read with.

    Attributes:
        domains: The domain labels, in config order; ties break towards the first.
        exact: Whole-word entries to the (domain, weight) pairs they score.
        prefixes: Prefix entries, without the `*`, to the pairs they score.
        prefix_lengths: Every prefix length present, shortest first.
        fallback: Source labels outside the label set to the label they count as.
        rule: The decision rule.
    """

    domains: Tuple[str, ...]
    exact: Mapping[str, DomainWeights]
    prefixes: Mapping[str, DomainWeights]
    prefix_lengths: Tuple[int, ...]
    fallback: Mapping[str, str]
    rule: DecisionRule

    def word_weights(self, word: str) -> DomainWeights:
        """Return each domain the word scores for, at its strongest matching entry.

        Args:
            word: One lowercased word.

        Returns:
            (domain, weight) pairs, empty when no entry matches.
        """
        return _match_word(word, self)

    def fallback_label(self, source_domain: str) -> str:
        """Return the label a document keeps when no domain clears the rule.

        Args:
            source_domain: The domain label of the document's source.

        Returns:
            The source label, mapped through the fallback table when it is not a domain.
        """
        return self.fallback.get(source_domain, source_domain)


def _require(config: Mapping[str, Any], key: str) -> Any:
    """Return `config[key]`, naming the key when it is missing."""
    if key not in config:
        raise ValueError(f"classifier config key '{key}' missing")
    return config[key]


def _domain_entries(domain: str, tiers: Mapping[str, List[str]], weights: Mapping[str, float]) -> Dict[str, float]:
    """Read one domain's entries with their weights, rejecting repeats and entries a shorter prefix covers."""
    entries: Dict[str, float] = {}
    for tier, tier_entries in tiers.items():
        if tier not in weights:
            raise ValueError(f"domain '{domain}' uses tier '{tier}', which has no weight")
        for entry in tier_entries:
            entry = str(entry).lower()
            if entry in entries:
                raise ValueError(f"domain '{domain}' lists '{entry}' twice")
            entries[entry] = weights[tier]
    prefixes = {entry.rstrip(PREFIX_MARK): weight for entry, weight in entries.items() if entry.endswith(PREFIX_MARK)}
    for entry, weight in entries.items():
        stem = entry.rstrip(PREFIX_MARK)
        for length in range(1, len(stem)):
            if prefixes.get(stem[:length], 0.0) >= weight:
                raise ValueError(f"domain '{domain}' entry '{entry}' is covered by its prefix '{stem[:length]}*'")
    return entries


def load_lexicons(config: Union[str, Path, Mapping[str, Any]]) -> DomainLexicons:
    """Compile the per-domain lexicons from the classifier config.

    The config has four keys: `weights` (tier name to weight), `lexicons`
    (domain to tier to list of entries), `fallback` (source label to domain)
    and `rule` (`threshold`, `margin`, `min_words`).

    Args:
        config: The config mapping, or the path of its YAML file.

    Returns:
        The compiled lexicons.

    Raises:
        ValueError: If a key is missing, a domain or fallback target is not one
            of the judge's labels, a tier has no weight, an entry repeats within
            a domain, or a shorter prefix of the same domain already covers it.
    """
    if isinstance(config, (str, Path)):
        config = load_yaml(Path(config))
    weights: Dict[str, float] = {tier: float(weight) for tier, weight in _require(config, "weights").items()}
    exact: Dict[str, List[Tuple[str, float]]] = {}
    prefixes: Dict[str, List[Tuple[str, float]]] = {}
    domains = tuple(_require(config, "lexicons"))
    for domain, tiers in config["lexicons"].items():
        if domain not in DOMAINS:
            raise ValueError(f"lexicon domain '{domain}' is not a judge label: {DOMAINS}")
        for entry, weight in _domain_entries(domain, tiers, weights).items():
            table = prefixes if entry.endswith(PREFIX_MARK) else exact
            table.setdefault(entry.rstrip(PREFIX_MARK), []).append((domain, weight))
    fallback: Dict[str, str] = dict(config.get("fallback") or {})
    for source_label, target in fallback.items():
        if target not in DOMAINS:
            raise ValueError(f"fallback '{source_label}' -> '{target}' is not a judge label: {DOMAINS}")
    rule = _require(config, "rule")
    return DomainLexicons(
        domains=domains,
        exact={entry: tuple(pairs) for entry, pairs in exact.items()},
        prefixes={entry: tuple(pairs) for entry, pairs in prefixes.items()},
        prefix_lengths=tuple(sorted({len(entry) for entry in prefixes})),
        fallback=fallback,
        rule=DecisionRule(
            threshold=float(_require(rule, "threshold")),
            margin=float(_require(rule, "margin")),
            min_words=int(_require(rule, "min_words")),
        ),
    )


@lru_cache(maxsize=_CACHE_SIZE)
def _match_word(word: str, lexicons: DomainLexicons) -> DomainWeights:
    """Look a word up in the exact and prefix tables, strongest entry per domain."""
    best: Dict[str, float] = {}
    matches: List[DomainWeights] = []
    if word in lexicons.exact:
        matches.append(lexicons.exact[word])
    for length in lexicons.prefix_lengths:
        if length > len(word):
            break
        pairs = lexicons.prefixes.get(word[:length])
        if pairs:
            matches.append(pairs)
    for pairs in matches:
        for domain, weight in pairs:
            if weight > best.get(domain, 0.0):
                best[domain] = weight
    return tuple(best.items())


def score_text(text: str, lexicons: DomainLexicons) -> Tuple[Dict[str, float], int]:
    """Score a text per domain as weighted lexicon hits per thousand words.

    Args:
        text: The document text.
        lexicons: The compiled lexicons.

    Returns:
        One score per domain, zero for a domain with no hit or an empty text,
        and the number of words scored.
    """
    hits: Dict[str, float] = dict.fromkeys(lexicons.domains, 0.0)
    words = 0
    for match in _WORD.finditer(text):
        words += 1
        for domain, weight in lexicons.word_weights(match.group(0).lower()):
            hits[domain] += weight
    if words == 0:
        return hits, 0
    return {domain: hit * PER_WORDS / words for domain, hit in hits.items()}, words


def assign_domain(scores: Mapping[str, float], words: int, rule: DecisionRule, fallback: str) -> str:
    """Pick the leading domain, or fall back to the source's label.

    Args:
        scores: One score per domain; ties break towards the earlier key.
        words: How many words the scores were read over.
        rule: The decision rule.
        fallback: The label the document keeps when the rule does not assign one.

    Returns:
        The assigned domain label.
    """
    if words < rule.min_words:
        return fallback
    ranked = sorted(scores.items(), key=lambda item: -item[1])
    leader, lead_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    if lead_score >= rule.threshold and lead_score - runner_up >= rule.margin:
        return leader
    return fallback


def label_document(text: str, source_domain: str, lexicons: DomainLexicons) -> str:
    """Label one document with the configured rule.

    Args:
        text: The document text.
        source_domain: The domain label of the document's source.
        lexicons: The compiled lexicons, carrying the rule.

    Returns:
        The assigned domain, or the source's label mapped through the fallback
        table when no domain clears the rule.
    """
    scores, words = score_text(text, lexicons)
    return assign_domain(scores, words, lexicons.rule, lexicons.fallback_label(source_domain))
