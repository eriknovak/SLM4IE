"""Adult/SEO-spam filter wrapped as a datatrove pipeline step.

The corpus statistics surfaced heavy adult and SEO-spam contamination in
the web sources (escort/porn/dating vocabulary among the most frequent
content words). Neither the language filter nor the Gopher heuristics
remove it, because it is grammatical text. `SpamFilter` drops such
documents using three complementary, language-aware signals:

* a per-language lexicon of unambiguous adult and SEO/scam terms
  (curated lists shipped under `slm4ie/data/spam/`, with LDNOOBW lists
  auto-loaded on demand for languages without a curated file);
* a language-agnostic URL/domain blocklist matched against
  `metadata.url`;
* an optional pluggable model scorer.

A document is flagged when any signal trips. Flagged documents are
dropped, except for a configurable `keep_fraction` that is retained
(and tagged with `metadata.spam_reason`) to preserve a controlled
sample. The lexicon is intentionally high-precision: only terms that
are overwhelmingly adult/spam in context are listed, so legitimate
health/dating/massage text is not discarded.
"""

# Datatrove probes installed dependencies via importlib.metadata at class
# definition time; both submodules must be imported explicitly under
# Python 3.13 before the datatrove imports (see language.py).
import importlib.metadata  # noqa: F401
import importlib.util  # noqa: F401
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Set, Tuple, Union
from urllib.parse import urlsplit

from numpy.random import default_rng

from datatrove.data import Document
from datatrove.io import cached_asset_path_or_download
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter

logger = logging.getLogger(__name__)

#: Languages without inter-word spaces, where lexicon matching must not
#: require non-word boundaries (mirrors datatrove's C4 badwords filter).
_NO_BOUNDARY_LANGS = frozenset({"ja", "th", "zh"})

#: LDNOOBW ("List of Dirty, Naughty, Obscene and Otherwise Bad Words")
#: base URL and the language codes it covers, used to auto-load adult
#: word lists for languages without a curated list shipped in-repo.
_LDNOOBW_BASE_URL = (
    "https://raw.githubusercontent.com/LDNOOBW/"
    "List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/"
    "5faf2ba42d7b1c0977169ec3611df25a3c08eb13/"
)
_LDNOOBW_EN_URL = (
    "https://raw.githubusercontent.com/LDNOOBW/"
    "List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/"
    "25e679f03d96baa721cde20db9944649e8d0a844/en"
)
_LDNOOBW_LANGS = frozenset(
    {
        "ar",
        "cs",
        "da",
        "de",
        "en",
        "eo",
        "es",
        "fa",
        "fi",
        "fil",
        "fr",
        "hi",
        "hu",
        "it",
        "ja",
        "kab",
        "ko",
        "nl",
        "no",
        "pl",
        "pt",
        "ru",
        "sv",
        "th",
        "tlh",
        "tr",
        "zh",
    }
)


@dataclass
class SpamConfig:
    """Output-affecting knobs for `SpamFilter`.

    Attributes:
        min_adult_hits: Drop a document once its adult-term occurrences
            reach this count.
        min_spam_hits: Drop a document once its SEO/scam-term
            occurrences reach this count.
        keep_fraction: Fraction of flagged documents to retain anyway,
            sampled from a seeded uniform distribution.
        default_language: Language assumed for documents lacking a
            `metadata.language` value.
        url_blocklist: Enable the URL/domain blocklist signal.
        use_ldnoobw: Auto-load LDNOOBW adult lists for a document's
            language when no curated list is available for it.
        model: Optional classifier spec resolved by the caller into a
            scorer; `None` disables the model signal.
        model_threshold: Score at or above which the model flags a
            document as spam.
    """

    min_adult_hits: int = 2
    min_spam_hits: int = 2
    keep_fraction: float = 0.0
    default_language: str = "sl"
    url_blocklist: bool = True
    use_ldnoobw: bool = True
    model: Optional[str] = None
    model_threshold: float = 0.5


def _parse_terms(raw: bytes) -> Set[str]:
    """Parse a term-list payload into a lowercased token set.

    Blank lines and lines whose first non-whitespace character is `#`
    are skipped; remaining lines are stripped and lowercased.

    Args:
        raw: UTF-8 encoded contents of a term-list file.

    Returns:
        Set of lowercased terms.
    """
    out: Set[str] = set()
    for line in raw.decode("utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.add(line.lower())
    return out


def _spam_dir() -> Path:
    """Return the directory holding the bundled spam assets.

    Returns:
        Path to `slm4ie/data/spam/`.
    """
    return Path(__file__).resolve().parent.parent / "spam"


def load_spam_lexicon(code: str) -> Tuple[Set[str], Set[str], bytes]:
    """Load the curated adult and SEO-spam term sets for a language.

    Resolves `<code>/adult.txt` and `<code>/spam.txt` under
    `slm4ie/data/spam/`, parsing each into a lowercased token set.

    Args:
        code: Language code identifying the bundled lists (e.g. `"sl"`).

    Returns:
        Tuple `(adult terms, spam terms, raw bytes)`. The bytes are the
        concatenated file contents, intended for stable sentinel hashing.

    Raises:
        ValueError: If no `<code>/` folder with the two files exists. The
            message lists the available codes.
    """
    folder = _spam_dir() / code
    adult_path = folder / "adult.txt"
    spam_path = folder / "spam.txt"
    if not adult_path.exists() or not spam_path.exists():
        available = sorted(p.name for p in _spam_dir().iterdir() if p.is_dir() and not p.name.startswith("_"))
        raise ValueError(f"unknown spam lexicon language {code!r}; available: {available}")
    adult_raw = adult_path.read_bytes()
    spam_raw = spam_path.read_bytes()
    return _parse_terms(adult_raw), _parse_terms(spam_raw), adult_raw + b"\x00" + spam_raw


def load_spam_domains() -> Tuple[Set[str], bytes]:
    """Load the bundled adult/spam domain blocklist.

    Resolves `domains.txt` under `slm4ie/data/spam/`.

    Returns:
        Tuple `(domain set, raw bytes)`. The bytes are the original
        file contents, intended for stable sentinel hashing.
    """
    path = _spam_dir() / "domains.txt"
    raw = path.read_bytes()
    return _parse_terms(raw), raw


@dataclass
class SpamAssets:
    """Resolved spam-filter assets plus bytes for sentinel hashing.

    Attributes:
        adult_words: Per-language adult-term sets, keyed by language code.
        spam_words: Per-language SEO/scam-term sets, keyed by language
            code.
        domains: Blocklisted registered domains (empty when the URL
            blocklist is disabled).
        raw_bytes: Stable concatenation of the loaded list contents, for
            folding into the spam stage's sentinel hash.
    """

    adult_words: Dict[str, Set[str]]
    spam_words: Dict[str, Set[str]]
    domains: Set[str]
    raw_bytes: bytes


def load_spam_assets(languages: Sequence[str], *, url_blocklist: bool = True) -> SpamAssets:
    """Load curated lexicons for several languages plus the domain blocklist.

    Args:
        languages: Language codes whose curated lists to load eagerly.
        url_blocklist: Load the domain blocklist when True; otherwise
            leave the domain set empty.

    Returns:
        A `SpamAssets` bundle. `raw_bytes` is deterministic in the
        language set (codes are sorted) so it is stable across runs.

    Raises:
        ValueError: If any requested language has no curated list.
    """
    adult_words: Dict[str, Set[str]] = {}
    spam_words: Dict[str, Set[str]] = {}
    chunks = []
    for code in sorted(set(languages)):
        adult, spam, raw = load_spam_lexicon(code)
        adult_words[code] = adult
        spam_words[code] = spam
        chunks.append(code.encode("utf-8") + b":" + raw)
    domains: Set[str] = set()
    if url_blocklist:
        domains, domains_raw = load_spam_domains()
        chunks.append(b"domains:" + domains_raw)
    return SpamAssets(adult_words, spam_words, domains, b"\x00".join(chunks))


def _compile_terms(terms: Set[str], language: str) -> Optional[re.Pattern]:
    """Compile a term set into a single occurrence-counting regex.

    Args:
        terms: Lowercased terms to match.
        language: Language code; languages in `_NO_BOUNDARY_LANGS` match
            without requiring non-word flanks.

    Returns:
        A compiled pattern whose capture group matches one term, or
        `None` when `terms` is empty.
    """
    if not terms:
        return None
    alternation = "|".join(re.escape(t) for t in sorted(terms))
    if language in _NO_BOUNDARY_LANGS:
        return re.compile(alternation)
    return re.compile(r"(?:\W|^)({})(?:\W|$)".format(alternation))


class SpamFilter(BaseFilter):
    """Drop adult/SEO-spam documents via lexicon, URL, and model signals.

    A document is flagged when any of these trip: its URL host is on the
    blocklist; adult-term occurrences reach `min_adult_hits`; SEO/scam
    occurrences reach `min_spam_hits`; or an optional model scorer
    returns at least `model_threshold`. Flagged documents are dropped
    with a reason, except for a seeded `keep_fraction` retained with
    `metadata.spam_reason` set.

    Attributes:
        config: The `SpamConfig` knob bundle.
        domains: Lowercased blocklisted registered domains.
        model_fn: Optional callable mapping document text to a score.
    """

    name = "🔞 Spam/Adult"

    def __init__(
        self,
        *,
        adult_words: Dict[str, Set[str]],
        spam_words: Dict[str, Set[str]],
        domains: Set[str],
        config: SpamConfig,
        seed: Optional[int] = None,
        model_fn: Optional[Callable[[str], float]] = None,
        exclusion_writer: Optional[DiskWriter] = None,
    ) -> None:
        """Initialize the filter from preloaded lexicons and config.

        Args:
            adult_words: Per-language adult-term sets, keyed by language
                code.
            spam_words: Per-language SEO/scam-term sets, keyed by
                language code.
            domains: Blocklisted registered domains.
            config: Output-affecting knob bundle.
            seed: Seed for the `keep_fraction` sampler.
            model_fn: Optional text-to-score callable; enables the model
                signal when provided.
            exclusion_writer: Optional datatrove writer for dropped docs.
        """
        super().__init__(exclusion_writer)
        self.config = config
        self._adult_words = dict(adult_words)
        self._spam_words = dict(spam_words)
        self.domains = {d.lower() for d in domains}
        self.model_fn = model_fn
        self.uniform = default_rng(seed).uniform
        self._adult_regex: Dict[str, Optional[re.Pattern]] = {}
        self._spam_regex: Dict[str, Optional[re.Pattern]] = {}

    def _adult_pattern(self, lang: str) -> Optional[re.Pattern]:
        """Return (and cache) the adult-term regex for a language.

        Falls back to an LDNOOBW list for languages without a curated
        set when `use_ldnoobw` is enabled.

        Args:
            lang: Language code.

        Returns:
            Compiled pattern, or `None` when no terms are available.
        """
        if lang not in self._adult_regex:
            terms = self._adult_words.get(lang)
            if terms is None and self.config.use_ldnoobw:
                terms = self._load_ldnoobw(lang)
            self._adult_regex[lang] = _compile_terms(terms or set(), lang)
        return self._adult_regex[lang]

    def _spam_pattern(self, lang: str) -> Optional[re.Pattern]:
        """Return (and cache) the SEO/scam-term regex for a language.

        Args:
            lang: Language code.

        Returns:
            Compiled pattern, or `None` when no terms are available.
        """
        if lang not in self._spam_regex:
            self._spam_regex[lang] = _compile_terms(self._spam_words.get(lang) or set(), lang)
        return self._spam_regex[lang]

    def _load_ldnoobw(self, lang: str) -> Set[str]:
        """Fetch and cache the LDNOOBW adult list for a language.

        Args:
            lang: Language code.

        Returns:
            The term set, or an empty set when the language is
            unsupported or the download is unavailable offline.
        """
        if lang not in _LDNOOBW_LANGS:
            self._adult_words[lang] = set()
            return set()
        try:
            local_path = cached_asset_path_or_download(
                _LDNOOBW_EN_URL if lang == "en" else _LDNOOBW_BASE_URL + lang,
                namespace="filters",
                subfolder="spam_ldnoobw",
            )
            with open(local_path, "rt", encoding="utf-8") as fh:
                terms = {line.strip().lower() for line in fh if line.strip()}
        except Exception:  # noqa: BLE001 - offline/download failure is non-fatal
            logger.warning("LDNOOBW list for %r unavailable; treating as empty.", lang)
            terms = set()
        self._adult_words[lang] = terms
        return terms

    def _host_blocked(self, url: str) -> bool:
        """Return whether a URL's host (or a parent domain) is blocklisted.

        Args:
            url: The document URL.

        Returns:
            True when the host equals a blocklisted domain or is a
            subdomain of one.
        """
        host = (urlsplit(url).hostname or "").lower().strip(".")
        if not host:
            return False
        parts = host.split(".")
        for i in range(len(parts) - 1):
            if ".".join(parts[i:]) in self.domains:
                return True
        return False

    def filter(self, doc: Document) -> Union[bool, Tuple[bool, str]]:
        """Flag adult/SEO-spam documents; keep everything else.

        Args:
            doc: The document to evaluate.

        Returns:
            `True` to keep the document, or `(False, reason)` to drop it.
            A flagged document retained by `keep_fraction` returns `True`
            after recording `metadata.spam_reason`.
        """
        lang = doc.metadata.get("language") or self.config.default_language
        text = doc.text.lower()
        reasons = []

        if self.config.url_blocklist and self.domains:
            url = doc.metadata.get("url")
            if url and self._host_blocked(url):
                reasons.append("spam_url")

        adult_pat = self._adult_pattern(lang)
        if adult_pat is not None and len(adult_pat.findall(text)) >= self.config.min_adult_hits:
            reasons.append("adult_lexicon")

        spam_pat = self._spam_pattern(lang)
        if spam_pat is not None and len(spam_pat.findall(text)) >= self.config.min_spam_hits:
            reasons.append("spam_lexicon")

        if self.model_fn is not None and self.model_fn(doc.text) >= self.config.model_threshold:
            reasons.append("spam_model")

        if not reasons:
            return True

        reason = ",".join(reasons)
        self.stat_update("flagged", f"flagged_{lang}")
        if self.config.keep_fraction > 0.0 and self.uniform() < self.config.keep_fraction:
            self.stat_update("kept_flagged")
            doc.metadata["spam_reason"] = reason
            return True
        return False, reason
