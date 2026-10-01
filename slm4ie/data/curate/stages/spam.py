"""Adult/SEO-spam filter wrapped as a datatrove pipeline step.

The corpus statistics surfaced heavy adult and SEO-spam contamination in
the web sources (escort/porn/dating vocabulary among the most frequent
content words). Neither the language filter nor the Gopher heuristics
remove it, because it is grammatical text. `SpamFilter` drops such
documents using three complementary, language-aware signals:

* a per-language lexicon of unambiguous adult and SEO/scam terms
  (curated lists shipped under `slm4ie/data/curate/resources/spam/`, with LDNOOBW lists
  auto-loaded on demand for languages without a curated file);
* a language-agnostic URL/domain blocklist matched against
  `metadata.url`;
* an optional pluggable model scorer.

Lexicon matching is stem-based so Slovene inflection neither evades nor
over-fires. Each entry is reduced to a stem key: the last token of the
entry drops one trailing vowel when it has five or more letters, and the
key then accepts up to three more word characters (`joški` → `jošk`
matches `joške`, `joškah`, `joškami`). Earlier tokens of a phrase match
literally. Entries whose keys coincide or extend one another by at most
three characters collapse to one key, and the hit thresholds count
distinct keys, so one word repeated or inflected counts once.

A document is flagged when any signal trips. Flagged documents carry
`metadata.spam_reason` and `metadata.spam_terms` (the matched stem
keys); they are dropped, except for a configurable `keep_fraction` that
is retained to preserve a controlled sample. The stage writes every
dropped document to `<dataset>/removed/<rank>.jsonl.gz` inside its
unit, which the unit's integrity check and document digest ignore. The
lexicon is intentionally high-precision: only terms that are
overwhelmingly adult/spam in context are listed, so legitimate
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
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
from urllib.parse import urlsplit

from numpy.random import default_rng

from datatrove.data import Document
from datatrove.executor import LocalPipelineExecutor
from datatrove.io import cached_asset_path_or_download
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.pipeline.writers.jsonl import JsonlWriter

from slm4ie.data.curate.paths import CuratePaths
from slm4ie.data.curate.stages import StageRun
from slm4ie.data.curate.stages.common import jsonl_reader, jsonl_writer, pipeline_io_counts

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
        min_adult_hits: Drop a document once the distinct adult stem
            keys it matches reach this count.
        min_spam_hits: Drop a document once the distinct SEO/scam stem
            keys it matches reach this count.
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
        Path to `slm4ie/data/curate/resources/spam/`.
    """
    return Path(__file__).resolve().parents[1] / "resources" / "spam"


def load_spam_lexicon(code: str) -> Tuple[Set[str], Set[str], bytes]:
    """Load the curated adult and SEO-spam term sets for a language.

    Resolves `<code>/adult.txt` and `<code>/spam.txt` under
    `slm4ie/data/curate/resources/spam/`, parsing each into a lowercased token set.

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

    Resolves `domains.txt` under `slm4ie/data/curate/resources/spam/`.

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


#: Tokens shorter than this keep their final vowel: a four-letter stem
#: such as `anal` would reach common words (`analiza`).
_MIN_STEM_TOKEN = 5

#: Word characters a stem key accepts after it, covering Slovene endings.
_MAX_SUFFIX = 3


def stem_key(term: str) -> str:
    """Reduce a lexicon entry to the stem key it is matched and counted by.

    Only the entry's last token inflects: it loses one trailing vowel when
    it has at least `_MIN_STEM_TOKEN` letters.

    Args:
        term: A lowercased lexicon entry, one word or a space-separated
            phrase.

    Returns:
        The entry with whitespace normalised and its last token stemmed.
    """
    tokens = term.split()
    last = tokens[-1]
    if len(last) >= _MIN_STEM_TOKEN and last[-1] in "aeiou":
        tokens[-1] = last[:-1]
    return " ".join(tokens)


def collapse_keys(terms: Set[str]) -> Set[str]:
    """Stem a term set and drop keys another key already matches.

    A key is covered when another key has the same phrase head and its last
    token is a prefix of this key's last token by at most `_MAX_SUFFIX`
    characters, so `kurbir` folds into `kurb`.

    Args:
        terms: Lowercased lexicon entries.

    Returns:
        The distinct stem keys the filter counts.
    """
    keys = {stem_key(t) for t in terms if t.strip()}
    split = {k: k.rpartition(" ") for k in keys}
    out = set()
    for key, (head, _, last) in split.items():
        covered = any(
            other != key and o_head == head and last.startswith(o_last) and len(last) - len(o_last) <= _MAX_SUFFIX
            for other, (o_head, _, o_last) in split.items()
        )
        if not covered:
            out.add(key)
    return out


@dataclass
class TermMatcher:
    """A compiled lexicon that reports the distinct terms a text matches.

    Attributes:
        pattern: Alternation over every key, capturing the matched surface.
        keys: The keys the pattern matches; surfaces map back to these.
        stemmed: Whether keys accept an inflectional suffix.
    """

    pattern: re.Pattern
    keys: Set[str]
    stemmed: bool

    def matched_keys(self, text: str) -> Set[str]:
        """Return the distinct keys whose forms occur in a lowercased text.

        Args:
            text: Lowercased document text.

        Returns:
            The set of matched keys.
        """
        found: Set[str] = set()
        for surface in self.pattern.findall(text):
            if not self.stemmed:
                found.add(surface)
                continue
            *head, last = surface.split()
            for cut in range(min(_MAX_SUFFIX, len(last)) + 1):
                key = " ".join([*head, last[: len(last) - cut]])
                if key in self.keys:
                    found.add(key)
                    break
        return found


def _compile_terms(terms: Set[str], language: str) -> Optional[TermMatcher]:
    """Compile a term set into a distinct-term matcher.

    Args:
        terms: Lowercased terms to match.
        language: Language code; languages in `_NO_BOUNDARY_LANGS` match
            the literal terms without word boundaries or stemming.

    Returns:
        A `TermMatcher`, or `None` when `terms` is empty.
    """
    if not terms:
        return None
    if language in _NO_BOUNDARY_LANGS:
        alternation = "|".join(re.escape(t) for t in sorted(terms, key=lambda t: (-len(t), t)))
        return TermMatcher(re.compile(f"({alternation})"), set(terms), stemmed=False)
    keys = collapse_keys(terms)
    # Longest first, so a phrase wins over its own leading word.
    branches = []
    for key in sorted(keys, key=lambda k: (-len(k), k)):
        tokens = key.split()
        branches.append(r"\s+".join(re.escape(t) for t in tokens) + r"\w{0,%d}" % _MAX_SUFFIX)
    pattern = re.compile(r"(?<!\w)({})(?!\w)".format("|".join(branches)))
    return TermMatcher(pattern, keys, stemmed=True)


class SpamFilter(BaseFilter):
    """Drop adult/SEO-spam documents via lexicon, URL, and model signals.

    A document is flagged when any of these trip: its URL host is on the
    blocklist; distinct adult stem keys reach `min_adult_hits`; distinct
    SEO/scam stem keys reach `min_spam_hits`; or an optional model scorer
    returns at least `model_threshold`. Every flagged document gets
    `metadata.spam_reason` and `metadata.spam_terms`; it is dropped,
    except for a seeded `keep_fraction` that is retained.

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
        self._adult_regex: Dict[str, Optional[TermMatcher]] = {}
        self._spam_regex: Dict[str, Optional[TermMatcher]] = {}

    def _adult_pattern(self, lang: str) -> Optional[TermMatcher]:
        """Return (and cache) the adult-term regex for a language.

        Falls back to an LDNOOBW list for languages without a curated
        set when `use_ldnoobw` is enabled.

        Args:
            lang: Language code.

        Returns:
            Compiled matcher, or `None` when no terms are available.
        """
        if lang not in self._adult_regex:
            terms = self._adult_words.get(lang)
            if terms is None and self.config.use_ldnoobw:
                terms = self._load_ldnoobw(lang)
            self._adult_regex[lang] = _compile_terms(terms or set(), lang)
        return self._adult_regex[lang]

    def _spam_pattern(self, lang: str) -> Optional[TermMatcher]:
        """Return (and cache) the SEO/scam-term regex for a language.

        Args:
            lang: Language code.

        Returns:
            Compiled matcher, or `None` when no terms are available.
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
            Any flagged document first gets `metadata.spam_reason` and
            `metadata.spam_terms` (the sorted stem keys it matched in
            either lexicon); one retained by `keep_fraction` returns
            `True`.
        """
        lang = doc.metadata.get("language") or self.config.default_language
        text = doc.text.lower()
        reasons = []

        if self.config.url_blocklist and self.domains:
            url = doc.metadata.get("url")
            if url and self._host_blocked(url):
                reasons.append("spam_url")

        adult_pat = self._adult_pattern(lang)
        adult = adult_pat.matched_keys(text) if adult_pat is not None else set()
        if len(adult) >= self.config.min_adult_hits:
            reasons.append("adult_lexicon")

        spam_pat = self._spam_pattern(lang)
        spam = spam_pat.matched_keys(text) if spam_pat is not None else set()
        if len(spam) >= self.config.min_spam_hits:
            reasons.append("spam_lexicon")

        if self.model_fn is not None and self.model_fn(doc.text) >= self.config.model_threshold:
            reasons.append("spam_model")

        if not reasons:
            return True

        reason = ",".join(reasons)
        doc.metadata["spam_reason"] = reason
        doc.metadata["spam_terms"] = sorted(adult | spam)
        self.stat_update("flagged", f"flagged_{lang}")
        if self.config.keep_fraction > 0.0 and self.uniform() < self.config.keep_fraction:
            self.stat_update("kept_flagged")
            return True
        return False, reason


def _build_spam_config(spcfg: Dict[str, Any]) -> SpamConfig:
    """Resolve a spam-stage config slice into a `SpamConfig`.

    Args:
        spcfg: The effective `spam` config slice for one bucket.

    Returns:
        The resolved `SpamConfig`, with defaults applied.

    Raises:
        ValueError: If `model` is set; no model resolver is wired.
    """
    if spcfg.get("model"):
        raise ValueError(
            "the pretrain config's spam.model is set, but no model resolver is "
            "configured. Leave it null, or wire a scorer before enabling it."
        )
    return SpamConfig(
        min_adult_hits=int(spcfg.get("min_adult_hits", 2)),
        min_spam_hits=int(spcfg.get("min_spam_hits", 2)),
        keep_fraction=float(spcfg.get("keep_fraction", 0.0)),
        default_language=str(spcfg.get("default_language", "sl")),
        url_blocklist=bool(spcfg.get("url_blocklist", True)),
        use_ldnoobw=bool(spcfg.get("use_ldnoobw", True)),
        model=spcfg.get("model"),
        model_threshold=float(spcfg.get("model_threshold", 0.5)),
    )


def build_spam_executors(
    paths: CuratePaths,
    *,
    tasks: int = 1,
    spam_config: Optional[SpamConfig] = None,
    adult_words: Optional[Dict[str, Set[str]]] = None,
    spam_words: Optional[Dict[str, Set[str]]] = None,
    domains: Optional[Set[str]] = None,
    model_fn: Optional[Callable[[str], float]] = None,
    seed: Optional[int] = None,
    input_override: Optional[Path] = None,
    output_override: Optional[Path] = None,
) -> List[LocalPipelineExecutor]:
    """Build the spam stage: read 01_language/ → SpamFilter → write 02_spam/.

    Dropped documents go to `<output>/<dataset>/removed/<rank>.jsonl.gz`,
    inside the unit so they are promoted and removed with it.

    Args:
        paths: Resolved input/output locations.
        tasks: Parallel worker count.
        spam_config: `SpamFilter` knob bundle; defaults to `SpamConfig()`.
        adult_words: Per-language adult-term sets, keyed by language code.
        spam_words: Per-language SEO/scam-term sets, keyed by language
            code.
        domains: Blocklisted registered domains.
        model_fn: Optional text-to-score callable enabling the model
            signal.
        seed: Seed for the `keep_fraction` sampler.
        input_override: Optional folder to read from instead of the
            language stage's output, used to restrict the stage to a
            symlinked subset of datasets.
        output_override: Optional folder to write to instead of the
            stage's output folder (the run loop's staging folder).

    Returns:
        A list with one `LocalPipelineExecutor`.
    """
    in_ = input_override if input_override is not None else paths.stage_dir("language")
    out = output_override if output_override is not None else paths.stage_dir("spam")
    executor = LocalPipelineExecutor(
        pipeline=[
            jsonl_reader(in_),
            SpamFilter(
                adult_words=adult_words or {},
                spam_words=spam_words or {},
                domains=domains or set(),
                config=spam_config or SpamConfig(),
                model_fn=model_fn,
                seed=seed,
                exclusion_writer=JsonlWriter(
                    output_folder=str(out), output_filename="${dataset}/removed/${rank}.jsonl.gz"
                ),
            ),
            jsonl_writer(out),
        ],
        tasks=tasks,
        workers=tasks,
        logging_dir=str(paths.logs_dir("spam")),
        skip_completed=False,
    )
    return [executor]


def run(job: StageRun) -> Tuple[int, int]:
    """Run the spam stage over the job's input view.

    Args:
        job: What to filter, and where to write it; `job.spam_assets` holds
            the loaded lexicons and domain blocklist.

    Returns:
        `(records_in, records_out)` from the run's datatrove stats.
    """
    assets = job.spam_assets
    execs = build_spam_executors(
        job.paths,
        tasks=job.workers,
        spam_config=_build_spam_config(job.config),
        adult_words=assets.adult_words,
        spam_words=assets.spam_words,
        domains=assets.domains,
        input_override=job.input_view,
        output_override=job.output_folder,
    )
    return pipeline_io_counts(execs[-1].run())
