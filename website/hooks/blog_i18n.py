"""MkDocs hook that hides the blog files from the i18n plugin.

mkdocs-static-i18n drops the pages the Material blog plugin generates, which
empties the news section in every language (ultrabug/mkdocs-static-i18n#283,
unfixed upstream). This hook lifts the blog files out of the file set before
the i18n plugin sees them and puts them back afterwards. Every language build
renders the news section under its own prefix: the index takes that language's
`index.<locale>.md` when one exists, and the posts stay in English.

Adapted from the workaround by Kamil Krzyśków (MIT), posted on that issue.
Wired through `hooks:` in `mkdocs.yml`.
"""

import posixpath
from typing import Any, Dict, List, Tuple
from urllib.parse import urlsplit

from mkdocs import plugins
from mkdocs.config.defaults import MkDocsConfig
from mkdocs.structure.files import File, Files
from mkdocs.structure.nav import Navigation
from material.plugins.blog.structure import View
from mkdocs.structure.pages import Page

# Build state, all set on every build so `mkdocs serve`, which rebuilds in the
# same process, never reuses the previous build's values.
_BLOG_FILES: List[File] = []
_BLOG_PREFIXES: Tuple[str, ...] = ()
_DEFAULT_LANGUAGE = "en"
_CURRENT_LANGUAGE = "en"


@plugins.event_priority(-95)
def _disconnect_blog_files(files: Files, config: MkDocsConfig, **kwargs: Any) -> Files:
    """Remove the blog files from the file set.

    Runs after the blog plugin has generated its pages (-50) and before the
    i18n plugin rewrites the file set (-100). A translated index such as
    `news/index.sl.md` is dropped too: the blog plugin would otherwise render it
    as a stray page, and `_connect_blog_files` reads it as the index source.

    Args:
        files: The file set as the blog plugin left it.
        config: The MkDocs config, read for the blog and i18n plugin settings.
        **kwargs: Remaining event arguments, unused.

    Returns:
        The file set without the blog files.
    """
    global _BLOG_FILES, _BLOG_PREFIXES, _DEFAULT_LANGUAGE, _CURRENT_LANGUAGE
    _BLOG_FILES = []
    kept: List[File] = []

    _BLOG_PREFIXES = tuple(
        instance.config.blog_dir.rstrip("/") + "/"
        for name, instance in config.plugins.items()
        if name.startswith("material/blog")
    )
    plugin = config.plugins["i18n"]
    locales = [language.locale for language in plugin.config.languages]
    _DEFAULT_LANGUAGE = next(
        (language.locale for language in plugin.config.languages if language.default),
        _DEFAULT_LANGUAGE,
    )
    _CURRENT_LANGUAGE = plugin.current_language or _DEFAULT_LANGUAGE
    translated = tuple(f".{locale}.md" for locale in locales)

    for file in files:
        if not file.src_uri.startswith(_BLOG_PREFIXES):
            kept.append(file)
        elif not file.src_uri.endswith(translated):
            _BLOG_FILES.append(file)

    return Files(kept)


@plugins.event_priority(-105)
def _connect_blog_files(files: Files, **kwargs: Any) -> Files:
    """Put the blog files back once the i18n plugin has run.

    In a non-default build each file moves under the language prefix, so it no
    longer overwrites the default build's copy, and the index reads its
    translation when one exists.

    Args:
        files: The file set as the i18n plugin left it.
        **kwargs: Remaining event arguments, unused.

    Returns:
        The file set with the blog files restored.
    """
    for file in _BLOG_FILES:
        if _CURRENT_LANGUAGE != _DEFAULT_LANGUAGE:
            _localise(file, _CURRENT_LANGUAGE)
        # The plugin's sitemap template reads `alternates` off every file; blog
        # files never passed through the plugin, so they carry none.
        file.alternates = {}
        files.append(file)

    return files


on_files = plugins.CombinedEvent(_disconnect_blog_files, _connect_blog_files)


def _localise(file: File, locale: str) -> None:
    """Move a blog file under a language prefix and point an index at its translation.

    Args:
        file: A blog file, as the blog plugin left it.
        locale: The language being built.
    """
    file.dest_uri = f"{locale}/{file.dest_uri}"
    file.url = f"{locale}/{file.url}"
    file.abs_dest_path = posixpath.join(file.dest_dir, file.dest_uri)
    if file.abs_src_path and file.src_uri.endswith("index.md"):
        translation = file.abs_src_path.removesuffix(".md") + f".{locale}.md"
        if posixpath.isfile(translation):
            file.abs_src_path = translation


@plugins.event_priority(-100)
def on_nav(nav: Navigation, config: MkDocsConfig, **kwargs: Any) -> Navigation:
    """Title the news entry in the language being built.

    The blog plugin replaces the entry the i18n plugin translated with an
    untitled view of its own, which then falls back to the English page title.

    Args:
        nav: The navigation for this language.
        config: The MkDocs config, read for the navigation and its translations.
        **kwargs: Remaining event arguments, unused.

    Returns:
        The navigation, with the news entry titled in this language.
    """
    plugin = config.plugins["i18n"]
    if plugin.is_default_language_build:
        return nav

    translations = plugin.current_language_config.nav_translations or {}
    titles = _nav_titles(config.nav or [])
    for item in nav.items:
        file = getattr(item, "file", None)
        if file is None or not file.src_uri.startswith(_BLOG_PREFIXES):
            continue
        title = titles.get(file.src_uri)
        if item.title is None and title in translations:
            item.title = translations[title]

    return nav


def _nav_titles(nav: List[Any]) -> Dict[str, str]:
    """Collect the title each navigation entry gives a source file.

    Args:
        nav: A navigation list as it appears in the config, before the i18n
            plugin translates the navigation it builds from it.

    Returns:
        Mapping of source path to the title the navigation gives it.
    """
    titles: Dict[str, str] = {}
    for entry in nav:
        if not isinstance(entry, dict):
            continue
        for title, target in entry.items():
            if isinstance(target, str):
                titles[target] = title
            elif isinstance(target, list):
                titles.update(_nav_titles(target))
    return titles


@plugins.event_priority(-50)
def on_page_context(context: Dict[str, Any], page: Page, config: MkDocsConfig, **kwargs: Any) -> Dict[str, Any]:
    """Point the language selector at the same blog page in each language.

    Archive and category pages also take the index's `hide` front matter, so
    they keep its layout.

    The plugin rewrites the selector per page from the page's translations.
    Blog pages have none, so it leaves the previous page's links in place and
    the selector sends the reader to an unrelated page. Every language builds
    the same blog pages under its own prefix, so the link swaps the prefix.

    Args:
        context: The Jinja context for the page about to be rendered.
        page: The page being rendered.
        config: The MkDocs config, whose `extra.alternate` the selector reads.
        **kwargs: Remaining event arguments, unused.

    Returns:
        The context, unchanged.
    """
    if not page.file.src_uri.startswith(_BLOG_PREFIXES):
        return context

    # Archive and category pages carry no front matter; the index's layout
    # keeps them from growing the sidebars the index hides.
    blog = config.plugins["material/blog"].blog
    if isinstance(page, View) and page.url != blog.url:
        page.meta.setdefault("hide", blog.meta.get("hide", []))

    # The theme reads the selector off the attribute, which the plugin sets
    # alongside the `extra` mapping's own entry; the two are separate objects.
    base = urlsplit(config.site_url or "/").path or "/"
    url = page.url.removeprefix(f"{_CURRENT_LANGUAGE}/")
    for alternate in getattr(config.extra, "alternate", []):
        prefix = "" if alternate["lang"] == _DEFAULT_LANGUAGE else f"{alternate['lang']}/"
        alternate["link"] = f"{base}{prefix}{url}"

    return context
