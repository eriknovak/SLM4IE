"""Bundled adult/SEO-spam term lists and a domain blocklist.

Each language has a folder `<code>/` holding two UTF-8 plain-text files:

* `adult.txt` — unambiguous adult/pornography terms.
* `spam.txt` — unambiguous SEO/scam terms (e.g. pill spam, fake casinos).

Both files use one token per line; blank lines and `#`-prefixed lines are
ignored, and tokens are lowercased on load. `domains.txt` next to this
module is a language-agnostic list of adult/spam domains (one registered
domain per line).

The lists are intentionally high-precision: only terms that are
overwhelmingly adult/spam in context belong here. Ambiguous common words
(e.g. body parts, "massage", "dates") are excluded — the URL blocklist
and the per-document hit threshold catch those contexts instead. New
languages are added by dropping a new `<code>/` folder next to this
module; no registry update is needed.

Loaders return the original file bytes alongside the parsed sets so
callers can fold the contents into downstream sentinel hashes.
"""
