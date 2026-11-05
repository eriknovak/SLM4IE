"""Page checks: one module per document kind, each registering its `@check`s.

`rules` imports every module here, so any runner that uses the rules sees
them. A module imports only the standard library, pyyaml and this package;
the report-guard hook runs the checker without jinja2.
"""
