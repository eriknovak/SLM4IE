"""Page checks: one module per document kind, each registering its `@check`s.

`figures` holds the Figure row checks the record, programme and entry share,
and the figure rules of figures.md (FIG-020..025) on the record's own figures.

`rules` imports every module here, so any runner that uses the rules sees
them. A module imports only the standard library, pyyaml and this package;
the report-guard hook runs the checker without jinja2. REC-160 also loads
labflow:analyze's table_rules.py, which uses the standard library only.
"""
