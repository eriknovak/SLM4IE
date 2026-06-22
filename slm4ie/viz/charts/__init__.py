"""Builders for chart types plotly does not provide natively.

Reserved home for constructed charts whose logic goes beyond styling a native
trace: bump plots (rank over an axis), Pareto-front scatters (compute the
non-dominated frontier and overlay it), and waffle charts (a grid of cells).

Each builder is added here when a concrete chart needs it, following the color
roles fixed in the design (categorical for tokenizers, accent for frontier and
guide lines). The package is intentionally empty until then.
"""
