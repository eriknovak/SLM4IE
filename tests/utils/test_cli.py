"""Tests for slm4ie.utils.cli helpers."""

import argparse

import pytest

from slm4ie.utils.cli import add_tags, parse_tags

# ===============================
# --tag
# ===============================


def test_tags_parse_to_a_dict_and_a_pair_without_equals_sign_is_refused():
    """Repeated --tag KEY=VALUE becomes a dict split at the first `=`; a bare word raises naming it."""
    parser = argparse.ArgumentParser()
    add_tags(parser)

    args = parser.parse_args(["--tag", "tier=scratch", "--tag", "note=a=b"])

    assert parse_tags(args.tag) == {"tier": "scratch", "note": "a=b"}
    with pytest.raises(ValueError, match="'scratch'"):
        parse_tags(["scratch"])
