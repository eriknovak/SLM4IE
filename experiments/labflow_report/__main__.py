"""The labflow_report command line: `python -m labflow_report <command>`."""

import sys
from argparse import ArgumentParser
from pathlib import Path

from . import backfill, check

# ===============================
# Main function
# ===============================


def get_args(argv: list[str] | None):
    parser = ArgumentParser(
        prog="python -m labflow_report", description="The spec-led labflow report"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    check_parser = commands.add_parser(
        "check", help="print each rule breach as 'file:line RULE-ID message'"
    )
    check_parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="the files to check (default: every document a spec declares)",
    )
    check_parser.add_argument(
        "--experiments",
        type=Path,
        help="the experiments folder (default: the nearest 'experiments' above the first path, else ./experiments)",
    )
    check_parser.add_argument(
        "--specs",
        type=Path,
        default=check.SPEC_FOLDER,
        help="the folder of *.yaml specs (default: specs/ inside this package)",
    )
    check_parser.add_argument(
        "--skip-stage",
        default="",
        help="comma-separated stages whose documents are left out, e.g. concluded,abandoned (default: none)",
    )
    backfill_parser = commands.add_parser(
        "backfill",
        help="rewrite what follows from the record, then list what a person must write",
    )
    backfill_parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="the files to backfill (default: every document a spec declares)",
    )
    backfill_parser.add_argument(
        "--experiments",
        type=Path,
        help="the experiments folder (default: the nearest 'experiments' above the first path, else ./experiments, else the current folder)",
    )
    backfill_parser.add_argument(
        "--specs",
        type=Path,
        default=check.SPEC_FOLDER,
        help="the folder of *.yaml specs (default: specs/ inside this package)",
    )
    backfill_parser.add_argument(
        "--write",
        action="store_true",
        help="rewrite the files (default: print the changes and write nothing)",
    )
    backfill_parser.add_argument(
        "--list",
        type=Path,
        help="write the person list as markdown to this file (default: none)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = get_args(argv)
    experiments = args.experiments.resolve() if args.experiments else None
    if args.command == "backfill":
        return backfill.main(args.paths, experiments, args.specs, args.write, args.list)
    skip_stages = frozenset(s for s in args.skip_stage.split(",") if s)
    return check.main(args.paths, experiments, args.specs, skip_stages)


if __name__ == "__main__":
    sys.exit(main())
