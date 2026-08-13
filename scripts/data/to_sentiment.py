"""Convert SA datasets into task-shaped per-split JSONL.

Thin CLI wrapper around `slm4ie.data.task_converter`. All CLI, orchestration,
and the per-family record transform live in the library; this script only names
the converter to run. Driven by `configs/data/tasks.yaml`.

Examples:
    Convert every sentiment entry declared in tasks.yaml:

        uv run python scripts/data/to_sentiment.py --all

    Convert just one entry:

        uv run python scripts/data/to_sentiment.py sentiment/sentinews
"""

from slm4ie.data.task_converter import run_converter

if __name__ == "__main__":
    run_converter("to_sentiment")
