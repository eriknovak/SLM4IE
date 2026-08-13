"""Convert extracted NER datasets into GLiNER-style task JSONL.

Thin CLI wrapper around `slm4ie.data.task_converter`. All CLI, orchestration,
and the per-family record transform live in the library; this script only names
the converter to run. Driven by `configs/data/tasks.yaml`.

Examples:
    Convert every NER entry declared in tasks.yaml:

        uv run python scripts/data/to_spans.py --all

    Convert just one entry:

        uv run python scripts/data/to_spans.py ner/ssj500k
"""

from slm4ie.data.task_converter import run_converter

if __name__ == "__main__":
    run_converter("to_spans")
