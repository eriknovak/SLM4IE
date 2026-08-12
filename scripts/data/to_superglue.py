"""Convert SuperGLUE-SL subtasks into task-shaped per-split JSONL.

Thin CLI wrapper around `slm4ie.data.task_converter`. All CLI, orchestration,
and the per-family record transforms live in the library; this script only names
the converter to run. Driven by `configs/data/tasks.yaml`. The `--variant` flag
(default `humant`) selects the translated SuperGLUE-SL bundle.

Examples:
    Convert every SuperGLUE entry declared in tasks.yaml:

        uv run python scripts/data/to_superglue.py --all

    Convert one entry from the Google-MT variant:

        uv run python scripts/data/to_superglue.py nli/cb --variant googlemt
"""

from slm4ie.data.task_converter import run_converter

if __name__ == "__main__":
    run_converter("to_superglue")
