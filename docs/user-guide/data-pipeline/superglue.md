---
title: SuperGLUE format
---

# SuperGLUE format

!!! warning "🚧 In progress"
    This workflow is implemented but has **not yet been run end-to-end**, so
    this page is a stub. Detailed usage and results will land here once the
    pipeline is verified. Until then, treat the entry point and config below as
    the source of truth.

`to_superglue.py` converts the SuperGLUE-SL subtasks into per-task evaluation
files, dissolved into the `nli/`, `qa/`, `coref/`, `wsd/`, and `commonsense/`
families under `tasks/`. A `--variant` flag selects the `humant` (default) or
`googlemt` translation.

**Entry point:** [`scripts/data/to_superglue.py`](https://github.com/eriknovak/SLM4IE/blob/main/scripts/data/to_superglue.py)
**Config:** [`configs/data/tasks.yaml`](https://github.com/eriknovak/SLM4IE/blob/main/configs/data/tasks.yaml)
