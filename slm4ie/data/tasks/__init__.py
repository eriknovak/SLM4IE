"""Convert extracted datasets into task-shaped, per-split JSONL.

Everything downstream of extraction that produces benchmark data lives here:

* `registry`: the `configs/data/tasks.yaml` loader and the `<task>/<dataset>`
  entries it declares, with their roles, sources, splits and labels.
* `driver`: the argv-free `convert_tasks` entry point, the `TaskConverter` ABC
  and its `@register_converter` registry, the role gate and the split policies.
* `converters`: one backend per task family (`spans`, `sentiment`, `superglue`),
  registered on import.
* `writer`: the per-split output convention shared by every family.
* `tracking`: post-hoc MLflow logging of what each conversion produced.

Every entry names the converter that reads it, so one `convert_tasks` call can
span every family and `tasks.yaml` stays the only place that mapping lives.
"""

from slm4ie.data.tasks.driver import (
    ConvertContext,
    SplitPolicy,
    TaskConversionSummary,
    TaskConverter,
    assign_hash_split,
    convert_entry,
    convert_tasks,
    get_converter,
    iter_extracted_records,
    label_allow_set,
    register_converter,
    resolve_keys,
    synthesize_id,
    target_splits,
)
from slm4ie.data.tasks.registry import (
    TaskEntry,
    TaskSource,
    TasksConfig,
    TasksRoots,
    load_tasks,
    resolve_output_dir,
    resolve_source_paths,
)

__all__ = [
    "ConvertContext",
    "SplitPolicy",
    "TaskConversionSummary",
    "TaskConverter",
    "TaskEntry",
    "TaskSource",
    "TasksConfig",
    "TasksRoots",
    "assign_hash_split",
    "convert_entry",
    "convert_tasks",
    "get_converter",
    "iter_extracted_records",
    "label_allow_set",
    "load_tasks",
    "register_converter",
    "resolve_keys",
    "resolve_output_dir",
    "resolve_source_paths",
    "synthesize_id",
    "target_splits",
]
