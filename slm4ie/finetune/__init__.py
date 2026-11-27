"""Per-entry fine-tuning of pretrained encoders on the task registry.

`config.py` loads the shared fine-tuning config and an experiment's dial, and `heads/` holds
one task head per family. `run.py`'s `finetune` trains, selects and scores one
encoder on one entry; `tracking.py`'s `track_finetune` wraps each such run in
one MLflow run.
"""
