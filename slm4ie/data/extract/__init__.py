"""Extract raw downloads into the unified JSONL form under `extracted/`.

* `config.py` — `configs/data/extract.yaml` loaded into `ExtractConfig`.
* `run.py` — `extract_datasets`, the serial and sharded extraction loops.
* `extractors/` — the backend registry, one module per input format.
* `metadata_table.py` — the per-document metadata table a few sources ship, merged by extractors.
* `records.py` — readers that join the text file with its annotations sidecar.
* `tracking.py` — post-hoc MLflow logging of the extracted tree.
"""
