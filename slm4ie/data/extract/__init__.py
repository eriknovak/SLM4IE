"""Extract raw downloads into the unified JSONL form under `extracted/`.

* `config.py` — `configs/data/extract.yaml` loaded into `ExtractionConfig`.
* `driver.py` — `extract_datasets`, the serial and sharded extraction loops.
* `extractors/` — the backend registry, one module per input format.
* `sidecar.py` — external per-document metadata tables merged by extractors.
* `records.py` — readers that join the text file with its annotations sidecar.
* `tracking.py` — post-hoc MLflow logging of the extracted tree.
"""
