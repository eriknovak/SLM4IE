"""Download raw datasets declared in `configs/data/download.yaml`.

* `config.py` — the dataset catalog loaded into `DatasetConfig` entries.
* `driver.py` — `download_datasets`, the per-dataset dispatch and its summary.
* `sources/` — one downloader backend per source kind (`http`, `huggingface`).
"""
