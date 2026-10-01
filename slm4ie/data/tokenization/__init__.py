"""Convert lexicon downloads into the tokenizer-quality datasets under `tokenization/`.

* `config.py` — `configs/data/tokenization.yaml` loaded into `TokenizationConfig`.
* `driver.py` — `convert_tokenization_datasets`, one gzipped JSONL per dataset.
* `readers/` — the backend registry, one reader per lexicon (`sloleks`, `sloleks_relations`).
"""
