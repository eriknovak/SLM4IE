"""Convert lexicon downloads into the tokenizer-quality datasets under `tokenization/`.

* `config.py` — `configs/data/tokenization.yaml` loaded into `TokenizationConfig`.
* `run.py` — `convert_tokenization_datasets`, one gzipped JSONL per dataset.
* `lexicons/` — the backend registry, one module per lexicon (`sloleks`, `sloleks_relations`).
"""
