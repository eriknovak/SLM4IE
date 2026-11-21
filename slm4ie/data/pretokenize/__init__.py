"""Tokenize the curated pretraining corpus into pretokenized outputs for other projects.

Model training runs outside this repository, and each training project reads
the corpus in its own layout. This package tokenizes a corpus stage once with
an exported tokenizer and writes every configured layout from the same samples,
split and chunking.

* `config`: the `configs/data/pretokenize.yaml` loader and its entries.
* `chunking`: sentence splitting and chunk packing of a tokenized document.
* `splits`: the seeded, per-document train and validation split.
* `formats`: the output-format registry (`mds`, `mds-streaming`, `parquet`).
* `run`: the argv-free `run_pretokenize` entry point, shard by shard with resume.
* `check`: `check_pretokenized`, which reads a finished output back against its manifest.
"""

from slm4ie.data.pretokenize.check import CheckReport, check_pretokenized
from slm4ie.data.pretokenize.config import PretokenizeConfig, PretokenizeEntry, load_pretokenize_config
from slm4ie.data.pretokenize.run import PretokenizeSummary, run_pretokenize

__all__ = [
    "CheckReport",
    "PretokenizeConfig",
    "PretokenizeEntry",
    "PretokenizeSummary",
    "check_pretokenized",
    "load_pretokenize_config",
    "run_pretokenize",
]
