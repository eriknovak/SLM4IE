---
title: Synthetic data
---

# Synthetic IE data

`scripts/data/generate_synthetic.py` generates synthetic IE training
data via LLM APIs, used to bootstrap low-resource tasks where real
annotated data is scarce.

```bash
uv run python scripts/data/generate_synthetic.py
```

Configuration lives in
[`configs/data/synthetic.yaml`](https://github.com/eriknovak/SLM4IE/blob/main/configs/data).

!!! warning "API credentials"
    Synthetic generation calls external LLM APIs. Configure credentials
    via the standard environment variables expected by the upstream SDK.
    Never check API keys into the repository.
