---
title: HuggingFace Authentication
---

# HuggingFace Authentication

Some datasets used by SLM4IE (e.g., `FineWeb-2` and other gated corpora)
require a HuggingFace access token.

## One-time login

Authenticate via the unified `hf` CLI shipped with `huggingface_hub`
≥ 0.34, which replaces the deprecated `huggingface-cli`:

```bash
hf auth login
```

Paste a token from
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
when prompted. The token is stored under `~/.cache/huggingface/` and
picked up automatically by `huggingface_hub` and `datasets` — no
`HF_TOKEN` environment variable or `.env` file is needed.

## Non-interactive login

For CI and SLURM jobs where stdin is not available, pass the token
directly:

```bash
hf auth login --token "$HF_TOKEN"
```

## Verify

```bash
hf auth whoami
```

If the username is printed, authentication succeeded and gated downloads
will work.

!!! warning "Do not commit tokens"
    Never check `.env` files or shell rc files containing your HF token
    into the repository. SLM4IE's `.gitignore` already excludes common
    secret files; respect it.
