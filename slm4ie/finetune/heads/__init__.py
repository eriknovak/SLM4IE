"""Task heads for per-entry fine-tuning.

Importing this package registers every head with the registry in `base`: token
classification for `ner` and `pos`, multiple choice for `commonsense`, and
sequence or sequence-pair classification for the other families. Look one up
with `get_head(family)`.
"""

from slm4ie.finetune.heads import choice, sequence, token  # noqa: F401
from slm4ie.finetune.heads.base import TaskHead, available_heads, get_head, register_head

__all__ = ["TaskHead", "available_heads", "get_head", "register_head"]
