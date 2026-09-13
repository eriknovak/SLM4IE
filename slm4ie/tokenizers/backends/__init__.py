"""Tokenizer backend implementations.

Importing this package registers every backend with the registry through the
`@register_tokenizer` decorator. Import it for its side effects before calling
`slm4ie.tokenizers.registry.get_tokenizer`.
"""

from slm4ie.tokenizers.backends import (  # noqa: F401
    char_bpe,
    hf_bpe,
    hf_wordpiece,
    morph_bpe,
    morph_piece,
    sp_unigram,
)
