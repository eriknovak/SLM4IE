"""Tests for slm4ie.evaluation.backends.huggingface."""

import sys
import types
from unittest.mock import MagicMock

import pytest

from slm4ie.evaluation.backends.huggingface import TransformersBackend

# ===============================
# Test Data
# ===============================

SCHEMA = {"type": "object"}
MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]


@pytest.fixture
def fakes(monkeypatch):
    """Replace torch, transformers and xgrammar with mocks, so no GPU or download is needed."""
    torch = MagicMock()
    torch.cuda.is_available.return_value = True
    torch.cuda.max_memory_allocated.return_value = 4096.0

    transformers = MagicMock()
    tokenizer = transformers.AutoTokenizer.from_pretrained.return_value
    tokenizer.apply_chat_template.return_value.to.return_value = {"input_ids": MagicMock(shape=(1, 5))}
    tokenizer.decode.return_value = '{"mentions": []}'
    lm = transformers.AutoModelForCausalLM.from_pretrained.return_value
    lm.generate.return_value = [list(range(9))]

    xgrammar = MagicMock()
    xgrammar_hf = types.SimpleNamespace(LogitsProcessor=MagicMock())

    modules = {
        "torch": torch,
        "transformers": transformers,
        "xgrammar": xgrammar,
        "xgrammar.contrib": MagicMock(),
        "xgrammar.contrib.hf": xgrammar_hf,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return types.SimpleNamespace(torch=torch, transformers=transformers, lm=lm, xgrammar=xgrammar, hf=xgrammar_hf)


# ===============================
# TransformersBackend
# ===============================


def test_rejects_an_unknown_quantization_before_loading_anything():
    """Rejects an unknown quantization before loading anything."""
    with pytest.raises(ValueError, match="2bit"):
        TransformersBackend("some/model", quantization="2bit")


def test_chat_constrains_decoding_with_the_compiled_schema(fakes):
    """Chat constrains decoding with the compiled schema."""
    backend = TransformersBackend("some/model")

    result = backend.chat(MESSAGES, SCHEMA)

    compiler = fakes.xgrammar.GrammarCompiler.return_value
    compiler.compile_json_schema.assert_called_once_with('{"type": "object"}')
    fakes.hf.LogitsProcessor.assert_called_once_with(compiler.compile_json_schema.return_value)
    kwargs = fakes.lm.generate.call_args.kwargs
    assert kwargs["logits_processor"] == [fakes.hf.LogitsProcessor.return_value]
    assert kwargs["do_sample"] is False
    assert (result.text, result.input_tokens, result.output_tokens) == ('{"mentions": []}', 5, 4)


def test_grammar_takes_the_vocabulary_size_of_the_text_config(fakes):
    """A multimodal config such as Gemma 3 keeps `vocab_size` on its text config only."""
    fakes.lm.config = types.SimpleNamespace(get_text_config=lambda: types.SimpleNamespace(vocab_size=262208))
    TransformersBackend("some/model").chat(MESSAGES, SCHEMA)

    assert fakes.xgrammar.TokenizerInfo.from_huggingface.call_args.kwargs["vocab_size"] == 262208


def test_chat_seeds_torch_before_generating(fakes):
    """Chat seeds torch before generating."""
    TransformersBackend("some/model", seed=11).chat(MESSAGES, SCHEMA)

    fakes.torch.manual_seed.assert_called_once_with(11)


def test_model_loads_on_a_single_gpu(fakes):
    """Model loads on a single gpu."""
    TransformersBackend("some/model")

    kwargs = fakes.transformers.AutoModelForCausalLM.from_pretrained.call_args.kwargs
    assert kwargs["device_map"] == {"": 0}


def test_peak_memory_starts_after_loading_and_resets_on_demand(fakes):
    """Peak memory starts after loading and resets on demand."""
    backend = TransformersBackend("some/model")
    assert fakes.torch.cuda.reset_peak_memory_stats.call_count == 1

    backend.reset_peak_gpu_memory()

    assert fakes.torch.cuda.reset_peak_memory_stats.call_count == 2
    assert backend.peak_gpu_memory_bytes() == 4096


def test_peak_memory_is_none_without_cuda(fakes):
    """Peak memory is none without cuda."""
    fakes.torch.cuda.is_available.return_value = False

    assert TransformersBackend("some/model").peak_gpu_memory_bytes() is None


def test_generation_stops_on_the_tokenizer_and_config_end_tokens(fakes):
    """A model whose configs disagree on the end token stops on either, and the grammar allows both."""
    fakes.lm.generation_config = types.SimpleNamespace(eos_token_id=248044)
    fakes.transformers.AutoTokenizer.from_pretrained.return_value.eos_token_id = 248046
    TransformersBackend("some/model").chat(MESSAGES, SCHEMA)

    assert fakes.lm.generate.call_args.kwargs["eos_token_id"] == [248044, 248046]
    assert fakes.xgrammar.TokenizerInfo.from_huggingface.call_args.kwargs["stop_token_ids"] == [248044, 248046]


def test_end_tokens_accept_a_list_or_no_config_value(fakes):
    """A config list of end tokens is merged with the tokenizer's; a missing one is skipped."""
    fakes.lm.generation_config = types.SimpleNamespace(eos_token_id=[1, 106])
    fakes.transformers.AutoTokenizer.from_pretrained.return_value.eos_token_id = 106
    TransformersBackend("some/model").chat(MESSAGES, SCHEMA)
    assert fakes.lm.generate.call_args.kwargs["eos_token_id"] == [1, 106]

    fakes.lm.generation_config = types.SimpleNamespace(eos_token_id=None)
    TransformersBackend("some/model").chat(MESSAGES, SCHEMA)
    assert fakes.lm.generate.call_args.kwargs["eos_token_id"] == [106]


def test_quantized_loading_quiets_the_bitsandbytes_cast_warning(fakes):
    """An 8-bit model silences bitsandbytes' warning, which it logs on every matmul."""
    import logging

    TransformersBackend("some/model", quantization="8bit")

    assert logging.getLogger("bitsandbytes.autograd._functions").level == logging.ERROR
