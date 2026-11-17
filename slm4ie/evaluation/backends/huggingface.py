"""Chat backend for models the vllm-service cannot serve, run in-process with transformers."""

import json
import logging
from typing import Any

from slm4ie.evaluation.backends.base import DEFAULT_SEED, MAX_NEW_TOKENS, ChatBackend, ChatResult, register_backend

# ===============================
# Defaults
# ===============================

# the whole model on GPU 0: a model that does not fit fails with out-of-memory instead of offloading to the CPU (D7)
SINGLE_GPU_MAP = {"": 0}

QUANTIZATIONS = ("4bit", "8bit")
# bitsandbytes logs a dtype-cast warning on every 8-bit matmul
BITSANDBYTES_CAST_LOGGER = "bitsandbytes.autograd._functions"

# ===============================
# Backend class
# ===============================


@register_backend("transformers")
class TransformersBackend(ChatBackend):
    """Chat with a HuggingFace causal LM, decoded greedily under the answer schema.

    The schema is compiled to a grammar with xgrammar and enforced token by token through
    its logits processor, as vllm does. `torch`, `transformers` and `xgrammar` are imported
    when the model loads, so importing this module needs none of them.

    Peak GPU memory is `torch.cuda.max_memory_allocated`, counted from the end of loading
    (weights included) or from the last `reset_peak_gpu_memory` call.

    Args:
        model: The HuggingFace model id or a local path.
        quantization: `"4bit"` or `"8bit"` (bitsandbytes) for a model that does not fit 16 GB, else `None`.
        enable_thinking: Passed to the chat template; `False` turns a reasoning model's thinking off.
        seed: Seeds torch before every generation.
        fold_system: Fold the system turn into the first user turn, for a template that rejects a system role.

    Raises:
        ValueError: If `quantization` is not one of `QUANTIZATIONS` or `None`.
    """

    def __init__(
        self,
        model: str,
        quantization: str | None = None,
        enable_thinking: bool = False,
        seed: int = DEFAULT_SEED,
        fold_system: bool = False,
    ):
        """Load the model, its tokenizer and the grammar compiler onto the GPU."""
        if quantization is not None and quantization not in QUANTIZATIONS:
            raise ValueError(f"quantization {quantization!r} must be one of {QUANTIZATIONS} or None")

        import torch
        import xgrammar
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from xgrammar.contrib.hf import LogitsProcessor

        super().__init__(model, seed, fold_system)
        self.enable_thinking = enable_thinking

        # load the tokenizer and the model, quantised when asked
        quantization_config = None
        if quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=quantization == "4bit", load_in_8bit=quantization == "8bit"
            )
            logging.getLogger(BITSANDBYTES_CAST_LOGGER).setLevel(logging.ERROR)
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.lm = AutoModelForCausalLM.from_pretrained(
            model, device_map=SINGLE_GPU_MAP, dtype="auto", quantization_config=quantization_config
        )

        # stop on every end token the generation config or the tokenizer names: they may disagree
        self.stop_token_ids = _stop_token_ids(self.lm, self.tokenizer)

        # build the grammar compiler for the model's vocabulary, ending on the same tokens
        self._logits_processor = LogitsProcessor
        tokenizer_info = xgrammar.TokenizerInfo.from_huggingface(
            self.tokenizer,
            vocab_size=self.lm.config.get_text_config().vocab_size,
            stop_token_ids=self.stop_token_ids,
        )
        self._compiler = xgrammar.GrammarCompiler(tokenizer_info)

        self.reset_peak_gpu_memory()

    def peak_gpu_memory_bytes(self) -> int | None:
        """Return the peak GPU memory this process allocated since the last reset, or `None` without CUDA."""
        if not self._torch.cuda.is_available():
            return None
        return int(self._torch.cuda.max_memory_allocated())

    def reset_peak_gpu_memory(self) -> None:
        """Start the peak at the memory in use now (the weights); a no-op without CUDA."""
        if self._torch.cuda.is_available():
            self._torch.cuda.reset_peak_memory_stats()

    def _complete(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> ChatResult:
        """Generate greedily under the schema grammar and decode only the new tokens."""
        # constrain the answer to the schema; the processor holds per-generation state
        grammar = self._compiler.compile_json_schema(json.dumps(schema))
        processor = self._logits_processor(grammar)

        # tokenize the chat
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=self.enable_thinking,
        ).to(self.lm.device)

        # generate and decode the new tokens
        self._torch.manual_seed(self.seed)
        with self._torch.no_grad():
            output_ids = self.lm.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=MAX_NEW_TOKENS,
                logits_processor=[processor],
                eos_token_id=self.stop_token_ids,
            )
        input_tokens = inputs["input_ids"].shape[1]
        new_ids = output_ids[0][input_tokens:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return ChatResult(text, input_tokens, len(new_ids), 0.0)


# ===============================
# Helper functions
# ===============================


def _stop_token_ids(lm: Any, tokenizer: Any) -> list[int]:
    """Return the sorted end token ids of the generation config and the tokenizer.

    A checkpoint without `generation_config.json` falls back to `config.json`, whose
    end token may differ from the chat template's (Qwen3.5); the grammar then ends
    on one token while `generate` waits for the other until `MAX_NEW_TOKENS`.
    """
    configured = getattr(getattr(lm, "generation_config", None), "eos_token_id", None)
    if configured is None:
        configured = []
    elif isinstance(configured, int):
        configured = [configured]
    ids = set(configured)
    if isinstance(tokenizer.eos_token_id, int):
        ids.add(tokenizer.eos_token_id)
    return sorted(ids)
