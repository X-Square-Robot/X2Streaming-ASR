from __future__ import annotations

from vllm.model_executor.models.mistral import MistralForCausalLM


class AdaptiveTextMistralForCausalLM(MistralForCausalLM):
    """LM-only adaptive X2Streaming-ASR runtime (weights from export_text_model.py).

    LISTEN audio vectors arrive through EmbedsPrompt; DECODE uses the tied token
    embedding normally. AdaRMS is intentionally absent because adaptive
    training/inference fixes its conditioning input to exact zero.
    """

