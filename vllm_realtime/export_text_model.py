#!/usr/bin/env python3
"""Export the text LM of an X2Streaming-ASR checkpoint for the vLLM server.

The vLLM server runs only the language model; audio frames arrive as prompt
embeddings from the encoder sidecar. This writes the (LoRA-merged) text LM as
a standard Mistral checkpoint plus ``export_meta.json``, which ``serve.py``
checks against the sidecar's checkpoint so the two halves always match.

    python vllm_realtime/export_text_model.py --model "$MODEL_PATH" --output "$TEXT_MODEL"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def export_text_model(
    model: Any,
    source_config: Path,
    output_dir: Path,
    *,
    lora_path: str = "",
) -> None:
    """Export the merged adaptive text LM as a standard HF Mistral model."""
    from huggingface_hub import split_torch_state_dict_into_shards

    from models.voxtral_asr.interleaved import _unwrap_voxtral_cg

    vox = _unwrap_voxtral_cg(model)
    text_cfg = json.loads(source_config.read_text(encoding="utf-8"))["text_config"]
    rope = text_cfg.get("rope_parameters") or {}
    config = {
        "architectures": ["AdaptiveTextMistralForCausalLM"],
        "model_type": "mistral",
        "attention_dropout": float(text_cfg.get("attention_dropout", 0.0)),
        "bos_token_id": int(text_cfg.get("bos_token_id", 1)),
        "eos_token_id": int(text_cfg.get("eos_token_id", 2)),
        "head_dim": int(text_cfg.get("head_dim", 128)),
        "hidden_act": str(text_cfg.get("hidden_act", "silu")),
        "hidden_size": int(text_cfg["hidden_size"]),
        "intermediate_size": int(text_cfg["intermediate_size"]),
        "max_position_embeddings": int(text_cfg["max_position_embeddings"]),
        "num_attention_heads": int(text_cfg["num_attention_heads"]),
        "num_hidden_layers": int(text_cfg["num_hidden_layers"]),
        "num_key_value_heads": int(text_cfg["num_key_value_heads"]),
        "rms_norm_eps": float(text_cfg["rms_norm_eps"]),
        "rope_theta": float(rope.get("rope_theta", 1_000_000.0)),
        "sliding_window": int(text_cfg.get("sliding_window", 8192)),
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
        "use_cache": True,
        "vocab_size": int(text_cfg["vocab_size"]),
    }

    # AdaRMS is dropped: adaptive inference fixes its conditioning to zero.
    state: dict[str, torch.Tensor] = {
        f"model.{name}": tensor.detach()
        for name, tensor in vox.model.language_model.state_dict().items()
        if not name.startswith("rotary_emb.") and ".ada_rms_norm." not in name
    }
    # Tied embeddings: vLLM reconstructs lm_head from embed_tokens.
    split = split_torch_state_dict_into_shards(
        state, filename_pattern="model{suffix}.safetensors", max_shard_size="2GB"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    weight_map: dict[str, str] = {}
    for filename, tensor_names in split.filename_to_tensors.items():
        shard = {
            name: state[name].to(device="cpu").contiguous()
            for name in tensor_names
        }
        save_file(shard, output_dir / filename)
        weight_map.update({name: filename for name in tensor_names})
        del shard

    if split.is_sharded:
        _write_json(
            output_dir / "model.safetensors.index.json",
            {
                "metadata": {"total_size": int(split.metadata["total_size"])},
                "weight_map": weight_map,
            },
        )
    _write_json(output_dir / "config.json", config)
    _write_json(
        output_dir / "export_meta.json",
        {
            "source": str(source_config.parent.resolve()),
            "lora": str(Path(lora_path).resolve()) if lora_path else "",
            "format": "hf_mistral_text_only",
            "adaptive_adarms": "removed because t_cond is identically zero",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="X2Streaming-ASR checkpoint")
    parser.add_argument("--output", required=True, help="Output text-LM directory")
    parser.add_argument("--lora", default="", help="Optional LoRA to merge first")
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    from models import build_model

    model_dir = Path(args.model).expanduser().resolve()
    wrapper = build_model("voxtral_asr").load_for_inference(
        str(model_dir),
        device=args.device,
        lora_path=args.lora,
        adaptive_latency=True,
    )
    export_text_model(
        wrapper["model"].eval(),
        model_dir / "config.json",
        Path(args.output).expanduser(),
        lora_path=args.lora,
    )
    print(f"exported text LM to {args.output}", flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
