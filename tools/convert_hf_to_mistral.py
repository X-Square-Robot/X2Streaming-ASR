#!/usr/bin/env python3
"""Convert HF Voxtral-Realtime ``model.safetensors`` → Mistral ``consolidated.safetensors``.

Needed because vLLM's VoxtralRealtime loader expects Mistral key names
(``layers.*``, ``mm_streams_embeddings.*``), while the published checkpoint
ships HF names (``language_model.*``, ``audio_tower.*``).

By default the converted file is written into the same directory:

    python tools/convert_hf_to_mistral.py <downloaded-model-dir>
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

TEXT_HEADS, TEXT_KV, TEXT_HD = 32, 8, 128
AUDIO_HEADS, AUDIO_HD = 32, 64


def rope_pack_weight(w: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    hidden = w.shape[-1]
    w = w.view(n_heads, 2, head_dim // 2, hidden)
    return w.transpose(1, 2).contiguous().reshape(n_heads * head_dim, hidden)


def rope_pack_bias(b: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    b = b.view(n_heads, 2, head_dim // 2)
    return b.transpose(1, 2).contiguous().reshape(n_heads * head_dim)


def map_hf_key(hk: str):
    """Return ``(mistral_key, transform)`` or ``None`` to skip."""
    if hk.startswith("language_model.model.layers."):
        m = re.match(r"language_model\.model\.layers\.(\d+)\.(.*)", hk)
        assert m is not None
        i, rest = m.group(1), m.group(2)
        table = {
            "self_attn.q_proj.weight": (
                f"layers.{i}.attention.wq.weight",
                ("rope", TEXT_HEADS, TEXT_HD),
            ),
            "self_attn.k_proj.weight": (
                f"layers.{i}.attention.wk.weight",
                ("rope", TEXT_KV, TEXT_HD),
            ),
            "self_attn.v_proj.weight": (f"layers.{i}.attention.wv.weight", None),
            "self_attn.o_proj.weight": (f"layers.{i}.attention.wo.weight", None),
            "mlp.gate_proj.weight": (f"layers.{i}.feed_forward.w1.weight", None),
            "mlp.up_proj.weight": (f"layers.{i}.feed_forward.w3.weight", None),
            "mlp.down_proj.weight": (f"layers.{i}.feed_forward.w2.weight", None),
            "input_layernorm.weight": (f"layers.{i}.attention_norm.weight", None),
            "post_attention_layernorm.weight": (f"layers.{i}.ffn_norm.weight", None),
            "ada_rms_norm.linear1.weight": (
                f"layers.{i}.ada_rms_norm_t_cond.0.weight",
                None,
            ),
            "ada_rms_norm.linear2.weight": (
                f"layers.{i}.ada_rms_norm_t_cond.2.weight",
                None,
            ),
        }
        return table.get(rest)

    if hk == "language_model.model.embed_tokens.weight":
        return (
            "mm_streams_embeddings.embedding_module.tok_embeddings.weight",
            None,
        )
    if hk == "language_model.model.norm.weight":
        return ("norm.weight", None)
    if hk == "language_model.lm_head.weight":
        return None  # tied to tok embeddings

    if hk == "multi_modal_projector.linear_1.weight":
        return (
            "mm_streams_embeddings.embedding_module.audio_language_projection.0.weight",
            None,
        )
    if hk == "multi_modal_projector.linear_2.weight":
        return (
            "mm_streams_embeddings.embedding_module.audio_language_projection.2.weight",
            None,
        )

    if hk == "audio_tower.norm.weight":
        return (
            "mm_streams_embeddings.embedding_module.whisper_encoder.transformer.norm.weight",
            None,
        )

    if hk.startswith("audio_tower.embedder.conv1."):
        suf = hk.rsplit(".", 1)[-1]
        return (
            f"mm_streams_embeddings.embedding_module.whisper_encoder.conv_layers.0.conv.{suf}",
            None,
        )
    if hk.startswith("audio_tower.embedder.conv2."):
        suf = hk.rsplit(".", 1)[-1]
        return (
            f"mm_streams_embeddings.embedding_module.whisper_encoder.conv_layers.1.conv.{suf}",
            None,
        )

    if hk.startswith("audio_tower.layers."):
        m = re.match(r"audio_tower\.layers\.(\d+)\.(.*)", hk)
        assert m is not None
        i, rest = m.group(1), m.group(2)
        pref = (
            "mm_streams_embeddings.embedding_module.whisper_encoder."
            f"transformer.layers.{i}"
        )
        table = {
            "self_attn.q_proj.weight": (
                f"{pref}.attention.wq.weight",
                ("rope", AUDIO_HEADS, AUDIO_HD),
            ),
            "self_attn.q_proj.bias": (
                f"{pref}.attention.wq.bias",
                ("rope_bias", AUDIO_HEADS, AUDIO_HD),
            ),
            "self_attn.k_proj.weight": (
                f"{pref}.attention.wk.weight",
                ("rope", AUDIO_HEADS, AUDIO_HD),
            ),
            "self_attn.k_proj.bias": (
                f"{pref}.attention.wk.bias",
                ("rope_bias", AUDIO_HEADS, AUDIO_HD),
            ),
            "self_attn.v_proj.weight": (f"{pref}.attention.wv.weight", None),
            "self_attn.v_proj.bias": (f"{pref}.attention.wv.bias", None),
            "self_attn.o_proj.weight": (f"{pref}.attention.wo.weight", None),
            "self_attn.o_proj.bias": (f"{pref}.attention.wo.bias", None),
            "mlp.gate_proj.weight": (f"{pref}.feed_forward.w1.weight", None),
            "mlp.gate_proj.bias": (f"{pref}.feed_forward.w1.bias", None),
            "mlp.up_proj.weight": (f"{pref}.feed_forward.w3.weight", None),
            "mlp.up_proj.bias": (f"{pref}.feed_forward.w3.bias", None),
            "mlp.down_proj.weight": (f"{pref}.feed_forward.w2.weight", None),
            "mlp.down_proj.bias": (f"{pref}.feed_forward.w2.bias", None),
            "self_attn_layer_norm.weight": (f"{pref}.attention_norm.weight", None),
            "final_layer_norm.weight": (f"{pref}.ffn_norm.weight", None),
            "input_layernorm.weight": (f"{pref}.attention_norm.weight", None),
            "post_attention_layernorm.weight": (f"{pref}.ffn_norm.weight", None),
        }
        return table.get(rest)

    return None


def apply_transform(t: torch.Tensor, xf) -> torch.Tensor:
    if xf is None:
        return t
    kind, n_heads, head_dim = xf
    if kind == "rope":
        return rope_pack_weight(t, n_heads, head_dim)
    if kind == "rope_bias":
        return rope_pack_bias(t, n_heads, head_dim)
    raise ValueError(xf)


def convert_dir(src: Path, dst: Path, *, force: bool = False) -> Path:
    src = src.resolve()
    dst = dst.resolve()
    src_weights = src / "model.safetensors"
    if not src_weights.is_file():
        raise FileNotFoundError(f"missing HF weights: {src_weights}")

    out_weights = dst / "consolidated.safetensors"
    if out_weights.is_file() and not force:
        print(f"[convert] reuse existing {out_weights}")
        return dst

    dst.mkdir(parents=True, exist_ok=True)
    tensors = {}
    skipped = []
    with safe_open(str(src_weights), framework="pt") as f:
        for hk in f.keys():
            mapped = map_hf_key(hk)
            if mapped is None:
                skipped.append(hk)
                continue
            ck, xf = mapped
            tensors[ck] = apply_transform(f.get_tensor(hk), xf).contiguous()

    print(f"[convert] writing {len(tensors)} tensors → {out_weights} "
          f"(skipped {len(skipped)})")
    save_file(tensors, str(out_weights))

    for name in ("params.json", "tekken.json", "generation_config.json",
                 "processor_config.json"):
        s = src / name
        target = dst / name
        if s.is_file() and s.resolve() != target.resolve():
            shutil.copy2(s, target)

    # Prefer not to copy HF config.json (forces Transformers path).
    # Keep a note file.
    (dst / "README_ADAPTIVE_VLLM.txt").write_text(
        f"Converted from HF merge:\n  {src}\n"
        "Use with vLLM --config-format mistral --tokenizer-mode mistral\n"
    )
    return dst


def ensure_vllm_dir(model_path: str | Path, *, force: bool = False) -> Path:
    """If ``model_path`` is HF-only, convert into ``<path>_vllm`` and return it."""
    p = Path(model_path).resolve()
    if (p / "consolidated.safetensors").is_file():
        return p
    if not (p / "model.safetensors").is_file():
        raise FileNotFoundError(
            f"No consolidated.safetensors or model.safetensors under {p}"
        )
    dst = p.parent / f"{p.name}_vllm"
    return convert_dir(p, dst, force=force)


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Write consolidated.safetensors for vLLM from a downloaded "
            "Hugging Face checkpoint directory."
        )
    )
    ap.add_argument(
        "src",
        type=Path,
        help="Downloaded model directory that contains model.safetensors",
    )
    ap.add_argument(
        "--dst",
        type=Path,
        default=None,
        help="Output directory (default: the same directory as src)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing consolidated.safetensors",
    )
    args = ap.parse_args()
    dst = args.dst or args.src
    convert_dir(args.src, dst, force=args.force)
    print(f"[convert] done → {dst}")


if __name__ == "__main__":
    main()
