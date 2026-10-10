"""Gated language-id embedding (zh/en/codeswitch). Independent of AdaRMS τ.

Default-off: callers must attach this module and pass ``language_ids`` only
when ``language_tag.enabled`` is true. Omitted / disabled → no extra tensor
and no embed add.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Mapping, Optional, Union

import torch
import torch.nn as nn

from models.voxtral_asr.data import (
    LANG_ID_CS,
    LANG_ID_EN,
    LANG_ID_ZH,
    infer_lang_id_from_path,
)

NUM_LANGS = 3
LANG_NAME_TO_ID = {
    "zh": LANG_ID_ZH,
    "en": LANG_ID_EN,
    "cs": LANG_ID_CS,
    "zh-cn": LANG_ID_ZH,
    "chinese": LANG_ID_ZH,
    "english": LANG_ID_EN,
    "codeswitch": LANG_ID_CS,
    "code-switch": LANG_ID_CS,
    "code_switch": LANG_ID_CS,
}


class LanguageTagEmbed(nn.Module):
    """``nn.Embedding(3, hidden_size)`` for zh=0, en=1, cs=2.

    Residual-style: zero-init so the add is a no-op at step 0. Default
    ``nn.Embedding`` is N(0, 1) and would OOD-shift every token, including
    frozen text embeds the encoder cannot correct.
    """

    def __init__(self, hidden_size: int, num_langs: int = NUM_LANGS):
        super().__init__()
        self.num_langs = int(num_langs)
        self.hidden_size = int(hidden_size)
        self.embed = nn.Embedding(self.num_langs, self.hidden_size)
        nn.init.zeros_(self.embed.weight)

    def forward(self, language_ids: torch.Tensor) -> torch.Tensor:
        """``language_ids`` ``[B]`` → ``[B, H]``."""
        return self.embed(language_ids)


def language_tag_cfg(cfg: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Read ``language_tag`` from top-level or ``model.language_tag``."""
    if not cfg:
        return {}
    raw = cfg.get("language_tag")
    if raw is None:
        raw = (cfg.get("model") or {}).get("language_tag")
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return dict(raw)
    return {"enabled": bool(raw)}


def language_tag_enabled(cfg: Optional[Mapping[str, Any]]) -> bool:
    return bool(language_tag_cfg(cfg).get("enabled", False))


def resolve_language_id(
    value: Any,
    cfg: Optional[Mapping[str, Any]] = None,
) -> int:
    """Map yaml ``zh`` / ``en`` / ``cs`` / int to an embedding row."""
    if value is None or value == "":
        return LANG_ID_ZH
    if isinstance(value, bool):
        raise ValueError(f"language id must not be bool, got {value!r}")
    if isinstance(value, int):
        return int(value)
    key = str(value).strip().lower()
    ids = (cfg or {}).get("ids") if cfg else None
    if isinstance(ids, Mapping) and key in ids:
        return int(ids[key])
    if key in LANG_NAME_TO_ID:
        return LANG_NAME_TO_ID[key]
    return infer_lang_id_from_path(key)


def language_id_name(language_id: Any) -> str:
    """Map embedding row / alias back to ``zh`` / ``en`` / ``cs``."""
    lid = resolve_language_id(language_id)
    if lid == LANG_ID_EN:
        return "en"
    if lid == LANG_ID_CS:
        return "cs"
    return "zh"


def set_infer_language_id(model: Any, language_id: Any) -> int:
    """Stamp ``_infer_language_id`` so live / offline both add the tag."""
    lid = int(resolve_language_id(language_id))
    cg = _unwrap_cg(model)
    for host in (cg, model):
        if host is None:
            continue
        try:
            host._infer_language_id = lid
            host._infer_language_ids = None
        except Exception:
            pass
    return lid


def detect_hidden_size(model: Any) -> int:
    """Text hidden size for the language-tag embedding."""
    cfg = getattr(model, "config", None)
    if cfg is not None:
        text_cfg = getattr(cfg, "text_config", None)
        for owner in (text_cfg, cfg):
            if owner is None:
                continue
            for attr in ("hidden_size", "dim", "d_model"):
                val = getattr(owner, attr, None)
                if val is not None:
                    return int(val)
    emb = None
    if hasattr(model, "get_input_embeddings"):
        try:
            emb = model.get_input_embeddings()
        except Exception:
            emb = None
    if emb is None:
        inner = getattr(model, "model", None)
        if inner is not None and hasattr(inner, "get_input_embeddings"):
            try:
                emb = inner.get_input_embeddings()
            except Exception:
                emb = None
    if emb is not None and hasattr(emb, "weight") and emb.weight is not None:
        return int(emb.weight.shape[-1])
    raise ValueError(
        "cannot detect hidden_size for language_tag "
        "(no config.text_config.hidden_size / embed weight)"
    )


def _unwrap_cg(model: Any) -> Any:
    m = model
    if hasattr(m, "module"):
        m = m.module
    if hasattr(m, "get_base_model"):
        try:
            m = m.get_base_model()
        except Exception:
            pass
    if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
        inner = m.base_model.model
        if hasattr(inner, "lm_head") and hasattr(inner, "model"):
            m = inner
    return m


def _extract_tag_state(state: Mapping[str, Any], module: nn.Module) -> Dict[str, torch.Tensor]:
    want = set(module.state_dict().keys())
    extracted: Dict[str, torch.Tensor] = {}
    for key, val in state.items():
        if "language_tag_embed." not in key:
            continue
        suffix = key.split("language_tag_embed.", 1)[1]
        if suffix in want:
            extracted[suffix] = val
        elif suffix == "weight" and "embed.weight" in want:
            extracted["embed.weight"] = val
    return extracted


def _iter_checkpoint_states(model_path: str):
    if not model_path or not os.path.isdir(model_path):
        return
    st = os.path.join(model_path, "model.safetensors")
    if os.path.isfile(st):
        from safetensors.torch import load_file

        yield load_file(st, device="cpu")
        return
    idx = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.isfile(idx):
        with open(idx, "r", encoding="utf-8") as f:
            weight_map = json.load(f).get("weight_map", {})
        shards = sorted(
            {
                name
                for key, name in weight_map.items()
                if "language_tag_embed" in key
            }
        )
        if not shards:
            return
        from safetensors.torch import load_file

        for shard in shards:
            path = os.path.join(model_path, shard)
            if os.path.isfile(path):
                yield load_file(path, device="cpu")
        return
    bin_path = os.path.join(model_path, "pytorch_model.bin")
    if os.path.isfile(bin_path):
        try:
            yield torch.load(bin_path, map_location="cpu", weights_only=True)
        except TypeError:
            yield torch.load(bin_path, map_location="cpu")


def load_language_tag_weights(module: nn.Module, model_path: str) -> bool:
    """Copy ``language_tag_embed.*`` from a HF checkpoint if present."""
    merged: Dict[str, torch.Tensor] = {}
    for state in _iter_checkpoint_states(model_path) or ():
        merged.update(_extract_tag_state(state, module))
    if not merged:
        return False
    missing, unexpected = module.load_state_dict(merged, strict=False)
    return not missing


def _local_adapter_weight_keys(adapter_path: str) -> Optional[set[str]]:
    """Return local adapter keys without materializing safetensor values."""
    if not os.path.isdir(adapter_path):
        return None
    safetensors_path = os.path.join(adapter_path, "adapter_model.safetensors")
    if os.path.isfile(safetensors_path):
        from safetensors import safe_open

        with safe_open(safetensors_path, framework="pt", device="cpu") as state:
            return set(state.keys())
    index_path = os.path.join(
        adapter_path, "adapter_model.safetensors.index.json"
    )
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return set(json.load(f).get("weight_map", {}))
    bin_path = os.path.join(adapter_path, "adapter_model.bin")
    if os.path.isfile(bin_path):
        try:
            state = torch.load(bin_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(bin_path, map_location="cpu")
        return set(state)
    return set()


def adapter_saves_language_tag(adapter_path: str) -> bool:
    """Inspect PEFT metadata and validate a declared language-tag payload."""
    from peft import PeftConfig

    config = PeftConfig.from_pretrained(adapter_path)
    modules_to_save = set(getattr(config, "modules_to_save", None) or ())
    declared = "language_tag_embed" in modules_to_save
    if not declared:
        return False

    keys = _local_adapter_weight_keys(adapter_path)
    if keys is not None and not any(
        "language_tag_embed." in key for key in keys
    ):
        raise RuntimeError(
            "PEFT adapter declares language_tag_embed in modules_to_save, "
            f"but no language_tag_embed weight was found in {adapter_path!r}"
        )
    return True


def mark_language_tag_adapter_source(model: Any, adapter_path: str) -> None:
    """Record/log that PEFT's merged modules_to_save copy is authoritative."""
    tag = getattr(_unwrap_cg(model), "language_tag_embed", None)
    if tag is None:
        raise RuntimeError(
            "PEFT adapter saved language_tag_embed, but the merged model "
            "does not contain that module"
        )
    tag._language_tag_source = "adapter modules_to_save"  # type: ignore[attr-defined]
    tag._loaded_from_adapter = True  # type: ignore[attr-defined]
    print(
        "[language_tag] source=adapter modules_to_save "
        f"adapter={adapter_path}"
    )


def attach_language_tag_embed(
    model: Any,
    *,
    model_path: str = "",
    hidden_size: Optional[int] = None,
    num_langs: int = NUM_LANGS,
    require_checkpoint_weights: bool = False,
) -> LanguageTagEmbed:
    """Register ``model.language_tag_embed``; restore ckpt weights when present.

    Base Realtime checkpoints have no tag weights: zero init, do not crash.
    Stage1/2 ``final`` saves restore the learned embedding.
    """
    cg = _unwrap_cg(model)
    existing = getattr(cg, "language_tag_embed", None)
    if isinstance(existing, LanguageTagEmbed):
        tag = existing
        if getattr(tag, "_language_tag_initialized", False):
            loaded = bool(
                getattr(tag, "_loaded_from_checkpoint", False)
                or getattr(tag, "_loaded_from_adapter", False)
            )
            if require_checkpoint_weights and not loaded:
                raise RuntimeError(
                    "language_tag.require_checkpoint_weights=true, but the "
                    "existing language_tag_embed has no checkpoint weights"
                )
            print(
                "[language_tag] source="
                f"{getattr(tag, '_language_tag_source', 'existing module')} "
                "(already initialized)"
            )
            return tag
    else:
        if hidden_size is None:
            hidden_size = detect_hidden_size(cg)
        tag = LanguageTagEmbed(hidden_size, num_langs=num_langs)
        dtype = None
        device = None
        try:
            param = next(cg.parameters())
            dtype = param.dtype
            device = param.device
        except StopIteration:
            pass
        if dtype is not None or device is not None:
            tag = tag.to(device=device, dtype=dtype)
        cg.language_tag_embed = tag
        if model is not cg and getattr(model, "language_tag_embed", None) is None:
            # Keep a handle on the wrapper so save/load + freeze helpers see it.
            try:
                model.language_tag_embed = tag
            except Exception:
                pass
    loaded = False
    if model_path:
        loaded = load_language_tag_weights(tag, model_path)
    source = "base checkpoint" if loaded else "zero initialization"
    tag._language_tag_initialized = True  # type: ignore[attr-defined]
    tag._loaded_from_checkpoint = loaded  # type: ignore[attr-defined]
    tag._language_tag_source = source  # type: ignore[attr-defined]
    if require_checkpoint_weights and not loaded:
        raise RuntimeError(
            "language_tag.require_checkpoint_weights=true, but checkpoint "
            f"{model_path!r} has no language_tag_embed weights"
        )
    print(f"[language_tag] source={source} checkpoint={model_path or '<none>'}")
    return tag


def _coerce_language_ids(
    language_ids: Any,
    batch_size: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if language_ids is None:
        return None
    if not torch.is_tensor(language_ids):
        if isinstance(language_ids, (list, tuple)):
            language_ids = torch.tensor(
                language_ids, device=device, dtype=torch.long
            )
        else:
            return torch.full(
                (batch_size,), int(language_ids), device=device, dtype=torch.long
            )
    else:
        language_ids = language_ids.to(device=device, dtype=torch.long)
    if language_ids.dim() == 0:
        return language_ids.view(1).expand(batch_size)
    if language_ids.shape[0] == 1 and batch_size > 1:
        return language_ids.expand(batch_size)
    if language_ids.shape[0] != batch_size:
        if language_ids.numel() == 1:
            return language_ids.reshape(1).expand(batch_size)
        return None
    return language_ids


def apply_language_tag(
    inputs_embeds: torch.Tensor,
    model: Any,
    language_ids: Optional[Union[torch.Tensor, int]] = None,
) -> torch.Tensor:
    """Broadcast-add ``lang_emb[lang_id]`` over the sequence. No-op if disabled."""
    cg = _unwrap_cg(model)
    tag = getattr(cg, "language_tag_embed", None) or getattr(
        model, "language_tag_embed", None
    )
    if tag is None:
        return inputs_embeds
    ids = language_ids
    if ids is None:
        ids = getattr(cg, "_infer_language_ids", None)
        if ids is None:
            ids = getattr(model, "_infer_language_ids", None)
    if ids is None:
        scalar = getattr(cg, "_infer_language_id", None)
        if scalar is None:
            scalar = getattr(model, "_infer_language_id", None)
        ids = scalar
    ids = _coerce_language_ids(ids, inputs_embeds.shape[0], inputs_embeds.device)
    if ids is None:
        return inputs_embeds
    lang_emb = tag(ids)
    if lang_emb.dim() == 2:
        lang_emb = lang_emb.unsqueeze(1)
    return inputs_embeds + lang_emb.to(dtype=inputs_embeds.dtype)
