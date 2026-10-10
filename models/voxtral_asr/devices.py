"""Split the audio encoder and LM across two GPUs (one model copy).

``DEVICE`` / ``lm_device`` stays the default (LM, embeddings, KV).
``ENC_DEVICE`` / ``VOXTRAL_ENC_DEVICE`` optionally places ``audio_tower`` +
``multi_modal_projector`` (everything ``get_audio_features`` uses) on another
card. Unset encoder env → both stay on ``lm_device`` (old single-GPU path).
"""

from __future__ import annotations

import os
import types
from typing import Any, Iterable, List, Optional, Tuple

import torch


def _as_device(device: Any) -> torch.device:
    if isinstance(device, torch.device):
        return device
    return torch.device(str(device))


def resolve_enc_device(lm_device: Any, explicit: Any = None) -> torch.device:
    """``explicit`` > ``ENC_DEVICE`` > ``VOXTRAL_ENC_DEVICE`` > ``lm_device``."""
    raw = explicit
    if raw is None or not str(raw).strip():
        raw = os.environ.get("ENC_DEVICE") or os.environ.get("VOXTRAL_ENC_DEVICE") or ""
    raw = str(raw).strip()
    if not raw:
        return _as_device(lm_device)
    return _as_device(raw)


def _core_model(model: Any) -> Any:
    """``VoxtralRealtimeModel`` (has ``audio_tower``), unwrapping PEFT/CG."""
    m = model
    get_base = getattr(m, "get_base_model", None)
    if callable(get_base):
        try:
            m = get_base()
        except Exception:
            pass
    if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
        inner = m.base_model.model
        if hasattr(inner, "model") and hasattr(inner, "lm_head"):
            m = inner
    if hasattr(m, "model") and hasattr(m.model, "audio_tower"):
        return m.model
    return m


def _cg_model(model: Any) -> Any:
    """Prefer the CG wrapper so ``.device`` / attrs land where callers look."""
    m = model
    get_base = getattr(m, "get_base_model", None)
    if callable(get_base):
        try:
            m = get_base()
        except Exception:
            pass
    if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
        inner = m.base_model.model
        if hasattr(inner, "lm_head") and hasattr(inner, "model"):
            m = inner
    return m


def enc_device_of(model: Any) -> torch.device:
    for obj in (model, _cg_model(model), _core_model(model)):
        pinned = getattr(obj, "_enc_device", None)
        if pinned is not None:
            return _as_device(pinned)
    return _as_device(getattr(model, "device", "cpu"))


def lm_device_of(model: Any) -> torch.device:
    for obj in (model, _cg_model(model), _core_model(model)):
        pinned = getattr(obj, "_lm_device", None)
        if pinned is not None:
            return _as_device(pinned)
    return _as_device(getattr(model, "device", "cpu"))


def devices_are_split(model: Any) -> bool:
    return enc_device_of(model) != lm_device_of(model)


def encoder_modules(model: Any) -> List[Tuple[str, torch.nn.Module]]:
    """Submodules that ``get_audio_features`` actually runs."""
    core = _core_model(model)
    found: List[Tuple[str, torch.nn.Module]] = []
    tower = getattr(core, "audio_tower", None)
    if tower is not None:
        found.append(("audio_tower", tower))
    enc = getattr(core, "audio_encoder", None)
    if enc is not None and enc is not tower:
        found.append(("audio_encoder", enc))
    proj = getattr(core, "multi_modal_projector", None)
    if proj is not None:
        found.append(("multi_modal_projector", proj))
    return found


def _pin_reported_device(module: Any, lm_device: torch.device) -> None:
    """Make ``module.device`` keep reporting the LM card after the split.

    HF ``.device`` is ``next(parameters()).device``. ``audio_tower`` is first,
    so moving it would otherwise flip ``vox.device`` to the encoder GPU.
    """
    module._lm_device = lm_device
    cls = module.__class__
    if getattr(cls, "_voxtral_device_pinned", False):
        return
    orig = getattr(cls, "device", None)
    orig_fget = orig.fget if isinstance(orig, property) else None

    def _fget(self):
        pinned = getattr(self, "_lm_device", None)
        if pinned is not None:
            return _as_device(pinned)
        if orig_fget is not None:
            return orig_fget(self)
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    cls.device = property(_fget)
    cls._voxtral_device_pinned = True


def _wrap_get_audio_features(core: Any, enc_device: torch.device) -> None:
    """Keep encoder inputs on ``enc_device`` even if the caller used ``vox.device``."""
    if getattr(core, "_voxtral_enc_features_wrapped", False):
        return
    orig = core.get_audio_features
    orig_fn = getattr(orig, "__func__", orig)

    def wrapped(
        self,
        input_features=None,
        padding_cache=None,
        encoder_inputs_embeds=None,
        past_key_values=None,
        use_cache=None,
        **kwargs,
    ):
        if (
            input_features is not None
            and torch.is_tensor(input_features)
            and input_features.device != enc_device
        ):
            input_features = input_features.to(device=enc_device, non_blocking=True)
        if (
            encoder_inputs_embeds is not None
            and torch.is_tensor(encoder_inputs_embeds)
            and encoder_inputs_embeds.device != enc_device
        ):
            encoder_inputs_embeds = encoder_inputs_embeds.to(
                device=enc_device, non_blocking=True,
            )
        return orig_fn(
            self,
            input_features=input_features,
            padding_cache=padding_cache,
            encoder_inputs_embeds=encoder_inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

    core.get_audio_features = types.MethodType(wrapped, core)
    core._voxtral_enc_features_wrapped = True


def apply_encoder_lm_split(
    model: Any,
    *,
    lm_device: Any,
    enc_device: Any = None,
    verbose: bool = True,
) -> Tuple[torch.device, torch.device]:
    """Move encoder submodules to ``enc_device``; leave LM on ``lm_device``.

    One 4B copy. Returns ``(lm_device, enc_device)``.
    """
    lm_dev = _as_device(lm_device)
    enc_dev = resolve_enc_device(lm_dev, enc_device)
    cg = _cg_model(model)
    core = _core_model(model)

    for obj in (model, cg, core):
        obj._lm_device = lm_dev
        obj._enc_device = enc_dev

    if enc_dev != lm_dev:
        moved = []
        for name, mod in encoder_modules(model):
            mod.to(enc_dev)
            moved.append(name)
        _pin_reported_device(cg, lm_dev)
        if core is not cg:
            _pin_reported_device(core, lm_dev)
        _wrap_get_audio_features(core, enc_dev)
        if verbose:
            print(
                f"[voxtral] encoder on {enc_dev} ({', '.join(moved) or 'none'}); "
                f"LM on {lm_dev}",
                flush=True,
            )
    elif verbose:
        print(
            f"[voxtral] encoder on {enc_dev}, LM on {lm_dev} (same device)",
            flush=True,
        )
    return lm_dev, enc_dev


def to_device(
    tensor: torch.Tensor,
    device: Any,
    *,
    dtype: Optional[torch.dtype] = None,
    non_blocking: Optional[bool] = None,
) -> torch.Tensor:
    dest = _as_device(device)
    if non_blocking is None:
        non_blocking = tensor.device != dest
    if tensor.device == dest and (dtype is None or tensor.dtype == dtype):
        return tensor
    if dtype is None:
        return tensor.to(device=dest, non_blocking=bool(non_blocking))
    return tensor.to(device=dest, dtype=dtype, non_blocking=bool(non_blocking))


def _iter_cuda_devices(*devices: Any) -> Iterable[torch.device]:
    seen = set()
    for raw in devices:
        if raw is None:
            continue
        dev = _as_device(raw)
        if dev.type != "cuda" or dev in seen:
            continue
        seen.add(dev)
        yield dev


def synchronize_model_devices(model: Any = None, *extra: Any) -> None:
    if not torch.cuda.is_available():
        return
    devices: List[Any] = list(extra)
    if model is not None:
        devices.extend((lm_device_of(model), enc_device_of(model)))
    synced = False
    for dev in _iter_cuda_devices(*devices):
        torch.cuda.synchronize(dev)
        synced = True
    if not synced:
        torch.cuda.synchronize()


def empty_cache_model_devices(model: Any = None, *extra: Any) -> None:
    if not torch.cuda.is_available():
        return
    devices: List[Any] = list(extra)
    if model is not None:
        devices.extend((lm_device_of(model), enc_device_of(model)))
    for dev in _iter_cuda_devices(*devices):
        with torch.cuda.device(dev):
            torch.cuda.empty_cache()
