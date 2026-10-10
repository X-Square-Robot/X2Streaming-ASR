"""ASR model package — factory + auto-registration."""

from models.base import BaseASRModel, TranscribeResult, build_model, register_model

import models.voxtral_asr  # noqa: F401 — triggers @register_model

__all__ = ["BaseASRModel", "TranscribeResult", "build_model", "register_model"]
