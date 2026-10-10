"""ASR model abstraction layer — base class and factory."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class TranscribeResult:
    text: str
    language: str = ""
    time_stamps: Optional[List] = None
    emits: Optional[List[Dict[str, Any]]] = None
    n_emits: Optional[int] = None


class BaseASRModel(ABC):

    @abstractmethod
    def load_for_inference(self, model_path: str, device: str = "cuda:0", **kwargs):
        """Load model for inference. Returns a wrapper object."""

    @abstractmethod
    def transcribe(self, wrapper, audio, **kwargs) -> List[TranscribeResult]:
        """Run inference. Returns a list of TranscribeResult."""


_REGISTRY: Dict[str, type] = {}


def register_model(name: str):
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator


def build_model(model_type: str) -> BaseASRModel:
    if model_type not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model_type '{model_type}'. Available: {available}"
        )
    return _REGISTRY[model_type]()
