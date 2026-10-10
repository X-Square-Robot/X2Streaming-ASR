"""Audio preparation and runtime constants for Voxtral Realtime inference."""

import inspect
from typing import Any, List, Optional, Sequence

import librosa
import numpy as np
import torch
from transformers import BatchFeature


STREAMING_PAD_ID = 32
STREAMING_WORD_ID = 33
# Reserved control slot (tekken rank 4 = \\u0004). Do not use 34 ('"').
STREAMING_WORD_OFF_ID = 4
# Protocol-only delay for adaptive mode (mistral_common requires delay_ms > 0).
# Not a user-facing latency knob; the model decides when to wait or emit.
ADAPTIVE_PROTOCOL_DELAY_MS = 80

# language_tag ids (new embedding; not STREAMING_PAD / AdaRMS τ).
LANG_ID_ZH = 0
LANG_ID_EN = 1
LANG_ID_CS = 2

# Conservative path substrings. CS before EN/ZH; tal asr_* is zh, csasr is cs.
_LANG_EN_MARKERS = (
    "gigaspeech",
    "librispeech",
    "tedlium",
    "voxpopuli",
)
_LANG_ZH_MARKERS = (
    "aishell",
    "wenetspeech",
    "kespeech",
    "alimeeting",
    "aidatatang",
    "magicdata",
    "easy-turn",
    "easy_turn",
)


def infer_lang_id_from_path(path: str) -> int:
    """Map an audio / jsonl path to language_tag id (zh=0, en=1, cs=2).

    Unknown paths default to zh (CJK-heavy corpus). Be explicit: only listed
    substrings count as en/cs.
    """
    p = str(path or "").replace("\\", "/").lower()
    if not p:
        return LANG_ID_ZH
    if "csasr" in p:
        return LANG_ID_CS
    if any(marker in p for marker in _LANG_EN_MARKERS):
        return LANG_ID_EN
    if any(marker in p for marker in _LANG_ZH_MARKERS):
        return LANG_ID_ZH
    if "tal_data" in p and "asr_" in p:
        return LANG_ID_ZH
    return LANG_ID_ZH


def encode_audio_with_delay(
    processor: Any,
    audio_array: np.ndarray,
    sampling_rate: int,
    delay_ms: int,
) -> BatchFeature:
    """Encode waveform with a specific streaming delay (matches inference τ)."""
    return encode_audios_with_delay(
        processor, [audio_array], sampling_rate, delay_ms,
    )[0]


def _prepare_audio_with_delay(
    processor: Any,
    audio_array: np.ndarray,
    sampling_rate: int,
    delay_ms: int,
):
    """Apply the per-utterance transcription protocol before mel extraction."""
    from mistral_common.audio import Audio
    from mistral_common.protocol.instruct.chunk import RawAudio
    from mistral_common.protocol.transcription.request import StreamingMode, TranscriptionRequest

    audio = Audio(
        audio_array=np.asarray(audio_array, dtype=np.float32),
        sampling_rate=sampling_rate,
        format="wav",
    )
    request = TranscriptionRequest(
        audio=RawAudio.from_audio(audio),
        streaming=StreamingMode.OFFLINE,
        target_streaming_delay_ms=delay_ms,
    )
    tokenized = processor.tokenizer.tokenizer.encode_transcription(request)
    audio_arrays = [a.audio_array for a in tokenized.audios]
    text_encoding = processor.tokenizer(
        tokenized.tokens,
        add_special_tokens=False,
        return_tensors="pt",
    )
    num_delay_tokens = processor.mistral_common_audio_config.get_num_delay_tokens(delay_ms)
    return audio_arrays, text_encoding, num_delay_tokens


def _sample_count(audio_array: Any) -> int:
    arr = np.asarray(audio_array)
    return int(arr.shape[0]) if arr.ndim >= 1 else 0


def _mel_frames_from_samples(n_samples: int, hop_length: int, *, drop_last: bool) -> int:
    """STFT frames for ``center=True``: ``1 + n_samples // hop``, optionally drop last."""
    hop = max(1, int(hop_length))
    n_frames = 1 + max(0, int(n_samples)) // hop
    if drop_last:
        n_frames = max(1, n_frames - 1)
    return max(1, n_frames)


def _infer_drop_last_mel_frames(t_got: int, max_samples: int, hop_length: int) -> Optional[bool]:
    """Match batched T to center-STFT with or without the trailing-bin drop."""
    if int(t_got) == _mel_frames_from_samples(max_samples, hop_length, drop_last=True):
        return True
    if int(t_got) == _mel_frames_from_samples(max_samples, hop_length, drop_last=False):
        return False
    return None


def _extractor_honors_center(extractor: Any) -> bool:
    fn = getattr(type(extractor), "__call__", None)
    if fn is None:
        return False
    try:
        return "center" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _center_reflect_pad(audio_array: Any, n_fft: int) -> np.ndarray:
    """Same edge pad ``torch.stft(..., center=True, pad_mode='reflect')`` applies."""
    pad = max(0, int(n_fft) // 2)
    arr = np.asarray(audio_array, dtype=np.float32).reshape(-1)
    if pad == 0 or arr.size == 0:
        return arr
    tensor = torch.from_numpy(np.ascontiguousarray(arr)).view(1, 1, -1)
    try:
        padded = torch.nn.functional.pad(tensor, (pad, pad), mode="reflect")
    except RuntimeError:
        padded = torch.nn.functional.pad(tensor, (pad, pad), mode="constant")
    return padded.view(-1).detach().cpu().numpy()


def encode_audios_with_delay(
    processor: Any,
    audio_arrays: Sequence[np.ndarray],
    sampling_rate: int,
    delay_ms: int,
) -> List[BatchFeature]:
    """Encode waveforms with one batched feature-extractor call.

    Protocol construction and tokenization remain per utterance. Only mel
    extraction is fused. Extractors that honor ``center`` (VoxtralRealtime)
    get a reflect-prepad + ``center=False`` batch so longest-pad STFT matches
    per-utt ``center=True``. Whisper ``max_length`` padding is left as-is.
    """
    if not audio_arrays:
        return []

    prepared = [
        _prepare_audio_with_delay(
            processor, audio_array, sampling_rate, delay_ms,
        )
        for audio_array in audio_arrays
    ]
    flat_audio_arrays = [
        audio_array
        for delayed_arrays, _text_encoding, _num_delay_tokens in prepared
        for audio_array in delayed_arrays
    ]
    extractor = processor.feature_extractor
    hop = int(getattr(extractor, "hop_length", 160) or 160)
    n_fft = int(getattr(extractor, "n_fft", 400) or 400)
    orig_lens = [_sample_count(arr) for arr in flat_audio_arrays]
    honors_center = _extractor_honors_center(extractor) and len(flat_audio_arrays) > 1
    if honors_center:
        feat_inputs = [_center_reflect_pad(arr, n_fft) for arr in flat_audio_arrays]
        audio_encoding = extractor(
            feat_inputs, center=False, sampling_rate=sampling_rate,
        )
    else:
        audio_encoding = extractor(
            flat_audio_arrays, center=True, sampling_rate=sampling_rate,
        )
    batched_audio = BatchFeature(data=audio_encoding, tensor_type="pt")

    feats = batched_audio.get("input_features")
    drop_last: Optional[bool] = None
    if torch.is_tensor(feats) and feats.ndim >= 2 and orig_lens:
        drop_last = _infer_drop_last_mel_frames(
            int(feats.shape[-1]), max(orig_lens), hop,
        )

    results: List[BatchFeature] = []
    offset = 0
    for delayed_arrays, text_encoding, num_delay_tokens in prepared:
        count = len(delayed_arrays)
        per_audio = {}
        for key, value in batched_audio.items():
            if not torch.is_tensor(value):
                per_audio[key] = value
                continue
            sliced = value[offset : offset + count]
            if (
                key == "input_features"
                and drop_last is not None
                and sliced.ndim >= 2
                and delayed_arrays
            ):
                native_t = max(
                    _mel_frames_from_samples(
                        _sample_count(arr), hop, drop_last=drop_last,
                    )
                    for arr in delayed_arrays
                )
                native_t = min(native_t, int(sliced.shape[-1]))
                sliced = sliced[..., :native_t]
            per_audio[key] = sliced.contiguous()
        offset += count
        encoding = {
            **text_encoding,
            **per_audio,
            "num_delay_tokens": num_delay_tokens,
        }
        results.append(BatchFeature(data=encoding, tensor_type="pt"))
    return results


def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


# Attached in DataLoader workers so collator can skip serial load_audio+mel.
AUDIO_ENCODING_KEY = "audio_encoding"



def _find_time_embedding_owner(model: Any) -> Any:
    """Locate the module that owns ``time_embedding`` (unwrap PEFT / DDP).

    Layouts:
      - Bare CG: ``VoxtralRealtimeForConditionalGeneration.model.time_embedding``
      - PeftModel: ``base_model.model.model.time_embedding``
        (PeftModel.model is the CG; CG.model is VoxtralRealtimeModel)
    """
    m = model
    if hasattr(m, "module"):
        m = m.module

    # BFS one/two levels through common PEFT/HF wrappers.
    queue: List[Any] = [m]
    seen: set = set()
    while queue:
        cur = queue.pop(0)
        cid = id(cur)
        if cid in seen or cur is None:
            continue
        seen.add(cid)
        if hasattr(cur, "time_embedding"):
            return cur
        for attr in ("base_model", "model"):
            child = getattr(cur, attr, None)
            if child is not None and id(child) not in seen:
                queue.append(child)
    return None


def disable_adarms_delay_conditioning(model: Any) -> None:
    """Make AdaRMS delay conditioning an identity (scale = 1).

    Replaces ``time_embedding`` so ``ada_rms_norm(t_cond)`` is always zero
    (AdaRmsNorm has no bias). Adaptive models must not be locked to a fixed τ.
    Safe to call multiple times. Works on bare or PEFT-wrapped models.
    """
    root = _find_time_embedding_owner(model)
    if root is None:
        # Should not happen for VoxtralRealtime; skip with a clear log rather
        # than crashing so callers can still install interleaved forward.
        print(
            "[adaptive] warning: no time_embedding found after unwrap "
            "(PEFT/DDP); AdaRMS disable skipped"
        )
        return

    # Already replaced (idempotent).
    if type(root.time_embedding).__name__ == "_ZeroTimeEmbedding":
        return

    dim = int(root.time_embedding.dim)

    class _ZeroTimeEmbedding(torch.nn.Module):
        def __init__(self, hidden_dim: int):
            super().__init__()
            self.dim = hidden_dim

        def forward(self, time_tensor: torch.Tensor) -> torch.Tensor:
            return torch.zeros(
                self.dim, device=time_tensor.device, dtype=time_tensor.dtype,
            )

    root.time_embedding = _ZeroTimeEmbedding(dim)
