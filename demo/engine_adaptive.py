"""Adaptive ASR session: live online decode + full-context offline inference.

1. **Live** — chunked streaming features + listen↔decode while PCM arrives
   (decode tokens stream out as they are generated). LISTEN is frame-synchronous
   by default (``listen_chunk_size=1``, one ~80 ms frame per LM step).
2. **Offline** — after stop/EOF, ``generate_interleaved_offline``: listen all
   audio frames → force one ``W_off`` → decode to EOS (same as infer_offline_*.yaml).
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.voxtral_asr.data import (  # noqa: E402
    ADAPTIVE_PROTOCOL_DELAY_MS,
    STREAMING_PAD_ID,
    STREAMING_WORD_ID,
    encode_audio_with_delay,
)
from models.voxtral_asr.devices import (  # noqa: E402
    empty_cache_model_devices,
    enc_device_of,
    lm_device_of,
    synchronize_model_devices,
    to_device,
)
from models.voxtral_asr.interleaved import (  # noqa: E402
    _adaptive_n_left_pad,
    _adaptive_seconds_per_token,
    _crop_past_kv,
    _lm_step_embeds,
    _past_seq_len,
    _unwrap_voxtral_cg,
    generate_interleaved_offline,
)
from engine import _AUDIO_ENCODER_LOCK, _GENERATION_LOCK  # noqa: E402
from data_pipeline.join_emit import join_emit_texts  # noqa: E402

_SILENCE_PAD_CACHE: Dict[int, torch.Tensor] = {}
_EMBED_Q_MAX = 64
_FEAT_Q_MAX = 64


def _env_truthy(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _pooler_frames(pooler_output: torch.Tensor) -> torch.Tensor:
    if pooler_output.dim() == 3:
        return pooler_output[0]
    if pooler_output.dim() == 2:
        return pooler_output
    raise ValueError(f"Unexpected pooler_output shape: {tuple(pooler_output.shape)}")


def _get_silence_pad_frames(
    vox: Any, processor: Any, n_left_pad: int, dtype: torch.dtype,
) -> torch.Tensor:
    """Return ``[n_left_pad, H]`` silence pooler frames for warmup prefill.

    Runs actual silence through ``encode_audio_with_delay`` and
    ``get_audio_features`` so the left-pad frames match model inputs.
    Cached per model instance.
    """
    cache_key = id(vox)
    cached = _SILENCE_PAD_CACHE.get(cache_key)
    if cached is not None and cached.shape[0] >= n_left_pad:
        return cached[:n_left_pad].to(device=lm_device_of(vox), dtype=dtype)

    sr = processor.feature_extractor.sampling_rate
    alt = int(getattr(processor, "audio_length_per_tok", 8) or 8)
    hop = int(processor.feature_extractor.hop_length)
    # Generate enough silence to produce ≥ n_left_pad pooler frames (2× margin).
    n_samples = (n_left_pad + 4) * alt * hop * 2
    n_samples = max(n_samples, sr)  # at least 1 second
    silence = np.zeros(n_samples, dtype=np.float32)

    enc = encode_audio_with_delay(processor, silence, sr, ADAPTIVE_PROTOCOL_DELAY_MS)
    input_features = enc.input_features.to(device=enc_device_of(vox), dtype=vox.dtype)

    with torch.no_grad():
        audio_out = vox.model.get_audio_features(input_features=input_features)
    pooled = audio_out.pooler_output  # [1, T, H]

    frames = pooled[0, :n_left_pad].to(dtype=dtype).contiguous()
    _SILENCE_PAD_CACHE[cache_key] = frames
    return frames


class AdaptiveSession:
    """Live adaptive decoding plus full-context offline decoding."""

    def __init__(
        self,
        model,
        processor,
        *,
        emit_word_prob_threshold: Optional[float] = None,
        max_decode_tokens: int = 32,
        max_new_tokens: Optional[int] = None,
        verbose: bool = True,
        enable_live: bool = True,
        enable_offline: bool = True,
        listen_chunk_size: int = 1,
        offline_listen_chunk_size: int = 16,
        feature_prefetch: bool = True,
        live_emit_word_prob_threshold: Optional[float] = None,
        gpu_encode_overlap: Optional[bool] = None,
    ):
        self.model = model
        self.processor = processor
        self.sr = processor.feature_extractor.sampling_rate
        self.emit_word_prob_threshold = emit_word_prob_threshold
        self.live_emit_word_prob_threshold = None
        self.max_decode_tokens = max_decode_tokens
        self.max_new_tokens = max_new_tokens
        self.verbose = verbose
        self.enable_live = enable_live
        self.enable_offline = enable_offline
        self.listen_chunk_size = max(1, int(listen_chunk_size))
        self.offline_listen_chunk_size = max(1, int(offline_listen_chunk_size))
        self.feature_prefetch = bool(feature_prefetch)
        if gpu_encode_overlap is None:
            gpu_encode_overlap = _env_truthy("VOXTRAL_GPU_ENCODE_OVERLAP", True)
        self.gpu_encode_overlap = bool(gpu_encode_overlap)
        self._use_enc_overlap = (
            bool(self.feature_prefetch)
            and bool(self.gpu_encode_overlap)
            and torch.cuda.is_available()
        )
        vox0 = _unwrap_voxtral_cg(model)
        self._lm_device = lm_device_of(vox0)
        self._enc_device = enc_device_of(vox0)
        self._split_devices = torch.device(self._lm_device) != torch.device(self._enc_device)
        from engine import encode_first_streaming_chunk, get_streaming_delay_config

        self._encode_first_streaming_chunk = encode_first_streaming_chunk
        self._delay_cfg = get_streaming_delay_config(processor, ADAPTIVE_PROTOCOL_DELAY_MS)

        self._buffer = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()
        self._gpu_span_lock = threading.Lock()
        self._embed_q: queue.Queue = self._new_embed_q()
        self._feat_q: queue.Queue = queue.Queue(maxsize=_FEAT_Q_MAX)
        self._event_q: queue.Queue = queue.Queue()
        self._text_q = self._event_q

        self._closed = threading.Event()
        self._abort = threading.Event()
        self._feat_closed = threading.Event()
        self._started = False
        self._prod_started = False
        self._live_thread: Optional[threading.Thread] = None
        self._feat_thread: Optional[threading.Thread] = None
        self._enc_thread: Optional[threading.Thread] = None
        self._finish_thread: Optional[threading.Thread] = None
        self._encode_stream: Optional[torch.cuda.Stream] = None
        self._enc_busy = False
        self._enc_done = threading.Event()
        self._enc_done.set()

        self._padding_cache = None
        self._audio_past = None
        self._mel_frame_idx: Optional[int] = None
        self._next_start = 0
        self._prod_next_start = 0
        self._frame_i = 0
        self._preheat_samples = 0
        self._warmup_done = False
        self._enc_inited = False
        self._pad_frames_pushed = 0
        self._n_left_pad = _adaptive_n_left_pad(processor)

        self._live_text = ""
        self._live_parts: List[str] = []
        self._live_emits_timed: List[Dict[str, Any]] = []
        self._pending_emit_time_s: Optional[float] = None
        self._pending_emit_frame: Optional[int] = None
        self._last_wav: Optional[np.ndarray] = None
        self._offline_text = ""   # 离线推理：全量 prefill → 一次 W → decode
        self._recheck_text = ""   # alias of offline_text (compat)
        self._final_text = ""
        self._error: Optional[str] = None
        self._last_meta: Dict[str, Any] = {}
        self._leftover_frames: List[torch.Tensor] = []

        # ── RTF timing ────────────────────────────────────────────────
        self._t_live_start: Optional[float] = None  # wall-clock of first user PCM
        self._t_last_emit: Optional[float] = None   # wall-clock of previous emit (for delta)
        self._enc_wall_s: float = 0.0               # accumulated GPU time in get_audio_features
        self._lm_wall_s: float = 0.0                # accumulated GPU time in LM forward (listen + decode)
        self._n_frames_processed: int = 0            # total audio frames consumed by LM
        self._gpu_spans: List[tuple] = []           # pending CUDA events (kind, start, end)
        self.language_id = 0
        self.set_language_id(0)

    def _new_embed_q(self) -> queue.Queue:
        return queue.Queue(maxsize=_EMBED_Q_MAX if self._use_enc_overlap else 0)

    def set_language_id(self, language_id: Any) -> bool:
        """Set language_tag id. Returns True if the id changed (need re-warmup)."""
        from models.voxtral_asr.language_tag import set_infer_language_id

        lid = set_infer_language_id(self.model, language_id)
        changed = int(getattr(self, "language_id", lid)) != lid
        self.language_id = lid
        return changed

    # ── public ────────────────────────────────────────────────────────

    def reset(self):
        self.abort()
        self._buffer = np.zeros(0, dtype=np.float32)
        self._embed_q = self._new_embed_q()
        self._leftover_frames: List[torch.Tensor] = []
        self._feat_q = queue.Queue(maxsize=_FEAT_Q_MAX)
        self._event_q = queue.Queue()
        self._text_q = self._event_q
        self._closed.clear()
        self._abort.clear()
        self._feat_closed.clear()
        self._started = False
        self._prod_started = False
        self._live_thread = None
        self._feat_thread = None
        self._enc_thread = None
        self._encode_stream = None
        self._enc_busy = False
        self._enc_done.set()
        self._finish_thread = None
        self._padding_cache = None
        self._audio_past = None
        self._mel_frame_idx = None
        self._next_start = 0
        self._prod_next_start = 0
        self._frame_i = 0
        self._preheat_samples = 0
        self._warmup_done = False
        self._enc_inited = False
        self._pad_frames_pushed = 0
        self._live_text = ""
        self._live_parts = []
        self._live_emits_timed = []
        self._pending_emit_time_s = None
        self._pending_emit_frame = None
        self._last_wav = None
        self._offline_text = ""
        self._recheck_text = ""
        self._final_text = ""
        self._error = None
        self._last_meta = {}
        self._t_live_start = None
        self._t_last_emit = None
        self._enc_wall_s = 0.0
        self._lm_wall_s = 0.0
        self._n_frames_processed = 0
        self._gpu_spans = []

    def _streaming_left_pad_samples(self) -> int:
        """PCM samples of zeros that yield ``n_left_pad`` streaming frames."""
        n_left = int(self._n_left_pad)
        first = int(self._delay_cfg["num_samples_first_audio_chunk"])
        hop = int(self.processor.feature_extractor.hop_length)
        win = int(self.processor.feature_extractor.win_length)
        alt = int(self.processor.audio_length_per_tok)
        cfg = self.processor.mistral_common_audio_config
        n_first = int(cfg.get_num_delay_tokens(ADAPTIVE_PROTOCOL_DELAY_MS)) + 1
        extra = max(0, n_left - n_first)
        if extra == 0:
            return first
        mel_first = int(self._delay_cfg["num_mel_frames_first_audio_chunk"])
        next_start = mel_first * hop - win // 2
        last_start = next_start + (extra - 1) * alt * hop
        return int(last_start + alt * hop + win)

    def start(self):
        """Start live worker with streaming-encoder left-pad, then user audio.

        Protocol pad is zeros pushed through the same ONLINE encoder as speech
        (not a separate offline silence cache). Offline still sees only
        user PCM via ``_preheat_samples``.
        """
        if not self.enable_live:
            self._warmup_done = True
            self._event_q.put({
                "kind": "warmup_done",
                "n_left_pad": 0,
                "preheat_s": 0.0,
                "wall_s": 0.0,
            })
            return
        if self._live_thread is not None and self._live_thread.is_alive():
            return
        self._closed.clear()
        self._abort.clear()
        self._feat_closed.clear()
        self._started = False
        self._prod_started = False
        self._enc_inited = False
        self._pad_frames_pushed = 0
        self._padding_cache = None
        self._audio_past = None
        self._mel_frame_idx = None
        self._next_start = 0
        self._prod_next_start = 0
        self._frame_i = 0
        self._warmup_done = False
        self._embed_q = self._new_embed_q()
        self._leftover_frames: List[torch.Tensor] = []
        self._feat_q = queue.Queue(maxsize=_FEAT_Q_MAX)
        self._encode_stream = None
        self._enc_busy = False
        if self._use_enc_overlap:
            self._enc_done.clear()
        else:
            self._enc_done.set()
        self._live_parts = []
        self._live_emits_timed = []
        self._pending_emit_time_s = None
        self._pending_emit_frame = None
        self._live_text = ""
        self._t_live_start = None
        self._enc_wall_s = 0.0
        self._lm_wall_s = 0.0
        self._gpu_spans = []

        hop = int(self.processor.feature_extractor.hop_length)
        alt = int(self.processor.audio_length_per_tok)
        pad_n = self._streaming_left_pad_samples() + alt * hop
        with self._lock:
            self._buffer = np.zeros(pad_n, dtype=np.float32)
            self._preheat_samples = pad_n

        self._event_q.put({
            "kind": "warmup_start",
            "n_left_pad": self._n_left_pad,
            "preheat_s": round(pad_n / float(self.sr), 3),
        })
        if self.verbose:
            print(
                f"[adaptive-live] start streaming left-pad "
                f"n_left_pad={self._n_left_pad} preheat={pad_n / self.sr:.2f}s",
                flush=True,
            )

        self._live_thread = threading.Thread(target=self._run_live_worker, daemon=True)
        self._live_thread.start()
        if self.feature_prefetch:
            self._feat_thread = threading.Thread(target=self._run_feature_worker, daemon=True)
            self._feat_thread.start()
        if self._use_enc_overlap:
            self._enc_thread = threading.Thread(target=self._run_encoder_worker, daemon=True)
            self._enc_thread.start()
            if self.verbose:
                extra = (
                    f" enc={self._enc_device} lm={self._lm_device}"
                    if self._split_devices else ""
                )
                print(
                    f"[adaptive-live] encoder overlap on (side CUDA stream{extra})",
                    flush=True,
                )
        else:
            self._enc_thread = None
            self._enc_done.set()

    def _skip_remaining_preheat_pcm(self):
        """Stop encoding leftover warmup zeros; next chunk is user PCM.

        Feature worker is gated at ``_preheat_samples`` until warmup finishes,
        so leftover ``feat_q`` items are extra zero mels only.
        """
        with self._lock:
            pre = int(self._preheat_samples)
            if self._prod_started:
                self._prod_next_start = max(int(self._prod_next_start), pre)
            if self._started:
                self._next_start = max(int(self._next_start), pre)
        while True:
            try:
                self._feat_q.get_nowait()
            except queue.Empty:
                break

    def append_pcm(self, pcm: np.ndarray):
        pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if pcm.size == 0:
            return
        with self._lock:
            if self._t_live_start is None:
                self._t_live_start = time.time()
            self._buffer = np.concatenate([self._buffer, pcm])

    def buffered_seconds(self) -> float:
        with self._lock:
            return float(self._buffer.shape[0]) / float(self.sr)

    def finish_async(self, on_chunk: Optional[Callable[[str], None]] = None):
        """End the live path, then run optional full-context inference."""

        def _run():
            try:
                # 1) close live
                if self._live_thread is not None:
                    with self._lock:
                        self._pad_and_flush_tail_locked()
                        self._closed.set()
                        self._feat_closed.set()
                    if self._feat_thread is not None and self._feat_thread.is_alive():
                        self._feat_thread.join(timeout=30)
                    # Live consumes embeds and waits for enc_done. Join encoder
                    # after live so a full embed_q cannot stall this thread.
                    self._live_thread.join(timeout=600)
                    if self._enc_thread is not None and self._enc_thread.is_alive():
                        self._enc_thread.join(timeout=30)
                    self._live_text = join_emit_texts(self._live_parts)
                    if self.verbose:
                        print(f"[adaptive-live] done text={self._live_text!r}", flush=True)
                    self._event_q.put({
                        "kind": "live_done",
                        "text": self._live_text,
                        "emits": list(self._live_parts),
                        "emits_timed": list(self._live_emits_timed),
                    })

                    # Release live KV cache / audio encoder state before offline
                    self._padding_cache = None
                    self._audio_past = None
                    empty_cache_model_devices(
                        self.model, self._lm_device, self._enc_device,
                    )

                # 2) offline: listen all frames → force one W → decode to EOS
                with self._lock:
                    wav = self._buffer[self._preheat_samples:].copy()

                dur = float(wav.shape[0]) / float(self.sr) if wav.size else 0.0
                rms = float(np.sqrt(np.mean(np.square(wav)))) if wav.size else 0.0
                self._last_wav = wav
                self._last_meta = {
                    "duration_s": round(dur, 3),
                    "n_samples": int(wav.shape[0]),
                    "rms": round(rms, 6),
                    "n_emits": 0,
                    "n_audio_frames": 0,
                    "live_text": self._live_text,
                    "live_emits": list(self._live_parts),
                    "live_emits_timed": list(self._live_emits_timed),
                }

                if not self.enable_offline:
                    self._final_text = self._live_text
                    return

                if wav.size == 0:
                    self._error = "empty_audio"
                    self._final_text = ""
                    return

                if self.verbose:
                    print(
                        f"[adaptive-offline] start dur={dur:.2f}s rms={rms:.4f}",
                        flush=True,
                    )

                emit_parts: List[str] = []

                def _on_emit(rec: Dict[str, Any]) -> None:
                    chunk = (rec.get("text") or "").strip()
                    if not chunk:
                        return
                    emit_parts.append(chunk)
                    self._event_q.put({"kind": "offline_emit", "text": chunk})

                self._event_q.put({"kind": "offline_start"})
                with _GENERATION_LOCK:
                    offline_out = generate_interleaved_offline(
                        self.model,
                        self.processor,
                        wav,
                        max_new_tokens=self.max_new_tokens,
                        on_emit=_on_emit,
                    )
                self._offline_text = (offline_out.get("text") or "").strip()
                if not self._offline_text and emit_parts:
                    self._offline_text = "".join(emit_parts)
                self._recheck_text = self._offline_text
                self._final_text = self._offline_text
                self._event_q.put({
                    "kind": "offline_done",
                    "text": self._offline_text,
                    "emits": list(emit_parts),
                })
                self._last_meta.update({
                    "n_emits": int(offline_out.get("n_emits") or len(emit_parts)),
                    "n_audio_frames": int(offline_out.get("n_audio_frames") or 0),
                    "offline_text": self._offline_text,
                    "offline_emits": list(emit_parts),
                    "mode": "offline",
                })
                if self.verbose:
                    print(f"[adaptive-offline] done text={self._offline_text!r}", flush=True)
                if not self._offline_text:
                    self._error = (
                        f"no_text (n_emits={self._last_meta['n_emits']}, "
                        f"dur={dur:.2f}s, rms={rms:.4f})"
                    )
            except Exception as exc:  # noqa: BLE001
                self._error = str(exc)
                print(f"[adaptive] finish error: {exc}", flush=True)
            finally:
                self._event_q.put(None)

        self._finish_thread = threading.Thread(target=_run, daemon=True)
        self._finish_thread.start()
        return self._finish_thread

    def _pad_and_flush_tail_locked(self):
        """Pad tail so live worker can consume all buffered audio before close."""
        pad = int(self._delay_cfg["pad_samples"])
        if pad > 0:
            self._buffer = np.pad(self._buffer, (0, pad))

        first_need = int(self._delay_cfg["num_samples_first_audio_chunk"])
        window = int(self._delay_cfg["num_samples_per_audio_chunk"])

        # Ensure first chunk can be encoded.
        if self.feature_prefetch:
            started = bool(self._prod_started)
            next_start = int(self._prod_next_start)
        else:
            started = bool(self._started)
            next_start = int(self._next_start)

        if not started:
            if self._buffer.shape[0] < first_need:
                extra = first_need - int(self._buffer.shape[0])
                self._buffer = np.pad(self._buffer, (0, extra))
            return

        # Extract window is larger than hop; pad until one more full window fits.
        need_end = next_start + window
        total = int(self._buffer.shape[0])
        if need_end > total:
            self._buffer = np.pad(self._buffer, (0, need_end - total))

    def finish(self):
        t = self.finish_async()
        if t is not None:
            t.join()

    @property
    def final_text(self) -> str:
        return self._final_text

    @property
    def live_text(self) -> str:
        return self._live_text

    @property
    def offline_text(self) -> str:
        return self._offline_text

    @property
    def recheck_text(self) -> str:
        return self._recheck_text

    @property
    def error(self) -> Optional[str]:
        return self._error

    @property
    def last_meta(self) -> Dict[str, Any]:
        return dict(self._last_meta)

    @property
    def last_wav(self) -> Optional[np.ndarray]:
        return self._last_wav

    def _record_live_emit(self, text: str) -> Dict[str, Any]:
        rec = {
            "time_s": round(float(self._pending_emit_time_s or 0.0), 3),
            "frame": (
                int(self._pending_emit_frame)
                if self._pending_emit_frame is not None
                else -1
            ),
            "text": text,
        }
        self._live_emits_timed.append(rec)
        return rec

    @property
    def warmup_done(self) -> bool:
        return bool(self._warmup_done)

    def is_live_running(self) -> bool:
        return self._live_thread is not None and self._live_thread.is_alive()

    def join(self, timeout: Optional[float] = None):
        if self._finish_thread is not None:
            self._finish_thread.join(timeout=timeout)
        elif self._live_thread is not None:
            self._live_thread.join(timeout=timeout)

    def abort(self):
        self._abort.set()
        self._closed.set()
        self._feat_closed.set()
        try:
            self._embed_q.put_nowait(None)
        except Exception:  # noqa: BLE001
            pass
        try:
            self._feat_q.put_nowait(None)
        except Exception:  # noqa: BLE001
            pass
        if self._feat_thread is not None and self._feat_thread.is_alive():
            self._feat_thread.join(timeout=5)
        if self._enc_thread is not None and self._enc_thread.is_alive():
            self._enc_thread.join(timeout=5)
        if self._live_thread is not None and self._live_thread.is_alive():
            self._live_thread.join(timeout=5)
        if self._finish_thread is not None and self._finish_thread.is_alive():
            self._finish_thread.join(timeout=5)

    # ── live feature / state machine ──────────────────────────────────

    def _run_live_worker(self):
        try:
            self._run_live_state_machine()
        except Exception as exc:  # noqa: BLE001
            self._error = f"live: {exc}"
            print(f"[adaptive-live] error: {exc}", flush=True)

    def _run_feature_worker(self):
        while not self._abort.is_set():
            progressed = self._prepare_and_enqueue_features_once()
            if not progressed:
                if self._feat_closed.is_set() and not self._pending_pcm():
                    break
                time.sleep(0.005)

    def _run_encoder_worker(self):
        """Drain ``_feat_q`` and encode on a side CUDA stream.

        Only this thread writes ``_padding_cache`` / ``_audio_past`` /
        ``_enc_inited`` when overlap is on. Does not take ``_GENERATION_LOCK``.
        """
        device = self._enc_device
        try:
            if torch.device(device).type == "cuda":
                torch.cuda.set_device(device)
                self._encode_stream = torch.cuda.Stream(device=device)
            while not self._abort.is_set():
                try:
                    first = self._feat_q.get(timeout=0.05)
                except queue.Empty:
                    if self._feat_closed.is_set() and self._feat_q.empty():
                        break
                    continue
                if first is None:
                    if self._feat_closed.is_set() and self._feat_q.empty():
                        break
                    continue
                self._enc_busy = True
                try:
                    self._encode_feat_chunks(first)
                finally:
                    self._enc_busy = False
        except Exception as exc:  # noqa: BLE001
            self._error = f"enc: {exc}"
            print(f"[adaptive-live] encoder error: {exc}", flush=True)
        finally:
            self._enc_done.set()

    def _encode_feat_chunks(self, first: torch.Tensor):
        chunks: List[torch.Tensor] = [first]
        while True:
            try:
                item = self._feat_q.get_nowait()
            except queue.Empty:
                break
            if item is None:
                break
            chunks.append(item)
        stream = self._encode_stream
        ctx = torch.cuda.stream(stream) if stream is not None else nullcontext()
        with ctx:
            # First streaming chunk must init encoder cache alone (delay/prefill).
            if not self._enc_inited:
                self._push_features(chunks[0])
                self._enc_inited = True
                chunks = chunks[1:]
                if not chunks:
                    return
            if not self._warmup_done and self._pad_frames_pushed >= int(self._n_left_pad):
                self._skip_remaining_preheat_pcm()
                return
            batched = torch.cat(chunks, dim=-1)
            self._push_features(batched)

    def _prepare_and_enqueue_features_once(self) -> bool:
        with self._lock:
            pre = int(self._preheat_samples)
            if self._warmup_done:
                if self._prod_started and int(self._prod_next_start) < pre:
                    self._prod_next_start = pre
            if not self._prod_started:
                need = int(self._delay_cfg["num_samples_first_audio_chunk"])
                if self._buffer.shape[0] < need:
                    return False
                audio_chunk = self._buffer[:need].copy()
                is_first = True
            else:
                chunk_samples = int(self._delay_cfg["num_samples_per_audio_chunk"])
                end_idx = int(self._prod_next_start) + chunk_samples
                # Do not encode user PCM until LM has consumed the pad.
                # Otherwise prefetch dumps speech into feat_q and warmup
                # later drops it with leftover zeros.
                if not self._warmup_done and end_idx > pre:
                    return False
                if end_idx > self._buffer.shape[0]:
                    return False
                audio_chunk = self._buffer[int(self._prod_next_start):end_idx].copy()
                is_first = False

        if is_first:
            inputs = self._encode_first_streaming_chunk(
                self.processor,
                audio_chunk,
                ADAPTIVE_PROTOCOL_DELAY_MS,
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                audio_chunk,
                is_streaming=True,
                is_first_audio_chunk=False,
                return_tensors="pt",
            )

        feats = inputs.input_features
        if isinstance(feats, torch.Tensor):
            feats = feats.detach()
            if feats.device.type == "cpu":
                feats = feats.contiguous()
                if self._use_enc_overlap:
                    feats = feats.pin_memory()
        try:
            self._feat_q.put(feats, timeout=0.05)
        except queue.Full:
            return False

        hop = self.processor.feature_extractor.hop_length
        win = self.processor.feature_extractor.win_length
        with self._lock:
            if is_first and not self._prod_started:
                # Same cursor math as StreamingSession / non-prefetch path:
                # extract window is (alt*hop + win), but start advances by alt*hop.
                self._mel_frame_idx = int(
                    self._delay_cfg["num_mel_frames_first_audio_chunk"]
                )
                self._prod_next_start = self._mel_frame_idx * hop - win // 2
                self._prod_started = True
            elif self._prod_started:
                self._mel_frame_idx = int(self._mel_frame_idx or 0)
                self._mel_frame_idx += int(self.processor.audio_length_per_tok)
                self._prod_next_start = self._mel_frame_idx * hop - win // 2
        return True

    def _drain_feature_queue_to_embeds(self):
        """Batch all pending mel chunks into one ``get_audio_features`` call.

        Each chunk is ``[n_mels, T_i]``.  Concatenating along the time axis is
        equivalent to sequential ``use_cache=True`` calls because the Conformer
        is causal (causal conv + causal attention), but it amortises GPU
        kernel-launch overhead across many chunks — critical for RTF≪1.
        """
        chunks: List[torch.Tensor] = []
        while True:
            try:
                item = self._feat_q.get_nowait()
            except queue.Empty:
                break
            if item is None:
                break
            chunks.append(item)
        if not chunks:
            return
        # First streaming chunk must init encoder cache alone (delay/prefill).
        if not self._enc_inited:
            self._push_features(chunks[0])
            self._enc_inited = True
            chunks = chunks[1:]
            if not chunks:
                return
        if not self._warmup_done and self._pad_frames_pushed >= int(self._n_left_pad):
            self._skip_remaining_preheat_pcm()
            return
        batched = torch.cat(chunks, dim=-1)
        self._push_features(batched)

    def _feed_ready_chunks_locked(self):
        """Encode ready audio chunks while keeping lock hold short."""
        vox = _unwrap_voxtral_cg(self.model)
        device = self._enc_device
        dtype = vox.dtype
        hop = self.processor.feature_extractor.hop_length
        win = self.processor.feature_extractor.win_length

        while True:
            if self._abort.is_set():
                break
            with self._lock:
                if not self._started:
                    need = int(self._delay_cfg["num_samples_first_audio_chunk"])
                    if self._buffer.shape[0] < need:
                        return
                    audio_chunk = self._buffer[:need].copy()
                    is_first = True
                else:
                    chunk_samples = int(self._delay_cfg["num_samples_per_audio_chunk"])
                    end_idx = int(self._next_start) + chunk_samples
                    pre = int(self._preheat_samples)
                    if not self._warmup_done and end_idx > pre:
                        return
                    if end_idx > self._buffer.shape[0]:
                        return
                    audio_chunk = self._buffer[int(self._next_start):end_idx].copy()
                    is_first = False

            if is_first:
                inputs = self._encode_first_streaming_chunk(
                    self.processor,
                    audio_chunk,
                    ADAPTIVE_PROTOCOL_DELAY_MS,
                    return_tensors="pt",
                )
            else:
                inputs = self.processor(
                    audio_chunk,
                    is_streaming=True,
                    is_first_audio_chunk=False,
                    return_tensors="pt",
                )
            inputs = inputs.to(device, dtype=dtype)
            self._push_features(inputs.input_features)

            with self._lock:
                if is_first and not self._started:
                    self._mel_frame_idx = int(
                        self._delay_cfg["num_mel_frames_first_audio_chunk"]
                    )
                    self._next_start = self._mel_frame_idx * hop - win // 2
                    self._started = True
                    self._enc_inited = True
                    if self.verbose:
                        print("[adaptive-live] first chunk encoded", flush=True)
                elif self._started:
                    self._mel_frame_idx += self.processor.audio_length_per_tok
                    self._next_start = self._mel_frame_idx * hop - win // 2

    def _begin_gpu_span(self, *, device=None, stream=None):
        """Start a GPU timer without blocking. Pair with ``_end_gpu_span``."""
        dev = torch.device(device) if device is not None else None
        if torch.cuda.is_available() and (dev is None or dev.type == "cuda"):
            ctx = torch.cuda.device(dev) if dev is not None else nullcontext()
            with ctx:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                rec = stream
                if rec is None and dev is not None:
                    rec = torch.cuda.current_stream(device=dev)
                if rec is not None:
                    start.record(rec)
                else:
                    start.record()
            return ("cuda", start, end, rec)
        return ("cpu", time.time(), None, None)

    def _end_gpu_span(self, kind: str, handle):
        mode, start, end, stream = handle
        if mode == "cpu":
            dt = time.time() - start
            if kind == "enc":
                self._enc_wall_s += dt
            else:
                self._lm_wall_s += dt
            return
        if stream is not None:
            end.record(stream)
        else:
            end.record()
        with self._gpu_span_lock:
            self._gpu_spans.append((kind, start, end))

    def _flush_gpu_spans(self):
        """Block once, then fold pending CUDA events into enc/lm totals."""
        synchronize_model_devices(self.model, self._lm_device, self._enc_device)
        with self._gpu_span_lock:
            spans = self._gpu_spans
            self._gpu_spans = []
        for kind, start, end in spans:
            dt = float(start.elapsed_time(end)) / 1000.0
            if kind == "enc":
                self._enc_wall_s += dt
            else:
                self._lm_wall_s += dt

    @torch.inference_mode()
    def _push_features(self, input_features: torch.Tensor):
        n_left = int(self._n_left_pad)
        if not self._warmup_done and self._pad_frames_pushed >= n_left:
            # Extra warmup-zero mels after pad is complete: drop without
            # appending to the streaming encoder cache.
            self._skip_remaining_preheat_pcm()
            return
        vox = _unwrap_voxtral_cg(self.model)
        enc_device = self._enc_device
        if input_features.device != enc_device or input_features.dtype != vox.dtype:
            input_features = input_features.to(
                device=enc_device,
                dtype=vox.dtype,
                non_blocking=input_features.device.type == "cpu",
            )
        handle = self._begin_gpu_span(
            device=enc_device, stream=self._encode_stream,
        )
        with _AUDIO_ENCODER_LOCK:
            out = vox.model.get_audio_features(
                input_features=input_features,
                padding_cache=self._padding_cache,
                past_key_values=self._audio_past,
                use_cache=True,
            )
        self._end_gpu_span("enc", handle)
        self._padding_cache = getattr(out, "padding_cache", None)
        self._audio_past = getattr(out, "past_key_values", None)
        frames = _pooler_frames(out.pooler_output)
        ready_ev = None
        if torch.device(enc_device).type == "cuda":
            rec_stream = self._encode_stream
            with torch.cuda.device(enc_device):
                ready_ev = torch.cuda.Event()
                if rec_stream is not None:
                    ready_ev.record(rec_stream)
                else:
                    ready_ev.record(torch.cuda.current_stream(device=enc_device))
        jumped = False
        for i in range(int(frames.shape[0])):
            if not self._warmup_done and self._pad_frames_pushed >= n_left:
                jumped = True
                break
            self._put_embed(frames[i].detach(), ready_ev)
            if not self._warmup_done:
                self._pad_frames_pushed += 1
                if self._pad_frames_pushed >= n_left:
                    jumped = True
                    break
        if jumped:
            self._skip_remaining_preheat_pcm()

    def _put_embed(self, frame: torch.Tensor, ready_ev: Optional[torch.cuda.Event]):
        item: Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.cuda.Event]]]
        item = (frame, ready_ev) if ready_ev is not None else frame
        while not self._abort.is_set():
            try:
                self._embed_q.put(item, timeout=0.05)
                return
            except queue.Full:
                continue

    def _unwrap_embed(self, item: Any) -> Optional[torch.Tensor]:
        if item is None:
            return None
        ev = None
        if isinstance(item, tuple):
            frame, ev = item[0], item[1] if len(item) > 1 else None
        else:
            frame = item
            ev = getattr(item, "_ready_event", None)
        if ev is not None and torch.cuda.is_available():
            torch.cuda.current_stream(device=self._lm_device).wait_event(ev)
        if isinstance(frame, torch.Tensor) and frame.device != self._lm_device:
            frame = to_device(frame, self._lm_device, non_blocking=True)
        return frame

    def _pump_features_from_live(self):
        """Encode from the live thread only when the encoder worker is off."""
        if self._use_enc_overlap:
            return
        if self.feature_prefetch:
            self._drain_feature_queue_to_embeds()
        else:
            self._feed_ready_chunks_locked()

    def _live_input_exhausted(self) -> bool:
        if not self._closed.is_set():
            return False
        if not self._embed_q.empty() or self._leftover_frames:
            return False
        if self._pending_pcm():
            return False
        if self._use_enc_overlap:
            if not self._feat_q.empty() or self._enc_busy:
                return False
            if (
                self._enc_thread is not None
                and self._enc_thread.is_alive()
                and not self._enc_done.is_set()
            ):
                return False
            return True
        if self.feature_prefetch and not self._feat_q.empty():
            return False
        return True

    def _pending_pcm(self) -> bool:
        with self._lock:
            if self.feature_prefetch:
                if not self._prod_started:
                    return self._buffer.shape[0] >= self._delay_cfg["num_samples_first_audio_chunk"]
                chunk_samples = self._delay_cfg["num_samples_per_audio_chunk"]
                return (self._prod_next_start + chunk_samples) <= self._buffer.shape[0]
            if not self._started:
                return self._buffer.shape[0] >= self._delay_cfg["num_samples_first_audio_chunk"]
            chunk_samples = self._delay_cfg["num_samples_per_audio_chunk"]
            return (self._next_start + chunk_samples) <= self._buffer.shape[0]

    def _wait_embed(self) -> Optional[torch.Tensor]:
        while not self._abort.is_set():
            self._pump_features_from_live()
            try:
                item = self._embed_q.get(timeout=0.05)
                return self._unwrap_embed(item)
            except queue.Empty:
                if self._live_input_exhausted():
                    return None
                time.sleep(0.01)
        return None

    def _drain_listen_embeds(self, max_n: int) -> Optional[List[torch.Tensor]]:
        """Gather up to ``max_n`` LISTEN frames, preferring a full chunk.

        Mid-chunk leftovers are consumed first. Before close we block until a
        full chunk is ready (matches offline ``listen_chunk_size`` behavior).
        After close we return whatever remains so the tail can flush.
        """
        frames: List[torch.Tensor] = []
        while self._leftover_frames and len(frames) < max_n:
            frames.append(self._leftover_frames.pop(0))

        while len(frames) < max_n and not self._abort.is_set():
            self._pump_features_from_live()
            try:
                item = self._embed_q.get(timeout=0.05)
            except queue.Empty:
                # Only allow a partial chunk once input is closed and drained.
                if self._live_input_exhausted() and not self._leftover_frames:
                    break
                if (
                    self._use_enc_overlap
                    and self._enc_done.is_set()
                    and self._embed_q.empty()
                    and not self._leftover_frames
                    and self._error
                ):
                    break
                continue
            frame = self._unwrap_embed(item)
            if frame is None:
                break
            frames.append(frame)

        if not frames:
            return None
        return frames

    @torch.inference_mode()
    def _run_live_state_machine(self):
        vox = _unwrap_voxtral_cg(self.model)
        embed_fn = vox.model.get_input_embeddings()
        bos_id = self.processor.tokenizer.bos_token_id or 1
        eos_id = self.processor.tokenizer.eos_token_id or 2
        device = self._lm_device
        dtype = next(embed_fn.parameters()).dtype
        n_left_pad = int(self._n_left_pad)
        seconds_per_token = _adaptive_seconds_per_token(self.processor)
        tok = self.processor.tokenizer
        listen_chunk = int(self.listen_chunk_size)

        _tok_cache: Dict[int, torch.Tensor] = {}
        bos_emb = None
        w_emb = None

        # ── timing helpers ─────────────────────────────────────────
        def _timed_lm_step(*args, **kwargs):
            handle = self._begin_gpu_span(device=self._lm_device)
            result = _lm_step_embeds(vox, *args, **kwargs)
            self._end_gpu_span("lm", handle)
            return result

        def _emit_rtf_event():
            if self._t_live_start is None:
                return
            self._flush_gpu_spans()
            n_speech = max(self._frame_i - n_left_pad, 0)
            audio_dur = n_speech * seconds_per_token
            total_wall = time.time() - self._t_live_start
            enc = self._enc_wall_s
            lm = self._lm_wall_s
            rtf_total = total_wall / audio_dur if audio_dur > 0.0 else -1.0
            rtf_enc = enc / audio_dur if audio_dur > 0.0 else -1.0
            rtf_lm = lm / audio_dur if audio_dur > 0.0 else -1.0
            self._event_q.put({
                "kind": "rtf",
                "frame": self._frame_i - 1,
                "n_frames": self._frame_i,
                "n_speech_frames": n_speech,
                "audio_s": round(audio_dur, 3),
                "wall_total_s": round(total_wall, 3),
                "wall_enc_s": round(enc, 4),
                "wall_lm_s": round(lm, 4),
                "rtf_total": round(rtf_total, 3),
                "rtf_enc": round(rtf_enc, 4),
                "rtf_lm": round(rtf_lm, 4),
            })

        def _tok_embed(tid: int) -> torch.Tensor:
            emb = _tok_cache.get(tid)
            if emb is None:
                t = torch.tensor([[tid]], device=device, dtype=torch.long)
                emb = embed_fn(t).to(dtype=dtype)
                if len(_tok_cache) < 4096:
                    _tok_cache[tid] = emb
            return emb

        def _listen_decision(logits_row: torch.Tensor):
            argmax_tok = int(torch.argmax(logits_row).item())
            do_emit = argmax_tok == STREAMING_WORD_ID
            return argmax_tok, do_emit

        # Warmup holds the shared model lock. After it, wait for the first
        with _GENERATION_LOCK:
            bos_emb = embed_fn(
                torch.tensor([[bos_id]], device=device, dtype=torch.long)
            ).to(dtype=dtype)
            w_emb = embed_fn(
                torch.tensor([[STREAMING_WORD_ID]], device=device, dtype=torch.long)
            ).to(dtype=dtype)
            _tok_cache[bos_id] = bos_emb
            _tok_cache[STREAMING_WORD_ID] = w_emb
            _logits, past = _lm_step_embeds(vox, bos_emb, None)
            seq_len = 1
            t_warm0 = time.time()
            pad_got = 0
            while pad_got < n_left_pad and not self._abort.is_set():
                need = n_left_pad - pad_got
                frames = self._drain_listen_embeds(need)
                if frames is None:
                    break
                stack = torch.stack(
                    [f.to(device=device, dtype=dtype) for f in frames], dim=0
                ).unsqueeze(0)
                _logits, past = _lm_step_embeds(vox, stack, past)
                pad_got += len(frames)
                seq_len += len(frames)
            self._frame_i = pad_got
            with self._gpu_span_lock:
                self._gpu_spans = []
                self._enc_wall_s = 0.0
                self._lm_wall_s = 0.0
            # Feature/encoder workers may start user PCM only after this flag.
            self._warmup_done = pad_got >= n_left_pad
            warm_s = time.time() - t_warm0
            self._event_q.put({
                "kind": "warmup_done",
                "frame": self._frame_i - 1,
                "n_left_pad": n_left_pad,
                "wall_s": round(warm_s, 3),
                "prefill": False,
                "listen_chunk_size": listen_chunk,
            })
            if self.verbose:
                print(
                    f"[adaptive-live] warmup streaming pad {pad_got}/{n_left_pad} "
                    f"wall={warm_s:.3f}s (listen_chunk_size={listen_chunk})",
                    flush=True,
                )

        pending_frames = self._drain_listen_embeds(listen_chunk)
        if pending_frames is None:
            return

        with _GENERATION_LOCK:
            self._run_live_decode_loop(
                past=past,
                seq_len=seq_len,
                pending_frames=pending_frames,
                _tok_embed=_tok_embed,
                _listen_decision=_listen_decision,
                _timed_lm_step=_timed_lm_step,
                _emit_rtf_event=_emit_rtf_event,
                w_emb=w_emb,
                tok=tok,
                eos_id=eos_id,
                bos_id=bos_id,
                n_left_pad=n_left_pad,
                seconds_per_token=seconds_per_token,
                listen_chunk=listen_chunk,
                device=device,
                dtype=dtype,
            )

    def _run_live_decode_loop(
        self,
        *,
        past: Any,
        seq_len: int,
        pending_frames: List[torch.Tensor],
        _tok_embed: Callable[[int], torch.Tensor],
        _listen_decision: Callable[[torch.Tensor], tuple],
        _timed_lm_step: Callable[..., tuple],
        _emit_rtf_event: Callable[[], None],
        w_emb: torch.Tensor,
        tok: Any,
        eos_id: int,
        bos_id: int,
        n_left_pad: int,
        seconds_per_token: float,
        listen_chunk: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Listen/decode after warmup. Caller holds ``_GENERATION_LOCK``."""
        state = "listen"
        emit_ids: List[int] = []
        next_tok: Optional[int] = None
        decode_count = 0
        decode_limit = int(self.max_decode_tokens)
        steps = 0
        hard_cap = self.max_new_tokens or 100000
        first_frames: Optional[List[torch.Tensor]] = pending_frames

        while steps < hard_cap and not self._abort.is_set():
            steps += 1
            self._pump_features_from_live()

            if state == "listen":
                if first_frames is not None:
                    frames = first_frames
                    first_frames = None
                else:
                    frames = self._drain_listen_embeds(listen_chunk)
                if frames is None:
                    break
                k = len(frames)
                stack = torch.stack(
                    [f.to(device=device, dtype=dtype) for f in frames], dim=0
                ).unsqueeze(0)  # [1, K, H]
                if k == 1:
                    logits, past = _timed_lm_step(stack, past)
                    logits_all = logits  # [1, 1, V]
                else:
                    logits_all, past = _timed_lm_step(
                        stack, past, return_all_logits=True,
                    )

                emit_j: Optional[int] = None
                argmax_tok = 0
                for j in range(k):
                    argmax_tok, do_emit = _listen_decision(logits_all[0, j])
                    if do_emit:
                        emit_j = j
                        break

                if emit_j is None:
                    self._frame_i += k
                    seq_len += k
                    continue

                # Crop KV if we looked past the emit frame inside this chunk.
                keep = seq_len + emit_j + 1
                if emit_j + 1 < k:
                    past = _crop_past_kv(past, keep)
                    # Save remaining frames so they are re-processed after DECODE
                    self._leftover_frames = list(frames[emit_j + 1:])
                seq_len = keep
                self._frame_i += emit_j + 1

                t_s = (self._frame_i - n_left_pad) * seconds_per_token
                self._pending_emit_time_s = float(t_s)
                self._pending_emit_frame = int(self._frame_i - 1)
                self._event_q.put({
                    "kind": "live_emit_start",
                    "frame": self._pending_emit_frame,
                    "time_s": round(float(t_s), 3),
                })
                emit_ids = []
                state = "decode"
                decode_count = 0
                # Keep decode budget stable. Adaptive backlog capping changes
                # early-stop behavior and shifts later emit frames under realtime.
                decode_limit = int(self.max_decode_tokens)
                logits, past = _timed_lm_step(w_emb, past)
                seq_len = _past_seq_len(past)
                next_tok = int(torch.argmax(logits[0, -1]).item())

            if next_tok == eos_id or decode_count >= decode_limit:
                chunk_text = tok.decode(emit_ids, skip_special_tokens=True).strip()
                if chunk_text:
                    self._live_parts.append(chunk_text)
                    self._live_text = join_emit_texts(self._live_parts)
                    rec = self._record_live_emit(chunk_text)
                    self._event_q.put({"kind": "live_emit", **rec})
                _emit_rtf_event()
                emit_ids = []
                state = "listen"
                next_tok = None
                continue

            if next_tok not in (STREAMING_PAD_ID, STREAMING_WORD_ID, bos_id):
                emit_ids.append(next_tok)
                piece = tok.decode([next_tok], skip_special_tokens=True)
                so_far = tok.decode(emit_ids, skip_special_tokens=True)
                self._event_q.put({
                    "kind": "live_decode",
                    "piece": piece,
                    "text": so_far,
                })
            decode_count += 1
            logits, past = _timed_lm_step(_tok_embed(next_tok), past)
            seq_len += 1
            next_tok = int(torch.argmax(logits[0, -1]).item())

        if emit_ids:
            chunk_text = tok.decode(emit_ids, skip_special_tokens=True).strip()
            if chunk_text:
                self._live_parts.append(chunk_text)
                self._live_text = join_emit_texts(self._live_parts)
                rec = self._record_live_emit(chunk_text)
                self._event_q.put({"kind": "live_emit", **rec})
            _emit_rtf_event()
