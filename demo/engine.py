"""Live streaming ASR session for Voxtral Realtime (transformers backend)."""

from __future__ import annotations

import queue
import threading
from typing import Optional

import numpy as np
from mistral_common.protocol.instruct.chunk import RawAudio
from mistral_common.protocol.transcription.request import StreamingMode, TranscriptionRequest
from mistral_common.tokens.tokenizers.audio import Audio
from transformers import TextIteratorStreamer
from transformers.feature_extraction_utils import BatchFeature

MIN_DELAY_MS = 80
MAX_DELAY_MS = 2400
DELAY_STEP_MS = 80
DEFAULT_DELAY_MS = 480

# The model is loaded once and shared across all websocket sessions. Running two
# time corrupts internal state, so we serialize with a process-wide lock.
_GENERATION_LOCK = threading.Lock()
# does *not* take this lock so encoder∥LM overlap stays legal.
_AUDIO_ENCODER_LOCK = threading.Lock()


def validate_delay_ms(delay_ms: int) -> int:
    delay_ms = int(delay_ms)
    if delay_ms < MIN_DELAY_MS or delay_ms > MAX_DELAY_MS:
        raise ValueError(
            f"delay_ms must be in [{MIN_DELAY_MS}, {MAX_DELAY_MS}], got {delay_ms}"
        )
    if delay_ms % DELAY_STEP_MS != 0:
        raise ValueError(f"delay_ms must be a multiple of {DELAY_STEP_MS}, got {delay_ms}")
    return delay_ms


def get_streaming_delay_config(processor, delay_ms: int) -> dict:
    """Delay-dependent chunk sizes (processor defaults are fixed at 480 ms)."""
    delay_ms = validate_delay_ms(delay_ms)
    cfg = processor.mistral_common_audio_config
    hop_length = processor.feature_extractor.hop_length
    win_length = processor.feature_extractor.win_length

    num_delay_tokens = cfg.get_num_delay_tokens(delay_ms)
    num_prefill_tokens = num_delay_tokens + 1
    num_mel_frames_first = num_prefill_tokens * cfg.audio_length_per_tok
    num_samples_first = (num_mel_frames_first - 1) * hop_length + win_length // 2
    num_samples_per_chunk = cfg.audio_length_per_tok * hop_length + win_length
    pad_samples = cfg.n_right_pad_tokens(delay_ms) * cfg.raw_audio_length_per_tok

    return {
        "delay_ms": delay_ms,
        "num_delay_tokens": num_delay_tokens,
        "num_mel_frames_first_audio_chunk": num_mel_frames_first,
        "num_samples_first_audio_chunk": num_samples_first,
        "num_samples_per_audio_chunk": num_samples_per_chunk,
        "pad_samples": pad_samples,
    }


def encode_first_streaming_chunk(processor, audio: np.ndarray, delay_ms: int, *, return_tensors: str = "pt"):
    """Encode the first streaming chunk with explicit target delay τ."""
    delay_ms = validate_delay_ms(delay_ms)
    sr = processor.feature_extractor.sampling_rate
    audio_obj = Audio(
        audio_array=np.asarray(audio, dtype=np.float32),
        sampling_rate=sr,
        format="wav",
    )
    request = TranscriptionRequest(
        audio=RawAudio.from_audio(audio_obj),
        streaming=StreamingMode.ONLINE,
        target_streaming_delay_ms=delay_ms,
        language=None,
    )
    tokenized = processor.tokenizer.tokenizer.encode_transcription(request)
    audio_arrays = [a.audio_array for a in tokenized.audios]
    text_encoding = processor.tokenizer(
        tokenized.tokens,
        add_special_tokens=False,
        return_tensors=return_tensors,
    )
    audio_encoding = processor.feature_extractor(
        audio_arrays,
        center=True,
        sampling_rate=sr,
    )
    num_delay_tokens = processor.mistral_common_audio_config.get_num_delay_tokens(delay_ms)
    encoding = {**text_encoding, **audio_encoding, "num_delay_tokens": num_delay_tokens}
    return BatchFeature(data=encoding, tensor_type=return_tensors)


class StreamingSession:
    """Feed microphone/file PCM incrementally and stream partial transcripts."""

    def __init__(self, model, processor, delay_ms: int = DEFAULT_DELAY_MS):
        self.model = model
        self.processor = processor
        self.sr = processor.feature_extractor.sampling_rate
        self.delay_ms = validate_delay_ms(delay_ms)
        self._delay_cfg = get_streaming_delay_config(processor, self.delay_ms)

        self._buffer = np.zeros(0, dtype=np.float32)
        self._feature_q: queue.Queue = queue.Queue()
        self._closed = threading.Event()
        self._started = False
        self._first_features = None
        self._first_inputs = None
        self._mel_frame_idx: Optional[int] = None
        self._next_start = 0
        self._lock = threading.Lock()

        self._gen_thread: Optional[threading.Thread] = None
        self._streamer: Optional[TextIteratorStreamer] = None

    def set_delay_ms(self, delay_ms: int):
        self.delay_ms = validate_delay_ms(delay_ms)
        self._delay_cfg = get_streaming_delay_config(self.processor, self.delay_ms)

    def reset(self):
        self._buffer = np.zeros(0, dtype=np.float32)
        self._feature_q = queue.Queue()
        self._closed.clear()
        self._started = False
        self._first_features = None
        self._first_inputs = None
        self._mel_frame_idx = None
        self._next_start = 0
        self._gen_thread = None
        self._streamer = None

    def append_pcm(self, pcm: np.ndarray):
        pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
        with self._lock:
            self._buffer = np.concatenate([self._buffer, pcm])
            if not self._started:
                self._try_start_locked()
            else:
                self._feed_ready_chunks_locked()

    def finish(self):
        with self._lock:
            pad = self._delay_cfg["pad_samples"]
            self._buffer = np.pad(self._buffer, (0, pad))
            if not self._started:
                self._try_start_locked()
            self._feed_ready_chunks_locked()
            self._closed.set()
            self._feature_q.put(None)

    def _try_start_locked(self):
        need = self._delay_cfg["num_samples_first_audio_chunk"]
        if self._buffer.shape[0] < need:
            return

        first_audio = self._buffer[:need]
        inputs = encode_first_streaming_chunk(
            self.processor,
            first_audio,
            self.delay_ms,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)
        self._first_inputs = inputs
        self._first_features = inputs.input_features

        self._mel_frame_idx = self._delay_cfg["num_mel_frames_first_audio_chunk"]
        hop = self.processor.feature_extractor.hop_length
        win = self.processor.feature_extractor.win_length
        self._next_start = self._mel_frame_idx * hop - win // 2

        self._started = True
        self._start_generation_locked()
        self._feed_ready_chunks_locked()

    def _feed_ready_chunks_locked(self):
        if not self._started:
            return

        hop = self.processor.feature_extractor.hop_length
        win = self.processor.feature_extractor.win_length
        chunk_samples = self._delay_cfg["num_samples_per_audio_chunk"]

        while True:
            end_idx = self._next_start + chunk_samples
            if end_idx > self._buffer.shape[0]:
                break

            chunk = self._buffer[self._next_start:end_idx]
            inputs = self.processor(
                chunk,
                is_streaming=True,
                is_first_audio_chunk=False,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device, dtype=self.model.dtype)
            self._feature_q.put(inputs.input_features)

            self._mel_frame_idx += self.processor.audio_length_per_tok
            self._next_start = self._mel_frame_idx * hop - win // 2

    def _feature_generator(self):
        yield self._first_features
        while True:
            item = self._feature_q.get()
            if item is None:
                return
            yield item

    def _start_generation_locked(self):
        self._streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        kwargs = {
            "input_ids": self._first_inputs.input_ids,
            "input_features": self._feature_generator(),
            "num_delay_tokens": self._first_inputs.num_delay_tokens,
            "streamer": self._streamer,
        }
        self._gen_thread = threading.Thread(
            target=self._run_generation,
            args=(kwargs,),
            daemon=True,
        )
        self._gen_thread.start()

    def _run_generation(self, kwargs: dict):
        # Serialize across sessions and clear any leaked streaming state on the
        # shared model before each run.
        with _GENERATION_LOCK:
            if hasattr(self.model, "_stream_exhausted"):
                self.model._stream_exhausted = False
            try:
                self.model.generate(**kwargs)
            except Exception as exc:  # noqa: BLE001 - keep the session alive
                print(f"[engine] generation error: {exc}")
            finally:
                if self._streamer is not None:
                    # Unblock any consumer still iterating the streamer.
                    self._streamer.end()

    def abort(self):
        """Stop an in-flight generation and reclaim its worker thread.

        Called on disconnect / restart so an interrupted session does not leave
        a generation thread blocked inside ``model.generate`` holding the model.
        """
        with self._lock:
            self._closed.set()
            if self._started:
                self._feature_q.put(None)
        thread = self._gen_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=10)
        self.reset()

    @property
    def streamer(self) -> Optional[TextIteratorStreamer]:
        return self._streamer

    def join(self, timeout: Optional[float] = None):
        if self._gen_thread is not None:
            self._gen_thread.join(timeout=timeout)


def pcm16_to_float32(pcm16: bytes) -> np.ndarray:
    arr = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32)
    return arr / 32768.0


def load_wav_float32(path: str, target_sr: int) -> np.ndarray:
    audio = Audio.from_file(path, strict=False)
    audio.resample(target_sr)
    return np.asarray(audio.audio_array, dtype=np.float32)
