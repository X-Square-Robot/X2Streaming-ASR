#!/usr/bin/env python3
"""Streaming audio encoder sidecar (Transformers side of the vLLM server).

Protocol is JSON-lines over stdin/stdout. Machine-readable responses are
prefixed with ``ENCODER_JSON:`` so model/library logs can share stdout.

Many live sessions share one loaded model.  Each command carries a ``sid`` and
each session event echoes it back.  Live sessions keep private encoder caches
(``get_audio_features`` calls are serialized by ``_AUDIO_ENCODER_LOCK``).
Only the audio encoder runs here; the LM runs in vLLM.
"""

from __future__ import annotations

import base64
import json
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_DEMO_DIR = _REPO_ROOT / "demo"
for path in (_REPO_ROOT, _DEMO_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from demo.engine_adaptive import AdaptiveSession  # noqa: E402
from models import build_model  # noqa: E402
from models.voxtral_asr.devices import (  # noqa: E402
    empty_cache_model_devices,
    enc_device_of,
    lm_device_of,
)

PREFIX = "ENCODER_JSON:"
_WRITE_LOCK = threading.Lock()


def _send(payload: dict[str, Any]) -> None:
    line = PREFIX + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    with _WRITE_LOCK:
        print(line, flush=True)


def _encode_frame(frame: torch.Tensor) -> tuple[str, float]:
    started = time.perf_counter()
    bits = (
        frame.detach()
        .to(device="cpu", dtype=torch.bfloat16)
        .contiguous()
        .view(torch.uint16)
        .numpy()
        .tobytes()
    )
    encoded = base64.b64encode(bits).decode("ascii")
    return encoded, (time.perf_counter() - started) * 1000.0


class EncoderOnlySession(AdaptiveSession):
    def __init__(
        self,
        model: Any,
        processor: Any,
        on_embedding: Callable[[torch.Tensor, int], None],
        *,
        sid: str,
        feature_prefetch: bool,
        gpu_encode_overlap: bool,
    ) -> None:
        self.sid = sid
        self.serialize_latencies_ms: list[float] = []
        super().__init__(
            model,
            processor,
            enable_live=True,
            enable_offline=False,
            listen_chunk_size=1,
            feature_prefetch=feature_prefetch,
            gpu_encode_overlap=gpu_encode_overlap,
            verbose=False,
        )
        self._on_embedding = on_embedding

    def set_language_id(self, language_id: Any) -> bool:
        # The sidecar only runs the audio encoder; vLLM adds the language tag.
        # Sessions share one model, so do not stamp its global language id.
        self.language_id = 0
        return False

    def _run_live_state_machine(self) -> None:
        # Drain the exact online encoder used by the HF demo, but do not run
        # its LM. The first n_left_pad rows are protocol silence embeddings.
        if torch.cuda.is_available():
            torch.cuda.set_device(self._lm_device)
        print(
            f"[adaptive-encoder] sid={self.sid} live start n_left_pad={self._n_left_pad} "
            f"preheat={self._preheat_samples}",
            flush=True,
        )
        count = 0
        while count < int(self._n_left_pad) and not self._abort.is_set():
            frame = self._wait_embed()
            if frame is None:
                print(
                    f"[adaptive-encoder] sid={self.sid} pad wait returned None",
                    flush=True,
                )
                return
            self._on_embedding(frame, count)
            count += 1

        self._frame_i = count
        self._warmup_done = count >= int(self._n_left_pad)
        self._skip_remaining_preheat_pcm()
        self._flush_gpu_spans()
        with self._gpu_span_lock:
            self._gpu_spans = []
            self._enc_wall_s = 0.0
            self._lm_wall_s = 0.0
        print(
            f"[adaptive-encoder] sid={self.sid} warmup embeds={count}",
            flush=True,
        )

        while not self._abort.is_set():
            frame = self._wait_embed()
            if frame is None:
                print(
                    f"[adaptive-encoder] sid={self.sid} live drain done "
                    f"frames={self._frame_i}",
                    flush=True,
                )
                return
            self._on_embedding(frame, self._frame_i)
            self._frame_i += 1


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lora", default="")
    parser.add_argument(
        "--feature-prefetch",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--gpu-encode-overlap",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    lora_path = args.lora.strip()
    if lora_path and not Path(lora_path).exists():
        print(f"[adaptive-encoder] lora missing, skip: {lora_path}", flush=True)
        lora_path = ""
    wrapper = build_model("voxtral_asr").load_for_inference(
        args.model,
        device=args.device,
        # Same preparation as the adaptive checkpoint everywhere else; it only
        # touches the LM, which this sidecar never runs.
        adaptive_latency=True,
        lora_path=lora_path,
        language_tag={"enabled": True, "require_checkpoint_weights": False},
    )
    model = wrapper["model"].eval()
    processor = wrapper["processor"]
    sessions: dict[str, EncoderOnlySession] = {}
    state_lock = threading.Lock()
    if lora_path:
        print(f"[adaptive-encoder] lora={lora_path}", flush=True)

    def maybe_empty_cache() -> None:
        # empty_cache stalls the device; skip it while other work is in flight.
        with state_lock:
            idle = not sessions
        if idle:
            empty_cache_model_devices(
                model, lm_device_of(model), enc_device_of(model)
            )

    def pop_session(sid: str) -> EncoderOnlySession | None:
        with state_lock:
            return sessions.pop(sid, None)

    def finish_session(sid: str, session: EncoderOnlySession) -> None:
        try:
            session.finish_async()
            session.join(timeout=60)
            if session.is_live_running():
                session.abort()
                session.join(timeout=5)
            session._flush_gpu_spans()
            latencies = session.serialize_latencies_ms
            _send(
                {
                    "type": "finished",
                    "sid": sid,
                    "error": session.error,
                    "frames": int(session._frame_i),
                    "enc_wall_s": round(float(session._enc_wall_s), 4),
                    "serialize_total_s": round(sum(latencies) / 1000.0, 4),
                    "serialize_mean_ms": round(
                        statistics.mean(latencies), 4
                    ) if latencies else 0.0,
                    "serialize_p90_ms": round(
                        sorted(latencies)[
                            min(len(latencies) - 1, int(len(latencies) * 0.9))
                        ],
                        4,
                    ) if latencies else 0.0,
                }
            )
        except Exception as exc:  # noqa: BLE001
            _send({"type": "error", "sid": sid, "message": str(exc)})
        maybe_empty_cache()

    def abort_session(sid: str, session: EncoderOnlySession | None) -> None:
        if session is not None:
            session.abort()
        _send({"type": "aborted", "sid": sid})
        maybe_empty_cache()

    _send({"type": "ready"})
    for raw in sys.stdin:
        sid = None
        try:
            message = json.loads(raw)
            kind = message.get("type")
            sid = message.get("sid")
            if kind == "start":
                if not sid:
                    raise ValueError("start requires a sid")
                stale = pop_session(sid)
                if stale is not None:
                    stale.abort()

                serialize_latencies_ms: list[float] = []

                def on_embedding(
                    frame: torch.Tensor,
                    index: int,
                    sid: str = sid,
                    latencies: list[float] = serialize_latencies_ms,
                ) -> None:
                    encoded, serialize_ms = _encode_frame(frame)
                    latencies.append(serialize_ms)
                    _send(
                        {
                            "type": "embedding",
                            "sid": sid,
                            "index": index,
                            "shape": list(frame.shape),
                            "bf16": encoded,
                            "serialize_ms": round(serialize_ms, 4),
                            "worker_send_ns": time.perf_counter_ns(),
                        }
                    )

                session = EncoderOnlySession(
                    model,
                    processor,
                    on_embedding,
                    sid=sid,
                    feature_prefetch=args.feature_prefetch,
                    gpu_encode_overlap=args.gpu_encode_overlap,
                )
                session.serialize_latencies_ms = serialize_latencies_ms
                with state_lock:
                    sessions[sid] = session
                session.start()
                _send({
                    "type": "started",
                    "sid": sid,
                    "feature_prefetch": args.feature_prefetch,
                    "gpu_encode_overlap": args.gpu_encode_overlap,
                    "active_sessions": len(sessions),
                })
            elif kind == "append":
                session = sessions.get(sid)
                if session is None:
                    raise RuntimeError(f"encoder session {sid!r} not started")
                pcm = base64.b64decode(message["pcm16"])
                wav = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
                session.append_pcm(wav)
            elif kind == "finish":
                session = pop_session(sid)
                if session is None:
                    raise RuntimeError(f"encoder session {sid!r} not started")
                # Draining can take a while; keep serving other sessions.
                threading.Thread(
                    target=finish_session, args=(sid, session), daemon=True
                ).start()
            elif kind == "abort":
                threading.Thread(
                    target=abort_session,
                    args=(sid, pop_session(sid)),
                    daemon=True,
                ).start()
            elif kind == "shutdown":
                with state_lock:
                    live = list(sessions.values())
                    sessions.clear()
                for session in live:
                    session.abort()
                _send({"type": "shutdown"})
                return
            else:
                raise ValueError(f"unknown command: {kind!r}")
        except Exception as exc:  # noqa: BLE001
            event = {"type": "error", "message": str(exc)}
            if sid is not None:
                event["sid"] = sid
            _send(event)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
