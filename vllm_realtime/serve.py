#!/usr/bin/env python3
"""Adaptive streaming ASR server on vLLM (true incremental decoding).

- vLLM text LM with one persistent KV request per session; audio frames enter
  as prompt embeddings, LISTEN/DECODE runs without re-prefilling.
- Online audio encoder in a sidecar process (Transformers), by default on the
  same GPU as the LM.
- ``/v1/realtime``: the WebSocket protocol used by ``demo/app.py``.

Concurrent WebSocket sessions (up to --max-sessions) each own one persistent
vLLM request and one encoder stream multiplexed over the shared sidecar.
The final result is the streaming (live) transcript; there is no offline
full-utterance pass. Launch through ``vllm_realtime/serve.sh``.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
from collections import deque
import json
import os
import statistics
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

_HERE = Path(__file__).resolve().parent

if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
# vLLM's EngineCore subprocess imports the model class and scheduler by name.
_pythonpath = str(_HERE)
os.environ["PYTHONPATH"] = (
    f"{_pythonpath}{os.pathsep}{os.environ['PYTHONPATH']}"
    if os.environ.get("PYTHONPATH")
    else _pythonpath
)

from encoder_bridge import EncoderBridge, EncoderStream  # noqa: E402
from session import (  # noqa: E402
    N_LEFT_PAD,
    SECONDS_PER_FRAME,
    PersistentEmbedsSession,
    RestoreFrame,
    _append_emit_text,
)


def _resolved(path: str | Path) -> str:
    text = str(path or "").strip()
    return str(Path(text).expanduser().resolve()) if text else ""


def _env_truthy(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _latency_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "n": 0,
            "mean": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
        }
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 4),
        "p50": round(statistics.median(values), 4),
        "p90": round(
            ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))],
            4,
        ),
        "p95": round(
            ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
            4,
        ),
    }


def _validate_export_alignment(
    text_model: str | Path,
    encoder_model: str | Path,
    encoder_lora: str | Path,
) -> None:
    """Fail fast when the vLLM text LM and the sidecar checkpoint differ."""
    meta_path = Path(text_model) / "export_meta.json"
    if not meta_path.exists():
        print(
            f"[adaptive-vllm-true] WARNING: {meta_path} is missing; "
            "cannot verify LM/sidecar weight alignment",
            flush=True,
        )
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected_source = _resolved(meta.get("source", ""))
    expected_lora = _resolved(meta.get("lora", ""))
    actual_source = _resolved(encoder_model)
    actual_lora = _resolved(encoder_lora)
    mismatches = []
    if expected_source and expected_source != actual_source:
        mismatches.append(
            f"base export={expected_source!r} sidecar={actual_source!r}"
        )
    if "lora" in meta and expected_lora != actual_lora:
        mismatches.append(
            f"lora export={expected_lora or '(none)'!r} "
            f"sidecar={actual_lora or '(none)'!r}"
        )
    if mismatches:
        raise RuntimeError(
            "vLLM text LM and sidecar weights do not match: "
            + "; ".join(mismatches)
            + ". Re-export MODEL or set ENCODER_MODEL/ENCODER_LORA consistently."
        )


def _load_embedding_table(model_dir: Path):
    import torch
    from safetensors import safe_open

    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    name = "model.embed_tokens.weight"
    shard = model_dir / index["weight_map"][name]
    with safe_open(str(shard), framework="pt") as handle:
        return handle.get_tensor(name)


_LANG_NAME_TO_ID = {
    "zh": 0,
    "en": 1,
    "cs": 2,
    "zh-cn": 0,
    "chinese": 0,
    "english": 1,
    "codeswitch": 2,
    "code-switch": 2,
    "code_switch": 2,
}


def _resolve_language_id(value) -> int:
    if value is None or value == "":
        return 0
    if isinstance(value, bool):
        raise ValueError(f"language id must not be bool, got {value!r}")
    if isinstance(value, int):
        return int(value)
    return _LANG_NAME_TO_ID.get(str(value).strip().lower(), 0)


def _language_id_name(language_id) -> str:
    lid = _resolve_language_id(language_id)
    if lid == 1:
        return "en"
    if lid == 2:
        return "cs"
    return "zh"


def _pick_language_tag_weight(state, hidden_size: int):
    import torch

    for key, val in state.items():
        if "language_tag_embed" not in key or not str(key).endswith("weight"):
            continue
        if not isinstance(val, torch.Tensor) or val.ndim != 2:
            continue
        if int(val.shape[-1]) != int(hidden_size):
            continue
        return val.detach().cpu()
    return None


def _iter_weight_states(root: str):
    import torch

    if not root or not os.path.isdir(root):
        return
    st = os.path.join(root, "model.safetensors")
    adapter = os.path.join(root, "adapter_model.safetensors")
    from safetensors.torch import load_file

    if os.path.isfile(st):
        yield load_file(st, device="cpu")
        return
    if os.path.isfile(adapter):
        yield load_file(adapter, device="cpu")
        return
    idx = os.path.join(root, "model.safetensors.index.json")
    if os.path.isfile(idx):
        with open(idx, "r", encoding="utf-8") as handle:
            weight_map = json.load(handle).get("weight_map", {})
        shards = sorted(
            {
                name
                for key, name in weight_map.items()
                if "language_tag_embed" in key
            }
        )
        for shard in shards:
            path = os.path.join(root, shard)
            if os.path.isfile(path):
                yield load_file(path, device="cpu")
        return
    for name in ("pytorch_model.bin", "adapter_model.bin"):
        path = os.path.join(root, name)
        if not os.path.isfile(path):
            continue
        try:
            yield torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            yield torch.load(path, map_location="cpu")
        return


def _load_language_tag_table(hidden_size: int, model_path: str, lora_path: str):
    """Load language_tag_embed[3, H] without importing the repo's models package."""
    import torch

    table = None
    source = None
    for state in _iter_weight_states(model_path) or ():
        picked = _pick_language_tag_weight(state, hidden_size)
        if picked is not None:
            table = picked
            source = "checkpoint"
            break
    adapter = (lora_path or "").strip()
    if adapter:
        for state in _iter_weight_states(adapter) or ():
            picked = _pick_language_tag_weight(state, hidden_size)
            if picked is not None:
                table = picked
                source = "adapter"
                break
    if table is None:
        print(
            "[adaptive-vllm-true] language_tag weights missing; "
            "UI language pills will not change decoding",
            flush=True,
        )
        return None
    print(f"[adaptive-vllm-true] language_tag source={source}", flush=True)
    return table.to(torch.bfloat16).contiguous()


def _clamp_gpu_mem(requested: float, headroom: float = 0.02) -> float:
    """Clamp without importing torch, so vLLM can still fork EngineCore."""
    try:
        import subprocess

        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
                "-i",
                os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0],
            ],
            text=True,
        ).strip()
        free_mib, total_mib = [float(x) for x in raw.split(",")]
        free_frac = free_mib / max(total_mib, 1.0)
        capped = max(0.05, min(requested, free_frac - headroom))
        if capped + 1e-6 < requested:
            print(
                f"[adaptive-vllm-true] GPU0 free={free_frac:.3f}; "
                f"clamping gpu_memory_utilization {requested:.3f} → {capped:.3f}",
                flush=True,
            )
        return capped
    except Exception as exc:  # noqa: BLE001
        print(f"[adaptive-vllm-true] mem probe failed ({exc})", flush=True)
        return requested


def _load_engine(app: FastAPI) -> None:
    from vllm import ModelRegistry
    from vllm.config import CompilationConfig
    from vllm.config.compilation import CUDAGraphMode
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.tokenizers import get_tokenizer

    from streaming_async_llm import StreamingEmbedsAsyncLLM

    ModelRegistry.register_model(
        "AdaptiveTextMistralForCausalLM",
        "adaptive_text_model:AdaptiveTextMistralForCausalLM",
    )
    gpu_mem = _clamp_gpu_mem(float(app.state.gpu_mem))
    cudagraph = not bool(app.state.enforce_eager)
    # On a shared GPU, vLLM's profiler charges other processes' allocation
    # swings to this engine; an explicit KV size skips that estimate.
    kv_cache_gib = float(app.state.kv_cache_gib)
    kv_cache_bytes = int(kv_cache_gib * (1 << 30)) if kv_cache_gib > 0 else None
    print(
        f"[adaptive-vllm-true] Loading text LM {app.state.model_path} "
        f"(gpu_mem={gpu_mem:.3f}, kv_cache_gib={kv_cache_gib or 'auto'}, "
        f"cudagraph={cudagraph})",
        flush=True,
    )
    app.state.engine = StreamingEmbedsAsyncLLM.from_engine_args(
        AsyncEngineArgs(
            model=str(app.state.model_path),
            skip_tokenizer_init=True,
            enable_prompt_embeds=True,
            enforce_eager=not cudagraph,
            hf_overrides={"architectures": ["MistralForCausalLM"]},
            compilation_config=CompilationConfig(
                cudagraph_mode=(
                    CUDAGraphMode.PIECEWISE if cudagraph else CUDAGraphMode.NONE
                )
            ),
            max_model_len=int(app.state.max_model_len),
            gpu_memory_utilization=gpu_mem,
            kv_cache_memory_bytes=kv_cache_bytes,
            scheduler_cls="streaming_embed_scheduler.StreamingEmbedsScheduler",
        )
    )
    app.state.tokenizer = get_tokenizer(
        str(app.state.tokenizer_path), tokenizer_mode="mistral"
    )
    app.state.table = _load_embedding_table(Path(app.state.model_path))
    hidden = int(app.state.table.shape[-1])
    app.state.language_tag_table = _load_language_tag_table(
        hidden,
        str(getattr(app.state, "encoder_model", "") or ""),
        str(getattr(app.state, "encoder_lora", "") or ""),
    )
    print("[adaptive-vllm-true] vLLM text LM ready", flush=True)


async def _start_encoder(app: FastAPI) -> None:
    bridge = EncoderBridge(
        python=str(app.state.encoder_python),
        worker=_HERE / "encoder_worker.py",
        model=str(app.state.encoder_model),
        gpu=str(app.state.encoder_gpu),
        lora=str(getattr(app.state, "encoder_lora", "") or ""),
        feature_prefetch=bool(app.state.encoder_feature_prefetch),
        gpu_encode_overlap=bool(app.state.encoder_gpu_overlap),
    )
    print(
        f"[adaptive-vllm-true] starting encoder sidecar "
        f"gpu={app.state.encoder_gpu} model={app.state.encoder_model} "
        f"lora={getattr(app.state, 'encoder_lora', '') or '(none)'}",
        flush=True,
    )
    await bridge.start()
    app.state.encoder = bridge
    print("[adaptive-vllm-true] encoder sidecar ready", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _start_encoder(app)
    _load_engine(app)
    yield
    encoder = getattr(app.state, "encoder", None)
    if encoder is not None:
        await encoder.shutdown()
    engine = getattr(app.state, "engine", None)
    if engine is not None and hasattr(engine, "shutdown"):
        try:
            engine.shutdown()
        except Exception:  # noqa: BLE001
            pass


app = FastAPI(title="Voxtral Adaptive Realtime (true vLLM)", lifespan=lifespan)
_active_sessions = 0


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "mode": "adaptive_vllm",
        "inference": "vllm_true_stream",
        "model": getattr(app.state, "model_path", ""),
        "encoder_model": getattr(app.state, "encoder_model", ""),
        "encoder_lora": getattr(app.state, "encoder_lora", ""),
        "encoder_gpu": getattr(app.state, "encoder_gpu", ""),
        "encoder_feature_prefetch": getattr(
            app.state, "encoder_feature_prefetch", False
        ),
        "encoder_gpu_overlap": getattr(
            app.state, "encoder_gpu_overlap", False
        ),
        "rollover_seconds": getattr(app.state, "rollover_seconds", 0.0),
        "rollover_context_seconds": getattr(
            app.state, "rollover_context_seconds", 0.0
        ),
        "rollover_max_kv_steps": getattr(
            app.state, "rollover_max_kv_steps", 0
        ),
        "active_sessions": _active_sessions,
        "max_sessions": getattr(app.state, "max_sessions", 0),
    }


@dataclass(frozen=True)
class _CommitRecord:
    global_frame: int
    text: str
    token_ids: tuple[int, ...]


def _make_restore_frames(
    frame_history: list[tuple[int, Any]],
    commits: list[_CommitRecord],
) -> list[RestoreFrame]:
    available = {frame_no for frame_no, _ in frame_history}
    commits_by_frame: dict[int, list[tuple[int, ...]]] = {}
    for record in commits:
        if record.global_frame in available:
            commits_by_frame.setdefault(record.global_frame, []).append(
                record.token_ids
            )
    return [
        RestoreFrame(
            embedding=embedding,
            commits=tuple(commits_by_frame.get(frame_no, ())),
        )
        for frame_no, embedding in frame_history
    ]


async def _pump_encoder(
    session: PersistentEmbedsSession,
    stream: EncoderStream,
    *,
    on_embedding: Callable[[Any, int], Awaitable[None]] | None = None,
) -> None:
    while True:
        event = await stream.events.get()
        kind = event.get("type")
        if kind == "embedding":
            enc_wall_s = event.get("enc_wall_s")
            embedding = EncoderBridge.decode_embedding(event)
            index = int(event.get("index", -1))
            if on_embedding is not None and index >= N_LEFT_PAD:
                await on_embedding(embedding, index)
            await session.feed_audio(
                embedding,
                enc_wall_s=float(enc_wall_s) if enc_wall_s is not None else None,
                timing={
                    key: event[key]
                    for key in (
                        "worker_send_ns",
                        "serialize_ms",
                        "_ipc_ms",
                        "_bridge_decode_ms",
                    )
                    if key in event
                },
            )
        elif kind == "finished":
            if event.get("error"):
                session.error = str(event["error"])
            enc_wall_s = event.get("enc_wall_s")
            if enc_wall_s is not None:
                session.enc_wall_s = float(enc_wall_s)
            session.encoder_stats = {
                key: event[key]
                for key in (
                    "frames",
                    "serialize_total_s",
                    "serialize_mean_ms",
                    "serialize_p90_ms",
                )
                if key in event
            }
            await session.finish()
            return
        elif kind == "error":
            session.error = str(event.get("message", "encoder error"))
            await session.abort()
            raise RuntimeError(session.error)


async def _run_realtime(ws: WebSocket) -> None:
    await ws.send_json(
        {
            "type": "session.created",
            "mode": "adaptive_vllm",
            "inference": "vllm_true_stream",
        }
    )

    model_name = app.state.model_path
    session: PersistentEmbedsSession | None = None
    pump: asyncio.Task | None = None
    enc_stream: EncoderStream | None = None
    deltas: list[str] = []
    pcm_tail = bytearray()
    pcm_tail_limit = int(app.state.rollover_context_seconds * 16000 * 2)
    context_frame_limit = max(
        1, int(round(app.state.rollover_context_seconds / SECONDS_PER_FRAME))
    )
    frame_history: deque[tuple[int, Any]] = deque(maxlen=context_frame_limit)
    recent_commits: deque[_CommitRecord] = deque()
    committed_text = ""
    global_audio_frames = 0
    epoch_global_start_frame = 0
    epoch_pcm_bytes = 0
    rollover_count = 0
    rollover_summaries: list[dict] = []
    enable_live = True
    language_id = int(_resolve_language_id("zh"))

    def _apply_language(payload: dict) -> bool:
        nonlocal language_id
        raw = payload.get("language", payload.get("language_id"))
        if raw is None or raw == "":
            return False
        new_id = int(_resolve_language_id(raw))
        current = (
            int(session.language_id) if session is not None else language_id
        )
        changed = new_id != current
        language_id = new_id
        if session is not None:
            session.set_language_id(language_id)
        return changed

    async def teardown_session() -> None:
        nonlocal session, pump, enc_stream, epoch_pcm_bytes
        if pump is not None and not pump.done():
            pump.cancel()
            await asyncio.wait({pump}, timeout=2.0)
        pump = None
        if session is not None:
            try:
                await asyncio.wait_for(session.abort(), timeout=5.0)
            except Exception:  # noqa: BLE001
                pass
            session = None
        if enc_stream is not None:
            try:
                await asyncio.wait_for(enc_stream.abort(), timeout=2.0)
            except Exception:  # noqa: BLE001
                enc_stream.close()
            enc_stream = None
        epoch_pcm_bytes = 0

    async def on_embedding(embedding: Any, _index: int) -> None:
        nonlocal global_audio_frames
        global_audio_frames += 1
        # EncoderBridge already materializes a private CPU tensor.  Keeping the
        # last few hundred frames is cheap and avoids re-encoding old PCM.
        frame_history.append((global_audio_frames, embedding))
        earliest = frame_history[0][0] if frame_history else global_audio_frames
        while recent_commits and recent_commits[0].global_frame < earliest:
            recent_commits.popleft()

    def reset_utterance() -> None:
        # One WebSocket carries many recordings; nothing may leak from the
        # previous one into the next transcript or its rollover context.
        nonlocal committed_text, global_audio_frames, epoch_global_start_frame
        nonlocal epoch_pcm_bytes, rollover_count
        deltas.clear()
        committed_text = ""
        frame_history.clear()
        recent_commits.clear()
        pcm_tail.clear()
        global_audio_frames = 0
        epoch_global_start_frame = 0
        epoch_pcm_bytes = 0
        rollover_count = 0
        rollover_summaries.clear()

    def build_restore_frames() -> list[RestoreFrame]:
        return _make_restore_frames(
            list(frame_history),
            list(recent_commits),
        )

    async def on_delta(text: str, extra: dict) -> None:
        nonlocal committed_text
        # Epoch timestamps are intentionally captured at the callback/send
        # boundary.  Unlike perf_counter(), they can be correlated by the demo
        # proxy and by a browser after websocket clock-offset calibration.
        algorithm_ready_epoch_ms = time.time_ns() / 1_000_000
        mode = extra.get("mode", "emit")
        global_frame = (
            epoch_global_start_frame + int(extra.get("live_audio_frames") or 0)
        )
        if mode == "decode":
            if not enable_live:
                return
            payload = {
                "type": "transcription.delta",
                "mode": "decode",
                "delta": extra.get("piece") or text,
                "block": extra.get("block") or text,
                "audio_frames": N_LEFT_PAD + global_frame,
                "algorithm_ready_epoch_ms": algorithm_ready_epoch_ms,
            }
            payload["server_emit_epoch_ms"] = time.time_ns() / 1_000_000
            await ws.send_json(payload)
            return
        if not enable_live:
            return
        token_ids = tuple(int(value) for value in extra.get("_token_ids", ()))
        deltas.append(text)
        committed_text = _append_emit_text(committed_text, text)
        recent_commits.append(
            _CommitRecord(
                global_frame=global_frame,
                text=text,
                token_ids=token_ids,
            )
        )
        payload = {
            "type": "transcription.delta",
            "mode": "emit",
            "delta": text,
            "algorithm_ready_epoch_ms": algorithm_ready_epoch_ms,
        }
        payload.update(
            {
                key: value
                for key, value in extra.items()
                if key not in {"mode", "_token_ids", "live_audio_frames"}
            }
        )
        payload["audio_frames"] = N_LEFT_PAD + global_frame
        payload["server_emit_epoch_ms"] = time.time_ns() / 1_000_000
        await ws.send_json(payload)
        if session is not None:
            await ws.send_json(session.rtf_event())

    async def ensure_session(
        *,
        internal_rollover: bool = False,
        restore_frames: list[RestoreFrame] | None = None,
    ) -> PersistentEmbedsSession:
        nonlocal session, pump, enc_stream, epoch_global_start_frame
        if session is not None:
            return session
        if not internal_rollover:
            await ws.send_json(
                {
                    "type": "status",
                    "message": "warmup_start",
                    "n_left_pad": N_LEFT_PAD,
                }
            )
        encoder: EncoderBridge = app.state.encoder
        enc_stream = await encoder.open_session()
        epoch_global_start_frame = global_audio_frames
        session = PersistentEmbedsSession(
            app.state.engine,
            app.state.tokenizer,
            app.state.table,
            on_delta=on_delta,
            max_decode_tokens=int(app.state.max_decode_tokens),
            n_left_pad=N_LEFT_PAD,
            continuous_decode=bool(app.state.continuous_decode),
            language_tag_table=getattr(app.state, "language_tag_table", None),
            language_id=language_id,
            restore_frames=restore_frames,
        )
        await session.start()
        pump = asyncio.create_task(
            _pump_encoder(session, enc_stream, on_embedding=on_embedding)
        )
        await session.wait_warmup()
        if not internal_rollover:
            await ws.send_json(
                {
                    "type": "status",
                    "message": "warmup_done",
                    "n_left_pad": N_LEFT_PAD,
                    "audio_frames": session.audio_frames,
                }
            )
        return session

    async def finish_epoch() -> PersistentEmbedsSession:
        nonlocal session, pump, enc_stream
        if session is None or pump is None or enc_stream is None:
            raise RuntimeError("cannot finish an inactive streaming epoch")
        old_session = session
        await enc_stream.finish()
        await pump
        enc_stream.close()
        session = None
        pump = None
        enc_stream = None
        return old_session

    async def rollover(reason: str) -> None:
        nonlocal rollover_count, epoch_pcm_bytes
        started = time.perf_counter()
        old_session = await finish_epoch()
        if old_session.error:
            raise RuntimeError(old_session.error)
        restore_frames = build_restore_frames()
        rollover_count += 1
        await ensure_session(
            internal_rollover=True,
            restore_frames=restore_frames,
        )
        assert session is not None
        elapsed = time.perf_counter() - started
        summary = {
            "rollover": rollover_count,
            "reason": reason,
            "old_session_id": old_session.request_id,
            "new_session_id": session.request_id,
            "old_live_audio_frames": old_session.live_audio_frames,
            "old_kv_steps": old_session.kv_steps,
            "old_rtf": old_session.rtf_event(),
            "global_audio_frames": global_audio_frames,
            "context_frames": len(restore_frames),
            "context_seconds": round(
                len(restore_frames) * SECONDS_PER_FRAME, 3
            ),
            "context_commits": sum(len(frame.commits) for frame in restore_frames),
            "context_tokens": session.restored_commit_tokens,
            "restore_prefill_s": round(session.restore_wall_s, 4),
            "rollover_wall_s": round(elapsed, 4),
            "pcm_tail_bytes": len(pcm_tail),
            "transport_buffers_during_rollover": True,
        }
        rollover_summaries.append(summary)
        print(
            "[adaptive-vllm-true] session_rollover "
            + json.dumps(summary, ensure_ascii=False, separators=(",", ":")),
            flush=True,
        )
        epoch_pcm_bytes = 0

    try:
        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)
            mtype = data.get("type")

            if mtype == "session.update":
                model_name = data.get("model") or model_name
                if "enable_live" in data:
                    enable_live = bool(data["enable_live"])
                _apply_language(data)
                await ws.send_json(
                    {
                        "type": "session.updated",
                        "model": model_name,
                        "mode": "adaptive_vllm",
                        "inference": "vllm_true_stream",
                        "language": _language_id_name(language_id),
                    }
                )
                continue

            if mtype == "session.configure":
                if "enable_live" in data:
                    enable_live = bool(data["enable_live"])
                lang_changed = _apply_language(data)
                print(
                    f"[adaptive-vllm-true] language={_language_id_name(language_id)}"
                    f"{' (rebuild warmup)' if lang_changed and session is not None else ''}",
                    flush=True,
                )
                if lang_changed and session is not None:
                    await teardown_session()
                    await ensure_session()
                await ws.send_json(
                    {
                        "type": "session.configured",
                        "enable_live": enable_live,
                        "enable_offline": False,
                        "language": _language_id_name(language_id),
                    }
                )
                continue

            if mtype == "input_audio_buffer.append":
                if session is None:
                    raise RuntimeError("encoder warmup is not finished")
                chunk = base64.b64decode(data.get("audio", ""))
                if chunk:
                    pcm_tail.extend(chunk)
                    if pcm_tail_limit > 0 and len(pcm_tail) > pcm_tail_limit:
                        del pcm_tail[: len(pcm_tail) - pcm_tail_limit]
                    if enable_live:
                        epoch_pcm_bytes += len(chunk)
                        assert enc_stream is not None
                        await enc_stream.append_pcm16(chunk)
                        rollover_reason = ""
                        epoch_audio_s = epoch_pcm_bytes / (16000.0 * 2.0)
                        if (
                            app.state.rollover_seconds > 0
                            and epoch_audio_s >= app.state.rollover_seconds
                        ):
                            rollover_reason = "audio_seconds"
                        elif (
                            app.state.rollover_max_kv_steps > 0
                            and session.kv_steps
                            >= app.state.rollover_max_kv_steps
                        ):
                            rollover_reason = "kv_steps"
                        if rollover_reason:
                            await rollover(rollover_reason)
                continue

            if mtype == "input_audio_buffer.commit":
                is_final = bool(data.get("final", False))
                if not is_final:
                    await ensure_session()
                    await ws.send_json(
                        {"type": "input_audio_buffer.committed", "final": False}
                    )
                    continue

                if session is None:
                    await ws.send_json({"type": "transcription.done", "text": ""})
                    continue

                assert pump is not None and enc_stream is not None
                await enc_stream.finish()
                await pump
                enc_stream.close()
                enc_stream = None
                live_text = committed_text or session.final_text or "".join(deltas)
                if session.error:
                    await ws.send_json(
                        {"type": "error", "message": session.error}
                    )
                stats = {}
                if session.latencies_ms:
                    ordered = sorted(session.latencies_ms)
                    stats = {
                        "session_id": session.request_id,
                        "audio_frames": session.audio_frames,
                        "lm_steps": len(session.latencies_ms),
                        "mean_ms": round(
                            statistics.mean(session.latencies_ms), 3
                        ),
                        "median_ms": round(
                            statistics.median(session.latencies_ms), 3
                        ),
                        "p90_ms": round(
                            ordered[
                                min(
                                    len(ordered) - 1,
                                    int(len(ordered) * 0.9),
                                )
                            ],
                            3,
                        ),
                        "decode_mode": (
                            "continuous"
                            if session.continuous_decode
                            else "step"
                        ),
                        "decode_blocks": session.decode_blocks,
                        "decode_tokens": session.decode_tokens,
                        "decode_wall_s": round(session.decode_wall_s, 4),
                        "decode_block_median_ms": (
                            round(
                                statistics.median(
                                    session.decode_block_latencies_ms
                                ),
                                3,
                            )
                            if session.decode_block_latencies_ms
                            else None
                        ),
                        "post_trigger_decode_ms": {
                            "trigger_to_first_token_ms": _latency_summary(
                                session.trigger_to_first_token_latencies_ms
                            ),
                            "trigger_to_block_done_ms": _latency_summary(
                                session.trigger_to_block_done_latencies_ms
                            ),
                            "per_token_decode_ms": _latency_summary(
                                session.decode_token_latencies_ms
                            ),
                            "blocks_without_text": (
                                session.decode_blocks_without_text
                            ),
                            "token_timing_unmeasured": (
                                session.decode_token_timing_unmeasured
                            ),
                            "first_block": (
                                session.decode_block_timings[0]
                                if session.decode_block_timings
                                else None
                            ),
                            "steady_state": {
                                "blocks": max(
                                    len(session.decode_block_timings) - 1,
                                    0,
                                ),
                                "trigger_to_first_token_ms": _latency_summary(
                                    [
                                        timing["trigger_to_first_token_ms"]
                                        for timing in session.decode_block_timings[1:]
                                        if timing["trigger_to_first_token_ms"]
                                        is not None
                                    ]
                                ),
                                "trigger_to_block_done_ms": _latency_summary(
                                    [
                                        timing["trigger_to_block_done_ms"]
                                        for timing in session.decode_block_timings[1:]
                                    ]
                                ),
                            },
                        },
                        "pipeline_timing_ms": {
                            "encoder_serialize": _latency_summary(
                                session.encoder_serialize_latencies_ms
                            ),
                            "encoder_ipc": _latency_summary(
                                session.encoder_ipc_latencies_ms
                            ),
                            "bridge_decode": _latency_summary(
                                session.bridge_decode_latencies_ms
                            ),
                            "encoder_queue": _latency_summary(
                                session.encoder_queue_latencies_ms
                            ),
                            "encoder_to_lm": _latency_summary(
                                session.encoder_to_lm_latencies_ms
                            ),
                        },
                        "encoder_stats": dict(session.encoder_stats),
                        "rollovers": rollover_count,
                        "rollover_summaries": list(rollover_summaries),
                    }
                summary = {
                    **session.rtf_event(),
                    **stats,
                    "live_emits": len(deltas),
                    "live_text_chars": len(live_text),
                    "error": session.error,
                }
                print(
                    "[adaptive-vllm-true] session_summary "
                    + json.dumps(summary, ensure_ascii=False, separators=(",", ":")),
                    flush=True,
                )
                if enable_live:
                    await ws.send_json(session.rtf_event())
                    await ws.send_json(
                        {
                            "type": "status",
                            "message": "live_done",
                            "text": live_text,
                            "emits": list(deltas),
                        }
                    )
                # The live transcript is the result; offline fields stay
                # empty so existing clients keep parsing the same payload.
                result_text = live_text if enable_live else ""
                final_payload = {
                    "type": "transcription.done",
                    "text": result_text,
                    "live_text": result_text,
                    "offline_text": "",
                    "live_emits": list(deltas) if enable_live else [],
                    "offline_emits": [],
                    **stats,
                }
                final_payload["server_emit_epoch_ms"] = (
                    time.time_ns() / 1_000_000
                )
                await ws.send_json(final_payload)
                session = None
                pump = None
                reset_utterance()
                continue

            if mtype == "session.end":
                break
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        try:
            await ws.send_json({"type": "error", "message": str(exc)})
        except Exception:  # noqa: BLE001
            pass
    finally:
        # A browser refresh can arrive while warmup is still running. Stop this
        # connection's encoder stream first, then bound every cleanup wait so a
        # wedged vLLM abort cannot hold a session slot forever.
        if enc_stream is not None:
            try:
                await asyncio.wait_for(enc_stream.abort(), timeout=2.0)
            except Exception:  # noqa: BLE001
                enc_stream.close()
        if pump is not None and not pump.done():
            pump.cancel()
            await asyncio.wait({pump}, timeout=2.0)
        if session is not None:
            try:
                await asyncio.wait_for(session.abort(), timeout=5.0)
            except Exception:  # noqa: BLE001
                pass


@app.websocket("/v1/realtime")
async def realtime(ws: WebSocket):
    global _active_sessions
    await ws.accept()
    # Each session holds a persistent vLLM request (KV cache) and an encoder
    # stream on the shared sidecar.  Beyond the cap, reject explicitly rather
    # than letting KV pressure preempt live requests.
    limit = int(app.state.max_sessions)
    if limit > 0 and _active_sessions >= limit:
        await ws.send_json(
            {
                "type": "error",
                "code": "session_busy",
                "message": f"all {limit} realtime sessions are active",
            }
        )
        await ws.close(code=1013, reason="realtime session busy")
        return
    _active_sessions += 1
    print(
        f"[adaptive-vllm-true] session open active={_active_sessions}/{limit or 'inf'}",
        flush=True,
    )
    try:
        await _run_realtime(ws)
    finally:
        _active_sessions -= 1
        print(
            f"[adaptive-vllm-true] session closed active={_active_sessions}",
            flush=True,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Adaptive streaming ASR on vLLM")
    parser.add_argument(
        "--model",
        default=os.environ.get("TEXT_MODEL", ""),
        help="Text LM exported by vllm_realtime/export_text_model.py.",
    )
    parser.add_argument(
        "--encoder-model",
        default=os.environ.get("MODEL_PATH", ""),
        help="Full X2Streaming-ASR checkpoint (audio encoder and language tag).",
    )
    parser.add_argument(
        "--tokenizer",
        default=os.environ.get("TOKENIZER", ""),
        help="Mistral tokenizer directory; defaults to --encoder-model.",
    )
    parser.add_argument(
        "--encoder-python",
        default=os.environ.get("ENCODER_PYTHON", sys.executable),
        help="Python of the env with requirements.txt (defaults to this one).",
    )
    parser.add_argument(
        "--encoder-gpu",
        default=os.environ.get(
            "ENCODER_GPU",
            os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0],
        ),
        help="Physical GPU of the encoder sidecar; defaults to the LM's GPU.",
    )
    parser.add_argument(
        "--encoder-lora",
        default=os.environ.get("LORA_PATH", ""),
    )
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=float(os.environ.get("GPU_MEM", "0.35")),
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=int(os.environ.get("MAX_MODEL_LEN", "9000")),
    )
    parser.add_argument(
        "--max-decode-tokens",
        type=int,
        default=int(os.environ.get("MAX_DECODE_TOKENS", "32")),
    )
    parser.add_argument(
        "--rollover-seconds",
        type=float,
        default=float(os.environ.get("X2_ROLLOVER_SECONDS", "180")),
        help="Start a fresh persistent request after this much live audio; <=0 disables.",
    )
    parser.add_argument(
        "--rollover-context-seconds",
        type=float,
        default=float(os.environ.get("X2_CONTEXT_SECONDS", "30")),
        help="Interleaved audio/text history restored into each fresh request.",
    )
    parser.add_argument(
        "--rollover-max-kv-steps",
        type=int,
        default=int(os.environ.get("X2_ROLLOVER_MAX_KV_STEPS", "3500")),
        help="Conservative KV row threshold; <=0 disables this trigger.",
    )
    parser.add_argument(
        "--kv-cache-gib",
        type=float,
        default=float(os.environ.get("KV_CACHE_GIB", "0")),
        help="Fixed KV cache size; <=0 lets vLLM size it from --gpu-memory-utilization.",
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=int(os.environ.get("MAX_SESSIONS", "4")),
        help="Concurrent /v1/realtime sessions; <=0 means unlimited.",
    )
    parser.add_argument(
        "--step-decode",
        action="store_true",
        default=os.environ.get("STEP_DECODE", "0") == "1",
        help="Use the legacy one-vLLM-request-per-token decode path.",
    )
    parser.add_argument(
        "--encoder-feature-prefetch",
        action=argparse.BooleanOptionalAction,
        default=_env_truthy("ENCODER_FEATURE_PREFETCH", True),
    )
    parser.add_argument(
        "--encoder-gpu-overlap",
        action=argparse.BooleanOptionalAction,
        default=_env_truthy("ENCODER_GPU_OVERLAP", True),
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        default=os.environ.get("ENFORCE_EAGER", "0") == "1",
    )
    args = parser.parse_args()
    if not args.model or not args.encoder_model:
        parser.error("--model (TEXT_MODEL) and --encoder-model (MODEL_PATH) are required")
    args.tokenizer = args.tokenizer or args.encoder_model
    return args


def main() -> None:
    args = parse_args()
    app.state.model_path = args.model
    app.state.tokenizer_path = args.tokenizer
    app.state.encoder_model = args.encoder_model
    app.state.encoder_python = args.encoder_python
    app.state.encoder_gpu = args.encoder_gpu
    app.state.encoder_lora = args.encoder_lora
    app.state.gpu_mem = args.gpu_memory_utilization
    app.state.kv_cache_gib = args.kv_cache_gib
    app.state.max_model_len = args.max_model_len
    app.state.max_decode_tokens = args.max_decode_tokens
    app.state.rollover_seconds = max(0.0, float(args.rollover_seconds))
    app.state.rollover_context_seconds = max(
        SECONDS_PER_FRAME, float(args.rollover_context_seconds)
    )
    app.state.rollover_max_kv_steps = max(
        0, int(args.rollover_max_kv_steps)
    )
    app.state.max_sessions = max(0, int(args.max_sessions))
    app.state.continuous_decode = not args.step_decode
    app.state.encoder_feature_prefetch = args.encoder_feature_prefetch
    app.state.encoder_gpu_overlap = args.encoder_gpu_overlap
    app.state.enforce_eager = args.enforce_eager

    _validate_export_alignment(
        args.model,
        args.encoder_model,
        args.encoder_lora,
    )
    print(f"Adaptive true-stream Realtime: ws://{args.host}:{args.port}/v1/realtime")
    print(f"Text LM : {args.model}")
    print(
        f"Encoder : {args.encoder_model}  gpu={args.encoder_gpu}  "
        f"lora={args.encoder_lora or '(none)'}"
    )
    print(
        "Rollover: "
        f"every={app.state.rollover_seconds:g}s  "
        f"context={app.state.rollover_context_seconds:g}s  "
        f"max_kv_steps={app.state.rollover_max_kv_steps}"
    )
    print(f"Sessions: max={app.state.max_sessions or 'unlimited'}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
