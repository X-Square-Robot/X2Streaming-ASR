"""Unified web demo for Voxtral Realtime streaming ASR.

Supports:
  --backend transformers | vllm
  --mode    fixed | adaptive

Adaptive interleaved ASR cannot run on stock vLLM; use
``MODE=adaptive_vllm bash vllm/serve.sh`` which starts the persistent-KV
Realtime server, then ``BACKEND=vllm MODE=adaptive bash demo/run.sh``.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
import tempfile
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.formparsers import MultiPartParser

# Default to GPU 0 only when a second encoder card was not requested.
# DEVICE=cuda:0 ENC_DEVICE=cuda:1 needs both devices visible.
if not (
    (os.environ.get("ENC_DEVICE") or os.environ.get("VOXTRAL_ENC_DEVICE") or "").strip()
):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

STATIC_DIR = Path(__file__).resolve().parent / "static"
UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
MAX_UPLOAD_BYTES = 512 * 1024 * 1024
_UPLOAD_TTL_S = 3600.0
_ALLOWED_AUDIO_SUFFIXES = {
    ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".webm", ".wma",
}
_UPLOADS: dict[str, dict] = {}

# Meeting-length files exceed Starlette's default 1MB multipart part cap.
if getattr(MultiPartParser, "max_part_size", None) is not None:
    MultiPartParser.max_part_size = MAX_UPLOAD_BYTES


def _ensure_upload_dir():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _safe_unlink(path: str | os.PathLike | None) -> None:
    if not path:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


def _cleanup_stale_uploads(max_age_s: float = _UPLOAD_TTL_S) -> None:
    now = time.time()
    for fid, rec in list(_UPLOADS.items()):
        created = float(rec.get("created") or 0)
        if now - created >= max_age_s:
            _safe_unlink(rec.get("path"))
            _UPLOADS.pop(fid, None)
    if not UPLOAD_DIR.is_dir():
        return
    for p in UPLOAD_DIR.iterdir():
        if not p.is_file():
            continue
        try:
            age = now - p.stat().st_mtime
        except OSError:
            continue
        if age >= max_age_s:
            _safe_unlink(p)


def _take_upload_path(data: dict) -> str:
    """Resolve a WS file notify to a local path. Caller must delete the file."""
    file_id = str(data.get("id") or data.get("file_id") or "").strip()
    if file_id:
        if len(file_id) != 32 or any(c not in "0123456789abcdef" for c in file_id.lower()):
            raise ValueError("invalid upload id")
        rec = _UPLOADS.pop(file_id, None)
        if rec is None:
            raise ValueError("unknown or already-used upload id")
        path = Path(rec.get("path") or "")
        try:
            path.resolve().relative_to(UPLOAD_DIR.resolve())
        except ValueError as exc:
            raise ValueError("upload path escaped") from exc
        if not path.is_file():
            raise ValueError("uploaded file missing")
        return str(path)

    raw_b64 = data.get("data")
    if raw_b64:
        raw = base64.b64decode(raw_b64)
        suffix = Path(data.get("name") or "audio.wav").suffix.lower() or ".wav"
        if suffix not in _ALLOWED_AUDIO_SUFFIXES:
            suffix = ".wav"
        _ensure_upload_dir()
        fd, tmp = tempfile.mkstemp(suffix=suffix, dir=UPLOAD_DIR)
        os.close(fd)
        Path(tmp).write_bytes(raw)
        return tmp

    raise ValueError("file message needs id (HTTP upload) or data")


# ---------------------------------------------------------------------------
# Transformers backend
# ---------------------------------------------------------------------------

_tf_model = None
_tf_processor = None
_tf_adaptive = False
_tf_emit_thr = None


def _language_tag_status(model) -> dict:
    from models.voxtral_asr.interleaved import _unwrap_voxtral_cg
    from models.voxtral_asr.language_tag import language_id_name

    if model is None:
        return {
            "language_tag_enabled": False,
            "language_tag_loaded": False,
            "language": "zh",
            "language_id": 0,
        }
    cg = _unwrap_voxtral_cg(model)
    tag = getattr(cg, "language_tag_embed", None)
    loaded = bool(
        tag is not None
        and (
            getattr(tag, "_loaded_from_checkpoint", False)
            or getattr(tag, "_loaded_from_adapter", False)
        )
    )
    lid = int(getattr(cg, "_infer_language_id", 0) or 0)
    return {
        "language_tag_enabled": tag is not None,
        "language_tag_loaded": loaded,
        "language": language_id_name(lid),
        "language_id": lid,
    }


def _parse_ws_language(data: dict | None) -> int:
    from models.voxtral_asr.language_tag import resolve_language_id

    raw = None
    if data:
        raw = data.get("language", data.get("language_id"))
    if raw is None or raw == "":
        raw = getattr(app.state, "default_language", "zh")
    return int(resolve_language_id(raw))


def _load_transformers(
    repo_id: str,
    device: str,
    *,
    adaptive: bool,
    lora_path: str = "",
    emit_thr=None,
    enc_device: str = "",
):
    global _tf_model, _tf_processor, _tf_adaptive, _tf_emit_thr

    if adaptive:
        from models import build_model

        print(f"[transformers/adaptive] Loading {repo_id} on {device} ...")
        backend = build_model("voxtral_asr")
        wrapper = backend.load_for_inference(
            repo_id,
            device=device,
            enc_device=enc_device or "",
            lora_path=lora_path or "",
            adaptive_latency=True,
            language_tag={"enabled": True, "require_checkpoint_weights": False},
        )
        _tf_model = wrapper["model"]
        _tf_model.eval()
        _tf_processor = wrapper["processor"]
        _tf_adaptive = True
        _tf_emit_thr = emit_thr
        from models.voxtral_asr.devices import enc_device_of, lm_device_of
        from models.voxtral_asr.language_tag import set_infer_language_id

        set_infer_language_id(_tf_model, getattr(app.state, "default_language", "zh"))
        tag_info = _language_tag_status(_tf_model)
        print(
            f"[transformers/adaptive] encoder on {enc_device_of(_tf_model)}, "
            f"LM on {lm_device_of(_tf_model)}",
            flush=True,
        )
        print(
            f"[transformers/adaptive] language_tag enabled={tag_info['language_tag_enabled']} "
            f"loaded={tag_info['language_tag_loaded']} "
            f"default={tag_info['language']}",
            flush=True,
        )
        print("[transformers/adaptive] Model ready.")
        return

    import torch
    from transformers import VoxtralRealtimeForConditionalGeneration, VoxtralRealtimeProcessor

    print(f"[transformers] Loading model from {repo_id} on {device} ...")
    _tf_processor = VoxtralRealtimeProcessor.from_pretrained(repo_id)
    _tf_model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
        repo_id,
        device_map=device,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    _tf_model.eval()
    _tf_adaptive = False
    _tf_emit_thr = None
    print("[transformers] Model ready.")


async def _handle_transformers_adaptive_ws(websocket: WebSocket):
    """Adaptive: live online decode while audio flows, then offline whole-utt."""
    from engine import load_wav_float32, pcm16_to_float32
    from engine_adaptive import AdaptiveSession

    session = AdaptiveSession(
        _tf_model,
        _tf_processor,
        emit_word_prob_threshold=_tf_emit_thr,
        max_decode_tokens=int(getattr(app.state, "max_decode_tokens", 32) or 32),
        enable_live=bool(getattr(app.state, "enable_live", True)),
        enable_offline=bool(getattr(app.state, "enable_offline", True)),
        listen_chunk_size=int(getattr(app.state, "listen_chunk_size", 1) or 1),
        offline_listen_chunk_size=int(
            getattr(app.state, "offline_listen_chunk_size", 16) or 16
        ),
    )
    pump_task: asyncio.Task | None = None

    async def pump_events(ws: WebSocket, sess: AdaptiveSession):
        from data_pipeline.join_emit import join_emit_texts

        live_parts: list[str] = []
        live_committed = ""
        offline_parts: list[str] = []
        while True:
            item = await asyncio.to_thread(sess._event_q.get)
            if item is None:
                break
            if not isinstance(item, dict):
                continue
            kind = item.get("kind")
            if kind == "warmup_start":
                await ws.send_json({
                    "type": "status",
                    "message": "warmup_start",
                    "n_left_pad": item.get("n_left_pad"),
                    "preheat_s": item.get("preheat_s"),
                })
            elif kind == "warmup_done":
                await ws.send_json({
                    "type": "status",
                    "message": "warmup_done",
                    "frame": item.get("frame"),
                    "n_left_pad": item.get("n_left_pad"),
                    "wall_s": item.get("wall_s"),
                    "prefill": item.get("prefill", True),
                })
            elif kind == "live_decode":
                await ws.send_json({
                    "type": "partial",
                    "phase": "live",
                    "mode": "decode",
                    "text": item.get("piece") or "",
                    "block": item.get("text") or "",
                    "committed": live_committed,
                })
            elif kind == "live_emit":
                text = (item.get("text") or "").strip()
                if text:
                    live_parts.append(text)
                    live_committed = join_emit_texts(live_parts)
                    await ws.send_json({
                        "type": "partial",
                        "phase": "live",
                        "mode": "emit",
                        "text": text,
                        "committed": live_committed,
                    })
            elif kind == "live_emit_start":
                await ws.send_json({
                    "type": "status",
                    "message": "live_emit_start",
                    "frame": item.get("frame"),
                    "time_s": item.get("time_s"),
                })
            elif kind == "rtf":
                await ws.send_json({
                    "type": "rtf",
                    "audio_s": item.get("audio_s"),
                    "wall_total_s": item.get("wall_total_s"),
                    "wall_enc_s": item.get("wall_enc_s"),
                    "wall_lm_s": item.get("wall_lm_s"),
                    "rtf_total": item.get("rtf_total"),
                    "rtf_enc": item.get("rtf_enc"),
                    "rtf_lm": item.get("rtf_lm"),
                    "frame": item.get("frame"),
                })
            elif kind == "live_done":
                await ws.send_json({
                    "type": "status",
                    "message": "live_done",
                    "text": item.get("text") or live_committed,
                    "emits": item.get("emits") or [],
                    "emits_timed": item.get("emits_timed") or [],
                })
            elif kind == "offline_start":
                await ws.send_json({
                    "type": "status",
                    "message": "offline_start",
                })
            elif kind == "offline_done":
                emits = item.get("emits") or offline_parts
                text = (item.get("text") or "".join(emits)).strip()
                await ws.send_json({
                    "type": "partial",
                    "phase": "offline",
                    "mode": "done",
                    "text": text,
                    "emits": emits,
                })
            elif kind == "offline_emit":
                text = (item.get("text") or "").strip()
                if text:
                    offline_parts.append(text)
                    await ws.send_json({
                        "type": "partial",
                        "phase": "offline",
                        "mode": "emit",
                        "text": text,
                        "emits": list(offline_parts),
                    })

        await asyncio.to_thread(sess.join)
        final_text = (sess.final_text or "".join(offline_parts)).strip()
        live_text = (sess.live_text or live_committed).strip()
        meta = sess.last_meta
        meta["live_text"] = live_text
        meta["offline_text"] = final_text
        if sess.error and not final_text:
            await ws.send_json({
                "type": "status",
                "message": f"adaptive: {sess.error}",
            })
        await ws.send_json({
            "type": "final",
            "text": final_text,
            "live_text": live_text,
            "offline_text": final_text,
            "live_emits": meta.get("live_emits") or [],
            "live_emits_timed": meta.get("live_emits_timed") or [],
            "offline_emits": meta.get("offline_emits") or offline_parts,
            "meta": meta,
            "empty": not bool(final_text),
        })

    async def _cancel_pump():
        nonlocal pump_task
        if pump_task is not None and not pump_task.done():
            pump_task.cancel()
            try:
                await pump_task
            except asyncio.CancelledError:
                pass
        pump_task = None

    session.set_language_id(getattr(app.state, "default_language", "zh"))

    def _apply_infer_flags(data: dict | None = None):
        if data is not None:
            if "enable_live" in data:
                session.enable_live = bool(data["enable_live"])
            if "enable_offline" in data:
                session.enable_offline = bool(data["enable_offline"])

    def _apply_language(data: dict | None = None) -> bool:
        return session.set_language_id(_parse_ws_language(data))

    async def _arm_live():
        """Keep the live worker warm; do not reset an already-running session."""
        nonlocal pump_task
        if session.enable_live:
            if not session.is_live_running():
                session.start()
        else:
            session.start()
        if pump_task is None or pump_task.done():
            pump_task = asyncio.create_task(pump_events(websocket, session))

    async def _rearm_live():
        """After an utterance, reset and pre-warm for the next mic/file click."""
        session.reset()
        await _arm_live()

    try:
        await _arm_live()
        while True:
            msg = await websocket.receive()

            if msg.get("type") == "websocket.disconnect":
                break

            if "bytes" in msg and msg["bytes"]:
                pcm = pcm16_to_float32(msg["bytes"])
                session.append_pcm(pcm)
                continue

            if "text" not in msg:
                continue

            data = json.loads(msg["text"])
            mtype = data.get("type")

            if mtype == "start":
                # Reuse the pre-warmed live worker unless language_tag changed;
                # pad KV already has the old lang embed and must be rebuilt.
                lang_changed = _apply_language(data)
                _apply_infer_flags(data)
                if lang_changed:
                    session.reset()
                await _arm_live()
                await websocket.send_json({
                    "type": "status",
                    "message": "session_started",
                    "mode": "adaptive",
                    "live": bool(session.enable_live),
                    "offline": bool(session.enable_offline),
                    "warmup_done": session.warmup_done,
                    "language": session.language_id,
                })
                continue

            if mtype == "stop":
                print(
                    f"[demo] stop buffered={session.buffered_seconds():.2f}s "
                    f"(live={session.enable_live} offline={session.enable_offline})",
                    flush=True,
                )
                session.finish_async()
                if pump_task is not None:
                    await pump_task
                    pump_task = None
                await websocket.send_json({
                    "type": "status",
                    "message": "session_finished",
                })
                await _rearm_live()
                continue

            if mtype == "file":
                tmp_path = None
                try:
                    lang_changed = _apply_language(data)
                    _apply_infer_flags(data)
                    if lang_changed:
                        session.reset()
                    tmp_path = _take_upload_path(data)
                    await _arm_live()
                    audio = load_wav_float32(
                        tmp_path, _tf_processor.feature_extractor.sampling_rate
                    )
                    sr = _tf_processor.feature_extractor.sampling_rate
                    if session.enable_live:
                        # Pace ~80ms chunks so live path can emit mid-file.
                        step = max(int(0.08 * sr), 1)
                        for i in range(0, len(audio), step):
                            session.append_pcm(audio[i : i + step])
                            await asyncio.sleep(0)
                    else:
                        session.append_pcm(audio)
                    session.finish_async()
                    if pump_task is not None:
                        await pump_task
                        pump_task = None
                    await websocket.send_json({
                        "type": "status",
                        "message": "file_done",
                    })
                    await _rearm_live()
                except Exception as exc:  # noqa: BLE001
                    traceback.print_exc()
                    try:
                        await websocket.send_json({
                            "type": "status",
                            "message": f"file_error: {exc}",
                        })
                    except Exception:
                        pass
                    try:
                        await _rearm_live()
                    except Exception:
                        pass
                finally:
                    _safe_unlink(tmp_path)
                continue

    except WebSocketDisconnect:
        pass
    finally:
        await _cancel_pump()
        session.abort()


async def _handle_transformers_ws(websocket: WebSocket, default_delay_ms: int):
    from engine import StreamingSession, load_wav_float32, pcm16_to_float32

    loop = asyncio.get_running_loop()
    session = StreamingSession(_tf_model, _tf_processor, delay_ms=default_delay_ms)
    pump_task: asyncio.Task | None = None

    async def pump_streamer(ws: WebSocket, sess: StreamingSession):
        streamer = sess.streamer
        if streamer is None:
            return

        def iterate():
            parts = []
            for chunk in streamer:
                parts.append(chunk)
                asyncio.run_coroutine_threadsafe(
                    ws.send_json({"type": "partial", "text": chunk}),
                    loop,
                )
            return "".join(parts)

        final_text = await asyncio.to_thread(iterate)
        await ws.send_json({"type": "final", "text": final_text})

    try:
        while True:
            msg = await websocket.receive()

            if msg.get("type") == "websocket.disconnect":
                break

            if "bytes" in msg and msg["bytes"]:
                pcm = pcm16_to_float32(msg["bytes"])
                session.append_pcm(pcm)
                if pump_task is None and session.streamer is not None:
                    pump_task = asyncio.create_task(pump_streamer(websocket, session))
                continue

            if "text" not in msg:
                continue

            data = json.loads(msg["text"])
            mtype = data.get("type")

            if mtype == "start":
                if pump_task is not None and not pump_task.done():
                    pump_task.cancel()
                await asyncio.to_thread(session.abort)
                pump_task = None
                delay_ms = int(data.get("delay_ms", default_delay_ms))
                session.set_delay_ms(delay_ms)
                session.reset()
                await websocket.send_json({
                    "type": "status",
                    "message": "session_started",
                    "delay_ms": session.delay_ms,
                })
                continue

            if mtype == "stop":
                session.finish()
                if pump_task is None and session.streamer is not None:
                    pump_task = asyncio.create_task(pump_streamer(websocket, session))
                if pump_task is not None:
                    await pump_task
                await websocket.send_json({"type": "status", "message": "session_finished"})
                session.reset()
                pump_task = None
                continue

            if mtype == "file":
                tmp_path = None
                try:
                    delay_ms = int(data.get("delay_ms", default_delay_ms))
                    session.set_delay_ms(delay_ms)
                    tmp_path = _take_upload_path(data)
                    session.reset()
                    audio = load_wav_float32(tmp_path, _tf_processor.feature_extractor.sampling_rate)
                    session.append_pcm(audio)
                    session.finish()
                    if session.streamer is not None:
                        await pump_streamer(websocket, session)
                    await websocket.send_json({"type": "status", "message": "file_done"})
                    session.reset()
                except Exception as exc:  # noqa: BLE001
                    traceback.print_exc()
                    try:
                        await websocket.send_json({
                            "type": "status",
                            "message": f"file_error: {exc}",
                        })
                    except Exception:
                        pass
                    try:
                        session.reset()
                    except Exception:
                        pass
                finally:
                    _safe_unlink(tmp_path)
                continue

    except WebSocketDisconnect:
        pass
    finally:
        if pump_task is not None and not pump_task.done():
            pump_task.cancel()
        # Reclaim any generation thread left blocked inside model.generate so the
        # shared model is free for the next connection.
        await asyncio.to_thread(session.abort)


# ---------------------------------------------------------------------------
# vLLM / adaptive-serve backend (same Realtime WS protocol)
# ---------------------------------------------------------------------------

async def _handle_vllm_ws(websocket: WebSocket, vllm_url: str, model: str):
    import websockets

    def _pcm16_bytes_from_wav(path: str, sr: int = 16000) -> bytes:
        import librosa
        audio, _ = librosa.load(path, sr=sr, mono=True)
        return (audio * 32767).astype(np.int16).tobytes()

    try:
        vllm_ws_cm = websockets.connect(vllm_url, max_size=None)
        vllm_ws = await vllm_ws_cm.__aenter__()
    except OSError as exc:
        await websocket.send_json({
            "type": "status",
            "message": (
                f"error: 连不上 {vllm_url}（{exc}）。"
                "vLLM 还在加载时不会监听 8000，等日志出现 "
                "Uvicorn running on http://0.0.0.0:8000 后再开麦"
            ),
        })
        return

    try:
        created = json.loads(await vllm_ws.recv())
        if created.get("type") != "session.created":
            raise RuntimeError(f"Unexpected vLLM handshake: {created}")
        supports_warmup = created.get("inference") == "vllm_true_stream"
        await vllm_ws.send(json.dumps({
            "type": "session.update",
            "model": model,
            "language": getattr(app.state, "default_language", "zh"),
        }))

        warmup_ev = asyncio.Event()
        done_ev = asyncio.Event()
        upstream_error: str | None = None
        if supports_warmup:
            await websocket.send_json({
                "type": "status",
                "message": "warmup_start",
                "n_left_pad": 32,
            })
        else:
            # HF adaptive-serve has no left-pad warmup. Tell the page so
            # the record button is not stuck on warmupReady=false.
            warmup_ev.set()
            await websocket.send_json({
                "type": "status",
                "message": "warmup_done",
                "n_left_pad": 0,
                "prefill": False,
            })

        async def pump() -> None:
            nonlocal upstream_error
            try:
                async for raw in vllm_ws:
                    data = json.loads(raw)
                    mtype = data.get("type")
                    if mtype == "transcription.delta":
                        mode = data.get("mode", "emit")
                        if mode == "decode":
                            await websocket.send_json({
                                "type": "partial",
                                "phase": "live",
                                "mode": "decode",
                                "text": data.get("delta", ""),
                                "block": data.get("block") or data.get("delta", ""),
                            })
                        else:
                            await websocket.send_json({
                                "type": "partial",
                                "phase": "live",
                                "mode": "emit",
                                "text": data.get("delta", ""),
                            })
                    elif mtype == "offline_start":
                        await websocket.send_json({
                            "type": "status",
                            "message": "offline_start",
                        })
                    elif mtype == "offline_emit":
                        await websocket.send_json({
                            "type": "partial",
                            "phase": "offline",
                            "mode": "emit",
                            "text": data.get("text", ""),
                            "emits": data.get("emits") or [],
                        })
                    elif mtype == "offline_done":
                        await websocket.send_json({
                            "type": "partial",
                            "phase": "offline",
                            "mode": "done",
                            "text": data.get("text", ""),
                            "emits": data.get("emits") or [],
                        })
                    elif mtype == "transcription.done":
                        await websocket.send_json({
                            "type": "final",
                            "text": data.get("text", ""),
                            "live_text": data.get("live_text"),
                            "offline_text": data.get("offline_text") or data.get("text", ""),
                            "live_emits": data.get("live_emits") or [],
                            "offline_emits": data.get("offline_emits") or [],
                        })
                        done_ev.set()
                    elif mtype == "rtf":
                        await websocket.send_json(data)
                    elif mtype == "status":
                        await websocket.send_json(data)
                        if data.get("message") == "warmup_done":
                            warmup_ev.set()
                    elif mtype == "error":
                        upstream_error = str(data.get("message") or data)
                        await websocket.send_json({
                            "type": "status",
                            "message": f"error: {data}",
                        })
                        warmup_ev.set()
                        done_ev.set()
            except websockets.ConnectionClosed as exc:
                upstream_error = f"upstream websocket closed: {exc}"
                warmup_ev.set()
                done_ev.set()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # Browser refresh can break send_json mid-warmup.
                upstream_error = f"browser websocket closed: {exc}"
                # Unblock prewarm()/commit immediately so their finally block can
                # send session.end and release the single-session vLLM lock.
                warmup_ev.set()
                done_ev.set()

        def raise_upstream_error() -> None:
            if upstream_error:
                raise RuntimeError(upstream_error)

        async def prewarm() -> None:
            if not supports_warmup:
                return
            warmup_ev.clear()
            await vllm_ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            await asyncio.wait_for(warmup_ev.wait(), timeout=180)
            raise_upstream_error()

        async def append_pcm(pcm: bytes) -> None:
            await vllm_ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm).decode(),
            }))

        pump_task = asyncio.create_task(pump())
        try:
            await prewarm()
            while True:
                msg = await websocket.receive()
                if msg.get("type") == "websocket.disconnect":
                    break

                if "bytes" in msg and msg["bytes"]:
                    if not warmup_ev.is_set():
                        await websocket.send_json({
                            "type": "status",
                            "message": "send start before audio",
                        })
                        continue
                    await append_pcm(msg["bytes"])
                    continue

                if "text" not in msg:
                    continue

                data = json.loads(msg["text"])
                mtype = data.get("type")

                if mtype == "start":
                    if not warmup_ev.is_set():
                        await prewarm()
                    enable_live = bool(
                        data.get("enable_live", app.state.enable_live)
                    )
                    enable_offline = bool(
                        data.get("enable_offline", app.state.enable_offline)
                    )
                    lang = data.get("language", data.get("language_id"))
                    if lang is None or lang == "":
                        lang = getattr(app.state, "default_language", "zh")
                    cfg = {
                        "type": "session.configure",
                        "enable_live": enable_live,
                        "enable_offline": enable_offline,
                        "language": lang,
                    }
                    await vllm_ws.send(json.dumps(cfg))
                    await websocket.send_json({
                        "type": "status",
                        "message": "session_started",
                        "mode": app.state.mode,
                        "live": enable_live,
                        "offline": enable_offline,
                        "warmup_done": True,
                        "language": lang if lang not in (None, "") else "zh",
                    })
                    continue

                if mtype == "stop":
                    try:
                        raise_upstream_error()
                        done_ev.clear()
                        await vllm_ws.send(json.dumps({
                            "type": "input_audio_buffer.commit",
                            "final": True,
                        }))
                        await asyncio.wait_for(done_ev.wait(), timeout=90)
                        raise_upstream_error()
                    except Exception as exc:  # noqa: BLE001
                        traceback.print_exc()
                        try:
                            await websocket.send_json({
                                "type": "status",
                                "message": f"error: {exc}",
                            })
                        except Exception:  # noqa: BLE001
                            pass
                    try:
                        await websocket.send_json({
                            "type": "status",
                            "message": "session_finished",
                        })
                    except Exception:  # noqa: BLE001
                        pass
                    try:
                        await prewarm()
                    except Exception as exc:  # noqa: BLE001
                        traceback.print_exc()
                        warmup_ev.set()
                        try:
                            await websocket.send_json({
                                "type": "status",
                                "message": f"error: warmup {exc}",
                            })
                        except Exception:  # noqa: BLE001
                            pass
                    continue

                if mtype == "file":
                    tmp = None
                    try:
                        if not warmup_ev.is_set():
                            await prewarm()
                        file_lang = data.get("language", data.get("language_id"))
                        if file_lang is None or file_lang == "":
                            file_lang = getattr(app.state, "default_language", "zh")
                        await vllm_ws.send(json.dumps({
                            "type": "session.configure",
                            "enable_live": bool(
                                data.get("enable_live", app.state.enable_live)
                            ),
                            "enable_offline": bool(
                                data.get(
                                    "enable_offline",
                                    app.state.enable_offline,
                                )
                            ),
                            "language": file_lang,
                        }))
                        tmp = _take_upload_path(data)
                        pcm = _pcm16_bytes_from_wav(tmp)
                        chunk_bytes = 4096
                        for i in range(0, len(pcm), chunk_bytes):
                            await append_pcm(pcm[i : i + chunk_bytes])
                        raise_upstream_error()
                        done_ev.clear()
                        await vllm_ws.send(json.dumps({
                            "type": "input_audio_buffer.commit",
                            "final": True,
                        }))
                        await asyncio.wait_for(done_ev.wait(), timeout=180)
                        raise_upstream_error()
                        await websocket.send_json({
                            "type": "status",
                            "message": "file_done",
                        })
                        await prewarm()
                    except Exception as exc:  # noqa: BLE001
                        traceback.print_exc()
                        await websocket.send_json({
                            "type": "status",
                            "message": f"file_error: {exc}",
                        })
                    finally:
                        _safe_unlink(tmp)
                    continue
        finally:
            pump_task.cancel()
            try:
                await vllm_ws.send(json.dumps({"type": "session.end"}))
            except Exception:  # noqa: BLE001
                pass
    finally:
        try:
            await vllm_ws_cm.__aexit__(None, None, None)
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    if app.state.backend == "transformers":
        _load_transformers(
            app.state.repo_id,
            app.state.device,
            adaptive=(app.state.mode == "adaptive"),
            lora_path=app.state.lora_path,
            emit_thr=app.state.emit_thr,
            enc_device=str(getattr(app.state, "enc_device", "") or ""),
        )
    else:
        kind = "adaptive-serve" if app.state.mode == "adaptive" else "vLLM"
        print(f"[{kind}] Will proxy to {app.state.vllm_url} (mode={app.state.mode})")
    yield
    _cleanup_stale_uploads(max_age_s=0)


app = FastAPI(title="Voxtral Realtime Demo", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

def _upload_suffix(filename: str | None) -> str:
    suffix = Path(filename or "audio.wav").suffix.lower()
    if suffix in _ALLOWED_AUDIO_SUFFIXES:
        return suffix
    return ""


def _content_type_ok(content_type: str | None) -> bool:
    ct = (content_type or "").split(";")[0].strip().lower()
    if not ct:
        return True
    if ct.startswith("audio/"):
        return True
    return ct in {"application/octet-stream", "application/ogg", "video/webm"}


@app.post("/api/file")
async def upload_infer_file(file: UploadFile = File(...)):
    """Accept a full audio file over HTTP; WS later references it by id."""
    _ensure_upload_dir()
    _cleanup_stale_uploads()

    name = file.filename or "audio.wav"
    suffix = _upload_suffix(name)
    if not suffix:
        raise HTTPException(400, f"unsupported audio type: {Path(name).suffix or '(none)'}")
    if not _content_type_ok(file.content_type):
        raise HTTPException(400, f"unsupported content-type: {file.content_type}")

    cl = file.headers.get("content-length")
    if cl:
        try:
            if int(cl) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"file too large (max {MAX_UPLOAD_BYTES} bytes)")
        except ValueError:
            pass

    fid = uuid.uuid4().hex
    dest = UPLOAD_DIR / f"{fid}{suffix}"
    size = 0
    try:
        with dest.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, f"file too large (max {MAX_UPLOAD_BYTES} bytes)")
                out.write(chunk)
    except HTTPException:
        _safe_unlink(dest)
        raise
    except Exception:
        _safe_unlink(dest)
        raise
    finally:
        await file.close()

    if size == 0:
        _safe_unlink(dest)
        raise HTTPException(400, "empty file")

    _UPLOADS[fid] = {
        "path": str(dest),
        "name": name,
        "created": time.time(),
        "size": size,
    }
    return {"id": fid, "name": name, "size": size}


@app.get("/")
async def index():
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/info")
async def info():
    from engine import DEFAULT_DELAY_MS, DELAY_STEP_MS, MAX_DELAY_MS, MIN_DELAY_MS

    return {
        "backend": app.state.backend,
        "mode": app.state.mode,
        "adaptive": app.state.mode == "adaptive",
        "delay_ms": app.state.delay_ms,
        "delay_ms_default": DEFAULT_DELAY_MS,
        "delay_ms_min": MIN_DELAY_MS,
        "delay_ms_max": MAX_DELAY_MS,
        "delay_ms_step": DELAY_STEP_MS,
        "delay_ms_enabled": app.state.mode != "adaptive",
        "enable_live": bool(getattr(app.state, "enable_live", True)),
        "enable_offline": bool(getattr(app.state, "enable_offline", True)),
        "languages": ["zh", "en", "cs"],
        "language": getattr(app.state, "default_language", "zh"),
        **(
            _language_tag_status(_tf_model)
            if app.state.backend == "transformers" and app.state.mode == "adaptive"
            else {
                # vLLM proxy forwards language; HF/true-stream serve apply the tag.
                "language_tag_enabled": app.state.backend == "vllm",
                "language_tag_loaded": app.state.backend == "vllm",
            }
        ),
    }


@app.websocket("/ws")
async def websocket_asr(websocket: WebSocket):
    await websocket.accept()
    if app.state.backend == "transformers":
        if app.state.mode == "adaptive":
            await _handle_transformers_adaptive_ws(websocket)
        else:
            await _handle_transformers_ws(websocket, app.state.delay_ms)
    else:
        await _handle_vllm_ws(websocket, app.state.vllm_url, app.state.model)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Voxtral Realtime unified demo (transformers / vllm, fixed / adaptive)",
    )
    parser.add_argument(
        "--backend",
        choices=["transformers", "vllm"],
        default="transformers",
        help="ASR backend (default: transformers)",
    )
    parser.add_argument(
        "--mode",
        choices=["fixed", "adaptive"],
        default="fixed",
        help="fixed τ streaming, or interleaved adaptive latency",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Local model directory or Hugging Face model ID",
    )
    parser.add_argument("--lora-path", default="", help="Optional LoRA adapter (transformers/adaptive)")
    parser.add_argument(
        "--language",
        default=os.environ.get("LANGUAGE", "zh"),
        help="Default language_tag: zh / en / cs (LANGUAGE=)",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", default="cuda:0", help="LM / default device (transformers)")
    parser.add_argument(
        "--enc-device",
        default=os.environ.get("ENC_DEVICE") or os.environ.get("VOXTRAL_ENC_DEVICE") or "",
        help="Audio encoder GPU (ENC_DEVICE / VOXTRAL_ENC_DEVICE). Empty = same as --device",
    )
    parser.add_argument(
        "--delay-ms",
        type=int,
        default=480,
        help="Streaming target latency τ in ms (fixed mode only; 80–2400, step 80)",
    )
    parser.add_argument(
        "--emit-word-prob-threshold",
        type=float,
        default=None,
        help="Live LISTEN: emit when P(W)>=threshold (default argmax==W)",
    )
    parser.add_argument(
        "--listen-chunk-size",
        type=int,
        default=1,
        help="Live LISTEN frames per LM step (1=every ~80ms; streaming ASR default)",
    )
    parser.add_argument(
        "--max-decode-tokens",
        type=int,
        default=32,
        help="Adaptive DECODE cap per emit segment (smaller is faster for live)",
    )
    parser.add_argument(
        "--live",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("ENABLE_LIVE", True),
        help="Adaptive live streaming decode (ENABLE_LIVE=0 / --no-live to skip)",
    )
    parser.add_argument(
        "--offline",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("ENABLE_OFFLINE", True),
        help="Adaptive offline whole-utt decode (ENABLE_OFFLINE=0 / --no-offline to skip)",
    )
    parser.add_argument("--vllm-host", default="127.0.0.1", help="vLLM / adaptive-serve host")
    parser.add_argument("--vllm-port", type=int, default=8000, help="vLLM / adaptive-serve port")
    return parser.parse_args()


def main():
    from engine import validate_delay_ms

    args = parse_args()
    if args.mode == "fixed":
        validate_delay_ms(args.delay_ms)

    repo_id = args.repo_id

    app.state.backend = args.backend
    app.state.mode = args.mode
    app.state.repo_id = repo_id
    app.state.lora_path = args.lora_path
    from models.voxtral_asr.language_tag import language_id_name, resolve_language_id

    app.state.default_language = language_id_name(resolve_language_id(args.language))
    app.state.device = args.device
    app.state.enc_device = str(args.enc_device or "").strip()
    app.state.delay_ms = args.delay_ms
    rec_thr = args.emit_word_prob_threshold
    app.state.emit_thr = None if rec_thr is None or float(rec_thr) <= 0 else float(rec_thr)
    app.state.listen_chunk_size = max(1, int(args.listen_chunk_size))
    app.state.offline_listen_chunk_size = 16
    app.state.max_decode_tokens = max(8, int(args.max_decode_tokens))
    app.state.enable_live = bool(args.live)
    app.state.enable_offline = bool(args.offline)
    if not app.state.enable_live and not app.state.enable_offline:
        print("WARN: --no-live and --no-offline are both set; nothing will decode")
    app.state.model = repo_id
    app.state.vllm_url = f"ws://{args.vllm_host}:{args.vllm_port}/v1/realtime"

    print(f"Backend: {args.backend}")
    print(f"Mode:    {args.mode}")
    print(f"Model:   {repo_id}")
    print(
        f"Devices: encoder={app.state.enc_device or args.device}  "
        f"LM={args.device}"
        + ("" if app.state.enc_device and app.state.enc_device != args.device else " (same GPU)")
    )
    if args.mode == "fixed":
        print(f"Delay τ: {args.delay_ms} ms")
    if args.mode == "adaptive":
        print(f"LISTEN chunk: live={app.state.listen_chunk_size} offline=16")
        print(
            f"Infer:    live={'on' if app.state.enable_live else 'off'}  "
            f"offline={'on' if app.state.enable_offline else 'off'}  "
            f"language={app.state.default_language}"
        )
    print(f"Demo:    http://{args.host}:{args.port}")
    if args.backend == "vllm" and args.mode == "adaptive":
        print("Hint:    start server with  MODE=adaptive_vllm bash vllm/serve.sh")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
