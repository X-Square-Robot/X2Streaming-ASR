"""Async client for the streaming-encoder sidecar subprocess.

One sidecar process serves many concurrent sessions.  Every command carries a
``sid`` and the worker tags every session event with it; the reader task routes
events to the per-session queue of the matching :class:`EncoderStream`.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import torch

PREFIX = "ENCODER_JSON:"


class EncoderStream:
    """Handle for one live encoder session on the shared sidecar."""

    def __init__(self, bridge: "EncoderBridge", sid: str) -> None:
        self.bridge = bridge
        self.sid = sid
        self.events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def append_pcm16(self, pcm16: bytes) -> None:
        await self.bridge.send(
            "append",
            sid=self.sid,
            pcm16=base64.b64encode(pcm16).decode("ascii"),
        )

    async def finish(self) -> None:
        await self.bridge.send("finish", sid=self.sid)

    async def abort(self) -> None:
        try:
            await self.bridge.send("abort", sid=self.sid)
        finally:
            self.close()

    def close(self) -> None:
        """Stop routing events; late events for this sid are dropped."""
        self.bridge._routes.pop(self.sid, None)


class EncoderBridge:
    def __init__(
        self,
        *,
        python: str,
        worker: Path,
        model: str,
        gpu: str,
        lora: str = "",
        feature_prefetch: bool = True,
        gpu_encode_overlap: bool = True,
    ) -> None:
        self.python = python
        self.worker = worker
        self.model = model
        self.gpu = gpu
        self.lora = lora
        self.feature_prefetch = bool(feature_prefetch)
        self.gpu_encode_overlap = bool(gpu_encode_overlap)
        self.process: asyncio.subprocess.Process | None = None
        # Process-level events (ready / shutdown); session events are routed.
        self.events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._routes: dict[str, asyncio.Queue[dict[str, Any]]] = {}
        self._reader_task: asyncio.Task | None = None
        self._write_lock = asyncio.Lock()

    async def start(self) -> None:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.gpu
        cmd = [
            self.python,
            str(self.worker),
            "--model",
            self.model,
            "--device",
            "cuda:0",
            (
                "--feature-prefetch"
                if self.feature_prefetch
                else "--no-feature-prefetch"
            ),
            (
                "--gpu-encode-overlap"
                if self.gpu_encode_overlap
                else "--no-gpu-encode-overlap"
            ),
        ]
        if self.lora:
            cmd.extend(["--lora", self.lora])
        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )
        self._reader_task = asyncio.create_task(self._read_stdout())
        event = await asyncio.wait_for(self.events.get(), timeout=180)
        if event.get("type") != "ready":
            raise RuntimeError(f"encoder failed to start: {event}")

    async def _read_stdout(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        while line := await self.process.stdout.readline():
            text = line.decode("utf-8", errors="replace").rstrip()
            if text.startswith(PREFIX):
                event = json.loads(text[len(PREFIX) :])
                event["_bridge_receive_ns"] = time.perf_counter_ns()
                sid = event.get("sid")
                if sid is None:
                    if event.get("type") == "error":
                        print(f"[adaptive-encoder] error: {event}", flush=True)
                    await self.events.put(event)
                    continue
                route = self._routes.get(sid)
                if route is not None:
                    route.put_nowait(event)
            elif text:
                print(f"[adaptive-encoder] {text}", flush=True)
        exited = {
            "type": "error",
            "message": f"encoder process exited with {self.process.returncode}",
        }
        await self.events.put(exited)
        for route in list(self._routes.values()):
            route.put_nowait(dict(exited))

    async def send(self, kind: str, **payload: Any) -> None:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("encoder process is not running")
        message = json.dumps(
            {"type": kind, **payload}, separators=(",", ":")
        ).encode("utf-8") + b"\n"
        async with self._write_lock:
            self.process.stdin.write(message)
            await self.process.stdin.drain()

    @property
    def active_sessions(self) -> int:
        return len(self._routes)

    async def open_session(self) -> EncoderStream:
        stream = EncoderStream(self, uuid.uuid4().hex)
        self._routes[stream.sid] = stream.events
        try:
            await self.send("start", sid=stream.sid)
            while True:
                event = await asyncio.wait_for(stream.events.get(), timeout=60)
                kind = event.get("type")
                if kind == "started":
                    return stream
                if kind == "embedding":
                    # Left-pad frames can win the race against the started ack.
                    stream.events.put_nowait(event)
                    return stream
                if kind == "error":
                    raise RuntimeError(
                        event.get("message", "encoder start failed")
                    )
        except BaseException:
            try:
                await asyncio.wait_for(stream.abort(), timeout=2.0)
            except Exception:  # noqa: BLE001
                stream.close()
            raise

    @staticmethod
    def decode_embedding(event: dict[str, Any]) -> torch.Tensor:
        started = time.perf_counter_ns()
        raw = base64.b64decode(event["bf16"])
        bits = torch.frombuffer(bytearray(raw), dtype=torch.uint16)
        embedding = bits.view(torch.bfloat16).reshape(event["shape"]).clone()
        finished = time.perf_counter_ns()
        event["_bridge_decode_ms"] = (finished - started) / 1_000_000.0
        worker_send_ns = event.get("worker_send_ns")
        bridge_receive_ns = event.get("_bridge_receive_ns")
        if worker_send_ns is not None and bridge_receive_ns is not None:
            event["_ipc_ms"] = (
                int(bridge_receive_ns) - int(worker_send_ns)
            ) / 1_000_000.0
        return embedding

    async def shutdown(self) -> None:
        if self.process is None:
            return
        if self.process.returncode is None:
            try:
                await self.send("shutdown")
                await asyncio.wait_for(self.process.wait(), timeout=10)
            except Exception:  # noqa: BLE001
                self.process.terminate()
        if self._reader_task is not None:
            self._reader_task.cancel()
