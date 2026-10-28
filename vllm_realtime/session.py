"""Persistent-KV listen↔decode session over streaming prompt embeddings."""

from __future__ import annotations

import asyncio
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

import torch

PAD_ID = 32
WORD_ID = 33
BOS_ID = 1
EOS_ID = 2
N_LEFT_PAD = 32
SECONDS_PER_FRAME = 0.08


@dataclass(frozen=True)
class RestoreFrame:
    """One historical audio frame and the commits made after that frame."""

    embedding: torch.Tensor
    commits: tuple[tuple[int, ...], ...] = ()


def _edge_script(value: str, *, from_end: bool) -> str:
    """Return the script of the first/last letter-like character.

    Punctuation and whitespace at the edge are ignored.  Looking at one
    character instead of a whitespace-delimited word is important for mixed
    text such as ``你说TODAY``: the trailing edge is still Latin.
    """
    chars = list(str(value or "").strip())
    if from_end:
        chars.reverse()
    for char in chars:
        if re.match(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", char):
            return "cjk"
        if re.match(r"[0-9A-Za-z\u00c0-\u024f]", char):
            return "latin"
        if char.isalnum():
            return "other"
    return ""


def _needs_script_separator(previous: str, current: str) -> bool:
    left = _edge_script(previous, from_end=True)
    right = _edge_script(current, from_end=False)
    return (
        (left == "latin" and right in {"latin", "cjk"})
        or (left == "cjk" and right == "latin")
    )


def _append_emit_text(previous: str, current: str) -> str:
    text = str(current or "").strip()
    if not text:
        return previous or ""
    previous = previous or ""
    separator = " " if previous and _needs_script_separator(previous, text) else ""
    return previous + separator + text


def as_prompt_embeds(embed: torch.Tensor) -> torch.Tensor:
    value = embed.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    if value.ndim == 1:
        value = value.unsqueeze(0)
    elif value.ndim > 2:
        value = value.view(-1, value.shape[-1])
    return value


class PersistentEmbedsSession:
    """One resumable vLLM request for a whole utterance."""

    def __init__(
        self,
        engine: Any,
        tokenizer: Any,
        table: torch.Tensor,
        *,
        on_delta: Optional[Callable[[str, dict[str, Any]], Awaitable[None]]] = None,
        max_decode_tokens: int = 32,
        n_left_pad: int = N_LEFT_PAD,
        continuous_decode: bool = True,
        language_tag_table: Optional[torch.Tensor] = None,
        language_id: int = 0,
        restore_frames: Optional[list[RestoreFrame]] = None,
    ) -> None:
        self.engine = engine
        self.tokenizer = tokenizer
        self.table = table
        self.language_tag_table = language_tag_table
        self.language_id = int(language_id)
        self.on_delta = on_delta
        self.max_decode_tokens = max_decode_tokens
        self.n_left_pad = int(n_left_pad)
        self.continuous_decode = bool(continuous_decode)
        self.restore_frames = list(restore_frames or [])
        self.request_id = f"adaptive-true-{uuid.uuid4()}"
        self.final_text = ""
        self.emits: list[str] = []
        self.latencies_ms: list[float] = []
        self.audio_frames = 0
        self.live_audio_frames = 0
        self.restored_audio_frames = len(self.restore_frames)
        self.restored_commit_tokens = sum(
            1 + len(token_ids)
            for frame in self.restore_frames
            for token_ids in frame.commits
        )
        self.restore_wall_s = 0.0
        self.enc_wall_s = 0.0
        self.lm_wall_s = 0.0
        self.error: Optional[str] = None
        self.warmup_event = asyncio.Event()
        self._speech_t0: Optional[float] = None
        self._count_lm = False
        self._in_q: asyncio.Queue = asyncio.Queue()
        self._out_q: asyncio.Queue = asyncio.Queue()
        self._audio_q: asyncio.Queue = asyncio.Queue()
        self._consume_task: Optional[asyncio.Task] = None
        self._state_task: Optional[asyncio.Task] = None
        self._last_output_ready_ns = 0
        self._pending_context_ids: list[int] = []
        self.decode_blocks = 0
        self.decode_tokens = 0
        self.decode_wall_s = 0.0
        self.decode_block_latencies_ms: list[float] = []
        self.trigger_to_first_token_latencies_ms: list[float] = []
        self.trigger_to_block_done_latencies_ms: list[float] = []
        self.decode_token_latencies_ms: list[float] = []
        self.decode_block_timings: list[dict[str, Any]] = []
        self.decode_blocks_without_text = 0
        self.decode_token_timing_unmeasured = 0
        self.encoder_serialize_latencies_ms: list[float] = []
        self.encoder_ipc_latencies_ms: list[float] = []
        self.bridge_decode_latencies_ms: list[float] = []
        self.encoder_queue_latencies_ms: list[float] = []
        self.encoder_to_lm_latencies_ms: list[float] = []
        self.encoder_stats: dict[str, Any] = {}

    @property
    def kv_steps(self) -> int:
        """Conservative count of rows retained by the persistent request."""
        return (
            1
            + self.n_left_pad
            + self.restored_audio_frames
            + self.restored_commit_tokens
            + self.live_audio_frames
            + self.decode_tokens
        )

    async def start(self) -> None:
        from vllm.engine.protocol import StreamingInput
        from vllm.inputs import EmbedsPrompt
        from vllm.sampling_params import RequestOutputKind, SamplingParams

        self._streaming_input = StreamingInput
        self._embeds_prompt = EmbedsPrompt
        self._decode_sampling = SamplingParams(
            temperature=0.0,
            # Match the legacy loop: keep at most max_decode_tokens text
            # tokens and sample one extra token so EOS can terminate the block.
            max_tokens=self.max_decode_tokens + 1,
            stop_token_ids=[EOS_ID],
            ignore_eos=True,
            detokenize=False,
            skip_special_tokens=False,
            output_kind=RequestOutputKind.DELTA,
        )

        async def inputs():
            while True:
                yield await self._in_q.get()

        async def consume() -> None:
            async for output in self.engine.generate(
                prompt=inputs(),
                sampling_params=SamplingParams(
                    temperature=0.0,
                    max_tokens=1,
                    output_kind=RequestOutputKind.DELTA,
                ),
                request_id=self.request_id,
            ):
                # Capture availability at the engine iterator boundary.  The
                # state machine may be busy forwarding an earlier preview, so
                # timing at _out_q.get() would include websocket backpressure.
                await self._out_q.put((output, time.perf_counter_ns()))

        self._consume_task = asyncio.create_task(consume())
        self._state_task = asyncio.create_task(self._state_machine())

    async def wait_warmup(self, timeout: float = 120.0) -> None:
        await asyncio.wait_for(self.warmup_event.wait(), timeout=timeout)

    async def feed_audio(
        self,
        embed: torch.Tensor,
        *,
        enc_wall_s: Optional[float] = None,
        timing: Optional[dict[str, Any]] = None,
    ) -> None:
        if enc_wall_s is not None:
            self.enc_wall_s = float(enc_wall_s)
        timing = dict(timing or {})
        timing["_session_enqueue_ns"] = time.perf_counter_ns()
        await self._audio_q.put((embed, timing))

    async def _next_audio_embed(self) -> Optional[torch.Tensor]:
        item = await self._audio_q.get()
        if item is None:
            return None
        embed, timing = item
        consumed_ns = time.perf_counter_ns()
        enqueue_ns = timing.get("_session_enqueue_ns")
        if enqueue_ns is not None:
            self.encoder_queue_latencies_ms.append(
                (consumed_ns - int(enqueue_ns)) / 1_000_000.0
            )
        worker_send_ns = timing.get("worker_send_ns")
        if worker_send_ns is not None:
            self.encoder_to_lm_latencies_ms.append(
                (consumed_ns - int(worker_send_ns)) / 1_000_000.0
            )
        for key, target in (
            ("serialize_ms", self.encoder_serialize_latencies_ms),
            ("_ipc_ms", self.encoder_ipc_latencies_ms),
            ("_bridge_decode_ms", self.bridge_decode_latencies_ms),
        ):
            value = timing.get(key)
            if value is not None:
                target.append(float(value))
        return embed

    def rtf_event(self) -> dict[str, Any]:
        n_speech = max(self.audio_frames - self.n_left_pad, 0)
        audio_s = n_speech * SECONDS_PER_FRAME
        wall = (
            time.perf_counter() - self._speech_t0 if self._speech_t0 is not None else 0.0
        )
        enc = self.enc_wall_s
        lm = self.lm_wall_s
        return {
            "type": "rtf",
            "session_id": self.request_id,
            "audio_s": round(audio_s, 3),
            "wall_total_s": round(wall, 3),
            "wall_enc_s": round(enc, 4),
            "wall_lm_s": round(lm, 4),
            "rtf_total": round(wall / audio_s, 3) if audio_s > 0.0 else -1.0,
            "rtf_enc": round(enc / audio_s, 4) if audio_s > 0.0 else -1.0,
            "rtf_lm": round(lm / audio_s, 4) if audio_s > 0.0 else -1.0,
            "frame": n_speech - 1,
            "audio_frames": self.audio_frames,
            "lm_steps": len(self.latencies_ms),
            "decode_blocks": self.decode_blocks,
            "decode_tokens": self.decode_tokens,
            "audio_queue_depth": self._audio_q.qsize(),
        }

    async def finish(self) -> str:
        await self._audio_q.put(None)
        if self._state_task is not None:
            await self._state_task
        await self._stop_engine()
        return self.final_text

    async def abort(self) -> None:
        self.error = self.error or "aborted"
        try:
            self._audio_q.put_nowait(None)
        except Exception:  # noqa: BLE001
            pass
        for task in (self._state_task, self._consume_task):
            if task is not None and not task.done():
                task.cancel()
        await self._stop_engine()

    async def _stop_engine(self) -> None:
        if self._consume_task is not None and not self._consume_task.done():
            self._consume_task.cancel()
            try:
                await self._consume_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        abort = getattr(self.engine, "abort", None)
        if abort is not None:
            try:
                result = abort(self.request_id)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:  # noqa: BLE001
                pass

    async def _step(self, embed: torch.Tensor) -> int:
        started = time.perf_counter()
        pending_context_ids = list(self._pending_context_ids)
        prompt_rows = [
            as_prompt_embeds(self._with_language(self._token_embed(token_id)))
            for token_id in pending_context_ids
        ]
        prompt_rows.append(as_prompt_embeds(self._with_language(embed)))
        self._pending_context_ids.clear()
        await self._in_q.put(
            self._streaming_input(
                prompt=self._embeds_prompt(
                    prompt_embeds=torch.cat(prompt_rows, dim=0)
                )
            )
        )
        output, output_ready_ns = await self._out_q.get()
        dt = time.perf_counter() - started
        self.latencies_ms.append(dt * 1000.0)
        if self._count_lm:
            self.lm_wall_s += dt
        self._last_output_ready_ns = output_ready_ns
        return int(output.outputs[0].token_ids[-1])

    async def _restore_interleaved_context(self) -> None:
        """Prefill historical audio and commit tokens in their original order."""
        if not self.restore_frames:
            return
        rows: list[torch.Tensor] = []
        for frame in self.restore_frames:
            rows.append(as_prompt_embeds(self._with_language(frame.embedding)))
            for token_ids in frame.commits:
                rows.append(
                    as_prompt_embeds(
                        self._with_language(self._token_embed(WORD_ID))
                    )
                )
                rows.extend(
                    as_prompt_embeds(
                        self._with_language(self._token_embed(int(token_id)))
                    )
                    for token_id in token_ids
                )
        if not rows:
            return
        started = time.perf_counter()
        await self._in_q.put(
            self._streaming_input(
                prompt=self._embeds_prompt(prompt_embeds=torch.cat(rows, dim=0))
            )
        )
        # This sampled decision belongs to historical context.  Consume and
        # discard it: only the supplied interleaved rows must enter the KV.
        await self._out_q.get()
        self.restore_wall_s = time.perf_counter() - started

    async def _decode_to_eos(self, trigger_ns: int) -> list[int]:
        """Let vLLM decode one complete emit block without frontend round trips."""
        previous_ready_ns = trigger_ns
        first_text_ns: Optional[int] = None
        block_done_ns = trigger_ns
        await self._in_q.put(
            self._streaming_input(
                prompt=self._embeds_prompt(
                    prompt_embeds=as_prompt_embeds(
                        self._with_language(self._token_embed(WORD_ID))
                    )
                ),
                sampling_params=self._decode_sampling,
            )
        )

        emit_ids: list[int] = []
        context_ids: list[int] = []
        finished = False
        while not finished:
            output, output_ready_ns = await self._out_q.get()
            block_done_ns = output_ready_ns
            token_dt_ms = (output_ready_ns - previous_ready_ns) / 1_000_000.0
            previous_ready_ns = output_ready_ns
            self.latencies_ms.append(token_dt_ms)
            if self._count_lm:
                self.lm_wall_s += token_dt_ms / 1000.0

            candidate = output.outputs[0]
            token_ids = [int(token_id) for token_id in candidate.token_ids]
            if len(token_ids) == 1:
                self.decode_token_latencies_ms.append(token_dt_ms)
            else:
                self.decode_token_timing_unmeasured += len(token_ids)
            for token in token_ids:
                self.decode_tokens += 1
                if token == EOS_ID:
                    finished = True
                    break
                if len(context_ids) >= self.max_decode_tokens:
                    # This is the legacy loop's final look-ahead token. vLLM
                    # leaves it uncomputed and the next streaming update drops it.
                    finished = True
                    break
                context_ids.append(token)
                if token not in (PAD_ID, WORD_ID, BOS_ID):
                    if first_text_ns is None:
                        first_text_ns = output_ready_ns
                    emit_ids.append(token)
                    piece = self.tokenizer.decode(
                        [token], skip_special_tokens=True
                    )
                    block = self.tokenizer.decode(
                        emit_ids, skip_special_tokens=True
                    )
                    if block and "\ufffd" not in block:
                        await self._preview(piece, block)

            if candidate.finish_reason is not None:
                finished = True

        block_ms = (block_done_ns - trigger_ns) / 1_000_000.0
        first_token_ms = (
            (first_text_ns - trigger_ns) / 1_000_000.0
            if first_text_ns is not None
            else None
        )
        self.decode_blocks += 1
        self.decode_wall_s += block_ms / 1000.0
        self.decode_block_latencies_ms.append(block_ms)
        self.trigger_to_block_done_latencies_ms.append(block_ms)
        if first_token_ms is None:
            self.decode_blocks_without_text += 1
        else:
            self.trigger_to_first_token_latencies_ms.append(first_token_ms)
        self.decode_block_timings.append(
            {
                "trigger_to_first_token_ms": first_token_ms,
                "trigger_to_block_done_ms": block_ms,
                "tokens": len(context_ids),
            }
        )
        self._pending_context_ids = context_ids
        return emit_ids

    def _token_embed(self, token_id: int) -> torch.Tensor:
        return self.table[int(token_id)]

    def set_language_id(self, language_id: int) -> None:
        self.language_id = int(language_id)

    def _with_language(self, embed: torch.Tensor) -> torch.Tensor:
        table = self.language_tag_table
        if table is None:
            return embed
        tag = table[int(self.language_id)].to(dtype=embed.dtype)
        if embed.ndim == 1:
            return embed + tag
        return embed + tag.view(*([1] * (embed.ndim - 1)), -1)

    async def _preview(self, piece: str, block: str) -> None:
        if self.on_delta is None:
            return
        await self.on_delta(
            piece,
            {
                "mode": "decode",
                "piece": piece,
                "block": block,
                "audio_frames": self.audio_frames,
            },
        )

    async def _emit(self, token_ids: list[int], extra: dict[str, Any]) -> None:
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        if not text:
            return
        self.emits.append(text)
        self.final_text = _append_emit_text(self.final_text, text)
        extra = {
            "mode": "emit",
            "_token_ids": tuple(int(token_id) for token_id in token_ids),
            **extra,
        }
        if self.on_delta is not None:
            await self.on_delta(text, extra)

    async def _state_machine(self) -> None:
        try:
            await self._step(self._token_embed(BOS_ID))
            while self.audio_frames < self.n_left_pad:
                embed = await self._next_audio_embed()
                if embed is None:
                    return
                self.audio_frames += 1
                await self._step(embed)
            await self._restore_interleaved_context()
            # Startup BOS + left-pad steps are preheating, not user inference.
            # Keep completion statistics scoped to post-warmup live work.
            self.latencies_ms.clear()
            self.lm_wall_s = 0.0
            self.encoder_serialize_latencies_ms.clear()
            self.encoder_ipc_latencies_ms.clear()
            self.bridge_decode_latencies_ms.clear()
            self.encoder_queue_latencies_ms.clear()
            self.encoder_to_lm_latencies_ms.clear()
            self._count_lm = True
            self.warmup_event.set()
            while True:
                embed = await self._next_audio_embed()
                if embed is None:
                    return
                self.audio_frames += 1
                self.live_audio_frames += 1
                if self._speech_t0 is None:
                    self._speech_t0 = time.perf_counter()
                decision = await self._step(embed)
                if decision != WORD_ID:
                    continue
                trigger_ns = time.perf_counter_ns()
                if self.continuous_decode:
                    emit_ids = await self._decode_to_eos(trigger_ns)
                    await self._emit(
                        emit_ids,
                        {
                            "audio_frames": self.audio_frames,
                            "live_audio_frames": self.live_audio_frames,
                            "lm_steps": len(self.latencies_ms),
                        },
                    )
                    continue
                block_tokens = 1
                token = await self._step(self._token_embed(WORD_ID))
                token_ready_ns = self._last_output_ready_ns
                previous_ready_ns = trigger_ns
                first_text_ns: Optional[int] = None
                emit_ids: list[int] = []
                for _ in range(self.max_decode_tokens):
                    if token == EOS_ID:
                        break
                    token_dt_ms = (
                        token_ready_ns - previous_ready_ns
                    ) / 1_000_000.0
                    self.decode_token_latencies_ms.append(token_dt_ms)
                    previous_ready_ns = token_ready_ns
                    if token not in (PAD_ID, WORD_ID, BOS_ID):
                        if first_text_ns is None:
                            first_text_ns = token_ready_ns
                        emit_ids.append(token)
                        piece = self.tokenizer.decode(
                            [token], skip_special_tokens=True
                        )
                        block = self.tokenizer.decode(
                            emit_ids, skip_special_tokens=True
                        )
                        if block and "\ufffd" not in block:
                            await self._preview(piece, block)
                    token = await self._step(self._token_embed(token))
                    token_ready_ns = self._last_output_ready_ns
                    block_tokens += 1
                block_done_ns = token_ready_ns
                block_ms = (block_done_ns - trigger_ns) / 1_000_000.0
                first_token_ms = (
                    (first_text_ns - trigger_ns) / 1_000_000.0
                    if first_text_ns is not None
                    else None
                )
                self.decode_blocks += 1
                self.decode_tokens += block_tokens
                self.decode_wall_s += block_ms / 1000.0
                self.decode_block_latencies_ms.append(block_ms)
                self.trigger_to_block_done_latencies_ms.append(block_ms)
                if first_token_ms is None:
                    self.decode_blocks_without_text += 1
                else:
                    self.trigger_to_first_token_latencies_ms.append(first_token_ms)
                self.decode_block_timings.append(
                    {
                        "trigger_to_first_token_ms": first_token_ms,
                        "trigger_to_block_done_ms": block_ms,
                        "tokens": block_tokens,
                    }
                )
                await self._emit(
                    emit_ids,
                    {
                        "audio_frames": self.audio_frames,
                        "live_audio_frames": self.live_audio_frames,
                        "lm_steps": len(self.latencies_ms),
                    },
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)
            raise
