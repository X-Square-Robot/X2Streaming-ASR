"""Scheduler extension adding resumable prompt-embedding sessions."""

from __future__ import annotations

import torch

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.request import RequestStatus, StreamingUpdate


_original_from_request = StreamingUpdate.from_request.__func__


def _from_request_with_embeds(cls, request):
    update = _original_from_request(cls, request)
    if update is not None:
        update.prompt_embeds = request.prompt_embeds
    return update


StreamingUpdate.from_request = classmethod(_from_request_with_embeds)


def _append_embed_rows(session, rows: torch.Tensor) -> torch.Tensor:
    """Append rows to ``session.prompt_embeds`` in amortized O(len(rows)).

    Re-concatenating the whole history on every streaming update is
    O(history) per audio frame; it dominated the EngineCore busy loop and
    serialized concurrent sessions behind each other.  Rows live in a
    geometrically grown buffer and ``prompt_embeds`` is a view of its prefix.
    Existing rows are never rewritten and growth allocates a new buffer, so
    views already handed to the model runner stay valid.
    """
    old = session.prompt_embeds
    used = int(old.shape[0])
    need = used + int(rows.shape[0])
    buf = getattr(session, "_prompt_embeds_buf", None)
    if buf is None or buf.data_ptr() != old.data_ptr() or need > buf.shape[0]:
        buf = old.new_empty((max(need, 2 * used, 256), *old.shape[1:]))
        buf[:used].copy_(old)
        session._prompt_embeds_buf = buf
    buf[used:need].copy_(rows)
    return buf[:need]


class StreamingEmbedsScheduler(Scheduler):
    def _update_request_as_session(self, session, update) -> None:
        prompt_embeds = getattr(update, "prompt_embeds", None)
        if prompt_embeds is None:
            return super()._update_request_as_session(session, update)
        if session.prompt_embeds is None:
            raise ValueError("cannot switch a token session to prompt embeddings")

        # The final sampled token was not computed into KV and is intentionally
        # discarded, matching the token-session implementation in the base
        # scheduler. Continuous decode can leave earlier generated tokens in KV.
        num_computed_tokens = session.num_computed_tokens
        del session._all_token_ids[num_computed_tokens:]
        session._output_token_ids.clear()
        old_prompt_tokens = int(session.prompt_embeds.shape[0])
        missing = num_computed_tokens - old_prompt_tokens
        if missing < 0 or int(prompt_embeds.shape[0]) <= missing:
            raise ValueError(
                "streaming prompt-embedding continuation is missing generated "
                f"token rows: computed={num_computed_tokens}, "
                f"prompt={old_prompt_tokens}, update={prompt_embeds.shape[0]}"
            )
        # Generated tokens already have KV entries but no prompt-embedding rows.
        # The frontend replays exactly those rows before the next new audio row.
        session.prompt_embeds = _append_embed_rows(session, prompt_embeds)
        new_rows = int(prompt_embeds.shape[0]) - missing
        session._all_token_ids.extend([0] * new_rows)
        session.update_block_hashes()
        session.num_prompt_tokens = int(session.prompt_embeds.shape[0])
        session.arrival_time = update.arrival_time
        session.sampling_params = update.sampling_params
        assert session.sampling_params.max_tokens is not None
        session.max_tokens = session.sampling_params.max_tokens
        if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            self.num_waiting_for_streaming_input -= 1
        session.status = RequestStatus.WAITING
        if self.log_stats:
            session.record_event(EngineCoreEventType.QUEUED)
