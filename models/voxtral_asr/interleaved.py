"""Interleaved adaptive-latency path (listen ↔ decode state machine).

True sequence interleaving — audio and text occupy **different** positions
(not ``text_emb + audio_emb`` at the same index).

Inference mirrors the same two-phase machine:
  LISTEN — feed next audio frame, predict PAD (wait) or W (emit)
  DECODE — feed text tokens until EOS, then return to LISTEN

Only used when ``adaptive_latency=True``. Fixed-delay path never imports this.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from models.voxtral_asr.data import (
    ADAPTIVE_PROTOCOL_DELAY_MS,
    STREAMING_PAD_ID,
    STREAMING_WORD_ID,
    STREAMING_WORD_OFF_ID,
    encode_audios_with_delay,
    encode_audio_with_delay,
    load_audio,
)
from models.voxtral_asr.devices import enc_device_of, lm_device_of, to_device


def _unwrap_voxtral_cg(model: Any) -> Any:
    """Return ``VoxtralRealtimeForConditionalGeneration`` (unwrap PEFT/DDP)."""
    m = model
    if hasattr(m, "module"):
        m = m.module
    if hasattr(m, "get_base_model"):
        try:
            m = m.get_base_model()
        except Exception:
            pass
    # PeftModel sometimes nests as .base_model.model
    if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
        inner = m.base_model.model
        if hasattr(inner, "lm_head") and hasattr(inner, "model"):
            m = inner
    return m


def _lm_step_embeds(
    voxtral_cg: Any,
    inputs_embeds: torch.Tensor,
    past_key_values=None,
    num_delay_tokens: int = 1,
    attention_mask: Optional[torch.Tensor] = None,
    language_ids: Optional[torch.Tensor] = None,
    *,
    return_all_logits: bool = False,
):
    """LM forward from embeds (no audio add).

    ``inputs_embeds`` may be ``[B, 1, H]`` (one step) or ``[B, T, H]`` (prefill).
    ``attention_mask`` optional ``[B, S]`` (full length including past when
    continuing with cache — same convention as HF generate).
    When ``return_all_logits=True``, returns ``[B, T, V]`` (all new positions);
    otherwise ``[B, 1, V]`` for the last position only.
    """
    if inputs_embeds.dim() == 2:
        inputs_embeds = inputs_embeds.unsqueeze(0)
    from models.voxtral_asr.language_tag import apply_language_tag

    inputs_embeds = apply_language_tag(inputs_embeds, voxtral_cg, language_ids)
    time_tensor = torch.full(
        (1,),
        int(num_delay_tokens),
        device=inputs_embeds.device,
        dtype=inputs_embeds.dtype,
    )
    # AdaRMS delay conditioning is an identity in adaptive inference.
    t_cond = voxtral_cg.model.time_embedding(time_tensor)[None, ...]

    outputs = voxtral_cg.model.language_model(
        inputs_embeds=inputs_embeds,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        use_cache=True,
        t_cond=t_cond,
    )
    hidden = outputs.last_hidden_state
    if return_all_logits:
        logits = voxtral_cg.lm_head(hidden)
    else:
        logits = voxtral_cg.lm_head(hidden[:, -1:, :])
    return logits, outputs.past_key_values


def _crop_past_kv(past: Any, keep_len: int) -> Any:
    """Crop KV cache to absolute sequence length ``keep_len``."""
    if past is None:
        return None
    keep_len = int(keep_len)
    if hasattr(past, "crop"):
        past.crop(keep_len)
        return past
    # Legacy tuple[(k,v), ...] with seq on dim 2.
    cropped = []
    for layer in past:
        if isinstance(layer, (tuple, list)) and len(layer) >= 2:
            k, v = layer[0], layer[1]
            cropped.append((k[:, :, :keep_len, :], v[:, :, :keep_len, :]))
        else:
            raise TypeError(f"unsupported past_key_values layer type: {type(layer)}")
    return tuple(cropped)


def _past_seq_len(past: Any) -> int:
    if past is None:
        return 0
    if hasattr(past, "get_seq_length"):
        return int(past.get_seq_length())
    layer0 = past[0]
    k = layer0[0] if isinstance(layer0, (tuple, list)) else layer0
    return int(k.shape[2])



def _adaptive_n_left_pad(processor: Any) -> int:
    cfg = getattr(processor, "mistral_common_audio_config", None)
    if cfg is None:
        return 32
    return int(getattr(cfg, "streaming_n_left_pad_tokens", None) or 32)


def _adaptive_seconds_per_token(processor: Any) -> float:
    """Audio-frame duration in seconds (typically 0.08 = 80 ms)."""
    sr = float(processor.feature_extractor.sampling_rate)
    hop = float(processor.feature_extractor.hop_length)
    alt = float(
        getattr(processor, "audio_length_per_tok", None)
        or getattr(
            getattr(processor, "mistral_common_audio_config", None),
            "audio_length_per_tok",
            8,
        )
        or 8
    )
    return alt * hop / sr


@torch.no_grad()
def _generate_interleaved_adaptive_from_embeds(
    model: Any,
    processor: Any,
    audio_embeds: torch.Tensor,
    *,
    max_decode_tokens: int = 64,
    max_new_tokens: Optional[int] = None,
    emit_word_prob_threshold: Optional[float] = None,
    emit_word_prob_space: str = "full_vocab",
    on_emit: Optional[Callable[[Dict[str, Any]], None]] = None,
    listen_chunk_size: int = 16,
) -> Dict[str, Any]:
    """LISTEN↔DECODE state machine given precomputed ``audio_embeds`` ``[T, H]``.

    ``listen_chunk_size`` > 1 feeds multiple audio frames per LM forward and
    crops the KV cache if an emit occurs mid-chunk (same decisions as step-by-step).
    Left-pad frames are always prefilled in one shot when ``listen_chunk_size > 1``.

    """
    vox = _unwrap_voxtral_cg(model)
    n_left_pad = _adaptive_n_left_pad(processor)
    seconds_per_token = _adaptive_seconds_per_token(processor)
    n_frames = int(audio_embeds.shape[0])
    chunk = max(1, int(listen_chunk_size))


    embed_fn = vox.model.get_input_embeddings()
    bos_id = processor.tokenizer.bos_token_id or 1
    eos_id = processor.tokenizer.eos_token_id or 2
    device = lm_device_of(vox)
    dtype = next(embed_fn.parameters()).dtype
    audio_embeds = to_device(audio_embeds, device, dtype=dtype)

    # Cache special-token embeds (decode path hits these often).
    bos_emb = embed_fn(
        torch.tensor([[bos_id]], device=device, dtype=torch.long)
    ).to(dtype=dtype)
    w_emb = embed_fn(
        torch.tensor([[STREAMING_WORD_ID]], device=device, dtype=torch.long)
    ).to(dtype=dtype)
    _tok_cache: Dict[int, torch.Tensor] = {
        bos_id: bos_emb,
        STREAMING_WORD_ID: w_emb,
    }

    def _tok_embed(tid: int) -> torch.Tensor:
        emb = _tok_cache.get(tid)
        if emb is None:
            t = torch.tensor([[tid]], device=device, dtype=torch.long)
            emb = embed_fn(t).to(dtype=dtype)
            # Bound cache size; specials already pinned.
            if len(_tok_cache) < 4096:
                _tok_cache[tid] = emb
        return emb

    def _finalize_emit(emit_meta: Optional[Dict[str, Any]], emit_ids: List[int]) -> None:
        if emit_meta is None:
            return
        chunk_text = processor.tokenizer.decode(
            emit_ids, skip_special_tokens=True,
        ).strip()
        rec = {
            "time_s": float(emit_meta["time_s"]),
            "frame": int(emit_meta["frame"]),
            "text": chunk_text,
        }
        if "p_word" in emit_meta:
            rec["p_word"] = emit_meta["p_word"]
        if "forced_by_prob" in emit_meta:
            rec["forced_by_prob"] = emit_meta["forced_by_prob"]
        emits.append(rec)
        if on_emit is not None:
            on_emit(rec)

    def _listen_decision(logits_row: torch.Tensor) -> Tuple[int, bool, Optional[float]]:
        argmax_tok = int(torch.argmax(logits_row).item())
        p_word: Optional[float] = None
        if emit_word_prob_threshold is None:
            do_emit = argmax_tok == STREAMING_WORD_ID
        else:
            if emit_word_prob_space == "pad_word":
                pair = logits_row[
                    [STREAMING_PAD_ID, STREAMING_WORD_ID]
                ].float()
                p_word = float(torch.softmax(pair, dim=-1)[1].item())
            else:
                probs = torch.softmax(logits_row.float(), dim=-1)
                p_word = float(probs[STREAMING_WORD_ID].item())
            do_emit = p_word >= float(emit_word_prob_threshold)
        return argmax_tok, do_emit, p_word

    def _enter_decode(
        frame_i_after: int,
        argmax_tok: int,
        p_word: Optional[float],
        past_in: Any,
    ) -> Tuple[Any, int]:
        nonlocal state, emit_meta, emit_ids, n_emits, decode_count, next_tok
        gen_ids.append(STREAMING_WORD_ID)
        emit_frame = frame_i_after - 1
        emit_meta = {
            "frame": emit_frame,
            "time_s": (frame_i_after - n_left_pad) * seconds_per_token,
        }
        if p_word is not None:
            emit_meta["p_word"] = round(p_word, 6)
            emit_meta["forced_by_prob"] = argmax_tok != STREAMING_WORD_ID
        emit_ids = []
        state = "decode"
        n_emits += 1
        decode_count = 0
        logits, past_out = _lm_step_embeds(vox, w_emb, past_in)
        next_tok = int(torch.argmax(logits[0, -1]).item())
        gen_ids.append(next_tok)
        return past_out, frame_i_after

    _logits, past = _lm_step_embeds(vox, bos_emb, None)
    seq_len = 1

    frame_i = 0
    state = "listen"
    text_ids: List[int] = []
    gen_ids: List[int] = []
    emits: List[Dict[str, Any]] = []
    emit_meta: Optional[Dict[str, Any]] = None
    emit_ids: List[int] = []
    n_emits = 0
    steps = 0
    hard_cap = max_new_tokens or (n_frames * 4 + 256)
    next_tok: Optional[int] = None
    decode_count = 0

    # Fast path: multi-frame LISTEN forwards (+ left-pad prefill).
    use_chunk = chunk > 1 and n_frames > 0

    while steps < hard_cap:
        steps += 1

        if state == "listen":
            if frame_i >= n_frames:
                break

            if use_chunk:
                # Prefer large first chunk covering left pad.
                if frame_i == 0 and n_left_pad > 0:
                    take = min(n_frames, max(chunk, n_left_pad))
                else:
                    take = min(chunk, n_frames - frame_i)
                emb = audio_embeds[frame_i : frame_i + take].unsqueeze(0).to(dtype=dtype)
                logits, past = _lm_step_embeds(
                    vox, emb, past, return_all_logits=True,
                )
                # logits: [1, take, V]
                emit_j: Optional[int] = None
                argmax_tok = 0
                p_word = None
                for j in range(take):
                    argmax_tok, do_emit, p_word = _listen_decision(logits[0, j])
                    if do_emit:
                        emit_j = j
                        break
                    gen_ids.append(argmax_tok)

                if emit_j is None:
                    frame_i += take
                    seq_len += take
                    continue

                # Keep wait tokens already recorded for j=0..emit_j-1; crop KV
                # to BOS + frames through the emit frame.
                keep = seq_len + emit_j + 1
                past = _crop_past_kv(past, keep)
                seq_len = keep
                frame_i = frame_i + emit_j + 1
                past, frame_i = _enter_decode(
                    frame_i, argmax_tok, p_word, past,
                )
                seq_len = _past_seq_len(past)
                # fall through to DECODE handling below
            else:
                emb = audio_embeds[frame_i : frame_i + 1].unsqueeze(0).to(dtype=dtype)
                logits, past = _lm_step_embeds(vox, emb, past)
                frame_i += 1
                seq_len += 1
                argmax_tok, do_emit, p_word = _listen_decision(logits[0, -1])
                if not do_emit:
                    gen_ids.append(argmax_tok)
                    continue
                past, frame_i = _enter_decode(
                    frame_i, argmax_tok, p_word, past,
                )
                seq_len = _past_seq_len(past)

        if next_tok == eos_id or decode_count >= max_decode_tokens:
            if next_tok != eos_id:
                gen_ids.append(eos_id)
            _finalize_emit(emit_meta, emit_ids)
            emit_meta = None
            emit_ids = []
            state = "listen"
            next_tok = None
            continue

        if next_tok not in (STREAMING_PAD_ID, STREAMING_WORD_ID, bos_id):
            text_ids.append(next_tok)
            emit_ids.append(next_tok)
        decode_count += 1
        step_emb = _tok_embed(next_tok)
        logits, past = _lm_step_embeds(vox, step_emb, past)
        seq_len += 1
        next_tok = int(torch.argmax(logits[0, -1]).item())
        gen_ids.append(next_tok)

    _finalize_emit(emit_meta, emit_ids)

    text = processor.tokenizer.decode(text_ids, skip_special_tokens=True).strip()
    out: Dict[str, Any] = {
        "text": text,
        "text_ids": text_ids,
        "gen_ids": gen_ids,
        "emits": emits,
        "n_audio_frames": n_frames,
        "n_emits": n_emits,
        "n_left_pad": n_left_pad,
        "seconds_per_token": seconds_per_token,
        "adaptive_latency": True,
        "interleaved": True,
        "mode": "streaming",
        "emit_word_prob_threshold": emit_word_prob_threshold,
        "emit_word_prob_space": emit_word_prob_space,
        "listen_chunk_size": chunk,
    }
    return out


@torch.no_grad()
def generate_interleaved_adaptive(
    model: Any,
    processor: Any,
    audio,
    *,
    max_decode_tokens: int = 64,
    max_new_tokens: Optional[int] = None,
    emit_word_prob_threshold: Optional[float] = None,
    emit_word_prob_space: str = "full_vocab",
    on_emit: Optional[Callable[[Dict[str, Any]], None]] = None,
    listen_chunk_size: int = 16,
) -> Dict[str, Any]:
    """Run listen↔decode state-machine inference on one utterance.

    Args:
        audio: wav path or float32 mono array at processor sample rate.
        emit_word_prob_threshold: If set (e.g. ``0.3``), LISTEN emits when
            ``P(STREAMING_WORD) >= threshold`` (force decode). If ``None``,
            keep the original argmax==W rule.
        on_emit: Optional callback invoked with each emit record as soon as a
            decode block finishes (useful for demo / realtime streaming).
        listen_chunk_size: LISTEN frames per LM forward (``1`` = legacy
            step-by-step). ``>1`` enables left-pad prefill + mid-chunk KV crop.

    Returns:
        dict including ``emits``: list of
        ``{time_s, frame, text}`` for each STREAMING_WORD cycle.
        ``time_s = (frame_i - n_left_pad) * seconds_per_token`` where
        ``frame_i`` is post-increment (after listening through the emit frame),
        i.e. the closed-interval end time of that audio frame.
    """
    vox = _unwrap_voxtral_cg(model)
    embed_fn = vox.model.get_input_embeddings()
    dtype = next(embed_fn.parameters()).dtype
    embeds_list, _nfr = _batch_encode_audio_embeds(
        vox, processor, _load_wavs(
            processor, [audio], processor.feature_extractor.sampling_rate,
        ),
        dtype=dtype,
    )
    return _generate_interleaved_adaptive_from_embeds(
        model,
        processor,
        embeds_list[0],
        max_decode_tokens=max_decode_tokens,
        max_new_tokens=max_new_tokens,
        emit_word_prob_threshold=emit_word_prob_threshold,
        emit_word_prob_space=emit_word_prob_space,
        on_emit=on_emit,
        listen_chunk_size=listen_chunk_size,
    )


@torch.no_grad()
def generate_interleaved_adaptive_pipelined(
    model: Any,
    processor: Any,
    audios: Sequence,
    *,
    pipeline_gpu_encode: bool = True,
    max_decode_tokens: int = 64,
    max_new_tokens: Optional[int] = None,
    emit_word_prob_threshold: Optional[float] = None,
    emit_word_prob_space: str = "full_vocab",
    on_emit: Optional[Callable[[Dict[str, Any]], None]] = None,
    on_utt: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    listen_chunk_size: int = 16,
) -> List[Dict[str, Any]]:
    """Streaming ASR over many utts with next-utt encode∥current state-machine.

    While the LISTEN↔DECODE machine runs on utterance ``i``:

    * a CPU worker prepares mels for ``i+1``;
    * optionally a side CUDA stream encodes ``i+1``.
    """
    if not audios:
        return []


    if len(audios) == 1:
        out = generate_interleaved_adaptive(
            model,
            processor,
            audios[0],
            max_decode_tokens=max_decode_tokens,
            max_new_tokens=max_new_tokens,
            emit_word_prob_threshold=emit_word_prob_threshold,
            emit_word_prob_space=emit_word_prob_space,
            on_emit=on_emit,
            listen_chunk_size=listen_chunk_size,
        )
        if on_utt is not None:
            on_utt(0, out)
        return [out]

    vox = _unwrap_voxtral_cg(model)
    embed_fn = vox.model.get_input_embeddings()
    dtype = next(embed_fn.parameters()).dtype
    sr = processor.feature_extractor.sampling_rate
    device = lm_device_of(vox)
    enc_device = enc_device_of(vox)
    use_cuda_pipe = bool(pipeline_gpu_encode) and torch.cuda.is_available() and (
        str(enc_device).startswith("cuda") or getattr(enc_device, "type", None) == "cuda"
    )
    encode_stream = torch.cuda.Stream(device=enc_device) if use_cuda_pipe else None

    def _encode_ready(
        feats: List[torch.Tensor], mel_lens: List[int],
    ) -> torch.Tensor:
        embeds_list, _ = _gpu_encode_mels_to_embeds(
            vox, processor, feats, mel_lens, dtype=dtype,
        )
        return embeds_list[0].contiguous()

    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=1) as pool:
        fut_mels: Future = pool.submit(
            _cpu_load_and_prepare_mels, processor, [audios[0]], sr,
        )
        feats0, mel_lens0 = fut_mels.result()
        fut_mels = pool.submit(
            _cpu_load_and_prepare_mels, processor, [audios[1]], sr,
        )
        cur_embeds = _encode_ready(feats0, mel_lens0)

        for i in range(len(audios)):
            next_embeds: Optional[torch.Tensor] = None
            next_encode_event: Optional[torch.cuda.Event] = None

            if i + 1 < len(audios) and use_cuda_pipe and encode_stream is not None:
                feats_n, mel_lens_n = fut_mels.result()
                next_encode_event = torch.cuda.Event()
                with torch.cuda.stream(encode_stream):
                    next_embeds = _encode_ready(feats_n, mel_lens_n)
                    next_encode_event.record(encode_stream)
                if i + 2 < len(audios):
                    fut_mels = pool.submit(
                        _cpu_load_and_prepare_mels,
                        processor,
                        [audios[i + 2]],
                        sr,
                    )

            out = _generate_interleaved_adaptive_from_embeds(
                model,
                processor,
                cur_embeds,
                max_decode_tokens=max_decode_tokens,
                max_new_tokens=max_new_tokens,
                emit_word_prob_threshold=emit_word_prob_threshold,
                emit_word_prob_space=emit_word_prob_space,
                on_emit=on_emit,
                listen_chunk_size=listen_chunk_size,
            )
            out["pipelined"] = True
            results.append(out)
            if on_utt is not None:
                on_utt(i, out)

            if i + 1 >= len(audios):
                break

            if use_cuda_pipe and next_embeds is not None:
                assert next_encode_event is not None
                torch.cuda.current_stream(device=device).wait_event(next_encode_event)
                cur_embeds = next_embeds
            else:
                feats_n, mel_lens_n = fut_mels.result()
                if i + 2 < len(audios):
                    fut_mels = pool.submit(
                        _cpu_load_and_prepare_mels,
                        processor,
                        [audios[i + 2]],
                        sr,
                    )
                cur_embeds = _encode_ready(feats_n, mel_lens_n)

    return results


@torch.no_grad()
def generate_interleaved_adaptive_batch(
    model: Any,
    processor: Any,
    audios: Sequence,
    *,
    batch_rows: int = 16,
    max_decode_tokens: int = 64,
    max_new_tokens: Optional[int] = None,
    emit_word_prob_threshold: Optional[float] = None,
    emit_word_prob_space: str = "full_vocab",
    on_utt: Optional[Callable[[int, Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    """Batched lockstep streaming ASR: many utts share one LM forward.

    每次迭代 batch 内所有行各前进恰好一个位置——LISTEN 行喂下一帧音频、
    DECODE 行喂下一个 token、完成的行喂 PAD 占位（输出忽略）。共享 KV
    cache 恒为矩形，无需 attention mask / KV crop，每行 logits 与逐句
    推理严格一致。决策规则与 ``generate_interleaved_adaptive`` 相同
    （全词表 argmax==W 才 emit；可选 ``emit_word_prob_threshold``）。

    与逐句版唯一的行为差异：左 pad 帧（音频时间开始之前）不做 emit
    决策——训练后的模型在该区间恒输出 PAD，实际输出一致。

    Args:
        audios: wav 路径或波形序列。
        batch_rows: 每个锁步 batch 的行数（一批跑完再跑下一批）。
    """
    if not audios:
        return []
    vox = _unwrap_voxtral_cg(model)
    embed_fn = vox.model.get_input_embeddings()
    dtype = next(embed_fn.parameters()).dtype
    sr = processor.feature_extractor.sampling_rate
    results: List[Optional[Dict[str, Any]]] = [None] * len(audios)
    rows = max(1, int(batch_rows))
    for c0 in range(0, len(audios), rows):
        chunk = list(audios[c0 : c0 + rows])
        embeds_list, _ = _batch_encode_audio_embeds(
            vox, processor, _load_wavs(processor, chunk, sr), dtype=dtype,
        )
        outs = _lockstep_adaptive_decode(
            vox,
            processor,
            embeds_list,
            max_decode_tokens=max_decode_tokens,
            max_new_tokens=max_new_tokens,
            emit_word_prob_threshold=emit_word_prob_threshold,
            emit_word_prob_space=emit_word_prob_space,
        )
        for i, out in enumerate(outs):
            out["lockstep_batch_rows"] = len(chunk)
            results[c0 + i] = out
            if on_utt is not None:
                on_utt(c0 + i, out)
    return results  # type: ignore[return-value]


def _lockstep_adaptive_decode(
    vox: Any,
    processor: Any,
    embeds_list: Sequence[torch.Tensor],
    *,
    max_decode_tokens: int = 64,
    max_new_tokens: Optional[int] = None,
    emit_word_prob_threshold: Optional[float] = None,
    emit_word_prob_space: str = "full_vocab",
) -> List[Dict[str, Any]]:
    """锁步 LISTEN↔DECODE：B 行共享一个 batched forward（见上方说明）。"""
    n_left_pad = _adaptive_n_left_pad(processor)
    seconds_per_token = _adaptive_seconds_per_token(processor)
    embed_fn = vox.model.get_input_embeddings()
    bos_id = processor.tokenizer.bos_token_id or 1
    eos_id = processor.tokenizer.eos_token_id or 2
    device = lm_device_of(vox)
    dtype = next(embed_fn.parameters()).dtype
    thr = emit_word_prob_threshold

    def _tok_embed_row(tid: int) -> torch.Tensor:
        t = torch.tensor([tid], device=device, dtype=torch.long)
        return embed_fn(t).to(dtype=dtype)[0]

    w_emb = _tok_embed_row(STREAMING_WORD_ID)
    pad_emb = _tok_embed_row(STREAMING_PAD_ID)
    bos_emb = _tok_embed_row(bos_id)

    aes = [a.to(device=device, dtype=dtype) for a in embeds_list]
    B = len(aes)
    n_frames = [int(a.shape[0]) for a in aes]
    H = int(aes[0].shape[-1])

    state = ["listen"] * B          # listen | decode | done
    frame_i = [0] * B               # 已喂入帧数
    decode_count = [0] * B
    steps_cnt = [0] * B
    stopped_by_cap = [False] * B
    gen_ids: List[List[int]] = [[] for _ in range(B)]
    text_ids: List[List[int]] = [[] for _ in range(B)]
    emit_ids: List[List[int]] = [[] for _ in range(B)]
    emits: List[List[Dict[str, Any]]] = [[] for _ in range(B)]
    emit_meta: List[Optional[Dict[str, Any]]] = [None] * B

    def _finalize(b: int) -> None:
        if emit_meta[b] is None:
            return
        txt = processor.tokenizer.decode(
            emit_ids[b], skip_special_tokens=True
        ).strip()
        rec = {
            "time_s": float(emit_meta[b]["time_s"]),
            "frame": int(emit_meta[b]["frame"]),
            "text": txt,
        }
        if "p_word" in emit_meta[b]:
            rec["p_word"] = emit_meta[b]["p_word"]
            rec["forced_by_prob"] = emit_meta[b]["forced_by_prob"]
        emits[b].append(rec)
        emit_meta[b] = None
        emit_ids[b] = []

    # ── prefill：BOS + 公共左 pad 帧（左 pad 内不做决策）──
    P = min([n_left_pad] + n_frames)
    prefill = torch.zeros((B, 1 + P, H), device=device, dtype=dtype)
    prefill[:, 0] = bos_emb
    for b in range(B):
        prefill[b, 1 : 1 + P] = aes[b][:P]
        frame_i[b] = P
    logits, past = _lm_step_embeds(vox, prefill, None, return_all_logits=True)
    last_logits = logits[:, -1]

    # ``max_new_tokens`` limits generated protocol/text positions, not audio
    # positions.  Unlike the single-utterance path (which consumes LISTEN
    # frames in chunks), lockstep advances only one audio/text position per
    # loop.  Counting the whole loop against max_new_tokens truncated long
    # utterances after roughly that many total positions, before their audio
    # tail was ever seen.  Give every row enough iterations to consume its
    # remaining audio plus its independent generation budget.
    if max_new_tokens is not None:
        caps = [
            max(0, nf - P) + int(max_new_tokens) + 1
            for nf in n_frames
        ]
    else:
        caps = [nf * 4 + 256 for nf in n_frames]

    global_cap = max(caps) + 8
    for _it in range(global_cap):
        rows_l = last_logits.float()                    # [B, V]
        argmax_full = torch.argmax(rows_l, dim=-1).tolist()
        p_word = None
        if thr is not None:
            if emit_word_prob_space == "pad_word":
                pair = rows_l[:, [STREAMING_PAD_ID, STREAMING_WORD_ID]]
                p_word = torch.softmax(pair, dim=-1)[:, 1].tolist()
            else:
                p_word = torch.softmax(
                    rows_l, dim=-1
                )[:, STREAMING_WORD_ID].tolist()

        next_embeds = torch.empty((B, 1, H), device=device, dtype=dtype)
        pending_toks: List[Tuple[int, int]] = []  # (row, tok) 批量取词嵌入
        any_active = False
        for b in range(B):
            if state[b] == "done":
                next_embeds[b, 0] = pad_emb
                continue
            steps_cnt[b] += 1
            if steps_cnt[b] > caps[b]:
                stopped_by_cap[b] = True
                _finalize(b)
                state[b] = "done"
                next_embeds[b, 0] = pad_emb
                continue

            if state[b] == "listen":
                a_tok = int(argmax_full[b])
                if thr is None:
                    do_emit = a_tok == STREAMING_WORD_ID
                else:
                    do_emit = p_word[b] >= float(thr)
                if do_emit:
                    gen_ids[b].append(STREAMING_WORD_ID)
                    emit_meta[b] = {
                        "frame": frame_i[b] - 1,
                        "time_s": (frame_i[b] - n_left_pad) * seconds_per_token,
                    }
                    if thr is not None:
                        emit_meta[b]["p_word"] = round(float(p_word[b]), 6)
                        emit_meta[b]["forced_by_prob"] = (
                            a_tok != STREAMING_WORD_ID
                        )
                    state[b] = "decode"
                    decode_count[b] = 0
                    emit_ids[b] = []
                    next_embeds[b, 0] = w_emb
                    any_active = True
                else:
                    gen_ids[b].append(a_tok)
                    if frame_i[b] >= n_frames[b]:
                        state[b] = "done"  # 音频耗尽（与逐句版一致）
                        next_embeds[b, 0] = pad_emb
                    else:
                        next_embeds[b, 0] = aes[b][frame_i[b]]
                        frame_i[b] += 1
                        any_active = True

            else:  # decode：rows_l[b] 是刚喂入的 W/内容 token 的 logits
                next_tok = int(argmax_full[b])
                gen_ids[b].append(next_tok)
                if next_tok == eos_id or decode_count[b] >= max_decode_tokens:
                    if next_tok != eos_id:
                        gen_ids[b].append(eos_id)
                    _finalize(b)
                    if frame_i[b] >= n_frames[b]:
                        state[b] = "done"
                        next_embeds[b, 0] = pad_emb
                    else:
                        state[b] = "listen"
                        next_embeds[b, 0] = aes[b][frame_i[b]]
                        frame_i[b] += 1
                        any_active = True
                else:
                    if next_tok not in (STREAMING_PAD_ID, STREAMING_WORD_ID, bos_id):
                        text_ids[b].append(next_tok)
                        emit_ids[b].append(next_tok)
                    decode_count[b] += 1
                    pending_toks.append((b, next_tok))
                    any_active = True

        if pending_toks:
            tok_t = torch.tensor(
                [t for _b, t in pending_toks], device=device, dtype=torch.long
            )
            tok_e = embed_fn(tok_t).to(dtype=dtype)
            for j, (b, _t) in enumerate(pending_toks):
                next_embeds[b, 0] = tok_e[j]

        if not any_active:
            break
        logits, past = _lm_step_embeds(vox, next_embeds, past)
        last_logits = logits[:, -1]

    outs: List[Dict[str, Any]] = []
    for b in range(B):
        _finalize(b)
        text = processor.tokenizer.decode(
            text_ids[b], skip_special_tokens=True
        ).strip()
        outs.append({
            "text": text,
            "text_ids": text_ids[b],
            "gen_ids": gen_ids[b],
            "emits": emits[b],
            "n_audio_frames": n_frames[b],
            "consumed_audio_frames": frame_i[b],
            "stopped_by_cap": stopped_by_cap[b],
            "n_emits": len(emits[b]),
            "n_left_pad": n_left_pad,
            "seconds_per_token": seconds_per_token,
            "adaptive_latency": True,
            "interleaved": True,
            "mode": "streaming",
            "emit_word_prob_threshold": thr,
            "emit_word_prob_space": emit_word_prob_space,
        })
    return outs


def resolve_offline_emit_id(token: Any = None) -> int:
    """Map yaml ``offline_emit_token`` to id. Default ``W_off`` (4)."""
    if token is None or token == "":
        return int(STREAMING_WORD_OFF_ID)
    if isinstance(token, bool):
        raise ValueError(f"offline_emit_token must be W or W_off (got {token!r})")
    if isinstance(token, (int, float)):
        return int(token)
    key = str(token).strip().lower().replace("-", "_")
    if key in ("w", "word", "streaming_word", "33"):
        return int(STREAMING_WORD_ID)
    if key in ("w_off", "woff", "word_off", "streaming_word_off", "4"):
        return int(STREAMING_WORD_OFF_ID)
    raise ValueError(f"offline_emit_token must be W or W_off (got {token!r})")


@torch.no_grad()
def generate_interleaved_offline(
    model: Any,
    processor: Any,
    audio,
    *,
    max_decode_tokens: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
    on_emit: Optional[Callable[[Dict[str, Any]], None]] = None,
    batch_audio_encode: bool = True,
    offline_emit_token: Any = None,
) -> Dict[str, Any]:
    """Adaptive-protocol offline ASR: listen to all frames, then one emit.

    Single-utterance wrapper around ``generate_interleaved_offline_batch``.
    Default force is ``W_off`` (id=4, asr_mix). Pass ``offline_emit_token='W'``
    to force streaming ``W`` (id=33), e.g. Stage1 checkpoints.
    """
    return generate_interleaved_offline_batch(
        model,
        processor,
        [audio],
        max_decode_tokens=max_decode_tokens,
        max_new_tokens=max_new_tokens,
        on_emit=on_emit,
        batch_audio_encode=batch_audio_encode,
        offline_emit_token=offline_emit_token,
    )[0]


def _cpu_prepare_mels(
    processor: Any,
    wavs: Sequence,
) -> Tuple[List[torch.Tensor], List[int]]:
    """CPU: wavs → mels ``[n_mels, T_i]`` via one batched feature_extractor."""
    if not wavs:
        return [], []
    sr = processor.feature_extractor.sampling_rate
    feats: List[torch.Tensor] = []
    mel_lens: List[int] = []
    encodings = encode_audios_with_delay(
        processor, wavs, sr, ADAPTIVE_PROTOCOL_DELAY_MS,
    )
    for enc in encodings:
        if enc.input_features.shape[0] != 1:
            raise RuntimeError(
                "expected one delayed audio per utterance, got "
                f"{enc.input_features.shape[0]}"
            )
        feat = enc.input_features.squeeze(0).contiguous()  # [n_mels, T]
        feats.append(feat)
        mel_lens.append(int(feat.shape[-1]))
    return feats, mel_lens


def _gpu_encode_mels_to_embeds(
    vox: Any,
    processor: Any,
    feats: Sequence[torch.Tensor],
    mel_lens: Sequence[int],
    *,
    dtype: torch.dtype,
) -> Tuple[List[torch.Tensor], List[int]]:
    """Pad mels on device → ``get_audio_features`` → slice to true frame lengths."""
    alt = int(
        getattr(processor, "audio_length_per_tok", None)
        or getattr(
            getattr(processor, "mistral_common_audio_config", None),
            "audio_length_per_tok",
            8,
        )
        or 8
    )
    n_mels = int(feats[0].shape[0])
    max_mel = max(int(t) for t in mel_lens)
    bsz = len(feats)
    enc_device = enc_device_of(vox)
    lm_device = lm_device_of(vox)
    batch_features = torch.zeros(
        (bsz, n_mels, max_mel),
        device=enc_device,
        dtype=vox.dtype,
    )
    for i, feat in enumerate(feats):
        t = int(mel_lens[i])
        batch_features[i, :, :t] = feat.to(
            device=enc_device, dtype=vox.dtype, non_blocking=feat.device.type == "cpu",
        )

    audio_out = vox.model.get_audio_features(input_features=batch_features)
    pooled = audio_out.pooler_output  # [B, T_max_frames, H]
    if pooled.dim() != 3:
        raise RuntimeError(
            f"expected pooler_output [B,T,H], got shape {tuple(pooled.shape)}"
        )

    embeds_list: List[torch.Tensor] = []
    n_frames_list: List[int] = []
    for i in range(bsz):
        n_frames = max(1, int(mel_lens[i]) // alt) if int(mel_lens[i]) > 0 else 0
        n_frames = min(n_frames, int(pooled.shape[1]))
        ae = pooled[i, :n_frames].to(
            device=lm_device, dtype=dtype, non_blocking=True,
        ).contiguous()
        embeds_list.append(ae)
        n_frames_list.append(int(ae.shape[0]))
    return embeds_list, n_frames_list


def _batch_encode_audio_embeds(
    vox: Any,
    processor: Any,
    wavs: Sequence,
    *,
    dtype: torch.dtype,
) -> Tuple[List[torch.Tensor], List[int]]:
    """Pad mel features → one ``get_audio_features`` → slice to true lengths."""
    feats, mel_lens = _cpu_prepare_mels(processor, wavs)
    return _gpu_encode_mels_to_embeds(
        vox, processor, feats, mel_lens, dtype=dtype,
    )


def _offline_decode_from_audio_embeds(
    vox: Any,
    processor: Any,
    audio_embeds_list: Sequence[torch.Tensor],
    n_frames_list: Sequence[int],
    *,
    bos_emb: torch.Tensor,
    w_emb: torch.Tensor,
    dtype: torch.dtype,
    hidden: int,
    n_left_pad: int,
    seconds_per_token: float,
    max_decode_tokens: Optional[int],
    max_new_tokens: Optional[int],
    on_emit: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    """LM prefill (BOS+audio) + force W_off + batched decode to EOS."""
    embed_fn = vox.model.get_input_embeddings()
    bos_id = processor.tokenizer.bos_token_id or 1
    eos_id = processor.tokenizer.eos_token_id or 2
    device = lm_device_of(vox)
    bsz = len(audio_embeds_list)
    if bsz == 0:
        return []

    seq_lens = [1 + n for n in n_frames_list]
    max_len = max(seq_lens)
    embeds = torch.zeros(bsz, max_len, hidden, device=device, dtype=dtype)
    attn = torch.zeros(bsz, max_len, device=device, dtype=torch.long)
    for i, ae in enumerate(audio_embeds_list):
        L = seq_lens[i]
        pad = max_len - L
        embeds[i, pad] = bos_emb
        if n_frames_list[i] > 0:
            embeds[i, pad + 1 : max_len] = ae
        attn[i, pad:] = 1

    _logits, past = _lm_step_embeds(vox, embeds, None, attention_mask=attn)

    w_step = w_emb.view(1, 1, -1).expand(bsz, 1, -1).contiguous()
    attn = torch.cat(
        [attn, torch.ones(bsz, 1, device=device, dtype=torch.long)], dim=1,
    )
    logits, past = _lm_step_embeds(vox, w_step, past, attention_mask=attn)
    next_toks = torch.argmax(logits[:, -1, :], dim=-1)

    decode_budget = max_decode_tokens
    if decode_budget is None:
        decode_budget = int(
            max_new_tokens if max_new_tokens else max(256, max(n_frames_list) or 1)
        )
    decode_budget = int(decode_budget)
    hard_cap = int(max_new_tokens or (decode_budget + 64))

    text_ids: List[List[int]] = [[] for _ in range(bsz)]
    emit_ids: List[List[int]] = [[] for _ in range(bsz)]
    finished = [False] * bsz
    decode_count = [0] * bsz

    steps = 0
    while steps < hard_cap and not all(finished):
        steps += 1
        any_active = False
        for i in range(bsz):
            if finished[i]:
                continue
            tok = int(next_toks[i].item())
            if tok == eos_id or decode_count[i] >= decode_budget:
                finished[i] = True
                continue
            if tok not in (
                STREAMING_PAD_ID, STREAMING_WORD_ID, STREAMING_WORD_OFF_ID, bos_id,
            ):
                text_ids[i].append(tok)
                emit_ids[i].append(tok)
            decode_count[i] += 1
            any_active = True
        if not any_active:
            break

        feed_ids = next_toks.clone()
        for i in range(bsz):
            if finished[i]:
                feed_ids[i] = eos_id
        step_emb = embed_fn(feed_ids.view(bsz, 1)).to(dtype=dtype)
        attn = torch.cat(
            [attn, torch.ones(bsz, 1, device=device, dtype=torch.long)], dim=1,
        )
        logits, past = _lm_step_embeds(vox, step_emb, past, attention_mask=attn)
        next_toks = torch.argmax(logits[:, -1, :], dim=-1)

    results: List[Dict[str, Any]] = []
    for i in range(bsz):
        n_frames = int(n_frames_list[i])
        emit_frame = max(n_frames - 1, 0)
        emit_time_s = max(n_frames - n_left_pad, 0) * seconds_per_token
        chunk_text = processor.tokenizer.decode(
            emit_ids[i], skip_special_tokens=True,
        ).strip()
        rec = {
            "time_s": float(emit_time_s),
            "frame": int(emit_frame),
            "text": chunk_text,
            "forced_offline": True,
        }
        if on_emit is not None:
            on_emit(rec)
        text = processor.tokenizer.decode(
            text_ids[i], skip_special_tokens=True,
        ).strip()
        results.append({
            "text": text,
            "text_ids": text_ids[i],
            "gen_ids": [],
            "emits": [rec],
            "n_audio_frames": n_frames,
            "n_emits": 1,
            "n_left_pad": n_left_pad,
            "seconds_per_token": seconds_per_token,
            "adaptive_latency": True,
            "interleaved": True,
            "mode": "offline",
            "batched": True,
        })
    return results


def _offline_shared_ctx(
    model: Any,
    processor: Any,
    *,
    offline_emit_token: Any = None,
) -> Dict[str, Any]:
    """Common tensors / scalars for offline batch / pipelined paths."""
    vox = _unwrap_voxtral_cg(model)
    embed_fn = vox.model.get_input_embeddings()
    bos_id = processor.tokenizer.bos_token_id or 1
    device = lm_device_of(vox)
    dtype = next(embed_fn.parameters()).dtype
    hidden = int(next(embed_fn.parameters()).shape[-1])
    bos_emb = embed_fn(
        torch.tensor([[bos_id]], device=device, dtype=torch.long)
    ).to(dtype=dtype).squeeze(0)
    # Default W_off (id=4, asr_mix). Stage1 / streaming emit is W (id=33).
    emit_id = resolve_offline_emit_id(offline_emit_token)
    w_emb = embed_fn(
        torch.tensor([[emit_id]], device=device, dtype=torch.long)
    ).to(dtype=dtype).squeeze(0)
    return {
        "vox": vox,
        "sr": processor.feature_extractor.sampling_rate,
        "n_left_pad": _adaptive_n_left_pad(processor),
        "seconds_per_token": _adaptive_seconds_per_token(processor),
        "dtype": dtype,
        "hidden": hidden,
        "bos_emb": bos_emb,
        "w_emb": w_emb,
    }


def _load_wavs(processor: Any, audios: Sequence, sr: int) -> List[Any]:
    wavs: List[Any] = []
    for audio in audios:
        if isinstance(audio, str):
            wavs.append(load_audio(audio, sr=sr))
        else:
            wavs.append(audio)
    return wavs


def _cpu_load_and_prepare_mels(
    processor: Any,
    audios: Sequence,
    sr: int,
) -> Tuple[List[torch.Tensor], List[int]]:
    """CPU thread work: load wavs (if paths) + extract mels."""
    wavs = _load_wavs(processor, audios, sr)
    return _cpu_prepare_mels(processor, wavs)


@torch.no_grad()
def generate_interleaved_offline_batch(
    model: Any,
    processor: Any,
    audios: Sequence,
    *,
    max_decode_tokens: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
    on_emit: Optional[Callable[[Dict[str, Any]], None]] = None,
    batch_audio_encode: bool = True,
    offline_emit_token: Any = None,
) -> List[Dict[str, Any]]:
    """Batched offline ASR under the adaptive interleaved protocol.

    For each utterance::

        Prefill [BOS] + all audio frames (left-padded across the batch)
        Force emit (default W_off) → DECODE until EOS (batched decode steps)

    When ``batch_audio_encode=True`` (default), mel features are right-padded
    and passed through ``get_audio_features`` once, then sliced to each utt's
    true frame count before LM prefill.
    """
    if not audios:
        return []

    ctx = _offline_shared_ctx(
        model, processor, offline_emit_token=offline_emit_token,
    )
    vox = ctx["vox"]
    wavs = _load_wavs(processor, audios, ctx["sr"])

    if batch_audio_encode:
        audio_embeds_list, n_frames_list = _batch_encode_audio_embeds(
            vox, processor, wavs, dtype=ctx["dtype"],
        )
    else:
        audio_embeds_list = []
        n_frames_list = []
        for wav in wavs:
            enc = encode_audio_with_delay(
                processor, wav, ctx["sr"], ADAPTIVE_PROTOCOL_DELAY_MS,
            )
            input_features = enc.input_features.to(
                device=enc_device_of(vox), dtype=vox.dtype,
            )
            audio_out = vox.model.get_audio_features(input_features=input_features)
            ae = audio_out.pooler_output[0].to(
                device=lm_device_of(vox), dtype=ctx["dtype"], non_blocking=True,
            )
            audio_embeds_list.append(ae)
            n_frames_list.append(int(ae.shape[0]))

    return _offline_decode_from_audio_embeds(
        vox,
        processor,
        audio_embeds_list,
        n_frames_list,
        bos_emb=ctx["bos_emb"],
        w_emb=ctx["w_emb"],
        dtype=ctx["dtype"],
        hidden=ctx["hidden"],
        n_left_pad=ctx["n_left_pad"],
        seconds_per_token=ctx["seconds_per_token"],
        max_decode_tokens=max_decode_tokens,
        max_new_tokens=max_new_tokens,
        on_emit=on_emit,
    )


@torch.no_grad()
def generate_interleaved_offline_pipelined(
    model: Any,
    processor: Any,
    audios: Sequence,
    *,
    batch_size: int = 8,
    pipeline_gpu_encode: bool = True,
    max_decode_tokens: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
    on_emit: Optional[Callable[[Dict[str, Any]], None]] = None,
    on_batch: Optional[Callable[[int, List[Dict[str, Any]]], None]] = None,
    offline_emit_token: Any = None,
) -> List[Dict[str, Any]]:
    """Offline ASR over many utts with encode∥decode pipeline.

    While LM decodes batch ``i`` on the default CUDA stream:

    * a CPU worker prepares mels for batch ``i+1`` (load wav + feature extract);
    * optionally a side CUDA stream runs ``get_audio_features`` for batch ``i+1``.

    Falls back to sequential ``generate_interleaved_offline_batch`` when there is
    only one micro-batch.
    """
    if not audios:
        return []
    bs = max(1, int(batch_size))
    chunks: List[Sequence] = [
        audios[i : i + bs] for i in range(0, len(audios), bs)
    ]
    if len(chunks) == 1:
        out = generate_interleaved_offline_batch(
            model,
            processor,
            chunks[0],
            max_decode_tokens=max_decode_tokens,
            max_new_tokens=max_new_tokens,
            on_emit=on_emit,
            batch_audio_encode=True,
            offline_emit_token=offline_emit_token,
        )
        if on_batch is not None:
            on_batch(0, out)
        return out

    ctx = _offline_shared_ctx(
        model, processor, offline_emit_token=offline_emit_token,
    )
    vox = ctx["vox"]
    dtype = ctx["dtype"]
    device = lm_device_of(vox)
    enc_device = enc_device_of(vox)
    use_cuda_pipe = bool(pipeline_gpu_encode) and torch.cuda.is_available() and (
        str(enc_device).startswith("cuda") or getattr(enc_device, "type", None) == "cuda"
    )
    encode_stream = torch.cuda.Stream(device=enc_device) if use_cuda_pipe else None

    def _encode_ready(
        feats: List[torch.Tensor], mel_lens: List[int],
    ) -> Tuple[List[torch.Tensor], List[int]]:
        return _gpu_encode_mels_to_embeds(
            vox, processor, feats, mel_lens, dtype=dtype,
        )

    def _decode_chunk(
        embeds: List[torch.Tensor], nfr: List[int],
    ) -> List[Dict[str, Any]]:
        batch_out = _offline_decode_from_audio_embeds(
            vox,
            processor,
            embeds,
            nfr,
            bos_emb=ctx["bos_emb"],
            w_emb=ctx["w_emb"],
            dtype=dtype,
            hidden=ctx["hidden"],
            n_left_pad=ctx["n_left_pad"],
            seconds_per_token=ctx["seconds_per_token"],
            max_decode_tokens=max_decode_tokens,
            max_new_tokens=max_new_tokens,
            on_emit=on_emit,
        )
        for rec in batch_out:
            rec["pipelined"] = True
        return batch_out

    all_results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=1) as pool:
        # Warm: CPU mel0 → kick mel1 → GPU encode0.
        fut_mels: Future = pool.submit(
            _cpu_load_and_prepare_mels, processor, chunks[0], ctx["sr"],
        )
        feats0, mel_lens0 = fut_mels.result()
        if len(chunks) > 1:
            fut_mels = pool.submit(
                _cpu_load_and_prepare_mels, processor, chunks[1], ctx["sr"],
            )
        cur_embeds, cur_nfr = _encode_ready(feats0, mel_lens0)

        for bi in range(len(chunks)):
            next_embeds: Optional[Tuple[List[torch.Tensor], List[int]]] = None
            next_encode_event: Optional[torch.cuda.Event] = None

            if bi + 1 < len(chunks) and use_cuda_pipe and encode_stream is not None:
                # Launch next GPU encode on side stream, then decode current
                # so the two overlap on the device.
                feats_n, mel_lens_n = fut_mels.result()
                next_encode_event = torch.cuda.Event()
                with torch.cuda.stream(encode_stream):
                    ne, nn = _encode_ready(feats_n, mel_lens_n)
                    next_embeds = ([e.contiguous() for e in ne], list(nn))
                    next_encode_event.record(encode_stream)
                if bi + 2 < len(chunks):
                    fut_mels = pool.submit(
                        _cpu_load_and_prepare_mels,
                        processor,
                        chunks[bi + 2],
                        ctx["sr"],
                    )

            batch_out = _decode_chunk(cur_embeds, cur_nfr)
            all_results.extend(batch_out)
            if on_batch is not None:
                on_batch(bi, batch_out)

            if bi + 1 >= len(chunks):
                break

            if use_cuda_pipe and next_embeds is not None:
                assert next_encode_event is not None
                torch.cuda.current_stream(device=device).wait_event(next_encode_event)
                cur_embeds, cur_nfr = next_embeds
            else:
                # CPU-only pipeline: encode next after decode (mel prep overlapped).
                feats_n, mel_lens_n = fut_mels.result()
                if bi + 2 < len(chunks):
                    fut_mels = pool.submit(
                        _cpu_load_and_prepare_mels,
                        processor,
                        chunks[bi + 2],
                        ctx["sr"],
                    )
                cur_embeds, cur_nfr = _encode_ready(feats_n, mel_lens_n)

    return all_results
