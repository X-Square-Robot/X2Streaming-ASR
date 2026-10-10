"""Voxtral Realtime model loading and inference."""

from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import (
    AutoProcessor,
    VoxtralRealtimeForConditionalGeneration,
)

from models.base import BaseASRModel, TranscribeResult, register_model
from models.voxtral_asr.data import (
    STREAMING_PAD_ID,
    STREAMING_WORD_ID,
    STREAMING_WORD_OFF_ID,
    disable_adarms_delay_conditioning,
    load_audio,
)
from models.voxtral_asr.interleaved import (
    generate_interleaved_adaptive,
    generate_interleaved_adaptive_batch,
    generate_interleaved_adaptive_pipelined,
    generate_interleaved_offline_batch,
    generate_interleaved_offline_pipelined,
)


@dataclass
class WordTimestamp:
    text: str
    start_time: float
    end_time: float


def extract_word_timestamps(
    gen_ids: List[int],
    tokenizer,
    ms_per_token: float = 80.0,
) -> List[WordTimestamp]:
    """Extract word-level timestamps from streaming token sequence.

    Each [STREAMING_WORD] (id=33) marks a word boundary. Text tokens between
    STREAMING_WORD markers belong to that word. STREAMING_PAD tokens advance
    the clock by ms_per_token each.
    """
    timestamps = []
    token_pos = 0
    current_word_ids = []
    current_word_start = None

    for tid in gen_ids:
        if tid in (STREAMING_WORD_ID, STREAMING_WORD_OFF_ID):
            if current_word_ids:
                text = tokenizer.decode(current_word_ids, skip_special_tokens=True)
                if text.strip():
                    timestamps.append(WordTimestamp(
                        text=text.strip(),
                        start_time=current_word_start / 1000.0,
                        end_time=token_pos * ms_per_token / 1000.0,
                    ))
                current_word_ids = []
            current_word_start = token_pos * ms_per_token
        elif tid == STREAMING_PAD_ID:
            pass
        else:
            current_word_ids.append(tid)
        token_pos += 1

    if current_word_ids:
        text = tokenizer.decode(current_word_ids, skip_special_tokens=True)
        if text.strip():
            timestamps.append(WordTimestamp(
                text=text.strip(),
                start_time=(current_word_start or 0) / 1000.0,
                end_time=token_pos * ms_per_token / 1000.0,
            ))

    return timestamps


@register_model("voxtral_asr")
class VoxtralASR(BaseASRModel):

    def _auto_dtype(self):
        if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
            return torch.bfloat16
        return torch.float16

    def prepare_adaptive_model(self, model):
        """Prepare delay conditioning for adaptive inference."""
        disable_adarms_delay_conditioning(model)
        return model

    def load_for_inference(self, model_path: str, device: str = "cuda:0", **kwargs):
        dtype = self._auto_dtype()
        processor = AutoProcessor.from_pretrained(model_path)
        model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device,
        )

        lora_path = kwargs.get("lora_path", "")
        language_tag = kwargs.get("language_tag") or {}
        language_tag_enabled = bool(
            language_tag.get("enabled", kwargs.get("language_tag_enabled", False))
        )
        adapter_saves_tag = False
        if lora_path:
            from models.voxtral_asr.language_tag import adapter_saves_language_tag

            adapter_saves_tag = adapter_saves_language_tag(lora_path)

        # A modules_to_save target must exist before PeftModel.from_pretrained:
        # PEFT wraps that module, restores its adapter copy, then keeps that
        # copy when merge_and_unload removes the wrapper.
        if language_tag_enabled or adapter_saves_tag:
            from models.voxtral_asr.language_tag import attach_language_tag_embed

            attach_language_tag_embed(
                model,
                model_path=model_path,
                require_checkpoint_weights=bool(
                    language_tag.get("require_checkpoint_weights", False)
                ),
            )

        if lora_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, lora_path)
            model = model.merge_and_unload()
            if adapter_saves_tag:
                from models.voxtral_asr.language_tag import (
                    mark_language_tag_adapter_source,
                )

                mark_language_tag_adapter_source(model, lora_path)

        adaptive_latency = bool(kwargs.get("adaptive_latency", False))
        if adaptive_latency:
            self.prepare_adaptive_model(model)

        from models.voxtral_asr.devices import apply_encoder_lm_split

        apply_encoder_lm_split(
            model,
            lm_device=device,
            enc_device=kwargs.get("enc_device") or kwargs.get("encoder_device"),
        )

        return {
            "model": model,
            "processor": processor,
            "adaptive_latency": adaptive_latency,
            "inference_mode": kwargs.get("inference_mode", "streaming"),
            "language_tag_enabled": language_tag_enabled,
            "language_id": kwargs.get("language_id"),
        }

    def transcribe(self, wrapper, audio, **kwargs) -> List[TranscribeResult]:
        model = wrapper["model"]
        processor = wrapper["processor"]
        adaptive_latency = bool(
            kwargs.get("adaptive_latency", wrapper.get("adaptive_latency", False))
        )
        mode = str(
            kwargs.get(
                "mode",
                kwargs.get(
                    "inference_mode",
                    wrapper.get("inference_mode", "streaming"),
                ),
            )
            or "streaming"
        ).lower()
        sr = processor.feature_extractor.sampling_rate
        return_timestamps = kwargs.get("return_timestamps", False)
        max_new_tokens = kwargs.get("max_new_tokens")

        hop = processor.feature_extractor.hop_length
        alt = processor.audio_length_per_tok
        ms_per_token = alt * hop / sr * 1000

        if isinstance(audio, str):
            audio = [audio]
        elif not isinstance(audio, list):
            audio = [audio]

        if not audio:
            return []

        wavs = [load_audio(audio_path, sr=sr) for audio_path in audio]

        language_tag_enabled = bool(
            kwargs.get(
                "language_tag_enabled",
                wrapper.get("language_tag_enabled", False),
            )
        )
        if language_tag_enabled:
            from models.voxtral_asr.data import infer_lang_id_from_path
            from models.voxtral_asr.interleaved import _unwrap_voxtral_cg

            cg = _unwrap_voxtral_cg(model)
            explicit = kwargs.get("language_id", wrapper.get("language_id"))
            path_ids = [
                infer_lang_id_from_path(p) if isinstance(p, str) else None
                for p in audio
            ]
            if explicit is not None:
                cg._infer_language_id = int(explicit)
                cg._infer_language_ids = None
            elif path_ids and all(i is not None for i in path_ids) and len(set(path_ids)) == 1:
                cg._infer_language_id = int(path_ids[0])
                cg._infer_language_ids = None
            elif path_ids and all(i is not None for i in path_ids):
                cg._infer_language_id = None
                cg._infer_language_ids = path_ids
            else:
                cg._infer_language_id = 0
                cg._infer_language_ids = None
        else:
            cg = None

        if adaptive_latency:
            emit_thr = kwargs.get(
                "emit_word_prob_threshold",
                wrapper.get("emit_word_prob_threshold"),
            )
            emit_prob_space = kwargs.get(
                "emit_word_prob_space",
                wrapper.get("emit_word_prob_space", "full_vocab"),
            )
            results = []
            if mode == "offline":
                pipeline_encode = bool(kwargs.get("pipeline_encode", False))
                micro_bs = int(kwargs.get("batch_size", 0) or 0)
                offline_emit_token = kwargs.get(
                    "offline_emit_token", wrapper.get("offline_emit_token"),
                )
                if pipeline_encode and micro_bs > 0 and len(wavs) > micro_bs:
                    outs = generate_interleaved_offline_pipelined(
                        model, processor, wavs, batch_size=micro_bs,
                        pipeline_gpu_encode=bool(kwargs.get("pipeline_gpu_encode", True)),
                        max_new_tokens=max_new_tokens, on_batch=kwargs.get("on_batch"),
                        offline_emit_token=offline_emit_token,
                    )
                else:
                    outs = generate_interleaved_offline_batch(
                        model, processor, wavs, max_new_tokens=max_new_tokens,
                        batch_audio_encode=bool(kwargs.get("batch_audio_encode", True)),
                        offline_emit_token=offline_emit_token,
                    )
                for out in outs:
                    ts = extract_word_timestamps(
                        out.get("gen_ids", []), processor.tokenizer, ms_per_token,
                    ) if return_timestamps else None
                    results.append(TranscribeResult(
                        text=out["text"], time_stamps=ts,
                        emits=out.get("emits"), n_emits=out.get("n_emits"),
                    ))
                return results

            if mode != "streaming":
                raise ValueError(f"adaptive inference mode must be streaming or offline, got {mode!r}")

            stream_rows = int(kwargs.get("stream_batch_rows", 0) or 0)
            if stream_rows > 1 and len(wavs) > 1:
                outs = generate_interleaved_adaptive_batch(
                    model, processor, wavs, batch_rows=stream_rows,
                    max_new_tokens=max_new_tokens,
                    emit_word_prob_threshold=emit_thr,
                    emit_word_prob_space=emit_prob_space,
                    on_utt=kwargs.get("on_utt"),
                )
            elif bool(kwargs.get("pipeline_encode", False)) and len(wavs) > 1:
                outs = generate_interleaved_adaptive_pipelined(
                    model, processor, wavs,
                    pipeline_gpu_encode=bool(kwargs.get("pipeline_gpu_encode", True)),
                    max_new_tokens=max_new_tokens,
                    emit_word_prob_threshold=emit_thr,
                    emit_word_prob_space=emit_prob_space,
                    on_utt=kwargs.get("on_utt"),
                    listen_chunk_size=int(kwargs.get("listen_chunk_size", 16) or 16),
                )
            else:
                outs = []
                for audio_path, wav in zip(audio, wavs):
                    if language_tag_enabled and cg is not None and isinstance(audio_path, str):
                        cg._infer_language_id = infer_lang_id_from_path(audio_path)
                        cg._infer_language_ids = None
                    outs.append(generate_interleaved_adaptive(
                        model, processor, wav, max_new_tokens=max_new_tokens,
                        emit_word_prob_threshold=emit_thr,
                        emit_word_prob_space=emit_prob_space,
                        listen_chunk_size=int(kwargs.get("listen_chunk_size", 16) or 16),
                    ))
            for out in outs:
                ts = extract_word_timestamps(
                    out.get("gen_ids", []), processor.tokenizer, ms_per_token,
                ) if return_timestamps else None
                results.append(TranscribeResult(
                    text=out["text"], time_stamps=ts,
                    emits=out.get("emits"), n_emits=out.get("n_emits"),
                ))
            return results

        inputs = processor(wavs, return_tensors="pt")
        inputs = inputs.to(model.device, dtype=model.dtype)
        prefix_len = inputs.input_ids.shape[1]

        gen_kwargs = {
            "input_ids": inputs.input_ids,
            "input_features": inputs.input_features,
            "num_delay_tokens": inputs.num_delay_tokens,
        }
        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.attention_mask
        if max_new_tokens:
            gen_kwargs["max_new_tokens"] = max_new_tokens

        with torch.no_grad():
            outputs = model.generate(**gen_kwargs)

        results = []
        for i, _audio_path in enumerate(audio):
            gen_ids = outputs[i][prefix_len:].tolist()
            decoded = processor.tokenizer.decode(
                [t for t in gen_ids if t not in (
                    STREAMING_PAD_ID, STREAMING_WORD_ID, STREAMING_WORD_OFF_ID,
                )],
                skip_special_tokens=True,
            )

            ts = None
            if return_timestamps:
                ts = extract_word_timestamps(gen_ids, processor.tokenizer, ms_per_token)

            results.append(TranscribeResult(
                text=decoded,
                time_stamps=ts,
            ))

        return results
