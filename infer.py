#!/usr/bin/env python3
"""Single-file and JSONL batch inference for X2StreamingASR.

Examples:
  python infer.py --config configs/infer.yaml --model_path MODEL --audio audio.wav
  python infer.py --config configs/infer.yaml --model_path MODEL --language en \
      --input_file audio.jsonl --output_file results.jsonl
"""

import argparse
import json
import os
import time

from src.utils import deep_update, load_config


def parse_args():
    parser = argparse.ArgumentParser("X2StreamingASR Inference")
    parser.add_argument("--config", default="")
    parser.add_argument("--model_path", default="")
    parser.add_argument("--lora_path", default="")
    parser.add_argument("--device", default="")
    parser.add_argument("--audio", default="")
    parser.add_argument("--input_file", default="")
    parser.add_argument("--output_file", default="")
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--stream-batch-rows", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--return_timestamps", action="store_true")
    parser.add_argument("--adaptive-latency", action="store_true")
    parser.add_argument("--emit-word-prob-threshold", type=float, default=None)
    parser.add_argument(
        "--emit-word-prob-space",
        choices=["full_vocab", "pad_word"],
        default="",
    )
    parser.add_argument(
        "--mode",
        choices=["", "streaming", "offline"],
        default="",
    )
    parser.add_argument(
        "--language",
        choices=["", "zh", "en", "cs"],
        default="",
        help="Language tag: zh, en, or cs (Chinese-English code-switching). "
        "Defaults to zh unless the input path names a known English corpus.",
    )
    return parser.parse_args()


def build_cfg(args):
    cfg = {
        "model": {
            "model_type": "voxtral_asr",
            "model_path": "",
            "lora_path": "",
            "device": "cuda:0",
            "max_new_tokens": 256,
            "adaptive_latency": False,
            "emit_word_prob_threshold": None,
            "emit_word_prob_space": "full_vocab",
        },
        "inference": {
            "audio": "",
            "return_timestamps": False,
            "mode": "streaming",
            "offline_emit_token": "W_off",
        },
        "batch": {
            "input_file": "",
            "output_file": "",
            "batch_size": 1,
            "pipeline_encode": False,
            "pipeline_gpu_encode": True,
            "listen_chunk_size": 16,
            "stream_batch_rows": 0,
            "limit": 0,
        },
    }
    if args.config:
        deep_update(cfg, load_config(args.config))

    cli_overrides = (
        ("model", "model_path", args.model_path),
        ("model", "lora_path", args.lora_path),
        ("model", "device", args.device),
        ("inference", "audio", args.audio),
        ("batch", "input_file", args.input_file),
        ("batch", "output_file", args.output_file),
    )
    for section, key, value in cli_overrides:
        if value:
            cfg[section][key] = value

    if args.batch_size > 0:
        cfg["batch"]["batch_size"] = args.batch_size
    if args.stream_batch_rows > 0:
        cfg["batch"]["stream_batch_rows"] = args.stream_batch_rows
    if args.limit > 0:
        cfg["batch"]["limit"] = args.limit
    if args.return_timestamps:
        cfg["inference"]["return_timestamps"] = True
    if args.adaptive_latency:
        cfg["model"]["adaptive_latency"] = True
    if args.emit_word_prob_threshold is not None:
        cfg["model"]["emit_word_prob_threshold"] = args.emit_word_prob_threshold
    if args.emit_word_prob_space:
        cfg["model"]["emit_word_prob_space"] = args.emit_word_prob_space
    if args.mode:
        cfg["inference"]["mode"] = args.mode
    if args.language:
        cfg["inference"]["language"] = args.language
    return cfg


def _load_batch(path, limit=0):
    samples = []
    with open(path, "r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            audio = item.get("audio") or item.get("wav") or item.get("wav_path")
            if not audio:
                raise ValueError(
                    f"{path}:{line_number}: expected audio, wav, or wav_path"
                )
            samples.append({"audio": audio})
            if limit > 0 and len(samples) >= limit:
                break
    return samples


def _result_record(audio, result, mode):
    record = {
        "audio": audio,
        "text": result.text.strip(),
        "mode": mode,
    }
    if result.emits is not None:
        record["emits"] = result.emits
        record["n_emits"] = result.n_emits
    if result.time_stamps:
        record["timestamps"] = [
            {
                "text": timestamp.text,
                "start_time": timestamp.start_time,
                "end_time": timestamp.end_time,
            }
            for timestamp in result.time_stamps
        ]
    return record


def main():
    args = parse_args()
    cfg = build_cfg(args)
    model_cfg = cfg["model"]
    infer_cfg = cfg["inference"]
    batch_cfg = cfg["batch"]

    if not model_cfg["model_path"]:
        raise ValueError("--model_path or model.model_path is required")

    mode = str(infer_cfg.get("mode") or "streaming").lower()
    if mode not in {"streaming", "offline"}:
        raise ValueError(f"inference.mode must be streaming or offline, got {mode!r}")

    adaptive = bool(model_cfg.get("adaptive_latency", False))
    if mode == "offline":
        adaptive = True
        model_cfg["adaptive_latency"] = True

    emit_threshold = model_cfg.get("emit_word_prob_threshold")
    if emit_threshold is not None:
        emit_threshold = float(emit_threshold)
        if not 0.0 <= emit_threshold <= 1.0:
            raise ValueError("emit_word_prob_threshold must be in [0, 1]")
        if not adaptive or mode != "streaming":
            emit_threshold = None

    emit_space = str(
        model_cfg.get("emit_word_prob_space", "full_vocab")
    ).lower()
    if emit_space not in {"full_vocab", "pad_word"}:
        raise ValueError("emit_word_prob_space must be full_vocab or pad_word")

    input_hint = (
        batch_cfg.get("input_file")
        or infer_cfg.get("audio")
        or ""
    )
    from models.voxtral_asr.data import infer_lang_id_from_path
    from models.voxtral_asr.language_tag import language_tag_cfg, resolve_language_id

    language_cfg = language_tag_cfg(cfg)
    language_enabled = bool(language_cfg.get("enabled", False))
    language_id = None
    if language_enabled:
        explicit_language = (
            batch_cfg.get("language")
            or infer_cfg.get("language")
            or language_cfg.get("language")
        )
        language_id = (
            resolve_language_id(explicit_language, language_cfg)
            if explicit_language not in (None, "")
            else infer_lang_id_from_path(input_hint)
        )

    from models import build_model

    backend = build_model(model_cfg.get("model_type", "voxtral_asr"))
    print(f"Loading model from {model_cfg['model_path']} ...")
    started = time.time()
    wrapper = backend.load_for_inference(
        model_cfg["model_path"],
        device=model_cfg.get("device", "cuda:0"),
        lora_path=model_cfg.get("lora_path", ""),
        max_new_tokens=model_cfg.get("max_new_tokens", 256),
        adaptive_latency=adaptive,
        inference_mode=mode,
        language_tag=language_cfg,
        language_id=language_id,
    )
    wrapper.update({
        "emit_word_prob_threshold": emit_threshold,
        "emit_word_prob_space": emit_space,
        "inference_mode": mode,
        "offline_emit_token": infer_cfg.get("offline_emit_token", "W_off"),
        "language_tag_enabled": language_enabled,
        "language_id": language_id,
    })
    print(f"Model loaded in {time.time() - started:.1f}s")

    transcribe_kwargs = {
        "return_timestamps": bool(infer_cfg.get("return_timestamps", False)),
        "adaptive_latency": adaptive,
        "emit_word_prob_threshold": emit_threshold,
        "emit_word_prob_space": emit_space,
        "mode": mode,
        "max_new_tokens": model_cfg.get("max_new_tokens", 256),
        "offline_emit_token": infer_cfg.get("offline_emit_token", "W_off"),
    }

    audio = infer_cfg.get("audio")
    if audio:
        for result in backend.transcribe(wrapper, audio, **transcribe_kwargs):
            print(f"Text: {result.text}")
            if result.emits is not None:
                print(f"Emits: {result.n_emits}")
            if result.time_stamps:
                for timestamp in result.time_stamps:
                    print(
                        f"  [{timestamp.start_time:.3f}s - "
                        f"{timestamp.end_time:.3f}s] {timestamp.text}"
                    )
        return

    input_file = batch_cfg.get("input_file")
    if not input_file:
        raise ValueError("provide inference.audio or batch.input_file")

    samples = _load_batch(input_file, int(batch_cfg.get("limit", 0) or 0))
    audio_paths = [sample["audio"] for sample in samples]
    batch_size = max(1, int(batch_cfg.get("batch_size", 1) or 1))
    pipeline_encode = bool(batch_cfg.get("pipeline_encode", False))
    pipeline_gpu_encode = bool(batch_cfg.get("pipeline_gpu_encode", True))
    stream_batch_rows = int(batch_cfg.get("stream_batch_rows", 0) or 0)
    listen_chunk_size = int(batch_cfg.get("listen_chunk_size", 16) or 16)
    use_pipeline = (
        (
            mode == "offline"
            and pipeline_encode
            and batch_size > 1
            and len(samples) > batch_size
        )
        or (
            mode == "streaming"
            and (pipeline_encode or stream_batch_rows > 1)
            and len(samples) > 1
        )
    )

    output_file = batch_cfg.get("output_file")
    if not output_file:
        output_file = f"{os.path.splitext(input_file)[0]}_results.jsonl"
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    started = time.time()
    written = 0
    with open(output_file, "w", encoding="utf-8") as output:
        if use_pipeline:
            results = backend.transcribe(
                wrapper,
                audio_paths,
                batch_size=batch_size,
                pipeline_encode=True,
                pipeline_gpu_encode=pipeline_gpu_encode,
                listen_chunk_size=listen_chunk_size,
                stream_batch_rows=stream_batch_rows,
                **transcribe_kwargs,
            )
            for audio_path, result in zip(audio_paths, results):
                output.write(
                    json.dumps(
                        _result_record(audio_path, result, mode),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                written += 1
        else:
            for offset in range(0, len(samples), batch_size):
                batch_paths = audio_paths[offset : offset + batch_size]
                results = backend.transcribe(
                    wrapper,
                    batch_paths,
                    listen_chunk_size=listen_chunk_size,
                    stream_batch_rows=stream_batch_rows,
                    **transcribe_kwargs,
                )
                for audio_path, result in zip(batch_paths, results):
                    output.write(
                        json.dumps(
                            _result_record(audio_path, result, mode),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    written += 1
                output.flush()
                print(f"[{written}/{len(samples)}] transcribed")

    elapsed = time.time() - started
    print(f"Transcribed {written} files in {elapsed:.1f}s")
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
