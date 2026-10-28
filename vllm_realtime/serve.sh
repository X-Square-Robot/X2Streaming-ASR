#!/usr/bin/env bash
# Adaptive streaming ASR server on vLLM (/v1/realtime for demo/app.py).
#
# Required:
#   MODEL_PATH=/path/to/X2Streaming-ASR-4B-1009 bash vllm_realtime/serve.sh
#
# Optional:
#   TEXT_MODEL=<dir>        exported text LM (default: $MODEL_PATH-vllm-text; exported on first run)
#   LORA_PATH=<dir>         LoRA merged into both the text LM export and the encoder
#   CUDA_VISIBLE_DEVICES=0  GPU for the vLLM LM
#   ENCODER_GPU=<id>        GPU for the encoder sidecar (default: same card as the LM)
#   MAX_SESSIONS=4          concurrent realtime sessions (6 is stable on one A800 80GB)
#   GPU_MEM=0.35            vLLM gpu_memory_utilization
#   KV_CACHE_GIB=<n>        fixed KV cache size; use on GPUs shared with other jobs, together
#                           with a lower GPU_MEM (vLLM needs GPU_MEM x total memory free at startup)
#   PYTHON=<python>         env with vLLM (default: python on PATH)
#   ENCODER_PYTHON=<python> env with requirements.txt (default: $PYTHON)
#   HOST=0.0.0.0 PORT=8000
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ -z "${MODEL_PATH:-}" ]]; then
  echo "ERROR: set MODEL_PATH to a local X2Streaming-ASR checkpoint directory"
  exit 1
fi
MODEL_PATH="$(cd "$MODEL_PATH" && pwd)"
TEXT_MODEL="${TEXT_MODEL:-${MODEL_PATH%/}-vllm-text}"
PY="${PYTHON:-$(command -v python)}"
ENCODER_PYTHON="${ENCODER_PYTHON:-$PY}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
ENCODER_GPU="${ENCODER_GPU:-${CUDA_VISIBLE_DEVICES%%,*}}"
export CUDA_VISIBLE_DEVICES

if [[ ! -f "$TEXT_MODEL/export_meta.json" ]]; then
  echo "Exporting text LM for vLLM -> $TEXT_MODEL"
  CUDA_VISIBLE_DEVICES="$ENCODER_GPU" "$ENCODER_PYTHON" "$SCRIPT_DIR/export_text_model.py" \
    --model "$MODEL_PATH" \
    --output "$TEXT_MODEL" \
    --lora "${LORA_PATH:-}"
fi

echo "Model  : $MODEL_PATH${LORA_PATH:+  lora=$LORA_PATH}"
echo "Text LM: $TEXT_MODEL  gpu=$CUDA_VISIBLE_DEVICES"
echo "Encoder: gpu=$ENCODER_GPU  python=$ENCODER_PYTHON"
echo "Listen : ws://${HOST:-0.0.0.0}:${PORT:-8000}/v1/realtime  max_sessions=${MAX_SESSIONS:-4}"

exec "$PY" "$SCRIPT_DIR/serve.py" \
  --model "$TEXT_MODEL" \
  --encoder-model "$MODEL_PATH" \
  --encoder-python "$ENCODER_PYTHON" \
  --encoder-gpu "$ENCODER_GPU" \
  --encoder-lora "${LORA_PATH:-}" \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-8000}" \
  --gpu-memory-utilization "${GPU_MEM:-0.35}" \
  --kv-cache-gib "${KV_CACHE_GIB:-0}" \
  --max-model-len "${MAX_MODEL_LEN:-9000}" \
  --max-sessions "${MAX_SESSIONS:-4}" \
  "$@"
