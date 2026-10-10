#!/usr/bin/env bash
# Public web demo launcher.
#
# Required:
#   REPO_ID=<local-model-directory-or-hugging-face-id> bash demo/run.sh
#
# Optional:
#   MODE=adaptive BACKEND=transformers DEVICE=cuda:0 bash demo/run.sh

set -euo pipefail

DEMO_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND="${BACKEND:-transformers}"
MODE="${MODE:-adaptive}"

if [[ -z "${REPO_ID:-}" ]]; then
  echo "ERROR: set REPO_ID to a local model directory or Hugging Face model ID"
  exit 1
fi

if [[ -n "${PYTHON:-}" ]]; then
  ENV_PYTHON="${PYTHON}"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  ENV_PYTHON="${CONDA_PREFIX}/bin/python"
else
  ENV_PYTHON="$(command -v python)"
fi

ARGS=(
  --backend "$BACKEND"
  --mode "$MODE"
  --repo-id "$REPO_ID"
  --host "${HOST:-0.0.0.0}"
  --port "${PORT:-7860}"
  --device "${DEVICE:-cuda:0}"
  --delay-ms "${DELAY_MS:-480}"
  --language "${LANGUAGE:-zh}"
  --vllm-host "${VLLM_HOST:-127.0.0.1}"
  --vllm-port "${VLLM_PORT:-8000}"
)

if [[ -n "${ENC_DEVICE:-}" ]]; then
  ARGS+=(--enc-device "$ENC_DEVICE")
fi
if [[ -n "${LORA_PATH:-}" ]]; then
  ARGS+=(--lora-path "$LORA_PATH")
fi
if [[ -n "${EMIT_WORD_PROB_THRESHOLD:-}" ]]; then
  ARGS+=(--emit-word-prob-threshold "$EMIT_WORD_PROB_THRESHOLD")
fi
if [[ "$MODE" == "adaptive" ]]; then
  ARGS+=(--listen-chunk-size "${LISTEN_CHUNK_SIZE:-1}")
  ARGS+=(--max-decode-tokens "${MAX_DECODE_TOKENS:-32}")
  [[ "${ENABLE_LIVE:-1}" == "0" ]] && ARGS+=(--no-live)
  [[ "${ENABLE_OFFLINE:-1}" == "0" ]] && ARGS+=(--no-offline)
fi

cd "$DEMO_DIR"
exec "$ENV_PYTHON" app.py "${ARGS[@]}" "$@"
