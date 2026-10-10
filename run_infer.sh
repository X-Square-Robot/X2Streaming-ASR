#!/usr/bin/env bash
# Voxtral Realtime ASR 推理启动脚本
# conda activate voxtral
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo ">>> Running inference ..."
python infer.py --config configs/infer.yaml "$@"
echo ">>> Done."
