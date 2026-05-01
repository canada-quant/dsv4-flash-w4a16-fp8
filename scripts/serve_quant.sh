#!/usr/bin/env bash
# Phase 4: serve AWQ-W4A16 quantized model on 8x H200 TP=8
set -euo pipefail
exec > >(tee -a /workspace/output/logs/quant-serve.log) 2>&1
echo "==== serve_quant.sh start $(date -u +%FT%TZ) ===="

source /opt/pytorch/bin/activate
export TORCH_CUDA_ARCH_LIST="9.0a"

vllm serve /workspace/model-awq-w4a16 \
  --tensor-parallel-size 8 \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.85 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --trust-remote-code \
  --port 8002 \
  --served-model-name DeepSeek-V4-Flash-AWQ
