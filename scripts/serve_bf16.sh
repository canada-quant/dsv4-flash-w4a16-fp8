#!/usr/bin/env bash
# Phase 2 verify: serve BF16 dequant model on 8x H200 TP=8 (port 8001)
set -euo pipefail
exec > >(tee -a /workspace/output/logs/bf16-serve.log) 2>&1
echo "==== serve_bf16.sh start $(date -u +%FT%TZ) ===="

source /opt/pytorch/bin/activate
export TORCH_CUDA_ARCH_LIST="9.0a"

vllm serve /workspace/model-bf16 \
  --tensor-parallel-size 8 \
  --kv-cache-dtype fp8 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.85 \
  --tokenizer-mode deepseek_v4 \
  --trust-remote-code \
  --port 8001 \
  --served-model-name DeepSeek-V4-Flash-BF16
