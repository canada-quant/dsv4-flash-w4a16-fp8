#!/usr/bin/env bash
# Phase 2: dequant DeepSeek-V4-Flash FP4/FP8 -> BF16 via flagos convert_weight.py
set -euo pipefail
exec > >(tee -a /workspace/output/logs/dequant.log) 2>&1
echo "==== dequant.sh start $(date -u +%FT%TZ) ===="

source /opt/pytorch/bin/activate
cd /workspace/flagos

python3 convert_weight.py \
  --input-fp4-hf-path /workspace/model-native \
  --output-bf16-hf-path /workspace/model-bf16 \
  --device cuda

echo "DEQUANT_DONE_$(date +%s)"
du -sh /workspace/model-bf16
ls /workspace/model-bf16 | head -10
