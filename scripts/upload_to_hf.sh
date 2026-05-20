#!/usr/bin/env bash
# Phase 5: upload AWQ-W4A16 model to HuggingFace canada-quant/DeepSeek-V4-Flash-W4A16-FP8
#
# Requires HF_TOKEN env var or `huggingface-cli login` already run.
# Never echo or log the token.
set -euo pipefail
LOG=/workspace/output/logs/hf-upload.log
exec > >(tee -a "$LOG") 2>&1
echo "==== upload_to_hf.sh start $(date -u +%FT%TZ) ===="

source /opt/pytorch/bin/activate

REPO="canada-quant/DeepSeek-V4-Flash-W4A16-FP8"
SRC="/workspace/model-awq-w4a16"

if [ -z "${HF_TOKEN:-}" ] && [ ! -f ~/.cache/huggingface/token ]; then
  echo "ERROR: no HF_TOKEN env var and no cached token. Run 'huggingface-cli login' or 'export HF_TOKEN=...' first."
  exit 2
fi

# Confirm source dir has weights + config + tokenizer
test -f "$SRC/config.json" || { echo "ERROR: $SRC/config.json missing"; exit 3; }
ls "$SRC"/*.safetensors > /dev/null || { echo "ERROR: no safetensors in $SRC"; exit 4; }
test -f "$SRC/README.md" || echo "WARN: README.md (model card) missing — should generate before upload"

du -sh "$SRC"
ls "$SRC" | head -20

echo "==== huggingface-cli upload ===="
huggingface-cli upload "$REPO" "$SRC" \
  --repo-type model \
  --commit-message "Initial AWQ-W4A16 quantization of DeepSeek-V4-Flash"

echo "UPLOAD_DONE_$(date +%s)"
