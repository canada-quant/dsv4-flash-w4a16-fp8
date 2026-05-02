#!/usr/bin/env bash
# Run the post-calibration pipeline on a freshly-saved llmcompressor output:
#   1. rewrite_for_vllm.py (header rename + shared_experts.down_proj refusion)
#   2. fix_dryrun_config.py (config field repair)
#   3. patch quantization_config.ignore to regex form
#   4. start vLLM serve in tmux
#
# Usage:
#   bash pipeline_post_calibrate.sh <input_dir> <output_dir> <port> <served_name>
set -euo pipefail
SRC="${1:?input dir}"
DST="${2:?output dir}"
PORT="${3:-8002}"
SERVED="${4:-DeepSeek-V4-Flash-AWQ-DRYRUN}"
LOGDIR=/workspace/output/logs
mkdir -p "$LOGDIR"

ts() { date -u +%FT%TZ; }
log() { echo "[pipeline $(ts)] $*"; }

log "step 1: rewrite_for_vllm $SRC -> $DST"
rm -rf "$DST"
source /workspace/.venv-quant/bin/activate
python3 /workspace/rewrite_for_vllm.py --input "$SRC" --output "$DST" 2>&1 | tee "$LOGDIR/rewrite.log" | tail -5
log "rewrite done"

log "step 2: fix_dryrun_config"
python3 /workspace/fix_dryrun_config.py --target "$DST/config.json"

log "step 3: patch ignore list to regex"
python3 - "$DST/config.json" << 'PY'
import json, sys
p = sys.argv[1]
c = json.load(open(p))
c["quantization_config"]["ignore"] = ["lm_head", "re:.*shared_experts.*"]
json.dump(c, open(p, "w"), indent=2)
print("ignore set to:", c["quantization_config"]["ignore"])
PY

log "step 4: launch serve in tmux phase-serve-${PORT}"
SESS="phase-serve-${PORT}"
tmux kill-session -t "$SESS" 2>/dev/null || true
pkill -9 -f "vllm serve" 2>/dev/null || true
sleep 3
SERVE_LOG="$LOGDIR/serve-${PORT}.log"
rm -f "$SERVE_LOG"
tmux new-session -d -s "$SESS" "source /opt/pytorch/bin/activate
export TORCH_CUDA_ARCH_LIST=9.0a
vllm serve $DST \
  --tensor-parallel-size 8 \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.85 \
  --compilation-config '{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"custom_ops\":[\"all\"]}' \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --trust-remote-code \
  --port $PORT \
  --served-model-name $SERVED 2>&1 | tee $SERVE_LOG
echo SERVE_EXIT=\$?
exec bash"
log "tmux session $SESS launched, serve log at $SERVE_LOG"
log "pipeline_post_calibrate done"
