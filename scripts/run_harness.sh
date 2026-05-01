#!/usr/bin/env bash
# Run the jasl ds4-sm120 harness against a vLLM endpoint.
#   $1 = label  (baseline | bf16 | quant)
#   $2 = base URL  (e.g. http://127.0.0.1:8000)
#   $3 = served-model-name (e.g. DeepSeek-V4-Flash)
set -euo pipefail
LABEL="${1:?label required}"
URL="${2:?base url required}"
MODEL="${3:?served model name required}"

source /opt/pytorch/bin/activate
cd /workspace/vllm-ds4-sm120-harness
mkdir -p /workspace/output/logs

ts() { date -u +%FT%TZ; }
log() { echo "[$(ts)] $*" | tee -a /workspace/output/logs/${LABEL}-harness.log; }

log "=== harness start label=${LABEL} url=${URL} model=${MODEL} ==="

log "-- health --"
python -m ds4_harness.cli health \
  --base-url "${URL}" \
  2>&1 | tee /workspace/output/${LABEL}-health.txt | tee -a /workspace/output/logs/${LABEL}-harness.log

for tag in quick quality coding; do
  log "-- chat-smoke ${tag} --"
  python -m ds4_harness.cli chat-smoke \
    --base-url "${URL}" \
    --model "${MODEL}" \
    --tag "${tag}" \
    --timeout 900 \
    --jsonl-output /workspace/output/${LABEL}-${tag}.jsonl \
    2>&1 | tee -a /workspace/output/logs/${LABEL}-harness.log
done

log "-- toolcall15 --"
python -m ds4_harness.cli toolcall15 \
  --base-url "${URL}" \
  --model "${MODEL}" \
  --json-output /workspace/output/${LABEL}-toolcall15.json \
  2>&1 | tee -a /workspace/output/logs/${LABEL}-harness.log

log "=== harness done label=${LABEL} ==="
