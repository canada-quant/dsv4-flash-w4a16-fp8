#!/usr/bin/env bash
# OOM characterization sweep for dual DGX Spark TP=2 DSv4 no-MTP.
#
# Drives the eugr/spark-vllm-docker pattern: tear down vllm_node on head+worker,
# relaunch with new --max-model-len / --max-num-seqs combo, watch for /health 200 OR
# container exit. Each cell takes 5-10 min (cold start + KV profile + cudagraph capture).
#
# Records: did it BOOT? did it SERVE a 256-tok request? if not, the exit code +
# last 100 lines of vllm_node logs.
#
# Usage:
#   bash oom_sweep.sh <head_host> <worker_host> <image_tag> <head_qsfp_ip> <worker_qsfp_ip> <qsfp_ifname> <out_dir>
#
# Then sweeps a hard-coded grid; tweak inline if you want to change axes.
set -uo pipefail

HEAD="${1:?head}"
WORKER="${2:?worker}"
IMAGE="${3:?image_tag}"
HEAD_IP="${4:?head_qsfp_ip}"
WORKER_IP="${5:?worker_qsfp_ip}"
QSFP_IF="${6:?qsfp_ifname}"
OUT="${7:?out_dir}"
mkdir -p "$OUT"

ENV_FLAGS=(
  -e VLLM_TRITON_MLA_SPARSE=1
  -e VLLM_TRITON_MLA_SPARSE_HEAD_BLOCK_SIZE=4
  -e VLLM_RPC_TIMEOUT=600000
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600
  -e TILELANG_CLEANUP_TEMP_FILES=1
  -e HF_HUB_OFFLINE=1
  -e NCCL_IB_DISABLE=0
  -e NCCL_NET_PLUGIN=none
  -e NCCL_IB_SUBNET_AWARE_ROUTING=1
  -e NCCL_IB_MERGE_NICS=0
  -e GLOO_SOCKET_IFNAME="$QSFP_IF"
  -e NCCL_SOCKET_IFNAME="$QSFP_IF"
)

DOCKER_BASE=(
  docker run -d --name vllm_node
  --gpus all --network=host --ipc=host
  --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface"
)

ENGINE_BASE=(
  vllm serve canada-quant/DeepSeek-V4-Flash-W4A16-FP8
  --served-model-name DSV4-W4A16-FP8 deepseek-ai/DeepSeek-V4-Flash deepseek-v4-flash
  --trust-remote-code
  --kv-cache-dtype fp8 --block-size 256
  --tokenizer-mode deepseek_v4
  --tool-call-parser deepseek_v4 --enable-auto-tool-choice
  --reasoning-parser deepseek_v4
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'
  --max-num-batched-tokens 8192
  --gpu-memory-utilization 0.90
  --host 0.0.0.0 --port 8888
  -tp 2 --nnodes 2
  --master-port 29501
)

bring_up_cell() {
  local mml="$1" mns="$2"
  # tear down
  ssh "pcozz@$HEAD"   "docker rm -f vllm_node 2>/dev/null || true"
  ssh "pcozz@$WORKER" "docker rm -f vllm_node 2>/dev/null || true"
  sleep 2
  # worker rank 1 first
  ssh "pcozz@$WORKER" "${DOCKER_BASE[*]} ${ENV_FLAGS[*]} -e VLLM_HOST_IP=$WORKER_IP $IMAGE bash -c '${ENGINE_BASE[*]} --master-addr $HEAD_IP --max-model-len $mml --max-num-seqs $mns --node-rank 1 --headless'" > /dev/null
  sleep 3
  ssh "pcozz@$HEAD" "${DOCKER_BASE[*]} ${ENV_FLAGS[*]} -e VLLM_HOST_IP=$HEAD_IP $IMAGE bash -c '${ENGINE_BASE[*]} --master-addr $HEAD_IP --max-model-len $mml --max-num-seqs $mns --node-rank 0'" > /dev/null
}

await_health_or_exit() {
  local timeout_s=900
  local waited=0
  while [[ $waited -lt $timeout_s ]]; do
    if ssh "pcozz@$HEAD" 'curl -sf http://localhost:8888/health > /dev/null'; then
      echo READY
      return 0
    fi
    HS=$(ssh "pcozz@$HEAD" 'docker inspect --format "{{.State.Status}}" vllm_node 2>/dev/null || echo missing')
    WS=$(ssh "pcozz@$WORKER" 'docker inspect --format "{{.State.Status}}" vllm_node 2>/dev/null || echo missing')
    if [[ "$HS" == "exited" || "$WS" == "exited" ]]; then
      echo "EXIT head=$HS worker=$WS"
      return 1
    fi
    sleep 15
    waited=$((waited + 15))
  done
  echo "TIMEOUT"
  return 2
}

run_cell() {
  local mml="$1" mns="$2"
  local tag="ctx${mml}_seq${mns}"
  local cell_log="$OUT/$tag.log"
  echo "=== cell mml=$mml mns=$mns at $(date -u +%FT%TZ) ===" | tee "$cell_log"
  bring_up_cell "$mml" "$mns"
  local t0=$(date +%s)
  local status
  status=$(await_health_or_exit | tail -1)
  local t1=$(date +%s)
  local boot_s=$((t1 - t0))
  echo "status=$status boot_s=$boot_s" | tee -a "$cell_log"
  if [[ "$status" == "READY" ]]; then
    # smoke a 256-token completion
    local out
    out=$(curl -fsS --max-time 90 "http://$HEAD:8888/v1/completions" \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"DSV4-W4A16-FP8\",\"prompt\":\"Hello, world.\",\"max_tokens\":256,\"temperature\":0}" 2>&1 | head -200)
    echo "smoke=$out" >> "$cell_log"
    echo "$tag READY ${boot_s}s" >> "$OUT/sweep_summary.txt"
  else
    # capture last 100 lines on both nodes
    ssh "pcozz@$HEAD"   'docker logs --tail 100 vllm_node 2>&1' >> "$cell_log" 2>&1 || true
    echo "----- WORKER -----" >> "$cell_log"
    ssh "pcozz@$WORKER" 'docker logs --tail 100 vllm_node 2>&1' >> "$cell_log" 2>&1 || true
    echo "$tag FAIL($status) ${boot_s}s" >> "$OUT/sweep_summary.txt"
  fi
  echo "  -> recorded to $cell_log"
}

# Grid sweep — context wall at seqs=1, then concurrency wall at ctx=256K
echo "=== context wall sweep (max_num_seqs=1) ==="
for mml in 262144 524288 1048576 1572864 2097152; do
  run_cell "$mml" 1
done

echo
echo "=== concurrency wall sweep (max_model_len=262144) ==="
for mns in 2 4 8 16; do
  run_cell 262144 "$mns"
done

echo
echo "=== sweep summary ==="
cat "$OUT/sweep_summary.txt"
