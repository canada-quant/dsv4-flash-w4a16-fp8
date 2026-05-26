#!/usr/bin/env bash
# Master orchestrator — drives the full bench suite across both pairs.
# Runs from basementdocker; uses spark-3 and spark-4 as bench drivers.
#
# Stage 1 (concurrent — both pairs steady-state):
#   - chat_smoke on each
#   - throughput @ 3 canonical configs on each
#   - long-reasoning think-max on each
#
# Stage 2 (sequential per pair — needs container restart):
#   - OOM characterization sweep on Pair B (HEAD jasl), then Pair A (baseline)
#
# Stage 3 (cross-SHA regression — Pair A re-launched with intermediate SHA-M):
#   - throughput at 3 canonical configs at SHA-M
#
# Stage 4: synthesize → commit → push branch → draft jasl comment
#
# Usage:
#   bash orchestrate_bench.sh
set -uo pipefail

ROOT=$HOME/dsv4-test-2gb10
RESULTS=$ROOT/results
SCRIPTS=$ROOT/scripts
LOGS=$ROOT/logs
mkdir -p $RESULTS $LOGS

# Pair config
PAIR_B_TAG="pairB_head_5d647"     # S5+S6, jasl HEAD
PAIR_B_HEAD_IP="100.87.54.16"
PAIR_B_HEAD="spark-5"
PAIR_B_WORKER="spark-6"
PAIR_B_QSFP_IF="enp1s0f0np0"
PAIR_B_HEAD_QSFP="192.168.101.1"
PAIR_B_WORKER_QSFP="192.168.101.2"
PAIR_B_IMAGE="vllm-w4a16-dsv4:head-5d647"

PAIR_A_TAG="pairA_baseline_428e08e"  # S1+S2, jasl baseline
PAIR_A_HEAD_IP="100.66.113.93"
PAIR_A_HEAD="spark-1"
PAIR_A_WORKER="spark-2"
PAIR_A_QSFP_IF="enp1s0f1np1"
PAIR_A_HEAD_QSFP="192.168.1.1"
PAIR_A_WORKER_QSFP="192.168.1.2"
PAIR_A_IMAGE="vllm-w4a16-dsv4:baseline-428e08e"

# Intermediate SHA-M (pre-building on spark-3 in parallel)
PAIR_M_TAG="pairA_intermediate_a937d4b28"  # reuse Pair A hardware
PAIR_M_IMAGE="vllm-w4a16-dsv4:intermediate-a937d4b28"

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a $LOGS/orchestrator.log; }

wait_for_health() {
  local ip=$1; local tag=$2; local timeout=${3:-1800}
  local waited=0
  while ! curl -fsS --max-time 5 "http://$ip:8888/health" > /dev/null 2>&1; do
    [[ $waited -ge $timeout ]] && { log "$tag /health timeout"; return 1; }
    sleep 30
    waited=$((waited + 30))
  done
  log "$tag /health 200 after ${waited}s"
}

# ============ STAGE 1: concurrent steady-state bench on both pairs ============
log "=== STAGE 1: concurrent bench on Pair A + Pair B ==="

# Pair B suite driven from spark-3
(
  TAG=$PAIR_B_TAG
  OUT=$RESULTS/$TAG
  mkdir -p $OUT
  log "STAGE 1 Pair B from spark-3 -> $TAG"
  ssh spark-3 "bash ~/dsv4-test-2gb10/scripts/run_bench_suite.sh $PAIR_B_HEAD_IP 8888 $TAG $OUT" \
    > $LOGS/stage1_pairB.log 2>&1
  log "STAGE 1 Pair B done (see $OUT/summary.md)"
) &
PID_B=$!

# Pair A suite driven from spark-4
(
  TAG=$PAIR_A_TAG
  OUT=$RESULTS/$TAG
  mkdir -p $OUT
  log "STAGE 1 Pair A from spark-4 -> $TAG"
  ssh spark-4 "bash ~/dsv4-test-2gb10/scripts/run_bench_suite.sh $PAIR_A_HEAD_IP 8888 $TAG $OUT" \
    > $LOGS/stage1_pairA.log 2>&1
  log "STAGE 1 Pair A done (see $OUT/summary.md)"
) &
PID_A=$!

wait $PID_B
wait $PID_A
log "STAGE 1 complete"

# ============ STAGE 2: OOM characterization (sequential — needs container restart) ============
log "=== STAGE 2: OOM sweep on Pair B then Pair A ==="

log "STAGE 2.1 OOM sweep Pair B (S5+S6, HEAD)"
bash $SCRIPTS/oom_sweep.sh \
  $PAIR_B_HEAD $PAIR_B_WORKER $PAIR_B_IMAGE \
  $PAIR_B_HEAD_QSFP $PAIR_B_WORKER_QSFP $PAIR_B_QSFP_IF \
  $RESULTS/$PAIR_B_TAG/oom_sweep \
  > $LOGS/stage2_pairB_oom.log 2>&1
log "STAGE 2.1 Pair B OOM sweep done"

log "STAGE 2.2 OOM sweep Pair A (S1+S2, baseline)"
bash $SCRIPTS/oom_sweep.sh \
  $PAIR_A_HEAD $PAIR_A_WORKER $PAIR_A_IMAGE \
  $PAIR_A_HEAD_QSFP $PAIR_A_WORKER_QSFP $PAIR_A_QSFP_IF \
  $RESULTS/$PAIR_A_TAG/oom_sweep \
  > $LOGS/stage2_pairA_oom.log 2>&1
log "STAGE 2.2 Pair A OOM sweep done"

# ============ STAGE 3: intermediate SHA-M regression ============
log "=== STAGE 3: intermediate SHA-M (a937d4b28) on Pair A ==="

# Stop existing Pair A containers
ssh $PAIR_A_HEAD   'docker rm -f vllm_node 2>/dev/null || true'
ssh $PAIR_A_WORKER 'docker rm -f vllm_node 2>/dev/null || true'
sleep 2

# Pull intermediate image from spark-3 (where we pre-built it)
log "  loading vllm-w4a16-dsv4:intermediate-a937d4b28 onto Pair A from spark-3..."
ssh spark-3 "docker save $PAIR_M_IMAGE | gzip -1" | \
  tee >(ssh $PAIR_A_HEAD   "gunzip | docker load") | \
       ssh $PAIR_A_WORKER "gunzip | docker load"

# Bring up Pair A with intermediate image (skip-build skip-download skip-network)
log "  launching intermediate SHA-M on Pair A..."
cd ~/dsv4-flash-w4a16-fp8
bash scripts/bootstrap_dsv4_spark.sh \
  --head-host $PAIR_A_HEAD --worker-host $PAIR_A_WORKER \
  --ssh-user pcozz \
  --vllm-ref a937d4b28 \
  --image-tag $PAIR_M_IMAGE \
  --qsfp-ifname $PAIR_A_QSFP_IF \
  --head-qsfp-ip $PAIR_A_HEAD_QSFP --worker-qsfp-ip $PAIR_A_WORKER_QSFP \
  --skip-build --skip-network --skip-download \
  > $LOGS/stage3_pairA_M_bootstrap.log 2>&1

wait_for_health $PAIR_A_HEAD_IP "Pair A SHA-M" 1800 || exit 5

log "  running throughput bench on SHA-M..."
OUT=$RESULTS/$PAIR_M_TAG
mkdir -p $OUT
ssh spark-4 "bash ~/dsv4-test-2gb10/scripts/run_bench_suite.sh $PAIR_A_HEAD_IP 8888 $PAIR_M_TAG $OUT" \
  > $LOGS/stage3_pairA_M_bench.log 2>&1
log "STAGE 3 Pair A SHA-M done"

# ============ STAGE 4: synthesize + commit + draft ============
log "=== STAGE 4: synthesize results + commit ==="
python3 $SCRIPTS/synthesize_results.py --root $RESULTS \
  --out $RESULTS/dual_spark_summary_2026-05-26.md
log "wrote $RESULTS/dual_spark_summary_2026-05-26.md"

cd ~/dsv4-flash-w4a16-fp8
RESULTS_TARGET=findings/dual_spark_jasl_sha_regression_2026-05-26
mkdir -p $RESULTS_TARGET
cp -r $RESULTS/$PAIR_B_TAG $RESULTS_TARGET/ 2>/dev/null || true
cp -r $RESULTS/$PAIR_A_TAG $RESULTS_TARGET/ 2>/dev/null || true
cp -r $RESULTS/$PAIR_M_TAG $RESULTS_TARGET/ 2>/dev/null || true
cp $RESULTS/dual_spark_summary_2026-05-26.md $RESULTS_TARGET/SUMMARY.md
git add $RESULTS_TARGET
git commit -m "Add bench results: Pair B (HEAD) + Pair A (baseline + intermediate SHA-M)

Three jasl/vllm SHAs benched on 2× DGX Spark TP=2:
  - SHA-E (HEAD 5d6479811) on Pair B (S5+S6)
  - SHA-A (baseline 428e08e) on Pair A (S1+S2)
  - SHA-M (intermediate a937d4b28) on Pair A re-launched

Stage 1: chat_smoke + throughput @ 3 configs (c=1 P=1K D=1K, c=1 P=64K D=2K, c=4 P=1K D=512) + AIME-2024 think-max (max_tokens=65536, reasoning_effort=high, 3 problems).
Stage 2: OOM sweep (max_model_len ∈ {256K, 512K, 1M, 1.5M, 2M} at seqs=1, then seqs ∈ {2,4,8,16} at 256K).
Stage 3: SHA-M throughput-only for regression localization.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>" 2>&1 | tail -3

log "=== orchestrator complete ==="
log "Next manual step: review SUMMARY.md, then merge dual-spark-bench-2026-05-26 to main and push to canada-quant/dsv4-flash-w4a16-fp8."
