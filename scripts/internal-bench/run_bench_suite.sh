#!/usr/bin/env bash
# Full bench suite for one (pair × jasl-SHA) combo. Runs when /health is 200.
#
# Usage:
#   bash run_bench_suite.sh <head_host> <port> <sha_tag> <out_dir>
#
# Suite (in order):
#   1. chat_smoke.sh (4 chat completions, ~1 min)
#   2. throughput_bench.py at 3 canonical configs:
#        c=1 prompt=1024 decode=1024 (single-stream long-out)
#        c=1 prompt=65536 decode=2048 (long-context retrieval-like)
#        c=4 prompt=1024 decode=512 (chat batching)
#   3. aime_think_max.py 3 problems, max_tokens=65536 (~30-60 min)
#
# Each output stored as JSON; final summary.md appended to OUT.
set -uo pipefail

HEAD="${1:?head_host}"
PORT="${2:?port}"
SHA_TAG="${3:?sha_tag}"
OUT="${4:?out_dir}"
mkdir -p "$OUT"
BASE="http://$HEAD:$PORT"
SCRIPTS="$HOME/dsv4-test-2gb10/scripts"

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$OUT/run.log"; }

log "=== bench suite for $SHA_TAG against $BASE ==="
log "out=$OUT"

# 0. Verify health
if ! curl -sf "$BASE/health" > /dev/null; then
  log "FAIL: /health not 200, aborting"
  exit 1
fi
log "/health ok"

# 1. chat smoke
log "step 1/4: chat_smoke"
bash "$SCRIPTS/chat_smoke.sh" "$BASE" DSV4-W4A16-FP8 > "$OUT/chat_smoke.log" 2>&1 \
  && log "  PASS" || log "  FAIL: see $OUT/chat_smoke.log"

# 2. throughput at 3 configs
log "step 2/4: throughput @ c=1 prompt=1024 decode=1024"
python3 "$SCRIPTS/throughput_bench.py" "$BASE" \
  --model DSV4-W4A16-FP8 --concurrency 1 --requests 10 \
  --prompt-tokens 1024 --decode-tokens 1024 \
  --out "$OUT/throughput_c1_p1k_d1k.json" 2>&1 | tee -a "$OUT/run.log"

log "step 2/4: throughput @ c=1 prompt=65536 decode=2048 (long ctx)"
python3 "$SCRIPTS/throughput_bench.py" "$BASE" \
  --model DSV4-W4A16-FP8 --concurrency 1 --requests 4 \
  --prompt-tokens 65536 --decode-tokens 2048 \
  --out "$OUT/throughput_c1_p64k_d2k.json" 2>&1 | tee -a "$OUT/run.log"

log "step 2/4: throughput @ c=4 prompt=1024 decode=512 (batched chat)"
python3 "$SCRIPTS/throughput_bench.py" "$BASE" \
  --model DSV4-W4A16-FP8 --concurrency 4 --requests 16 \
  --prompt-tokens 1024 --decode-tokens 512 \
  --out "$OUT/throughput_c4_p1k_d512.json" 2>&1 | tee -a "$OUT/run.log"

# 3. long-reasoning think-max
log "step 3/4: aime_think_max (3 problems, max_tokens=65536) — this is slow"
python3 "$SCRIPTS/aime_think_max.py" "$BASE" \
  --model DSV4-W4A16-FP8 --max-tokens 65536 --problems 3 \
  --out "$OUT/aime_think_max.json" 2>&1 | tee -a "$OUT/run.log"

# 4. summary
log "step 4/4: writing summary.md"
python3 - "$OUT" <<'PYEOF' | tee -a "$OUT/run.log"
import json, sys, os
out_dir = sys.argv[1]
def load(p):
    try:
        return json.load(open(os.path.join(out_dir, p)))
    except FileNotFoundError:
        return None
data = {
    "c1_p1k_d1k": load("throughput_c1_p1k_d1k.json"),
    "c1_p64k_d2k": load("throughput_c1_p64k_d2k.json"),
    "c4_p1k_d512": load("throughput_c4_p1k_d512.json"),
    "aime": load("aime_think_max.json"),
}
lines = [f"# Bench suite summary — {os.path.basename(out_dir)}", ""]
lines += ["## Throughput", "", "| config | aggregate tok/s | per-req median tok/s | TPOT median ms | successful |", "|---|---|---|---|---|"]
for name, key in [("c=1 prompt=1K decode=1K", "c1_p1k_d1k"),
                  ("c=1 prompt=64K decode=2K", "c1_p64k_d2k"),
                  ("c=4 prompt=1K decode=512", "c4_p1k_d512")]:
    d = data[key]
    if d is None:
        lines.append(f"| {name} | n/a | n/a | n/a | n/a |")
        continue
    s = d["summary"]
    lines.append(f"| {name} | {s['aggregate_tok_per_s']:.1f} | {s['per_request_tps']['median']:.1f} | {s['tpot_ms']['median']:.1f} | {s['successful']} |")
lines += ["", "## AIME-2024 think-max (max_tokens=65536, reasoning_effort=high)", ""]
if data["aime"]:
    s = data["aime"]["summary"]
    lines += [f"- problems run: {s['n']}, correct: {s['correct']}, truncated: {s['truncated']}",
              f"- avg wall: {s['avg_wall_s']:.0f}s, avg decode tps: {s['avg_decode_tps']:.1f}, avg tokens: {s['avg_completion_tokens']:.0f}",
              ""]
    lines += ["| problem | wall_s | decode_tps | tokens | truncated | extracted | correct |", "|---|---|---|---|---|---|---|"]
    for r in data["aime"]["results"]:
        lines.append(f"| {r.get('id','?')} | {r.get('wall_s',0):.1f} | {r.get('decode_tok_per_s',0):.1f} | {r.get('completion_tokens',0)} | {r.get('truncated','?')} | {r.get('extracted','?')} | {r.get('correct','?')} |")
with open(os.path.join(out_dir, "summary.md"), "w") as f:
    f.write("\n".join(lines))
print(f"  wrote {out_dir}/summary.md")
PYEOF

log "=== bench suite complete ==="
log "results in $OUT"
