#!/usr/bin/env bash
# Adapted from canada-quant/dsv4-flash-w4a16-fp8-mtp/scripts/chat_smoke.sh
# for the no-MTP variant on dual DGX Spark.
#
# Usage: bash chat_smoke.sh http://spark-5:8888 DSV4-W4A16-FP8
set -uo pipefail

BASE="${1:?usage: $0 <base_url> [model_name]}"
MODEL="${2:-DSV4-W4A16-FP8}"

echo "=== /health ==="
curl -fsS "$BASE/health" || { echo "FAIL: /health"; exit 1; }
echo "  OK"

echo
echo "=== /v1/models ==="
curl -fsS "$BASE/v1/models" | python3 -c "import sys,json; d=json.load(sys.stdin); print('  served:', [m['id'] for m in d['data']])"

echo
echo "=== 4 chat completions ==="
pass=0
total=0
for prompt in \
    "Hello! What is 1+1?" \
    "Write a haiku about Blackwell GPUs." \
    "What is the capital of France?" \
    "Briefly explain Mixture of Experts in one sentence."
do
    total=$((total + 1))
    echo "--- prompt: $prompt ---"
    t0=$(date +%s.%N)
    response=$(curl -fsS --max-time 120 "$BASE/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$MODEL\",
            \"messages\": [{\"role\": \"user\", \"content\": \"$prompt\"}],
            \"max_tokens\": 100,
            \"temperature\": 0
        }") || { echo "  FAIL: completion request"; continue; }
    t1=$(date +%s.%N)
    elapsed=$(echo "$t1 - $t0" | bc -l)
    content=$(echo "$response" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'][:200])" 2>/dev/null || echo "(parse failed)")
    tokens=$(echo "$response" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['usage'].get('completion_tokens',0))" 2>/dev/null || echo 0)
    tps=$(python3 -c "print(f'{$tokens / max($elapsed, 0.001):.1f}')")
    if [[ -n "$content" && "$content" != "(parse failed)" ]]; then
        echo "  ${content:0:150}..."
        echo "  -> $tokens tok in ${elapsed:0:5}s = ${tps} tok/s"
        pass=$((pass + 1))
    else
        echo "  FAIL: empty response"
    fi
done

echo
echo "=== Result: $pass/$total ==="
[[ $pass -eq $total ]] && exit 0 || exit 1
