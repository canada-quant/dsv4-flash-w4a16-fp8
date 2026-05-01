#!/usr/bin/env python3
"""Print compact bench summaries from vllm bench-serve JSON results."""
import json, sys, glob

paths = sys.argv[1:] or sorted(glob.glob("/workspace/output/baseline-bench-*.json"))
for p in paths:
    d = json.load(open(p))
    label = p.split("/")[-1]
    print(f"=== {label} ===")
    for k_print, k_json in [
        ("output_tok_s",  "output_throughput"),
        ("total_tok_s",   "total_token_throughput"),
        ("req_s",         "request_throughput"),
        ("TTFT_med_ms",   "median_ttft_ms"),
        ("TPOT_med_ms",   "median_tpot_ms"),
        ("duration_s",    "duration"),
    ]:
        v = d.get(k_json)
        if v is None:
            print(f"  {k_print}: missing")
        else:
            print(f"  {k_print}: {v:.2f}")
