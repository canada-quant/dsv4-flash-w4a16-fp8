#!/usr/bin/env python3
"""Update config_groups.targets to include all naming variants vLLM might
look up: both v5 (gate_proj/up_proj/down_proj) and native (w1/w2/w3) and
fused (gate_up_proj/fused_wqa_wkv/fused_wkv_wgate)."""
import json
import sys

p = sys.argv[1] if len(sys.argv) > 1 else "/workspace/model-w4a16-dryrun-vllm/config.json"
c = json.load(open(p))
qc = c["quantization_config"]

# Find the attention group and experts group by content
for g, gv in qc["config_groups"].items():
    targets = gv.get("targets", [])
    has_attn = any(("self_attn" in t) or ("attn." in t and "ffn" not in t) for t in targets)
    has_experts = any(("experts" in t) for t in targets)
    if has_attn and not has_experts:
        gv["targets"] = [
            r"re:.*attn\.(wq_a|wq_b|wkv|wo_a|wo_b|fused_wqa_wkv|q_a_proj|q_b_proj|kv_proj|o_a_proj|o_b_proj)$",
            r"re:.*attn\.compressor\.(wgate|wkv|fused_wkv_wgate|gate_proj|kv_proj)$",
            r"re:.*attn\.indexer\.(weights_proj|wq_b|q_b_proj)$",
            r"re:.*attn\.indexer\.compressor\.(wgate|wkv|gate_proj|kv_proj)$",
        ]
    elif has_experts:
        gv["targets"] = [
            r"re:.*experts\.\d+\.(w1|w2|w3|gate_proj|up_proj|down_proj|gate_up_proj)$",
        ]

json.dump(c, open(p, "w"), indent=2)
for g, gv in qc["config_groups"].items():
    print(g)
    for t in gv["targets"]:
        print("  ", t)
