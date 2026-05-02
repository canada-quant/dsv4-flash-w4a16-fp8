#!/usr/bin/env python3
"""Patch quantization_config.config_groups.targets in config.json to match
post-rewrite_for_vllm module names. The recipe writes targets using the
v5 names (self_attn.q_a_proj etc.), but after rewrite_for_vllm renames the
saved tensors, vLLM's V4 model uses native flat names (attn.wq_a etc.).
The find_matched_target() lookup compares against config_groups.targets, so
those targets must match the post-rename names.
"""
import json
import sys

p = sys.argv[1] if len(sys.argv) > 1 else "/workspace/model-w4a16-dryrun-vllm/config.json"
c = json.load(open(p))
qc = c["quantization_config"]

print("=== old config_groups targets ===")
for g, gv in qc["config_groups"].items():
    print(f"  {g}: {gv.get('targets')}")

if "attention" in qc["config_groups"]:
    qc["config_groups"]["attention"]["targets"] = [
        # Native flat attn names (post-rewrite_for_vllm.py)
        r"re:.*attn\.(wq_a|wq_b|wkv|wo_a|wo_b|fused_wqa_wkv)$",
        r"re:.*attn\.compressor\.(wgate|wkv|fused_wkv_wgate)$",
        r"re:.*attn\.indexer\.(weights_proj|wq_b)$",
        r"re:.*attn\.indexer\.compressor\.(wgate|wkv)$",
    ]
if "experts" in qc["config_groups"]:
    qc["config_groups"]["experts"]["targets"] = [
        # Routed experts in vLLM use w1/w2/w3 (post-rewrite); plus the FusedMoE
        # gate_up_proj fused name
        r"re:.*ffn\.experts\.\d+\.(w1|w2|w3)$",
        r"re:.*ffn.*gate_up_proj$",
    ]

json.dump(c, open(p, "w"), indent=2)
print("=== new config_groups targets ===")
for g, gv in qc["config_groups"].items():
    print(f"  {g}: {gv.get('targets')}")
