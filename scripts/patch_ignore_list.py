#!/usr/bin/env python3
"""Rewrite quantization_config.ignore in config.json to match the renamed
safetensors keys produced by rewrite_for_vllm.py."""
import json
import re
import sys


def rename(p: str) -> str:
    if p == "lm_head":
        return "lm_head"
    if p.startswith("model."):
        p = p[len("model."):]
    p = p.replace(".self_attn.", ".attn.")
    p = p.replace(".mlp.", ".ffn.")
    p = p.replace(".shared_experts.gate_proj", ".shared_experts.w1")
    p = p.replace(".shared_experts.up_proj", ".shared_experts.w3")
    p = p.replace(".shared_experts.down_proj", ".shared_experts.w2")
    p = re.sub(r"\.attn\.kv_proj$", ".attn.wkv", p)
    p = re.sub(r"\.attn\.q_a_proj$", ".attn.wq_a", p)
    p = re.sub(r"\.attn\.q_b_proj$", ".attn.wq_b", p)
    p = re.sub(r"\.attn\.o_a_proj$", ".attn.wo_a", p)
    p = re.sub(r"\.attn\.o_b_proj$", ".attn.wo_b", p)
    return p


def main():
    path = sys.argv[1]
    c = json.load(open(path))
    qc = c.get("quantization_config", {})
    old = qc.get("ignore", [])
    new = [rename(p) for p in old]
    qc["ignore"] = new
    c["quantization_config"] = qc
    with open(path, "w") as f:
        json.dump(c, f, indent=2)
    print(f"updated {len(new)} ignore entries")
    print("sample renames:")
    for o, n in zip(old[:6], new[:6]):
        print(f"  {o}\n  -> {n}")


if __name__ == "__main__":
    main()
