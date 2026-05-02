#!/usr/bin/env python3
"""Add packed_modules_mapping to DeepseekV4ForCausalLM in jasl/ds4-sm120 vLLM.

This is the piece kylesayrs's PR #41276 referenced (`self.packed_modules_mapping`
at line 205) but didn't actually define on the class. Without it, FP8_BLOCK
attn fails with:
    ValueError: Unable to find matching target for
    model.layers.0.attn.fused_wqa_wkv in the compressed-tensors config.
"""
import sys

F = sys.argv[1] if len(sys.argv) > 1 else "/workspace/vllm-source/vllm/model_executor/models/deepseek_v4.py"

with open(F) as f:
    src = f.read()

old = '''class DeepseekV4ForCausalLM(nn.Module, SupportsPP):
    model_cls = DeepseekV4Model

    # Default mapper assumes the original FP4-expert checkpoint layout.
    # Overridden per-instance in __init__ when expert_dtype != "fp4".
    hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper("fp4")'''

new = '''class DeepseekV4ForCausalLM(nn.Module, SupportsPP):
    model_cls = DeepseekV4Model

    # Default mapper assumes the original FP4-expert checkpoint layout.
    # Overridden per-instance in __init__ when expert_dtype != "fp4".
    hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper("fp4")

    # PATCH (paul/dsv4): mapping from fused module names to their constituent
    # shard names. Used by is_layer_skipped() and the compressed-tensors loader
    # to determine the quantization scheme for fused layers (which are constructed
    # at vLLM init from the underlying ColumnParallelLinear shards). Without this,
    # FP8_BLOCK on attn fails with "Unable to find matching target for
    # model.layers.0.attn.fused_wqa_wkv".
    packed_modules_mapping = {
        "fused_wqa_wkv": ["wq_a", "wkv"],
        "fused_wkv_wgate": ["wkv", "wgate"],
        "gate_up_proj": ["w1", "w3"],
    }'''

assert old in src, "anchor not found"
src = src.replace(old, new)
with open(F, "w") as f:
    f.write(src)
print(f"patched {F}")
