#!/usr/bin/env python3
"""Strip quantization_config and set expert_dtype=bf16 on dequantized BF16 model config."""
import json, shutil, sys

path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/model-bf16/config.json"
shutil.copy(path, path + ".bak")
c = json.load(open(path))
print("BEFORE:")
print("  quantization_config:", c.get("quantization_config"))
print("  expert_dtype:", c.get("expert_dtype"))
print("  torch_dtype:", c.get("torch_dtype"))
c.pop("quantization_config", None)
c["expert_dtype"] = "bf16"
c["torch_dtype"] = "bfloat16"
with open(path, "w") as f:
    json.dump(c, f, indent=2)
print("AFTER:")
print("  quantization_config:", c.get("quantization_config"))
print("  expert_dtype:", c.get("expert_dtype"))
print("  torch_dtype:", c.get("torch_dtype"))
