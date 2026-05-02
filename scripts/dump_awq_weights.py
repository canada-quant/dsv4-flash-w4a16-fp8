#!/usr/bin/env python3
import glob
import sys
from safetensors import safe_open

src = sys.argv[1] if len(sys.argv) > 1 else "/workspace/model-w4a16-dryrun"
out = sys.argv[2] if len(sys.argv) > 2 else "/workspace/output/awq-weight-names-sample.txt"

files = sorted(glob.glob(f"{src}/*.safetensors"))
keys = []
for f in files:
    with safe_open(f, framework="pt") as s:
        keys.extend(s.keys())
keys.sort()

with open(out, "w") as fh:
    fh.write(f"# AWQ output: {src}\n")
    fh.write(f"# total tensor keys: {len(keys)}\n\n")
    fh.write("=== first 60 keys ===\n")
    for k in keys[:60]:
        fh.write(k + "\n")
    fh.write("\n=== last 60 keys ===\n")
    for k in keys[-60:]:
        fh.write(k + "\n")
    fh.write("\n=== unique top-level prefixes (3 components) ===\n")
    prefixes = sorted(set(".".join(k.split(".")[:3]) for k in keys))
    for p in prefixes[:80]:
        fh.write(p + "\n")
    fh.write("\n=== sample layer-0 keys ===\n")
    for k in [x for x in keys if x.startswith("model.layers.0.") or x.startswith("layers.0.")][:80]:
        fh.write(k + "\n")
    fh.write("\n=== hc_head, embed, lm_head, head, norm keys ===\n")
    for k in keys:
        if any(s in k for s in ["hc_head", "embed", "lm_head", "head", "norm"]) and "layers." not in k:
            fh.write(k + "\n")

print(f"wrote {out} with {len(keys)} keys")
