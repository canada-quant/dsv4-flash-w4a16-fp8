#!/usr/bin/env python3
"""Diagnostic: figure out what the 4096 spurious shared_experts.X entries actually are."""
import glob
import json
import re
from collections import defaultdict
from safetensors import safe_open

SRC = "/workspace/model-w4a16-dryrun"
files = sorted(glob.glob(f"{SRC}/*.safetensors"))

INDEX_RE = re.compile(r"\.shared_experts\.(\d+)\.")

# Single pass: open each file once, scan all keys, collect shapes for layer-0 shared_experts
shapes = defaultdict(list)  # (shape, dtype) -> [keys]
non_indexed_keys = []  # (key, shape, dtype) for non-indexed
all_se_count = 0

for f in files:
    with safe_open(f, framework="pt") as sf:
        keys = sf.keys()
        # Filter once
        layer0_se = [k for k in keys if "layers.0." in k and "shared_experts" in k]
        if not layer0_se:
            continue
        print(f"# {f}: {len(layer0_se)} layer-0 shared_experts keys")
        for k in layer0_se:
            t = sf.get_tensor(k)
            shape_key = (tuple(t.shape), str(t.dtype))
            shapes[shape_key].append(k)
            all_se_count += 1
            if not INDEX_RE.search(k):
                non_indexed_keys.append((k, tuple(t.shape), str(t.dtype)))

print(f"\n=== layer-0 shared_experts total: {all_se_count} ===\n")

print("=== shape × dtype distribution ===")
for (shape, dtype), keys in sorted(shapes.items(), key=lambda x: -len(x[1])):
    print(f"  shape={shape}  dtype={dtype}  count={len(keys)}")
    print(f"    first: {keys[0]}")
    print(f"    last:  {keys[-1]}")
    indices = sorted(int(m.group(1)) for m in (INDEX_RE.search(k) for k in keys) if m)
    if indices:
        contig = indices == list(range(indices[0], indices[-1] + 1))
        print(f"    index range: [{indices[0]}..{indices[-1]}]  contiguous={contig}  count={len(indices)}")

print("\n=== NON-indexed shared_experts keys (the 'clean' candidates) ===")
for k, shape, dtype in sorted(non_indexed_keys):
    print(f"  {k}  shape={shape}  dtype={dtype}")

# Config dims
cfg = json.load(open(f"{SRC}/config.json"))
print(f"\n=== config dims ===")
for k in ("hidden_size", "intermediate_size", "moe_intermediate_size",
          "n_routed_experts", "n_shared_experts", "num_experts_per_tok"):
    print(f"  {k}={cfg.get(k)}")
