#!/usr/bin/env python3
import json
print("=== dryrun hc_head ===")
i = json.load(open("/workspace/model-w4a16-dryrun/model.safetensors.index.json"))
for k in sorted(k for k in i["weight_map"] if "hc_head" in k)[:10]:
    print(k)
print()
print("=== native hc_head ===")
i = json.load(open("/workspace/model-native/model.safetensors.index.json"))
for k in sorted(k for k in i["weight_map"] if "hc_head" in k)[:10]:
    print(k)
