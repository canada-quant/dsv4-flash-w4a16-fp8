#!/usr/bin/env python3
"""List keys present in native config but missing/None in target config."""
import json, sys

n = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "/workspace/model-native/config.json"))
d = json.load(open(sys.argv[2] if len(sys.argv) > 2 else "/workspace/model-w4a16-dryrun/config.json"))

diff = []
for k, v in n.items():
    if k not in d or (d.get(k) is None and v is not None):
        diff.append(k)

print(f"native has {len(n)} keys, dryrun has {len(d)} keys")
print(f"missing-or-None in dryrun ({len(diff)}):")
for k in diff:
    val = n[k]
    repr_val = (str(val)[:80] + "...") if len(str(val)) > 80 else str(val)
    print(f"  {k}: {repr_val}")

print()
print("--- in dryrun but NOT in native ---")
for k in d:
    if k not in n:
        print(f"  {k}: {str(d[k])[:80]}")
