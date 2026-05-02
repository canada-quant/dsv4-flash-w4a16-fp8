#!/usr/bin/env python3
"""Compare native vs dryrun configs and patch if compress_ratios is missing."""
import json
import sys


def show(path):
    c = json.load(open(path))
    print(f"=== {path} ===")
    print(f"  compress_ratios: {'YES (len=' + str(len(c.get('compress_ratios', []))) + ')' if 'compress_ratios' in c else 'no'}")
    print(f"  compress_rates: {c.get('compress_rates')}")
    return c


def main():
    native = show("/workspace/model-native/config.json")
    target_path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/model-w4a16-dryrun/config.json"
    target = show(target_path)
    if "compress_ratios" not in target and "compress_ratios" in native:
        # Patch: add compress_ratios from native (vLLM expects it)
        # We use native's compress_ratios since that's the original per-layer list
        # (compress_rates in new transformers is a dict keyed by layer-type, not the per-layer list)
        target["compress_ratios"] = native["compress_ratios"]
        json.dump(target, open(target_path, "w"), indent=2)
        print(f"PATCHED: added compress_ratios (len={len(target['compress_ratios'])}) to {target_path}")
    else:
        print("no patch needed")


if __name__ == "__main__":
    main()
