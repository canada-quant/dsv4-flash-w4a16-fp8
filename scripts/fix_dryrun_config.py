#!/usr/bin/env python3
"""
Patch a llmcompressor-saved DSV4 config.json so vLLM jasl/ds4-sm120 can load it.

The transformers `add-deepseek-v4` branch saves config keys that don't match
what vLLM's V4 model code (built against older transformers) expects:

  rope_scaling: None         (native has this; new transformers strips it)
  rope_parameters: {...nested dict-of-dicts of "main"/"compress"...}
  compress_ratios: <missing> (native has list; new transformers renamed to compress_rates)
  quantization_config["scale_fmt"]: <missing for W4A16>

Strategy: keep new-transformers fields for round-trip with the new
transformers, but ALSO inject the legacy fields vLLM reads. We pull the
canonical legacy values from the native model dir.
"""
import argparse
import json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="/workspace/model-w4a16-dryrun/config.json")
    p.add_argument("--native", default="/workspace/model-native/config.json")
    args = p.parse_args()

    target = json.load(open(args.target))
    native = json.load(open(args.native))

    changes = []

    # Map _legacy_* aliases that the new transformers add-deepseek-v4 branch
    # writes back to the names vLLM jasl/ds4-sm120 expects.
    legacy_alias_map = {
        "_legacy_compress_ratios": "compress_ratios",
        "_legacy_num_hash_layers": "num_hash_layers",
        "_legacy_qk_rope_head_dim": "qk_rope_head_dim",
        "_legacy_compress_rate_csa": "compress_rate_csa",
        "_legacy_compress_rate_hca": "compress_rate_hca",
    }
    for legacy, canonical in legacy_alias_map.items():
        v = target.get(legacy)
        if v is not None and target.get(canonical) is None:
            target[canonical] = v
            changes.append(f"{canonical} <- {legacy}")

    # Last-resort fallback: still missing -> copy from native
    for k in ["compress_ratios", "num_hash_layers", "qk_rope_head_dim"]:
        if target.get(k) is None and native.get(k) is not None:
            target[k] = native[k]
            changes.append(f"{k} (from native)")

    # torch_dtype: new transformers writes "dtype" instead. vLLM needs both.
    if not target.get("torch_dtype") and (target.get("dtype") or native.get("torch_dtype")):
        target["torch_dtype"] = target.get("dtype") or native["torch_dtype"]
        changes.append("torch_dtype")

    if not target.get("rope_scaling") and native.get("rope_scaling"):
        target["rope_scaling"] = native["rope_scaling"]
        changes.append("rope_scaling")

    # Drop rope_parameters so transformers 4.57.x in DLAMI venv re-derives it from
    # rope_scaling at load time (matches what worked for Phase 1 native serve).
    if "rope_parameters" in target:
        target.pop("rope_parameters")
        changes.append("rope_parameters (removed; will be re-derived from rope_scaling)")

    json.dump(target, open(args.target, "w"), indent=2)
    print(f"PATCHED {args.target}: {', '.join(changes) if changes else 'no changes'}")


if __name__ == "__main__":
    main()
