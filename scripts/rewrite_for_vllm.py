#!/usr/bin/env python3
"""
Rewrite an llmcompressor (kylesayrs/transformers-v5 branch) save_pretrained
output so that jasl/ds4-sm120 vLLM can load it.

Two layers of work:

1. NAME REWRITES (extend kylesayrs's fix_checkpoint_keys.py):
   - Strip "model." prefix
   - HC family: model.hc_head.{hc_base,hc_fn,hc_scale} -> hc_head_{base,fn,scale}
   - Per-layer HC: layers.X.attn_hc.{base,fn,scale} -> layers.X.hc_attn_{base,fn,scale}
                    layers.X.ffn_hc.{...} -> layers.X.hc_ffn_{...}
   - lm_head.weight -> head.weight
   - model.embed_tokens.weight -> embed.weight
   - .self_attn. -> .attn.
   - .mlp. -> .ffn.
   - .input_layernorm. -> .attn_norm.
   - .post_attention_layernorm. -> .ffn_norm.
   - shared_experts.gate_proj -> shared_experts.w1
   - shared_experts.up_proj -> shared_experts.w3
   (down_proj handled by the refuse step below)
   - attention proj renames: kv_proj -> wkv, q_a_proj -> wq_a, q_b_proj -> wq_b,
                             o_a_proj -> wo_a, o_b_proj -> wo_b, q_a_norm -> q_norm,
                             sinks -> attn_sink

2. SHARED_EXPERTS DOWN_PROJ REFUSION:
   transformers v5's V4 save_pretrained decomposes layers.X.mlp.shared_experts.down_proj.weight
   row-by-row into hidden_size separate keys named shared_experts.{i}.w2.weight.weight,
   each of shape [intermediate_size]. Stack them back into a single
   shared_experts.w2.weight tensor of shape [hidden_size, intermediate_size] per layer.
"""
import argparse
import json
import os
import re
import shutil
from collections import defaultdict

import torch
from safetensors import safe_open
from safetensors.torch import save_file


# Anything matching this pattern is a row of the original down_proj.weight that
# transformers v5 decomposed during save. We refuse them into a single tensor.
SHARED_EXPERTS_ROW_RE = re.compile(
    r"^(model\.layers\.\d+\.mlp\.shared_experts)\.(\d+)\.w2\.weight\.weight$"
)


def rename_key(key: str) -> str:
    """Rewrite a key from kylesayrs/transformers-v5 form to native (vLLM-mapper-input) form."""
    # Strip outer model. prefix (after this, top-level looks like layers./hc_head./embed_tokens./norm./head/lm_head)
    if key.startswith("model."):
        key = key[len("model."):]

    # Top-level renames for embedding/lm_head/norm/hc_head
    # (vLLM's _make_deepseek_v4_weights_mapper expects native names like embed.weight,
    # head.weight, norm.weight, hc_head_base.)
    if key == "embed_tokens.weight":
        return "embed.weight"
    if key == "lm_head.weight":
        return "head.weight"
    if key == "norm.weight":
        return "norm.weight"
    if key in ("hc_head.hc_base", "hc_head.hc_fn", "hc_head.hc_scale"):
        return key.replace("hc_head.hc_", "hc_head_")

    # Per-layer / per-mtp HC family
    # layers.X.attn_hc.{base,fn,scale} -> layers.X.hc_attn_{base,fn,scale}
    # layers.X.ffn_hc.{base,fn,scale}  -> layers.X.hc_ffn_{base,fn,scale}
    key = re.sub(r"\.attn_hc\.(base|fn|scale)$", r".hc_attn_\1", key)
    key = re.sub(r"\.ffn_hc\.(base|fn|scale)$", r".hc_ffn_\1", key)

    # Indexer compressor parts (must come before general compressor/attn renames)
    key = re.sub(r"\.compressor\.indexer\.(ape|wgate|wkv)", r".indexer.compressor.\1", key)
    key = key.replace(".compressor.indexer.kv_norm.", ".indexer.compressor.norm.")
    key = re.sub(r"\.compressor\.indexer\.(weights_proj|wq_b)", r".indexer.\1", key)
    key = key.replace(".compressor.kv_norm.", ".compressor.norm.")

    # Attention: self_attn -> attn
    key = key.replace(".self_attn.", ".attn.")
    # MLP -> FFN
    key = key.replace(".mlp.", ".ffn.")
    # Layer norms
    key = key.replace(".input_layernorm.", ".attn_norm.")
    key = key.replace(".post_attention_layernorm.", ".ffn_norm.")

    # Shared expert projection renames (gate_proj -> w1, up_proj -> w3, down_proj -> w2)
    key = key.replace(".shared_experts.gate_proj.", ".shared_experts.w1.")
    key = key.replace(".shared_experts.up_proj.", ".shared_experts.w3.")
    key = key.replace(".shared_experts.down_proj.", ".shared_experts.w2.")

    # Routed expert projection renames — same w1/w2/w3 convention.
    # These are indexed: experts.{N}.gate_proj.{weight_packed,weight_scale,...}.
    # vLLM's FusedMoE.make_expert_params_mapping enumerates patterns by w1/w2/w3
    # name; without this rename the loader's expert_mapping never matches our
    # gate_proj/up_proj/down_proj keys -> UnboundLocalError on name_mapped.
    key = re.sub(r"\.experts\.(\d+)\.gate_proj\.", r".experts.\1.w1.", key)
    key = re.sub(r"\.experts\.(\d+)\.up_proj\.", r".experts.\1.w3.", key)
    key = re.sub(r"\.experts\.(\d+)\.down_proj\.", r".experts.\1.w2.", key)

    # Compressor module renames (V4 MLA compressor + indexer-compressor).
    # vLLM expects native short names: wkv, wgate, wq_b, ape (for position_bias).
    # Applied AFTER mlp/experts renames so we don't accidentally hit .experts.N.gate_proj.
    # These match both `attn.compressor.X` and `attn.compressor.indexer.X` paths;
    # the kylesayrs indexer-rearrange rules at the top of this function then move
    # `compressor.indexer.X` → `indexer.compressor.X` (or `indexer.X`) for the
    # `wgate`/`wkv`/`wq_b`/`weights_proj`/`ape` cases. Order is fine because the
    # kylesayrs rules ran on the input shape; we re-apply them at the end.
    # Inner renames (the compressor's indexer): apply first so the kylesayrs
    # rearrange below can match the canonical wgate/wkv/wq_b/ape names.
    key = key.replace(".compressor.indexer.kv_proj.", ".compressor.indexer.wkv.")
    key = key.replace(".compressor.indexer.gate_proj.", ".compressor.indexer.wgate.")
    key = key.replace(".compressor.indexer.q_b_proj.", ".compressor.indexer.wq_b.")
    key = key.replace(".compressor.indexer.position_bias", ".compressor.indexer.ape")

    # Outer renames (the layer's compressor): apply AFTER inner so we don't
    # accidentally double-rename `.compressor.indexer.gate_proj.` via the
    # broader `.compressor.gate_proj.` rule (it wouldn't anyway because `.indexer.`
    # is between them, but explicit ordering keeps the intent clear).
    key = key.replace(".compressor.kv_proj.", ".compressor.wkv.")
    key = key.replace(".compressor.gate_proj.", ".compressor.wgate.")
    key = key.replace(".compressor.q_b_proj.", ".compressor.wq_b.")
    key = key.replace(".compressor.position_bias", ".compressor.ape")

    # Now apply kylesayrs indexer rearrange — moves compressor.indexer.{ape,wgate,wkv}
    # to indexer.compressor.{ape,wgate,wkv}, and compressor.indexer.{weights_proj,wq_b}
    # to indexer.{weights_proj,wq_b}.
    key = re.sub(r"\.compressor\.indexer\.(ape|wgate|wkv)", r".indexer.compressor.\1", key)
    key = re.sub(r"\.compressor\.indexer\.(weights_proj|wq_b)", r".indexer.\1", key)

    # Attention projection module renames (transformers V5 names -> native flat)
    # These appear under .attn. (post-rename); apply only inside that scope.
    key = re.sub(r"\.attn\.kv_proj\.", ".attn.wkv.", key)
    key = re.sub(r"\.attn\.q_a_proj\.", ".attn.wq_a.", key)
    key = re.sub(r"\.attn\.q_b_proj\.", ".attn.wq_b.", key)
    key = re.sub(r"\.attn\.o_a_proj\.", ".attn.wo_a.", key)
    key = re.sub(r"\.attn\.o_b_proj\.", ".attn.wo_b.", key)
    key = re.sub(r"\.attn\.q_a_norm\.", ".attn.q_norm.", key)
    key = key.replace(".attn.sinks", ".attn.attn_sink")

    return key


def collect_split_down_proj_groups(all_keys):
    """Identify groups of split shared_experts.down_proj rows.

    Returns: dict mapping `model.layers.X.mlp.shared_experts` -> sorted list of
    (row_index, original_key).
    """
    groups = defaultdict(list)
    for k in all_keys:
        m = SHARED_EXPERTS_ROW_RE.match(k)
        if m:
            base = m.group(1)
            row_idx = int(m.group(2))
            groups[base].append((row_idx, k))
    for base in groups:
        groups[base].sort()
    return groups


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="source dir with safetensors")
    p.add_argument("--output", required=True, help="dest dir (must not exist)")
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # 1. Read source index
    with open(os.path.join(args.input, "model.safetensors.index.json")) as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    metadata = index.get("metadata", {})

    # 2. Identify split-down_proj groups
    split_groups = collect_split_down_proj_groups(weight_map.keys())
    print(f"Detected {len(split_groups)} layer(s) with split shared_experts.down_proj")
    if split_groups:
        sample = next(iter(split_groups.items()))
        print(f"  example: {sample[0]} has {len(sample[1])} row tensors")

    # 3. Group source keys by file (for streaming)
    keys_by_file = defaultdict(list)
    for k, fn in weight_map.items():
        keys_by_file[fn].append(k)

    new_weight_map = {}

    # 4. Per-file rewrite. We do (a) renames for non-split keys, (b) refuse split
    #    groups when we encounter their LAST row in any shard. Simpler: for each
    #    output shard, write all renamed scalar tensors that belong here, then
    #    write any refused tensors whose LAST source row resides here.
    src_files = sorted({fn for fn in weight_map.values()})
    skipped_split_rows = set()
    for grp_base, rows in split_groups.items():
        for _, k in rows:
            skipped_split_rows.add(k)

    # Build per-output-file plan: native key -> source tensor reference
    # For renamed keys, source = (src_file, src_key).
    # For refused tensors, source = ("REFUSE", grp_base) with rows lookup.
    out_plan = defaultdict(dict)  # out_file -> {new_key: source}
    for src_file, keys in keys_by_file.items():
        for k in keys:
            if k in skipped_split_rows:
                continue
            new_k = rename_key(k)
            out_plan[src_file][new_k] = ("KEY", src_file, k)

    # Place refused tensors: write into shard where row 0 originally lived, by
    # convention. Build shard placement.
    refuse_assignments = {}  # grp_base -> out_file
    for grp_base, rows in split_groups.items():
        first_row_key = rows[0][1]
        out_file = weight_map[first_row_key]
        new_k = rename_key(grp_base + ".down_proj.weight")  # -> ...shared_experts.w2.weight
        out_plan[out_file][new_k] = ("REFUSE", grp_base)
        refuse_assignments[grp_base] = out_file

    # 5. Write each output shard
    for out_file in src_files:
        plan = out_plan.get(out_file, {})
        if not plan:
            continue
        print(f"writing {out_file} ({len(plan)} keys) ...")
        tensors_for_save = {}
        # Pre-open source files we need
        opened = {}

        def src_open(fn):
            if fn not in opened:
                opened[fn] = safe_open(os.path.join(args.input, fn), framework="pt")
            return opened[fn]

        for new_k, source in plan.items():
            if source[0] == "KEY":
                _, sf, sk = source
                t = src_open(sf).get_tensor(sk)
                tensors_for_save[new_k] = t
                new_weight_map[new_k] = out_file
            else:  # REFUSE
                _, grp_base = source
                rows = split_groups[grp_base]
                # Each row tensor lives in some src file (likely all in same)
                row_tensors = []
                for _, sk in rows:
                    sf = weight_map[sk]
                    t = src_open(sf).get_tensor(sk)
                    row_tensors.append(t)
                fused = torch.stack(row_tensors, dim=0).contiguous()
                tensors_for_save[new_k] = fused
                new_weight_map[new_k] = out_file
                print(f"  refused {grp_base} into {new_k} shape={tuple(fused.shape)}")
        save_file(tensors_for_save, os.path.join(args.output, out_file), metadata={"format": "pt"})
        for h in opened.values():
            try:
                h.__exit__(None, None, None)  # type: ignore
            except Exception:
                pass

    # 6. Write new index
    new_index = {
        "metadata": metadata,
        "weight_map": new_weight_map,
    }
    # update total_size if missing (vllm doesn't strictly require it)
    with open(os.path.join(args.output, "model.safetensors.index.json"), "w") as f:
        json.dump(new_index, f, indent=2)
    print(f"wrote new index with {len(new_weight_map)} keys")

    # 7. Copy non-safetensors files
    for fn in os.listdir(args.input):
        if fn.endswith(".safetensors") or fn == "model.safetensors.index.json":
            continue
        s = os.path.join(args.input, fn)
        d = os.path.join(args.output, fn)
        if os.path.isfile(s) and not os.path.exists(d):
            shutil.copy2(s, d)
    print("DONE")


if __name__ == "__main__":
    main()
