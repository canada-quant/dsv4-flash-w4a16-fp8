# DeepSeek-V4-Flash AWQ-W4A16 — H200 quantization

Working directory + artifacts for an in-progress AWQ-W4A16 quantization
of `deepseek-ai/DeepSeek-V4-Flash` on a single 8× H200 (`p5en.48xlarge`)
DLAMI box, intended for hak-uma Coder B (DGX Spark, TP=2) deployment via
Marlin W4A16.

## Layout

- `REPORT.md` — phase-by-phase mission report, updated after each phase.
- `scripts/` — driver shell scripts and the calibration recipe.
- `patches/` — provenance + the two source patches required to make
  kylesayrs's `transformers-v5` PR work end-to-end on V4-Flash:
  - `helpers.py.diff` — llm-compressor `SequentialTracer.create_arg`
    extension to handle `transformers.cache_utils.Cache` (the
    `PretrainedConfig` pattern, applied to Cache objects).
  - `modeling_deepseek_v4.py.diff` — skip auto-DynamicCache creation in
    `DeepseekV4Model.forward` so the compressor takes its no-cache
    fallback path (V4-Flash config has `layer_types=None` which makes
    `DynamicCache(config=...)` produce generic `DynamicLayer`s that
    lack `store_compression_weights`).
  - `quantize_v4_w4a16.py.snapshot` — the recipe (W4A16, dampening 0.1,
    ignore lm_head + self_attn + shared_experts; routed experts targeted).
  - `VERSIONS.md` — repo HEADs, package versions, reproduction checklist.

## Reproduction

See `patches/VERSIONS.md` "How to reproduce on a fresh box".
