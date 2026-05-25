# vllm-patches

Patches we apply on top of upstream `vllm-project/vllm` to support canada-quant's DeepSeek-V4 quantization artifacts on consumer Blackwell (SM 12.0, RTX PRO 6000) and similar non-mainstream hardware.

This directory is the **bridge** between our jasl/vllm runtime and our long-term goal of running on upstream mainline + documented patches. Every patch here has a row in the table below; every patch we depend on for production must either be merged upstream, have an open upstream PR, or have explicit rationale for indefinite carry.

## Patch series

| File | Title | Status | Upstream | Target minimum upstream | Rationale |
|---|---|---|---|---|---|
| `0001_marlin_moe_archs_40923.patch` | MARLIN_MOE_ARCHS gates 12.0a;12.1a cubins under CUDA 12.9 | open upstream | [PR #40923](https://github.com/vllm-project/vllm/pull/40923) | vllm-project/vllm post-#40923 merge | Without this, JIT-PTX fallback corrupts tokens on SM 12.0 concurrent decode. Approved by 1 reviewer, blocked on core-maintainer SM120 policy review. |
| `0002_marlin_moe_workspace_4x.patch` | Oversize Marlin MoE lock-array 4x | jasl-only (new) | (to file — follow-up to #40923) | needs new upstream PR | Resolves cudaErrorIllegalAddress on sm_120a after #40923 native cubins. Same root cause family as #26558, #36811, #35922. |
| `0003_marlin_moe_c_tmp_36889.patch` | Drop min() clamp on c_tmp FP32 reduce buffer | closed upstream (re-file needed) | [PR #36889](https://github.com/vllm-project/vllm/pull/36889) (closed unmerged) | needs re-file | The original author couldn't reproduce on A6000/DGX Spark; we reproduce on RTX PRO 6000. Re-file with concrete repro. |
| `0004_fp8_compressor_indexer_dequant.txt` | In-artifact BF16 dequant for compressor + indexer FP8 weights | artifact-level (documented) | n/a (artifact remediation, not vLLM patch) | n/a | One-time per artifact. Documented in each model card. |
| `0005_e_score_correction_bias_defensive.patch` | Make missing `e_score_correction_bias` non-fatal in DSv4 loader | jasl-only (new) | (to file) | needs new upstream PR | Allows older calibrations to load on current vLLM. Card A blocker. |
| `0006_sparse_mla_long_prefill_5d647981.patch` | Sparse MLA prefill stabilization | jasl-only | jasl/vllm@5d647981 | watch upstream | Lands in jasl ds4-sm120-preview-dev. Likely targets same race regime as #40923+#36889. |

## Apply order

```bash
# from a clean upstream vllm-project/vllm checkout at the minimum supported tag
cd vllm
for p in vllm-patches/0001_*.patch vllm-patches/0002_*.patch vllm-patches/0003_*.patch vllm-patches/0005_*.patch vllm-patches/0006_*.patch; do
    git apply --check "$p" && git apply "$p" || { echo "FAILED: $p"; exit 1; }
done
```

## Release gate

Each canada-quant artifact ships only after `vllm serve <artifact>` succeeds on:

1. The most recent tagged release of vllm-project/vllm (currently `v0.21.0`)
2. Plus this patch series applied cleanly
3. Plus the 3-prompt smoke test (sequential thinking, batched chat, MTP acceptance ≥ 75%)

Until that gate is met, the artifact's quickstart uses the jasl/vllm fork pin. The migration is tracked per-artifact in the model card "Reproducibility" section.

## Why this exists

Regulated buyers require supply-chain provenance. "We run upstream vLLM with N documented patches, each linked to an open PR" is defensible; "we run a maintainer's personal fork" is not. This directory makes every patch a real OSS contribution candidate instead of tribal knowledge.

The runtime stays on jasl/vllm for now because the patch series is large; as upstream catches up the series shrinks. When it reaches zero, the migration is complete and the gate switches to "mainline tagged release only".
