---
language:
  - en
base_model:
  - deepseek-ai/DeepSeek-V4-Flash
pipeline_tag: text-generation
library_name: transformers
tags:
  - deepseek_v4
  - deepseek
  - llmcompressor
  - quantized
  - INT4
  - W4A16
  - GPTQ
  - compressed-tensors
  - vllm
  - moe
  - hybrid-attention
  - csa
  - hca
license: mit
license_link: https://choosealicense.com/licenses/mit/
name: pastapaul/DeepSeek-V4-Flash-quantized.w4a16
description: >-
  W4A16 (INT4 weights, BF16 activations) GPTQ quantization of
  DeepSeek-V4-Flash, produced with LLM Compressor and saved in the
  compressed-tensors format. Quantizes attention and routed experts;
  shared experts and the LM head remain in BF16.
---

# DeepSeek-V4-Flash-quantized.w4a16

> **Status: <TBD - publication blocked until Phase 4 verification completes on Phase 3b full-run output.>**

## Model Overview

- **Model architecture:** `DeepseekV4ForCausalLM` (V4-Flash family, 284B total / 13B active MoE, hybrid CSA + HCA attention with mHC hyperconnections, hash-routed MoE)
- **Input:** Text
- **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** INT4 (W4A16, group size 128, GPTQ algorithm via LLM Compressor `GPTQModifier`)
  - **Activation quantization:** None (BF16 activations)
  - **Quantized layers:** Attention (CSA + HCA + indexer Linear modules) + routed experts
  - **Preserved (BF16):** Shared experts, LM head, embeddings, layer norms, MTP draft head, hyperconnection (hc) parameters
- **Format:** [compressed-tensors](https://github.com/neuralmagic/compressed-tensors) (Marlin / Machete kernel compatible)
- **Release date:** <TBD>
- **Version:** 0.1.0 (preview — see "Status" notice above)
- **Calibration data:** [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k), 1024 samples, max_seq_len 512
- **Quantization framework:** [LLM Compressor PR #2647](https://github.com/vllm-project/llm-compressor/pull/2647) (`kylesayrs/transformers-v5` branch)
- **Inference framework:** [vLLM PR #41276](https://github.com/vllm-project/vllm/pull/41276) (`neuralmagic:kylesayrs/deepseek-ct` branch)

## Status & Disclaimers

This model is part of an integration project documenting how to combine the
multiple in-flight upstream PRs that bring DeepSeek-V4-Flash compressed-tensors
support to vLLM. As of <TBD>, the relevant PRs are:

- vLLM #41276 (`neuralmagic:kylesayrs/deepseek-ct`): Draft, not yet merged. Adds
  `scale_fmt` fallback, `weight_scale` vs `weight_scale_inv` selector,
  `quant_config` plumbing for fused attention modules, and a
  `NotImplementedError` raise for unquantized BF16 attention.
- LLM Compressor #2647 (`kylesayrs/transformers-v5`): Adds V4 support to the
  GPTQ pipeline, including `linearize_moe_model` for routed expert calibration.

This model was produced and validated against those branches at the commits
referenced in the deployment instructions below. Upstream behavior may change.

**The reference deployment for this model is vLLM PR #41276 + LLM Compressor PR #2647.**
Loading in transformers directly works for inspection but does not exercise the
optimized kernels.

## Deployment

This model was deployed using the following branch with vLLM:
[#41276](https://github.com/vllm-project/vllm/pull/41276), commit
`f910a73a93c54d3a3139d64add5da4624d619603`.

```bash
# On 8× H200 (or any Hopper / Ampere with sufficient VRAM):
vllm serve pastapaul/DeepSeek-V4-Flash-quantized.w4a16 \
  --tensor-parallel-size 8 \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.85 \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --reasoning-parser deepseek_v4
```

For DGX Spark (SM 12.1a / GB10) at TP=2, see the canonical recipe and validation
report in [`findings/spark_tp2_deployment.md`](https://github.com/pasta-paul/dsv4-flash-w4a16-fp8/blob/main/findings/spark_tp2_deployment.md)
and the launch script [`scripts/serve_spark_tp2.sh`](https://github.com/pasta-paul/dsv4-flash-w4a16-fp8/blob/main/scripts/serve_spark_tp2.sh).
The Spark recipe additionally requires the workspace prereservation patch
([`scripts/patch_workspace_prereserve.py`](https://github.com/pasta-paul/dsv4-flash-w4a16-fp8/blob/main/scripts/patch_workspace_prereserve.py),
filed upstream as [`vllm-project/vllm#41700`](https://github.com/vllm-project/vllm/issues/41700)).

### Loading in transformers (inspection / debugging only)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "pastapaul/DeepSeek-V4-Flash-quantized.w4a16",
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=False,  # V4 is in transformers v5 main
)
tokenizer = AutoTokenizer.from_pretrained("pastapaul/DeepSeek-V4-Flash-quantized.w4a16")
# Note: transformers loads the compressed-tensors format but does not benefit
# from optimized W4A16 kernels. For production inference, use vLLM.
```

## Creation

This model was created by:

1. Loading the base `deepseek-ai/DeepSeek-V4-Flash` checkpoint (FP4 routed experts + FP8 non-experts, native format) and dequantizing the experts to BF16 using [`flagos-ai/DeepSeek-V4-FlagOS`](https://github.com/flagos-ai/DeepSeek-V4-FlagOS) `convert_weight.py`.
2. Running GPTQ-W4A16 calibration via LLM Compressor PR #2647 with the recipe below, on 8× H200 (TP=8, distributed via torchrun).
3. Post-processing the compressed-tensors output with a custom `rewrite_for_vllm.py` that:
   - Renames keys from transformers v5 save layout (e.g. `model.layers.X.self_attn.*`, `model.input_layernorm`, `model.hc_head.hc_*`) to the layout consumed by vLLM's `_make_deepseek_v4_weights_mapper` (e.g. `layers.X.attn.*`, `attn_norm`, `hc_head_*`).
   - Refuses the `shared_experts.down_proj` weight, which the GPTQ saver decomposes into 4096 row-shape (2048,) BF16 tensors per layer when `shared_experts` are excluded from quantization. The refusion stacks these back into the original `(hidden_size, moe_intermediate_size) = (4096, 2048)` weight.
   - Rewrites `quantization_config.ignore` patterns in `config.json` from literal module paths to regex patterns that match the post-rename layout.

The complete pipeline, including all integration scripts and findings, is open
sourced at [pasta-paul/dsv4-flash-awq-w4a16](https://github.com/pasta-paul/dsv4-flash-awq-w4a16).

### Recipe

```python
from llmcompressor.modifiers.quantization import GPTQModifier
from compressed_tensors.quantization import QuantizationScheme
from llmcompressor.modifiers.quantization.calibration import W4A16

recipe = GPTQModifier(
    config_groups={
        "default": QuantizationScheme(
            targets=["Linear"],
            **W4A16,
        ),
    },
    ignore=[
        "lm_head",
        "re:.*shared_experts.*",
    ],
    dampening_frac=0.1,
    sequential_targets=["DeepseekV4DecoderLayer"],
    batch_size=32,
    max_seq_len=512,
)
```

`re:.*self_attn.*` is **not** in the ignore list, on purpose. vLLM PR #41276
raises `NotImplementedError("DeepSeekV4 requires FP8 attention quantization")`
when attention weights have neither `weight_scale_inv` (FP8) nor `weight_scale`
(compressed-tensors). Including attention in the W4A16 quant produces the
required `weight_scale` attribute.

## Evaluation

> Evaluation results pending Phase 4 verification on the full 1024-sample run.

Will report once available:

- **Harness gates** (matching the H200 native FP4/FP8 baseline run):
  - chat-smoke quick (<TBD>/4)
  - chat-smoke quality (<TBD>/4)
  - chat-smoke coding (<TBD>/2 — hard requirement: 2/2)
  - toolcall15 (<TBD>/30 strict)
- **Bench** (8× H200, TP=8, random 128/512, 48 prompts, ignore-eos):
  - c=1: <TBD> input tok/s, <TBD> output tok/s, TTFT <TBD>ms, TPOT <TBD>ms
  - c=4: <TBD>
  - c=8: <TBD>

Native baseline for comparison (8× H200, TP=8, FP4/FP8 native):

- chat-smoke quick: 4/4 PASS
- chat-smoke quality: 4/4 PASS
- chat-smoke coding: **2/2 PASS** (notable — SM12x reports failures in the same eval, see [findings/SM90-vs-SM12x.md](https://github.com/pasta-paul/dsv4-flash-awq-w4a16/blob/main/findings/SM90-vs-SM12x.md) <TBD>)
- toolcall15: 23/30 = 77%, 11 strict pass / 4 fail
- c=8 bench: 777/970 input/output tok/s, TTFT 211.96ms, TPOT 9.24ms

Recovery percentages will be reported relative to this baseline.

## Known Limitations

1. **Reference deployment uses an unmerged draft PR.** This model's reference
   inference path is vLLM PR #41276, which is in WIP/Draft status as of <TBD>.
   When the PR merges (or is rebased), the deployment instructions above will
   be updated. Pin to the referenced commit for stable behavior.

2. **DGX Spark TP=2 deployment is validated** (2026-05-04). Marlin W4A16
   kernels select cleanly on SM 12.1a once the workspace prereservation patch
   is applied (see [`scripts/patch_workspace_prereserve.py`](https://github.com/pasta-paul/dsv4-flash-w4a16-fp8/blob/main/scripts/patch_workspace_prereserve.py)).
   Without the patch, `--enforce-eager` is required as a workaround
   (~4× decode penalty). With the patch, decode runs at ~14–17 tok/s with
   CUDA graphs enabled. Spark-side benchmarks: GSM8K 95.37% (vs 92.87% on H200),
   HumanEval pass@1 80.49% (vs 54.27% on H200 — methodology difference),
   harness toolcall15 41/45 (92%), and a budget-isolated 64K-context retest
   passes 9 / 10 think-max generation cases (the single non-pass is a
   client-side wall-clock timeout, not a model defect). Full report:
   [`findings/spark_tp2_deployment.md`](https://github.com/pasta-paul/dsv4-flash-w4a16-fp8/blob/main/findings/spark_tp2_deployment.md).

3. **Reasoning modes:** Non-think, Think High, and Think Max have all been
   exercised on Spark TP=2. The 64 K-context retest validates Think Max for
   all 9 generation cases that previously hit the 32 K reasoning budget,
   confirming those failures were budget-bound rather than model-bound.

4. **GPTQ vs AWQ:** The original brief targeted AWQ-W4A16. The shipped quant
   uses GPTQ via LLM Compressor's `GPTQModifier` with the W4A16 scheme — both
   produce W4A16 compressed-tensors output, but GPTQ was the path that
   converged in the integration timeframe. AWQ-specific calibration may
   produce different (likely better for some layers) accuracy and is a
   candidate for a follow-on revision.

5. **Shared experts are not quantized.** Quantizing them surfaces a save-side
   decomposition issue (the GPTQ saver splits `down_proj` into thousands of
   row-shape tensors per layer). The custom refusion script handles this for
   the unquantized case; quantizing shared experts would require additional
   refusion logic for the W4A16-packed format.

## References

- [Base model: deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
- [vLLM PR #41276 — DSV4 Compressed Tensors Support](https://github.com/vllm-project/vllm/pull/41276)
- [LLM Compressor PR #2647](https://github.com/vllm-project/llm-compressor/pull/2647)
- [Integration source code & findings](https://github.com/pasta-paul/dsv4-flash-awq-w4a16)
- [RedHat's NVFP4-FP8 V4-Flash quant (companion / FP8-attn variant)](https://huggingface.co/RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8)
- [vLLM v0.20.0 release notes](https://github.com/vllm-project/vllm/releases/tag/v0.20.0)
- [vLLM V4 Roadmap (issue #40902)](https://github.com/vllm-project/vllm/issues/40902)

## Citation

If this model or the associated integration work is useful in your research or
deployment, please cite:

```bibtex
@misc{cozzolino2026dsv4flashw4a16,
  title  = {{DeepSeek-V4-Flash-quantized.w4a16}: A W4A16 GPTQ quantization of
            DeepSeek-V4-Flash with kylesayrs PR #41276 integration},
  author = {Cozzolino, Paul},
  year   = {2026},
  month  = {5},
  howpublished = {Hugging Face},
  url    = {https://huggingface.co/pastapaul/DeepSeek-V4-Flash-quantized.w4a16}
}
```

## Contact

For issues with the model or the integration pipeline, open an issue at
[pasta-paul/dsv4-flash-awq-w4a16](https://github.com/pasta-paul/dsv4-flash-awq-w4a16/issues).
