# Phase 5 — Dual RTX PRO 6000 Blackwell deployment (2026-05-05)

End-to-end validation of `pastapaul/DeepSeek-V4-Flash-W4A16-FP8` on the harness HANDOFF's *primary development host*: dual RTX PRO 6000 Blackwell Server (SM 12.0, 96 GB × 2). Built on the **experimental superset** of `ds4-sm120` so we exercise commits that aren't in the public `ds4-sm120` PR branch yet — including the today-landed `e734ace5 (Release DeepSeek V4 protected prompt refs under pressure)` that we suspected was the fix for the Spark 256K×2 stall observed 2026-05-04.

## Hardware and toolchain

| Item | Value |
| --- | --- |
| Host | Brev `violent-azure-yak` (verda) |
| GPUs | 2× NVIDIA RTX PRO 6000 Blackwell Server (SM 12.0, 96 GB × 2 = 192 GB) |
| Driver | 580.126.09 |
| CUDA | 12.8 toolkit, `compute_120a` |
| CPU / RAM | 60-thread x86_64, 176 GB |
| OS | Ubuntu 24.04 (kernel 6.8.0-100) |

vLLM build (commit-stack on `/ephemeral/work/vllm`, branch `ds4-sm120-experimental`):

```
6a0b837da  Fix packed_modules_mapping: gate_up_proj uses w1/w3 not gate_proj/up_proj   ← local
7ffa27bdb  Apply packed_modules_mapping patch (re-applied after empirical confirmation)  ← local
395970dae  support ct quantization                                                        ← cherry-pick neuralmagic/kylesayrs/deepseek-ct@f910a73a
abad5dc71  Add gated split-KV sparse MLA decode                                           ← jasl/vllm ds4-sm120-experimental tip
de3a0ec3b  Add sparse MLA split-KV microbenchmark
b16048b2f  Stabilize DeepSeek V4 MTP draft sampling
c05ef3f3e  Tune SM12x sparse MLA graph defaults
16e681009  Refine DeepSeek V4 DSML tokenizer and parser
6e2fa4d4a  Add DeepSeek V4 SM12x smoke harness
a76b1a9be  Add GB10 fused MoE config aliases                                              ← Spark-relevant
890b010fc  Add RTX PRO 6000 Blackwell config aliases                                      ← directly for this hardware
e734ace5f  Release DeepSeek V4 protected prompt refs under pressure                       ← suspected 256K-stall fix
1d6f5c4eb  Reserve DeepSeek V4 prefill workspace during profiling                         ← replaces local patch_workspace_prereserve.py
a5ce0d7d0  Fix DeepSeek V4 MLA prefix cache reuse
```

Build env: `TORCH_CUDA_ARCH_LIST=12.0a CUDA_ARCH_LIST=120a MAX_JOBS=24 NVCC_THREADS=4`. Result: `vllm 0.1.dev16350+g395970dae` with `ENABLE_SCALED_MM_SM120 / NVFP4_SM120 / CUTLASS_MOE_SM120` flags. ~30 min compile.

## Two integration issues surfaced (worth flagging upstream)

### 1. `packed_modules_mapping` is still required

Despite the kylesayrs cherry-pick `f910a73a` and the experimental-branch reorganization, `DeepseekV4ForCausalLM` still does **not** carry the `packed_modules_mapping` class attribute that `compressed_tensors`'s `should_ignore_layer` / `find_matched_target` need to fan-out fused names to their underlying components. Without the patch, model load fails:

```
ValueError: Unable to find matching target for model.layers.0.ffn.shared_experts.gate_up_proj
in the compressed-tensors config.
```

Note **the recipe ignore list uses the `w1/w2/w3` naming** for shared experts (matching `transformers v5` save layout), so the correct fused-mapping value for `gate_up_proj` is `["w1", "w3"]`, not `["gate_proj", "up_proj"]`. Mapping wired automatically into the quant config via `vllm/model_executor/model_loader/utils.py:configure_quant_config` — no `SupportsQuant` inheritance needed. Ten-line drop-in (still in `patches/packed_modules_mapping.diff`):

```python
class DeepseekV4ForCausalLM(nn.Module, SupportsPP):
    ...
    hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper("fp4")

    packed_modules_mapping = {
        "fused_wqa_wkv": ["wq_a", "wkv"],
        "fused_wkv_wgate": ["wkv", "wgate"],
        "gate_up_proj": ["w1", "w3"],
    }
```

### 2. FlashInfer JIT mis-parses `TORCH_CUDA_ARCH_LIST=12.0a` on Blackwell

vLLM's top-p / top-k sampler routes through FlashInfer's JIT, whose `check_cuda_arch()` reads `TORCH_CUDA_ARCH_LIST` and refuses to compile for SM 12.0 — error message says **"FlashInfer requires GPUs with sm75 or higher"**, which is misleading; the actual GPU is sm_120 (way above sm75), the parser just doesn't recognize the `12.0a` arch token. Workaround: `VLLM_USE_FLASHINFER_SAMPLER=0` (vLLM has a graceful fall-back to PyTorch-native sampler when this env var is set explicitly to `0`). Should be auto-fall-back on Blackwell.

## Long-context probe — 256K×2 concurrent fixed

The headline test: **two concurrent 256K-context streams simultaneously**. This was the configuration that stalled on the Spark TP=2 deployment yesterday (2026-05-04) and was the open question we wanted `e734ace5` to answer.

Serve config: `--max-model-len 524288 --max-num-seqs 2 --gpu-memory-utilization 0.95 --kv-cache-dtype fp8`. KV cache: 1,087,973 tokens (2.08× concurrency at 524K).

| Probe | Prompt tokens | Elapsed | Result |
| --- | --- | --- | --- |
| 75K (line=2400) | 74,457 | 58.5 s | ✅ PASS |
| 128K (line=4000) | 124,057 | 119.9 s | ✅ PASS |
| 256K × 1 (line=8000) | 248,057 | 356.0 s | ✅ PASS |
| **256K × 2-A** (line=8000) | **248,057** | **377.2 s** | ✅ **PASS** ← concurrent |
| **256K × 2-B** (line=8000) | **248,057** | **377.1 s** | ✅ **PASS** ← concurrent |
| 500K × 1 (line=16000) | 496,057 | 1230.7 s (20m31s) | ✅ PASS |

Each probe matches all three sentinel terms in the KV-indexer NIAH test. **Concurrent 256K×2 finishes only ~21 s slower than single 256K×1** → batching is working correctly under memory pressure, no stall, no progression to OOM. With `e734ace5` confirmed effective on Blackwell sm_120, the path is now graphs-on, no `--enforce-eager`, no workspace-prereserve patch (`1d6f5c4` upstream), no special handling at 256K×2.

> Operational note: vLLM's throughput logger appears to suppress output during single very-long prefill chunked-prefill runs — the 500K×1 probe showed *no* server-side throughput log entries for 20 minutes despite GPUs at 98% / 418 W and the request actively progressing. Don't take logger silence as a sign of stall.

## Correctness

Harness HEAD `96785b9` (Spark match) for apples-to-apples vs the Spark numbers in the model card. All chat-smoke / generation-matrix runs use temperature=0.0, top-p=1.0 (deterministic).

| Test | Result on RTX PRO 6000 dual TP=2 |
| --- | --- |
| `chat-smoke quick` (4 cases) | **4 / 4 PASS** |
| `toolcall15` (15 cases × 1 round) | **27 / 30 = 90%** (1 PARTIAL: TC-07 Search-Read-Act, 1 FAIL: TC-06 Multi-Value Extraction) |
| **GSM8K** 8-shot, flexible-extract | **94.99% ±0.60%** |
| **GSM8K** 8-shot, strict-match | **95.07% ±0.60%** |
| **HumanEval** pass@1 (instruct, 0-shot, `--confirm_run_unsafe_code`) | **78.05% ±3.24%** |
| `generation-matrix` non-thinking en (18 prompts) | 18 entries, 31.7K total completion tokens (avg ~1.7K / prompt) |
| `generation-matrix` think-high en (18 × 3 rounds) | 54 entries, 169.6K total completion tokens (avg ~3.1K / entry) |
| `generation-matrix` think-max @ 32K en (18 × 3 rounds) | 54 entries, 274.7K total completion tokens (avg ~5.1K / entry) |

The `generation-matrix` runs do not have automatic pass/fail in this harness HEAD (validation is via manual review of the markdown reports, or via `oracle-compare` against DeepSeek's official API — both deferred for this run). All 126 invocations completed cleanly with `finish_reason=stop`.

### Comparison caveats

- The **8× H200** numbers in the published model card (toolcall15 26/30, GSM8K 92.87%, HumanEval 54.27%) come from harness HEAD `85aca32` and an earlier `jasl/vllm@428e08e`. They are **not on the same vllm rev** as today's RTX PRO 6000 numbers. Treat the H200 ↔ RTX 6000 comparison as informational, not as a "same software, different hardware" benchmark. The valid same-software comparison is **RTX 6000 ↔ Spark**.
- The Spark `toolcall15 41/45 (92%)` is **3 thinking-mode rounds × 15 cases**; today's RTX 6000 `27/30 (90%)` is a **single round**. Different denominators. Same effective level when normalized.

## Performance — `vllm bench serve`

Measured against the running serve, `--max-num-seqs 2 --max-model-len 524288`, ignore-eos, T=0.

| Concurrency | Input / Output | Duration | TTFT mean / p99 | TPOT mean / p99 | Output tok/s |
| --- | --- | --- | --- | --- | --- |
| 1 | 1024 / 1024 | 430.9 s | 237 ms / 711 ms | 20.8 ms / 21.7 ms | 47.5 |
| 2 | 2048 / 512 | 121.9 s | 1096 ms / 1900 ms | 21.7 ms / 23.0 ms | 84.0 |

Per-stream decode is rock-stable at ~47–48 tok/s (p99 TPOT only 22 ms — extremely tight distribution). Concurrent throughput scales 1.77× at concurrency=2. Aggregate input+output throughput at c=2 is 420 tok/s.

## Reproduction

Serve script as run:

```bash
#!/usr/bin/env bash
set -euo pipefail
export PATH="/usr/local/cuda/bin:$HOME/.local/bin:$PATH"
export CUDA_HOME="/usr/local/cuda"
export TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"
export CUDA_ARCH_LIST="120a"
export TORCH_CUDA_ARCH_LIST="12.0a"
unset PYTORCH_CUDA_ALLOC_CONF || true
export VLLM_USE_FLASHINFER_SAMPLER=0           # SM 12.0 / FlashInfer JIT workaround
. /ephemeral/work/.venv/bin/activate
exec vllm serve /local/path/to/DeepSeek-V4-Flash-W4A16-FP8 \
  --served-model-name DSV4-W4A16-FP8 \
  --tensor-parallel-size 2 \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --max-model-len 524288 \
  --max-num-seqs 2 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.95 \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --trust-remote-code \
  --host 0.0.0.0 --port 8000
```

For the 16K canonical recipe (matching the Spark TP=2 production target), drop `--max-model-len` to 16384, `--max-num-seqs` to 4, `--gpu-memory-utilization` to 0.92.
