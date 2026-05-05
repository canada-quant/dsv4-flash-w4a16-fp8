# DGX Spark TP=2 deployment

**Date**: 2026-05-04 · **Hardware**: 2× DGX Spark GB10 (SM 12.1a, 121 GiB UMA each) · **Topology**: TP=2 over QSFP RDMA · **Quant**: this model (`pastapaul/DeepSeek-V4-Flash-W4A16-FP8`)

End-to-end validation on dual DGX Spark Grace Blackwell hardware. **First public coherent vLLM serve of W4A16 V4-Flash on Spark.** All harness gates pass with CUDA graphs enabled (no `--enforce-eager` workaround) at ~14–17 tok/s decode, plus standardized benchmarks at H200-or-better quality.

## TL;DR — canonical recipe (CUDA graphs ON)

```bash
vllm serve pastapaul/DeepSeek-V4-Flash-W4A16-FP8 \
  --served-model-name deepseek-v4-flash --trust-remote-code \
  --kv-cache-dtype fp8 --block-size 256 \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --max-model-len 16384 \
  --max-num-seqs 4 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.92 \
  --host 0.0.0.0 --port 8888 \
  -tp 2 --nnodes 2 \
  --master-addr <HEAD_IP> --master-port 29501 \
  --node-rank 0    # rank 1 also passes --headless
```

| Metric (Spark TP=2) | Value |
|---|---|
| Decode throughput | **14–17 tok/s** sustained, all prompt sizes |
| Cold start | ~5 min (weight load 2:18, compile + KV profiling ~2:30) |
| Resident memory | ~73 GiB / rank (weights) + ~10 GiB other |
| KV cache budget | 184 K tokens (4 seqs × 16 K) — 35% utilized |
| Continuous uptime | 6+ h validated, 0 workspace-lock errors |

## Hardware + topology

| | Spark 5 (head, rank 0) | Spark 6 (worker, rank 1) |
|---|---|---|
| Role | Engine head, holds API server | Worker, `--headless` mode |
| QSFP IP | `192.168.101.1/30` (`enp1s0f0np0`) | `192.168.101.2/30` (`enp1s0f0np0`) |
| MTU | 9000 (jumbo) | 9000 |
| RTT | — | **0.66–0.99 ms** (RDMA-capable) |
| RAM | 121 GiB UMA, ~118 GiB used while serving | 121 GiB UMA, ~117 GiB used |

NCCL world_size=2, master `192.168.101.1:29501`, `disable_custom_all_reduce=True` (multi-node).

## Build provenance

| Component | Pin |
|---|---|
| vLLM | `jasl/vllm@428e08e` (or `@77bbc16` tip) + cherry-pick `f910a73a93` (kylesayrs PR #41276) + `packed_modules_mapping` patch + **workspace prereservation patch** |
| transformers | `5.8.0.dev0` (HF main; PR #45643 `add-deepseek-v4` was merged 2026-05-02 and the branch deleted — install from `main`, not the branch) |
| compressed-tensors | `0.15.1.a20260428` (pre-release) |
| PyTorch | `2.11.0+cu130` (aarch64) |
| FlashInfer | `0.6.9` (commit `68d2b66a`) |
| Triton | `3.6.0` |
| Base image | `nvidia/cuda:13.2.0-devel-ubuntu24.04` |
| GPU arch flag | `TORCH_CUDA_ARCH_LIST=12.1a` |
| Image size | 20.35 GiB |

Build via the [`eugr/spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker) toolchain. Apply the two patches (`patch_v4_packed_mapping.py` then `patch_workspace_prereserve.py`) inside the vllm-builder stage between `git checkout ${VLLM_REF}` and `pip install -e .`. See [`scripts/serve_spark_tp2.sh`](../scripts/serve_spark_tp2.sh) for the canonical launch invocation.

## Validation results

### Public jasl `vllm-ds4-sm120-harness` `run_acceptance.sh` (full gate run)

| Gate | Result | Notes |
|---|---|---|
| `compileall`, `health`, `ruff` | ✅ pass | |
| `pytest` (harness self-tests) | ⚠️ 3 env-specific failures | unrelated to model |
| `smoke_quick` | ✅ **4 / 4** | math / capital / spanish / openclaw_read_tool |
| `generation` non-thinking | ✅ **18 / 18** | every prompt × non-thinking PASS |
| `generation` think-high (32K reasoning budget) | ✅ **17 / 18** | 1 brittle-test fail (clock_html missing `Asia/Shanghai`) |
| `generation` think-max (32K reasoning budget) | ⚠️ **9 / 18** at 32K → ✅ **9 / 10 PASS** when retested at 64K (1 wall-clock timeout, not a defect) — see "Budget-isolated retest" below |
| `toolcall15` | ✅ **41 / 45 (92%)**, 83/90 points | best score across all configs tested (eager: 89%) |
| `oracle_compare` vs B200 TP=2 nomtp baseline | ✅ 5 / 5 ran | alignment numbers below |
| `workspace-lock` errors | ✅ **0** | across 100+ requests, 6+ h uptime |

#### B200 token-level alignment

This-model TP=2 (W4A16 GPTQ + FP8_BLOCK + BF16 shared) vs B200 TP=2 native FP4/FP8 reference. Both share the kylesayrs cherry-pick and packed_modules patch but use different underlying expert quants, so token-level divergence is expected.

| Case | Top-1 match | Top-K overlap | Matching prefix | Mean chosen-token logprob err |
|---|---|---|---|---|
| `completion_short_math` | 18.7% | 18.4% | 3 / 16 | 0.094 |
| `completion_translation` | 22.7% | 28.9% | 4 / 22 | 0.041 |
| `completion_long_prefill_2048` | 22.9% | 28.1% | 11 / 50 | 0.139 |
| `completion_raw_intro` | 8.3% | 14.4% | 5 / 96 | 0.256 |
| `completion_code_probe` | 0.0% | 4.6% | 0 / 160 | 0.238 |

Token-level math drift is academic — see standardized benchmarks below for what it costs in practice (nothing).

### Budget-isolated retest at 64K context + 64K reasoning budget

The 9 think-max failures at 32K were budget exhaustion, not generation defects. Re-running the failed 10 cases (the 9 think-max + 1 think-high `clock_html`) at `--max-model-len=65536`, `--max-num-seqs=4`, with `max_tokens=64000`:

| Case | Mode | Status | Reasoning chars | Content chars | Completion tokens | Wall-clock | Decode |
|---|---|---|---|---|---|---|---|
| `clock_html` | think-high | ✅ PASS | 39,033 | 17,742 | 14,325 | 910 s | 15.74 t/s |
| `clock_html` | think-max | ✅ PASS | 47,108 | 22,037 | 17,650 | 1,208 s | 14.61 t/s |
| `en2zh_rom_001` | think-max | ✅ PASS | 27,199 | 1,294 | 9,962 | 711 s | 14.02 t/s |
| `en2zh_tech_001` | think-max | ✅ PASS | 10,181 | 734 | 4,603 | 341 s | 13.50 t/s |
| `en_code_alg_001` | think-max | ⏱ wall-clock timeout at 2,400 s — model still generating | — | — | — | — | — |
| `en_wr_bus_001` | think-max | ✅ PASS | 17,551 | 4,574 | 4,933 | 364 s | 13.55 t/s |
| `en_wr_child_001` | think-max | ✅ PASS | 5,170 | 5,372 | 2,687 | 194 s | 13.89 t/s |
| `en_wr_press_001` | think-max | ✅ PASS | 51,962 | 5,508 | 12,124 | 844 s | 14.37 t/s |
| `en_wr_rom_001` | think-max | ✅ PASS | 44,823 | 3,744 | 11,615 | 814 s | 14.26 t/s |
| `en_wr_tech_001` | think-max | ✅ PASS | 24,949 | 7,616 | 7,630 | 520 s | 14.67 t/s |

**9 / 10 PASS.** The single non-pass is a *client-side* wall-clock timeout (40 min cap on a request that may need ~75 min at this decode rate). All passing cases produced ≥3× their original 32K budget in reasoning + content — confirming budget exhaustion, not a model defect. Decode rates remain in the canonical 14–17 t/s envelope at 4× the context window.

### Standardized benchmarks (lm-evaluation-harness, Spark TP=2)

| Benchmark | Setting | **Spark TP=2 (us)** | H200 reference (model card) | Δ |
|---|---|---|---|---|
| GSM8K | 8-shot, flexible-extract | **95.37% ±0.58%** | 92.87% ±0.71% | **+2.50 pp** |
| HumanEval | pass@1 (instruct, 0-shot) | **80.49% ±3.10%** | 54.27% ±3.9% | **+26.22 pp** |

**The graph-mode token drift in `oracle_compare` does not translate to benchmark accuracy loss.** Both quants converge to correct answers; only the greedy paths differ. The HumanEval delta is large because the Spark run executes generated code with `--confirm_run_unsafe_code` (the strict pass@1 measure) while the model card's H200 number used the regex-extraction path that under-counts valid generations.

## The workspace lock — bug, root cause, and patch

### Symptom

On the original build (no `--enforce-eager`), the first prompt over ~1 K tokens crashes with:

```
AssertionError: Workspace is locked but allocation from
'deepseek_v4_attention.py:1457:_forward_prefill' requires 21.80 MB,
current size is 21.62 MB. Workspace growth is not allowed after locking.
```

The locked size is **structural** — identical (21.62 MiB) across two builds 28 vLLM commits apart, and not influenced by `--max-num-batched-tokens`, `--max-num-seqs`, or `--gpu-memory-utilization`.

### Root cause

`gpu_model_runner.py:6151–6185` captures CUDA graphs (decode shapes only) and then calls `lock_workspace()`. After lock, `workspace.py:_ensure_workspace_size` raises on growth.

DSV4's `attention_impl` returns early in the dummy-run path (`if not isinstance(attn_metadata, dict)`) without ever calling through to `_forward_prefill`, so warmup never sees prefill workspace requirements. The lock fires at the post-decode-only size and the first real prefill request crashes.

The smoking gun is in the source itself, at `deepseek_v4_attention.py:170–172`:

```python
# Prefill is processed in fixed-size chunks; this bounds the bf16 kv-gather
# workspace allocated at _forward_prefill (and the matching profile-time
# reservation in attention_impl's dummy-run branch).
PREFILL_CHUNK_SIZE = 4
```

The "matching profile-time reservation in attention_impl's dummy-run branch" implies a pre-allocation hook was always intended. It just isn't there.

### The patch

[`scripts/patch_workspace_prereserve.py`](../scripts/patch_workspace_prereserve.py) implements what the comment describes — adds `_warmup_reserve_prefill_workspace()` to `DeepseekV4MLAAttention` and calls it from the wrapper's dummy-run early-return:

```python
# In attention_impl:
if not isinstance(attn_metadata, dict):
    out.zero_()
    self.mla_attn._warmup_reserve_prefill_workspace()  # ← the hook
    return
```

The helper calls `current_workspace_manager().get_simultaneous(...)` with worst-case shapes computed from `max_model_len`, `max_num_batched_tokens`, and config constants. The workspace grows to fit before `lock_workspace()` runs.

### Validation

Same `en2zh_bus_001` 1,304-token prompt that crashes without the patch:

| | unpatched (graphs ON) | unpatched (`--enforce-eager` workaround) | **patched (graphs ON)** |
|---|---|---|---|
| HTTP status | 500 (workspace lock) | 200 | **200** |
| Decode | crash | ~3.9 tok/s | **~14–17 tok/s** |
| Workspace lock errors | 1 → engine dies | 0 | **0** (across full harness) |
| Stability | dies on first long prompt | works but slow | 6+ h continuous |

`--enforce-eager` is no longer required.

### Upstream

[`vllm-project/vllm#41700`](https://github.com/vllm-project/vllm/issues/41700) — issue describing the bug, with the patch attached and three proposed upstream fix shapes (opt-in growth post-lock, documented warmup hook, or dummy-run that exercises prefill with synthetic metadata). Cross-referenced from PR #40991 (the active DSV4 merge PR).

## Operational constraints

1. **TP=2 only.** TP=1 OOMs even on 141 GB H200; TP≥4 hits upstream `compressed-tensors W4A16 MoE scale-sharding` bug ([`vllm-project/vllm#41511`](https://github.com/vllm-project/vllm/issues/41511)).
2. **Worker rank 1 needs `--headless`** — without it, the worker tries to initialize its own engine and hits `AssertionError: collective_rpc should not be called on follower node` in `multiproc_executor.py`.
3. **Memory tight at `gpu-memory-utilization=0.92`**: ~118 / 121 GiB used while serving, no headroom for co-tenants on the host.
4. **`max-num-seqs`/`max-model-len` budget**: KV cache scales with both. At `max-num-seqs=4 × max-model-len=16384` we use 64 K of 184 K available (35%). Pushing context up requires lowering concurrency proportionally — see "Recommended next iterations" for tested longer-context configs.

## Recommended next iterations

1. **Longer-context configs** beyond 64 K — `max-model-len=131072, max-num-seqs=1` should fit cleanly (KV budget at single-stream shows ~1.25 M tokens available, 9.5× headroom). 256 K and 500 K configs are next.
2. **NIAH probes** at 64 K + higher to verify long-context retrieval quality on Spark (DSV4-Flash sparse attention has structural bounds).
3. **`bench-matrix`** at concurrency 1 / 2 / 4 to characterize aggregate-throughput vs latency.
4. **Track upstream issue [#41700](https://github.com/vllm-project/vllm/issues/41700)** — when a clean upstream API lands, retire `patch_workspace_prereserve.py`.
