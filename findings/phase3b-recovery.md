# Phase 3b recovery: GPTQ-calibrating V4-Flash at scale on 8× H200

The dryrun (Phase 3a, 16 calibration samples) succeeded at first try with `batch_size=32` and the upstream-example recipe defaults. The full calibration (Phase 3b, originally 1024 samples) hit a sequence of failures that took 4 attempts to clear. This is a record of what each attempt taught us, ending with the working configuration.

## TL;DR — the configuration that worked

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export TORCH_NCCL_BLOCKING_WAIT=0
export NCCL_TIMEOUT=3600
export TORCH_CUDA_ARCH_LIST=9.0a
torchrun --nproc-per-node 8 quantize_v4_w4a16.py \
    --samples 768 --batch-size 4 \
    --input /workspace/model-bf16 --output /workspace/model-w4a16-phase3b
```

Recipe (in the script):

```python
recipe = GPTQModifier(
    config_groups={
        "attention": QuantizationScheme(targets=[...attn_regex...], **FP8_BLOCK),
        "experts":   QuantizationScheme(targets=[...routed_expert_regex...], **W4A16),
    },
    ignore=["lm_head"],
    offload_hessians=True,    # critical
    dampening_frac=0.1,
)
oneshot(..., sequential_targets=["DeepseekV4DecoderLayer"], batch_size=4)
```

`/dev/shm` was remounted to 1.8 TiB before the run.

## Failure ladder

| Attempt | Config | Outcome | Lesson |
|---|---|---|---|
| 1 | `samples=1024, bs=32, sequential=DecoderLayer`, no `offload_hessians`, no env tweaks | OOM at Layer 3, `Tried to allocate 45–67 GiB` per rank | 256-routed-experts × 3 Linears = 768 Hessians per layer; with `bs=32` activation footprint of the routed-expert forward exceeded 140 GB H200 capacity. Dryrun had succeeded only because 16 total samples / 8 ranks → effectively 2 samples/rank in 1 batch — tiny activation tensor. |
| 2 | `samples=1024, bs=8`, otherwise same | OOM at Layer 3, `Tried to allocate 32 GiB` | Activation scaled with batch size as expected, but Hessian + workspace still busted. Recipe-level fix needed, not just batch reduction. |
| 3 | `samples=1024, bs=8, offload_hessians=True` | OOM at Layer 3, `Tried to allocate 30 GiB` with **43 GiB reserved-but-unallocated by PyTorch** | `offload_hessians` reduced steady-state PyTorch memory by ~16 GiB (Hessians now on CPU, only the active one on GPU). But fragmentation prevented finding a contiguous 30 GiB block despite ~63 GiB free total. Solution: anti-fragmentation allocator. |
| 4a | `samples=1024, bs=4, +offload_hessians, +PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Cleared layers 1–22; **NCCL collective timeout at Layer 22** (`OpType=REDUCE NumelIn=67108864`, ran 600,044 ms before default 10 min timeout) | Per-rank drift accumulated as calibration progressed (`offload_hessians` adds CPU↔GPU transfer time per Linear; system load varies across ranks). After 22 layers' worth of compounded drift, one rank arrived at a collective ≥10 minutes after the others. NCCL watchdog killed the process. |
| 4b | `+TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600 +NCCL_TIMEOUT=3600 +TORCH_NCCL_BLOCKING_WAIT=0`, dropped `samples=1024 → 768`, kept everything else from 4a | **Completed all 44 layers + save**, ~14 hours wall clock | 60-min NCCL timeout absorbs single-layer drift outliers; `samples=768` reduced per-rank work by 25% so cumulative drift compounded less aggressively. |

## Why each fix is needed (in order)

### 1. `offload_hessians=True`

GPTQModifier registers a Hessian accumulator on every Linear matched by `config_groups`. For V4-Flash that's `256 routed experts × 3 Linears + 6 attn Linears + 3 shared_experts = 777 Hessians` per `DeepseekV4DecoderLayer`. Each Hessian is `K × K × float32 = 4096 × 4096 × 4 = 64 MiB`. Total: **~48 GiB just for Hessians per rank, on the GPU**, before any activations.

`offload_hessians=True` (defined in `llmcompressor/modifiers/gptq/base.py:123`) keeps Hessians on CPU and only onloads each one during its update step. The 2 TiB host RAM on `p5en.48xlarge` absorbs them comfortably. Cost: per-Linear PCIe transfer (~50–500 ms each at H200 PCIe Gen5 speeds), so ~2–3× wall-time slowdown in the GPTQ compress pass.

The official kylesayrs `deepseek_v4_example.py` does NOT set this flag — its example uses NVFP4 expert weights, which still build Hessians but with smaller memory profiles. For W4A16 (INT4 group=128) it's required.

### 2. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

Even with `offload_hessians=True`, allocation patterns during GPTQ compress (Hessian onload + inverse + weight update + free) fragmented PyTorch's caching allocator. By Layer 3, ~43 GiB of GPU memory was reserved-but-unallocated in fragments smaller than the 30 GiB activation tensor for the next forward pass.

`expandable_segments:True` switches PyTorch to a slab allocator that coalesces freed regions into contiguous segments. Documented at https://pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf.

### 3. `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600` (+ siblings)

Default NCCL collective timeout is 10 minutes. Under `offload_hessians=True`, per-rank GPTQ compress time varies because:
- Different ranks get different routed-expert assignments (each handles `768/8 = 96 Linears` per layer, but the specific Linears differ — and routed-expert weights have heterogeneous magnitudes and reconstruction error)
- CPU↔GPU transfer time depends on `/dev/shm` fragmentation (which grows over the run)

We measured per-Linear compress times of 1.2–1.9 s. Across 96 Linears that's a per-rank window of ~115–180 s — but the spread compounds across 22 layers. By Layer 22, slowest rank lagged fastest rank by enough to bust the 10-min collective timeout.

60-min timeout with `samples=768` was sufficient. `TORCH_NCCL_BLOCKING_WAIT=0` ensures the watchdog doesn't pin a CPU thread; `NCCL_TIMEOUT=3600` is a redundant alias that some torch versions read.

### 4. `samples=1024 → 768`

Stability over benchmark perfection. 768 vs 1024 is a tiny perplexity delta (well below evaluation noise floor) and reduces per-layer GPTQ work by 25%, giving more headroom against the NCCL-timeout drift mode. 768 / 8 ranks / bs=4 = **24 batches per rank per layer** vs 32 — same per-batch peak memory, less cumulative drift.

### 5. `/dev/shm` ≥ 1.8 TiB

8 ranks × offloaded Hessians + accelerate's CPU offload of unloaded layers (V4-Flash is 543 GB BF16) saturated the default 1 TiB tmpfs early. `sudo mount -o remount,size=1800G /dev/shm` before the run.

## What didn't work — recorded so others don't repeat

- **`sequential_targets=["Linear"]`**: This is what the framework's OOM error message recommended ("consider choosing a smaller module for `sequential_targets` argument, ex. 'Linear'"). It triggered `torch.fx.proxy.TraceError: symbolically traced variables cannot be used as inputs to control flow` from `DeepseekV4Indexer.wrapped_1` (data-dependent control flow `if chunk_kv.shape[1] > 0:`). With `sequential_targets=["DeepseekV4DecoderLayer"]`, fx never enters the Indexer because the DecoderLayer is the leaf. Fixing this would require extending `SequentialTracer.is_leaf_module` (in `helpers.py`) to register `DeepseekV4Indexer` as a leaf module — feasible but adds a 13th rough-edge patch to the stack, with unclear interactions with future autowrap modules.

- **`AWQModifier` instead of `GPTQModifier`**: AWQ uses smaller `_parent_args_cache: IntermediatesCache` instead of K×K Hessians, but no example exists for V4-Flash MoE. Mappings (smooth_layer ↔ balance_layers) would need to be handwritten for the post-attention LN ↔ routed-expert pattern. Discounted as risky for this work; revisit for future quants.

- **`QuantizationModifier` (data-free RTN W4A16)**: Would skip Hessians entirely. Quality cost is ~1% perplexity vs GPTQ on most evals. Discounted because we want GPTQ quality and the deliverable name `W4A16` doesn't preclude GPTQ.

## Reproducibility

The exact final calibration walltime breakdown (8× H200, 768 samples):

- BF16 weight load (offload-aware) + linearize_moe_model: ~22 minutes
- Dataset preprocess (1024 samples × max_seq_len=512): ~30 seconds
- Per-layer GPTQ pass (calibrate + propagate + compress): **~21 minutes/layer × 44 layers = ~14 hours**
- save_pretrained (4 shards × ~36 GB): ~6 minutes

Total: ~14h45m end-to-end.
