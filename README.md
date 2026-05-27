# dsv4-flash-w4a16-fp8

Reproduction repo for [`canada-quant/DeepSeek-V4-Flash-W4A16-FP8`](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-W4A16-FP8) — W4A16 INT4 routed experts + FP8 block 128×128 attention on DeepSeek-V4-Flash, vLLM-deployable at TP=2 on Hopper SM 9.0a (H200), Blackwell SM 12.1a (DGX Spark GB10), and Blackwell SM 12.0 (RTX PRO 6000).

> **Status (2026-05-26):** Production canonical on dual DGX Spark TP=2 graphs-ON at 1M-token context (image `vllm-w4a16-dsv4:baseline-428e08e` from `jasl/vllm@428e08e`). Bootstrap script is now version-pinned to `428e08e` after F004 — newer jasl SHAs (post 2026-05-19 model refactor) break the vendored `kylesayrs-deepseek-ct.patch`. Comprehensive 2× DGX Spark bench landed: **GSM8K 8-shot 94-98%**, AIME-2024 think-max **4/5 correct with 0 truncations** at max_tokens=65536 (longest reasoning 14,725 tokens), **concurrency aggregate scales 10× from seq=1 (12 tok/s) → seq=16 (125 tok/s)** at 16-128K ctx, peak 189.7 tok/s aggregate at 4K×32. **OOM ceiling: 1M × seq=1 fits; 1.5M and 256K × 2 don't.** RTX PRO 6000 Blackwell SM 12.0 validation also stands — chat-smoke 4/4, toolcall15 27/30 (90%), GSM8K 95.07%, NIAH 256K × 2 concurrent PASS. Workspace pre-reservation patch landed upstream as `jasl/vllm@1d6f5c4`.
>
> **Update (2026-05-25):** Two shipping issues surfaced when testing on *current* upstream vLLM (`jasl/vllm@a02a3778f`, post-PR-#40923 build that the MTP sibling now uses): **(1)** Same FP8 compressor/indexer load-failure as the [W4A16-MTP sibling](https://github.com/canada-quant/dsv4-flash-w4a16-fp8-mtp) had — fixable via the same in-artifact BF16 dequant (`scripts/dequant_compressor.py` in the MTP sibling repo); **not yet applied to this artifact**. **(2)** Architecture-drift `KeyError: 'layers.N.ffn.gate.e_score_correction_bias'` — older safetensors (calibrated 2026-05-06) don't contain a tensor that current vLLM's DSV4 loader requires; needs re-calibration or a defensive `.get()` loader patch upstream.
>
> **Update (2026-05-26a):** Investigated the same `.weight_scale` naming class of bug that the MTP sibling has. Card A's safetensors use the same `.weight_scale` naming (no `_inv`), 33,405 keys total (vs MTP sibling's 33,239). Naming verdict: **same as Card D** — the same vLLM upstream fix at [`vllm-project/vllm#43564`](https://github.com/vllm-project/vllm/issues/43564) (which we extended today with a deeper kernel-dispatch finding) will unblock both. A two-rename script (`.weight_scale` → `.weight_scale_inv` for FP8 layers + `ffn.gate.e_score_correction_bias` → `ffn.gate.bias` for architecture drift) sketched but **not pushed** since the deeper kernel-dispatch issue would still block runtime. Published RTX PRO 6000 numbers above remain valid for the historical `jasl/vllm@ds4-sm120-experimental@abad5dc71` build (2026-05-05); they do not reproduce on bleeding-edge vLLM without those upstream fixes. See the [W4A16-MTP sibling's findings docs](https://github.com/canada-quant/dsv4-flash-w4a16-fp8-mtp/tree/main/docs/findings) for the full investigation.

## vLLM patch series — [`vllm-patches/`](vllm-patches/)

Same patch series as the W4A16-MTP / NVFP4-MTP siblings — the minimum series we apply on top of `jasl/vllm@a02a3778f` for SM 12.0:

| Patch | Purpose | Upstream |
|---|---|---|
| `0001_marlin_moe_archs_40923.patch` | Native sm_120a Marlin MoE cubins | [PR #40923](https://github.com/vllm-project/vllm/pull/40923) (open) |
| `0002_marlin_moe_workspace_4x.patch` | Marlin MoE lock-array workspace 4× | (to file) |
| `0003_marlin_moe_c_tmp_36889.patch` | Drop `min()` clamp on c_tmp FP32 reduce buffer | [PR #36889](https://github.com/vllm-project/vllm/pull/36889) (closed; re-file candidate) |

These patches address the SM 12.0 W4A16 Marlin MoE concurrent-decode bug seen on the MTP sibling. Card A doesn't currently reproduce that bug since its loadability is blocked first (see Update 2026-05-25/26a above), but the patch series is identical so the install template at [`scripts/install_rtx6000pro.sh`](scripts/install_rtx6000pro.sh) sets the same vLLM build state. Card A patches will become functionally testable once the upstream issues at [`#43564`](https://github.com/vllm-project/vllm/issues/43564) are resolved.

> **Prereqs called out by the 2026-05-26b dual-spark OOB run** (all documented in [`findings/dual_spark_jasl_sha_regression_2026-05-26/`](findings/dual_spark_jasl_sha_regression_2026-05-26/)): (1) **uniform NVIDIA driver across the fleet** — heterogeneous 580.142 / 590.x cost ~20% throughput via NCCL slow-path fallback (F006); pin to a single Canonical-prebuilt-signed driver (we use `nvidia-driver-580-open=580.142-0ubuntu0.24.04.1`). Newer 595.71.05 needs MOK enrollment via firmware UI (F007). (2) **Passwordless sudo for `/usr/{bin,sbin}/ip`** on each node (F003); the bootstrap step 3 requires this. (3) PEP 668 + huggingface-cli legacy-removal workarounds are now inlined into the bootstrap (F001/F002 fixed in script).
>
> **Previously-flagged 2026-05-25 shipping issues** at the bench-pinned `428e08e` SHA: non-issues (`428e08e` is pre-refactor, before the FP8 compressor breakage landed in jasl HEAD).

## Family / related repos

| Repo | HF model card | Role |
|---|---|---|
| **this repo** (`dsv4-flash-w4a16-fp8`) | [W4A16-FP8](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-W4A16-FP8) | base recipe — no MTP — broadest hardware compatibility |
| [`canada-quant/dsv4-flash-w4a16-fp8-mtp`](https://github.com/canada-quant/dsv4-flash-w4a16-fp8-mtp) | [W4A16-FP8-MTP](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP) | successor — same W4A16 recipe + BF16 MTP retained — 1.49× spec-decode speedup at bs=1 |
| [`canada-quant/dsv4-flash-nvfp4-fp8-mtp`](https://github.com/canada-quant/dsv4-flash-nvfp4-fp8-mtp) | [NVFP4-FP8-MTP](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP) | sibling — NVFP4 routed experts (Blackwell-native) + MTP |
| [`canada-quant/dsv4-pro-nvfp4-fp8-mtp`](https://github.com/canada-quant/dsv4-pro-nvfp4-fp8-mtp) | [Pro NVFP4-FP8-MTP](https://huggingface.co/canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP) | larger sibling — V4-Pro NVFP4 + MTP, B300-only |

## Quickstart — dual DGX Spark TP=2

Single-file zero-to-serving:

```bash
curl -fsSLO https://raw.githubusercontent.com/canada-quant/dsv4-flash-w4a16-fp8/main/scripts/bootstrap_dsv4_spark.sh
chmod +x bootstrap_dsv4_spark.sh
./bootstrap_dsv4_spark.sh --head-host spark-a --worker-host spark-b
```

Idempotent — SSH-reachability check, model pre-cache, QSFP /30 setup, image build via `eugr/spark-vllm-docker` + our DSV4 patches (Dockerfile + kylesayrs patch + packed_modules patch curled into the build context), scp-distribute to worker, container launch on both nodes (head rank 0 + worker rank 1 `--headless`), waits for `/health=200`. ~30–50 min first run (mostly docker build), ~7 min on re-runs with `--skip-build`. On success writes `/workspace/build-metadata.yaml` to `/tmp/dsv4-spark-build-metadata-*.yaml` for bug reports; on failure dumps last-300-line container logs, env, `nvidia-smi`, and `dmesg` tail from **both nodes**.

Walk-through: [`findings/QUICKSTART_DUAL_SPARK.md`](findings/QUICKSTART_DUAL_SPARK.md). Manual build: [`scripts/Dockerfile.dsv4-spark`](scripts/Dockerfile.dsv4-spark) + [`scripts/kylesayrs-deepseek-ct.patch`](scripts/kylesayrs-deepseek-ct.patch) + [`scripts/patch_v4_packed_mapping.py`](scripts/patch_v4_packed_mapping.py).

## Why this exists

DeepSeek-V4-Flash launched April 24, 2026 (284 B total / 13 B active MoE, hybrid CSA + HCA attention, hash-routed experts). At release, no merged path through transformers + llm-compressor + vLLM existed for V4 quantization on Hopper or SM 12.x Blackwell:

- [`RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8`](https://huggingface.co/RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8) uses NVFP4 expert weights requiring SM 10.0+ tcgen05 instructions — unavailable on Hopper SM 9.0a and DGX Spark SM 12.1a.
- [`Intel/DeepSeek-V4-Flash-W4A16-AutoRound`](https://huggingface.co/Intel/DeepSeek-V4-Flash-W4A16-AutoRound) covered W4A16 but their card explicitly states *"vLLM and SGLang is not supported currently."*

This project produces a W4A16 GPTQ V4-Flash that **serves in vLLM at TP=2 on H200 (Hopper SM 9.0a), DGX Spark GB10 (Blackwell SM 12.1a), and RTX PRO 6000 (Blackwell SM 12.0)**, with attention quantized to FP8_BLOCK (mirroring RedHat's recipe topology, swapping NVFP4 → W4A16 for SM 9.x / 12.x compatibility).

## What's in this repo

| Path | What |
|---|---|
| **[`scripts/bootstrap_dsv4_spark.sh`](scripts/bootstrap_dsv4_spark.sh)** | **Single-file zero-to-serving script for dual DGX Spark TP=2.** SSH-orchestrated, idempotent, handles every step from network setup through engine boot. |
| [`scripts/Dockerfile.dsv4-spark`](scripts/Dockerfile.dsv4-spark) | Production Dockerfile — `jasl/vllm@${VLLM_REF}` + vendored kylesayrs deepseek-ct patch + `packed_modules_mapping` patch. Drop into an `eugr/spark-vllm-docker` checkout. |
| [`scripts/kylesayrs-deepseek-ct.patch`](scripts/kylesayrs-deepseek-ct.patch) | Vendored vLLM patch (kylesayrs PR #41276 work, pre-rebased onto `jasl/vllm@ds4-sm120`). Pinned by content, not SHA — see [`findings/kylesayrs-pr-41276-integration.md`](findings/kylesayrs-pr-41276-integration.md). |
| [`scripts/patch_v4_packed_mapping.py`](scripts/patch_v4_packed_mapping.py) | Local patch — adds `packed_modules_mapping` to `DeepseekV4ForCausalLM`. Still required (kylesayrs PR references but doesn't define it). |
| [`scripts/serve_spark_tp2.sh`](scripts/serve_spark_tp2.sh) | Standalone per-rank launch helper for manual operation. |
| [`findings/QUICKSTART_DUAL_SPARK.md`](findings/QUICKSTART_DUAL_SPARK.md) | Operator-facing manual quickstart with per-flag explanation. |
| [`findings/spark_tp2_deployment.md`](findings/spark_tp2_deployment.md) | Full DGX Spark validation report. |
| [`findings/rtxpro6000_blackwell_deployment.md`](findings/rtxpro6000_blackwell_deployment.md) | RTX PRO 6000 Blackwell validation. |
| [`findings/upstream-issue-marlin-tp-sharding.md`](findings/upstream-issue-marlin-tp-sharding.md) | Root-cause + filed bug for Marlin MoE TP scale-sharding ([`vllm-project/vllm#41511`](https://github.com/vllm-project/vllm/issues/41511)) — blocks W4A16 MoE under TP > 2. |
| [`findings/kylesayrs-pr-41276-integration.md`](findings/kylesayrs-pr-41276-integration.md) | Integration notes for kylesayrs's V4 attention path PR — 5 documented gaps with our patches. |
| [`findings/phase3b-recovery.md`](findings/phase3b-recovery.md) | H200 OOM + NCCL-timeout journey for the GPTQ calibration: which env vars, why each. |
| [`patches/`](patches/) | Static patches against upstream — calibration patches + `packed_modules_mapping.diff` for vLLM serving. |
| `REPORT.md` | Full mission log (setup → native baseline → dequant → calibration → serve attempts → harness results). |

## Headline validation

Same artifact, three SKUs:

| Test | 8× H200 TP=2 (older vLLM) | 2× DGX Spark TP=2 (current) | 2× RTX PRO 6000 TP=2 |
|---|---|---|---|
| `chat-smoke quick / quality / coding` | 4/4 · 4/4 · 2/2 | 4/4 · 4/4 · 2/2 | 4/4 · 4/4 · 2/2 |
| `toolcall15` | 26/30 (87%) | 41/45 (92%) | 27/30 (90%) |
| GSM8K 8-shot strict-match | see HF card "Changes" | 95.45% ± 0.57 | 95.07% ± 0.60 |
| HumanEval pass@1 (`--confirm_run_unsafe_code`) | 80.49% (corrected) | 80.49% ± 3.10 | 78.05% ± 3.24 |
| NIAH 75K → 500K single | — | 4/4 | 5/5 |
| NIAH 256K × 2 concurrent | — | fix in `jasl@e734ace5` | 4/4 (377 s) |
| Decode tok/s @ bs=1 (1024-in/1024-out) | — | 14–17 | 47.5 (TPOT 20.8 ms) |

Full table + methodology footnotes on the [HF model card](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-W4A16-FP8).

## Build for AWS calibration

Calibration runs against the BF16-dequantized base on 8× H200 with `kylesayrs/transformers-v5` llm-compressor branch.

```bash
# Clean venv (do NOT share the vLLM serve venv — pip cascades break vLLM's torch+cu13 pin)
pip install git+https://github.com/huggingface/transformers.git@add-deepseek-v4
pip install git+https://github.com/vllm-project/llm-compressor.git@kylesayrs/transformers-v5
pip install --pre 'compressed-tensors>=0.15.1a2'

# Calibration-time patches
patch -p1 -d "$(python -c 'import llmcompressor; print(llmcompressor.__path__[0])')" < patches/helpers.py.diff
patch -p1 -d "$(python -c 'import transformers; print(transformers.__path__[0])')" < patches/modeling_deepseek_v4.py.diff

# /dev/shm ≥ 1.8 TiB for 8-rank torchrun on a 543 GB BF16 model
sudo mount -o remount,size=1800G /dev/shm

# FP8_BLOCK attn + W4A16 GPTQ routed experts
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_TIMEOUT=3600
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_CUDA_ARCH_LIST=9.0a
torchrun --nproc-per-node 8 scripts/quantize_v4_w4a16.py \
    --samples 768 --batch-size 4 \
    --input  /path/to/DeepSeek-V4-Flash-bf16 \
    --output /path/to/DeepSeek-V4-Flash-W4A16-FP8
```

See [`findings/phase3b-recovery.md`](findings/phase3b-recovery.md) for **why** each env var is required.

## Build for vLLM serving

For dual DGX Spark TP=2 the bootstrap script above handles everything. Manual build for any other TP=2 hardware:

Required pieces (stacked):
- **`jasl/vllm@ds4-sm120-experimental`** — current branch with SM12x DSV4 + experimental superset (split-KV decode, GB10 fused-MoE config aliases, tuned MLA graph defaults). Use `ds4-sm120` instead for conservative PR-tracked branch.
- **Apply** [`scripts/kylesayrs-deepseek-ct.patch`](scripts/kylesayrs-deepseek-ct.patch) — vendored from `neuralmagic/vllm@kylesayrs/deepseek-ct` commit `d09eeb498` (rebased successor of `f910a73a93` which was force-pushed out of history — see [issue #1](https://github.com/canada-quant/dsv4-flash-w4a16-fp8/issues/1)).
- **Patch** [`scripts/patch_v4_packed_mapping.py`](scripts/patch_v4_packed_mapping.py) — adds `packed_modules_mapping` to `DeepseekV4ForCausalLM`.
- Workspace pre-reservation patch is **no longer needed** — landed upstream as `jasl/vllm@1d6f5c4`.

```bash
git clone https://github.com/jasl/vllm.git -b ds4-sm120-experimental vllm
cd vllm
git apply --check ../scripts/kylesayrs-deepseek-ct.patch
git am --keep-cr ../scripts/kylesayrs-deepseek-ct.patch
python3 ../scripts/patch_v4_packed_mapping.py vllm/model_executor/models/deepseek_v4.py
pip install -e . --no-build-isolation
```

Production canonical (1M context graphs-ON, single-stream):

```bash
vllm serve canada-quant/DeepSeek-V4-Flash-W4A16-FP8 \
    --served-model-name DSV4-W4A16-FP8 deepseek-ai/DeepSeek-V4-Flash deepseek-v4-flash \
    --tensor-parallel-size 2 \
    --kv-cache-dtype fp8 --block-size 256 \
    --max-model-len 1048576 \
    --max-num-seqs 1 --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.90 \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
    --trust-remote-code
```

**Three flag/env gotchas worth flagging:**
- `--served-model-name` takes multiple values per single flag, **not** repeated flags. `--served-model-name A --served-model-name B` silently keeps only the last value. Use space-separated form.
- `--gpu-memory-utilization=0.90` (not 0.92) on the experimental build — prefix-cache + split-KV reservations push past 0.92 on first boot.
- `VLLM_TRITON_MLA_SPARSE_HEAD_BLOCK_SIZE=4` is **mandatory at runtime** — without it sparse-MLA warmup crashes with `RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered` in `_dequantize_and_gather_k_kernel`. See [`findings/QUICKSTART_DUAL_SPARK.md §4`](findings/QUICKSTART_DUAL_SPARK.md#4-launch--head--worker) for the full env block.

**TP limit:** TP=1 OOMs on a single 141 GB H200. TP=2 works. TP ≥ 4 hits the upstream Marlin MoE TP scale-sharding bug ([`vllm-project/vllm#41511`](https://github.com/vllm-project/vllm/issues/41511)).

## Upstream contributions

| PR / Issue | Description | Status |
|---|---|---|
| [`vllm-project/vllm#41700`](https://github.com/vllm-project/vllm/issues/41700) | Workspace pre-reservation patch | **landed** as `jasl/vllm@1d6f5c4` |
| [`vllm-project/vllm#41511`](https://github.com/vllm-project/vllm/issues/41511) | Marlin MoE TP scale-sharding | open — blocks TP > 2 |
| [`vllm-project/vllm#40991`](https://github.com/vllm-project/vllm/pull/40991) → [`#41834`](https://github.com/vllm-project/vllm/pull/41834) | SM12x DSV4 base support + tuned autotune configs | open (jasl); our validation [comment](https://github.com/vllm-project/vllm/pull/41834#issuecomment-4550181916) 2026-05-26 |
| [`vllm-project/vllm#41276`](https://github.com/vllm-project/vllm/pull/41276) | compressed-tensors V4 attention path | open (kylesayrs) |
| [`vllm-project/vllm#43722`](https://github.com/vllm-project/vllm/pull/43722) | `MarlinFP8.can_implement` refuses block-FP8 layers — fixes load on SM 12.0 RTX PRO 6000 with the freshly-pulled artifact | **open, filed 2026-05-26** |
| [`vllm-project/vllm#43723`](https://github.com/vllm-project/vllm/pull/43723) | DSv4 `attention.py` `wo_a.weight_scale_inv` getattr fallback (companion to #43722) | **open, filed 2026-05-26** |
| [`vllm-project/vllm#43564`](https://github.com/vllm-project/vllm/issues/43564) | Parent tracker for SM 12.0 DSv4 quant scheme dispatch — see [Phase A summary](https://github.com/vllm-project/vllm/issues/43564#issuecomment-4550184475) for the working three-patch-set | open |

## Credits

- [@jasl](https://github.com/jasl) — DeepSeek-V4 vLLM SM12x base support; `e734ace5` memory-pressure-release fix that resolved the Blackwell 256K×2 stall.
- [@kylesayrs](https://github.com/kylesayrs) — compressed-tensors V4 attention path (PR [`#41276`](https://github.com/vllm-project/vllm/pull/41276)).
- [@aabbccddwasd](https://github.com/aabbccddwasd) — indexer KV cache layout fix.
- [@bbbearxyz](https://github.com/bbbearxyz) — SM12x Triton fallback kernels.
- [@wuwenthink](https://github.com/wuwenthink) — SM12x harness validation.
- [`RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8`](https://huggingface.co/RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8) — published reference for V4 mixed-precision attention topology.

## License

MIT, inherited from upstream `deepseek-ai/DeepSeek-V4-Flash`.
