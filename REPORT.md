# DeepSeek-V4-Flash AWQ-W4A16 Quantization — Mission Report

**Operator:** Paul Cozzolino (Sunday Market Media)
**Instance:** i-08141f6ceeb17e888 (p5en.48xlarge, us-west-2d, 8× H200)
**Reservation expires:** **2026-05-03 11:30 UTC** (extension `cbe-0118e77d18de359b7` payment-succeeded; original end was 2026-05-02 11:30 UTC).
**Mission start (UTC):** 2026-05-01T14:46:19Z

## Storage decision
**Workspace lives on ephemeral NVMe.** `/workspace` is a symlink to `/opt/dlami/nvme/workspace`, which sits on a 28 TB LVM volume across 8× 3.5 TB local NVMe drives (DLAMI pre-mount).

> **WARNING (ephemeral):** anything not uploaded to S3 or HuggingFace before instance termination is LOST. Operator artifacts also scp'd back to local `~/work/h200-quant/` after each phase.

EBS root (`/`) has 470 GB free and is reserved for OS / user home only.

## Pre-flight
- 8× H200 (143 GB each), SM 9.0
- PyTorch 2.10.0+cu130, CUDA 13.0, driver 580.126.16
- Memory 2.0 TiB
- Upstream resources confirmed reachable:
  - jasl/vllm@ds4-sm120 = `428e08ec`
  - jasl/vllm-ds4-sm120-harness HEAD = `85aca32`
  - flagos-ai/DeepSeek-V4-FlagOS HEAD = `f9846dc`
  - deepseek-ai/DeepSeek-V4-Flash HF page = HTTP 200

## PR #40991 notes
- New SM12x Triton kernels added; SM90 (Hopper) path preserved (existing FlashMLA).
- PR header lists PyTorch 2.11+ as required, but SM90-only build should work on 2.10. Will bump only if build fails.
- Caveat (per PR): reasoning-enabled tasks may exhaust token budget on coding tests. Matches wuwenthink finding.

## Phase plan
- Phase 0: Setup (in progress)
- Phase 1: Native baseline + harness
- Phase 2: Dequant FP4/FP8 → BF16
- Phase 3: AWQ-W4A16 quantize (4–6 h)
- Phase 4: Verify quantized + harness delta
- Phase 5: HF upload + cleanup

## Phase 0 — Setup & Verification (in progress)
- Started: 2026-05-01T14:46:19Z
- Decisions:
  - **Workspace = /opt/dlami/nvme/workspace** (28 TB ephemeral LVM, symlinked to `/workspace`).
  - **`CUDA_HOME` left as DLAMI default** (`/opt/pytorch/cuda`); brief's suggested `/usr/local/cuda` does not exist on this AMI. Removed wrong overrides from `~/.bashrc`.
  - **vLLM build needs `LIBRARY_PATH=$CUDA_HOME/lib`** so `ld` finds `libcudart_static.a` and `libcudadevrt.a`. Added to build script.
  - **Operator-approved (Option 1):** allow `pip install -e .` to upgrade torch 2.10 → 2.11 inside the DLAMI venv. jasl/ds4-sm120 hard-pins `torch==2.11.0`; PR description explicitly requires PyTorch ≥ 2.11. NCCL / cusparseLT pins already match DLAMI; cuDNN 9.15 → 9.19 expected.
- Outcomes so far:
  - 8× H200 (143 GB), SM 9.0, 2 TiB RAM, 28 TB ephemeral NVMe verified
  - Quant tooling installed: `llmcompressor`, `compressed-tensors`, `datasets`, `accelerate`. `autoawq` import broken vs DLAMI's transformers — non-blocking, llmcompressor recipe is primary.
  - Repos cloned: `vllm-source` (`428e08ec`), `vllm-ds4-sm120-harness` (`85aca32`), `flagos` (`f9846dc`).
  - Model download running in `phase-1-download` tmux (parallel scheduling optimization).
- Issues:
  - 1st build attempt: missing `setuptools_scm` → fixed.
  - 2nd build attempt: bad `CUDA_HOME=/usr/local/cuda` (path absent) → fixed.
  - 3rd build attempt: `ld` couldn't find `-lcudart_static`/`-lcudadevrt` → fixed via `LIBRARY_PATH`.
  - 4th build attempt: cmake configure failed — `CUDA_nvrtc_LIBRARY NOTFOUND` because DLAMI's `/opt/pytorch/cuda/lib` only ships versioned `libnvrtc.so.13` (no unversioned `libnvrtc.so` symlink that legacy FindCUDA wants). Created unversioned symlinks for `nvrtc`, `cublas`, `cublasLt`, `cudnn`, `cufft`, `curand`, `cusolver`, `cusparse`, `cupti`, `nvJitLink`, `nccl`. Also added `/opt/pytorch/cuda/lib/stubs/libcuda.so` → `/usr/lib/x86_64-linux-gnu/libcuda.so.1`.
  - 5th build attempt: completed compile (33 min), `BUILD_DONE_` printed, but `vllm._C.abi3.so` failed at import with `undefined symbol _ZN3c1013MessageLoggerC1EPKciib`. Root cause: pip's resolver upgraded torch 2.10 → 2.11 *after* nvcc was done compiling, so the .so was linked against c10 ABI from 2.10 while runtime imports torch 2.11.
  - 6th build attempt: SUCCESS. `import vllm._C` etc. clean against torch 2.11.
  - First serve attempt failed during profile run: flashinfer JIT-compile of `sampling.so` failed because it links with `-L/opt/pytorch/cuda/lib64`, but DLAMI ships `lib` (no `64`). Fix: `ln -sfn /opt/pytorch/cuda/lib /opt/pytorch/cuda/lib64`.
- HF auth: token set on remote (`huggingface-cli login`, file 600). Logged in as `pastapaul`. Gated dataset `bigcode/the-stack-smol` accessible.
- **HF destination confirmed:** `pastapaul/DeepSeek-V4-Flash-AWQ-W4A16` (no hyphen). Operator confirmed at 15:14 UTC.

## Phase 1 — Native V4-Flash Baseline (in progress)
- Started: 2026-05-01T16:25:33Z (serve up at 16:34:19Z, ~9 min cold start incl torch.compile + TileLang JIT)
- Smoke test ✓: `"The answer to 2+2 is 4."`
- jasl harness against native FP4/FP8 weights, 8× H200, TP=8, fp8 KV cache, max_model_len=16384:
  - **chat-smoke `quick`**: 4 / 4 PASS
  - **chat-smoke `quality`**: 4 / 4 PASS
  - **chat-smoke `coding`**: **2 / 2 PASS** (vs wuwenthink's 0/2 on SM120) — **major finding: SM12x kernel path has a bug not present in SM90 FlashMLA path**.
  - **toolcall15**: 23 / 30 points = 77 %, 11 strict pass / 4 fail. Matches PR #40991 caveat about non-deterministic tool-call patterns.
- Bench complete (random, 128 in / 512 out, 48 prompts, ignore-eos):

  | Concurrency | Output tok/s | Total tok/s | TTFT median (ms) | TPOT median (ms) | Duration (s) |
  |---|---|---|---|---|---|
  | 1 | 126.16 | 157.70 | 71.19 | 7.78 | 194.80 |
  | 4 | 438.14 | 547.68 | 199.54 | 8.65 | 56.09 |
  | 8 | 776.57 | 970.72 | 211.96 | 9.24 | 31.65 |

  PR #40991 reported ~478 tok/s @ C=8 on 2× RTX PRO 6000 (SM120). 8× H200 SM90 = ~1.6× that throughput, with the SM90 FlashMLA path passing 2/2 coding (vs 0/2 on SM120).
- Native server stopped, GPUs free at 0 MiB used.

## Phase 2 — Dequantize FP4/FP8 → BF16 (in progress)
- Started: 2026-05-01T18:22:37Z
- Tool: `flagos/convert_weight.py --device cuda`
- Source 153 GB → expected output ~600 GB BF16, plenty of headroom on the 28 TB ephemeral NVMe.
- 69 187 total tensors (33 792 FP4 expert weights + 375 FP8 non-expert + 34 167 scales + ~717 BF16/FP32 already-resolved).
- 46 shards to convert.
- **Result:** dequant completed in 7 min 31 s. 34 167 weights converted, 853 already-BF16/FP32 kept, 34 167 scale entries removed. Output: 543 GB across 46 shards, 35 020 keys total.

### Phase 2 verification — *vLLM BF16 path is structurally impossible in jasl/ds4-sm120*
- jasl's `vllm/model_executor/models/deepseek_v4.py:1017` unconditionally reads `config.quantization_config["scale_fmt"]`. There is no BF16-only inference path.
- Tried two workarounds; both fail:
  1. Strip `quantization_config` → `AttributeError: 'DeepseekV4Config' object has no attribute 'quantization_config'`.
  2. Keep `quantization_config: {fp8, ue8m0}` and serve BF16 weights → FP4/FP8 weight loaders expect packed int8 + E8M0 scales, not BF16 tensors. (Did not run; structural failure mode is obvious.)
- **Decision (operator-flagged 2026-05-01T18:34Z):** skip vLLM BF16 verify; rely on Phase 3 dry-run (16-sample llmcompressor calibration) as the BF16 sanity check. transformers' built-in `DeepseekV4ForCausalLM` will load the patched BF16 config cleanly. If dry-run fails fast, BF16 is bad. If it succeeds, BF16 is good and we proceed to full quant.
- BF16 config patched: `quantization_config` removed, `expert_dtype: bf16`, `torch_dtype: bfloat16`. Original preserved at `config.json.bak`.

## Phase 3 — AWQ-W4A16 Quantization (in progress)
- Started: 2026-05-01T18:35:00Z (first attempt blocked, see below)

### Tooling pivot — *DSV4 isn't in any released `transformers`*
- DeepSeek-V4-Flash released 2026-04-24; `DeepseekV4ForCausalLM` is in **none** of: `transformers` 4.52, 4.57, 5.7, or `main`. No DSV4 PR has been merged. PR #45643 (`add-deepseek-v4` branch) is open and supersedes the closed #45616.
- Root impact: brief's `llmcompressor.transformers.oneshot(...)` path (and AutoAWQ) both go through `transformers.AutoModelForCausalLM.from_pretrained`, which fails at architecture lookup.
- Operator-supplied intel: RedHat's `RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8` was produced via `vllm-project/llm-compressor` PR #2647 (branch `kylesayrs/transformers-v5`, commit `e03eb83` "dsv4 works", 2026-04-30). The PR adds `linearize_moe_model()` which converts V4's hash-routed FP4 expert modules into `nn.Linear` layers GPTQ can calibrate.
- **Tooling stack landed:**
  - **Isolated venv** at `/workspace/.venv-quant` (DLAMI venv reserved for vLLM serving). DLAMI venv was briefly broken by an earlier `pip install --force-reinstall llmcompressor` that cascade-downgraded torch to 2.10+cu128 and replaced cu13 NVIDIA packages with cu12; recovered by reinstalling cached torch 2.11.0 wheel + `nvidia-nccl-cu13==2.28.9` with `--no-deps`.
  - `transformers==5.8.0.dev0` from `huggingface/transformers@add-deepseek-v4` (PR #45643).
  - `compressed-tensors==0.15.1a20260428` (pre-release, required by kylesayrs branch).
  - `llmcompressor==0.10.1.dev109+ga308bc0e` from `vllm-project/llm-compressor@kylesayrs/transformers-v5`.
  - `torch==2.11.0+cu130` plus the cu13 NVIDIA stack.
- `DeepseekV4Config` loads via `AutoConfig.from_pretrained("/workspace/model-bf16", trust_remote_code=True)` — confirmed.

### Recipe pivot — *W4A16 instead of NVFP4 + FP8_BLOCK*
- kylesayrs's example uses `NVFP4` for experts + `FP8_BLOCK` for attention. We're keeping his `linearize_moe_model()` framework (the load-bearing V4 architecture support) but using `W4A16` for all `Linear` layers (operator-directed). Reasoning: hak-uma Coder B target deploys via Marlin W4A16 cleanly on DGX Spark; NVFP4 needs an `eugr/RobTand` patch stack on Spark.
- Calibration: `HuggingFaceH4/ultrachat_200k`, V4's manual chat encoding (BOS / `<｜User｜>` / `<｜Assistant｜>` / EOS). Sticking with kylesayrs's tested dataset + preprocessing rather than the brief's `bigcode/the-stack-smol` since that hasn't been validated against V4.
- Recipe: `GPTQModifier(config_groups={"default": QuantizationScheme(targets=["Linear"], **W4A16)}, ignore=["lm_head"])`, `sequential_targets=["DeepseekV4DecoderLayer"]`, `batch_size=32`, `max_seq_len=512`.
- **Dry-run launched 2026-05-01T18:57:56Z**, 16 samples, output `/workspace/model-w4a16-dryrun`. If dry-run succeeds, full run with 1024 samples; if W4A16 hits V4 edge cases, fall back to NVFP4+FP8_BLOCK (RedHat's exact recipe).
