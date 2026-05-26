# Comment to file on jasl/vllm (final draft, 2026-05-26)

**Target venue:** https://github.com/jasl/vllm/issues/12 (token-stream corruption thread) OR a new issue at https://github.com/jasl/vllm/issues/new — your call.

---

## 2× DGX Spark GB10 TP=2 validation of `jasl/vllm@428e08e` (pre-refactor) for DSv4-Flash W4A16

We re-ran the OOB recipe at https://github.com/canada-quant/dsv4-flash-w4a16-fp8 on 2× DGX Spark GB10 (Blackwell SM 12.1a, aarch64, 121 GiB UMA per node, QSFP 200 Gbps RDMA) at TP=2, with the goal of getting fresh fleet-aligned numbers + flagging the OOB blockers that current upstream users would hit.

### Headline result

`canada-quant/DeepSeek-V4-Flash-W4A16-FP8` serves cleanly on dual DGX Spark TP=2 at **--max-model-len 1048576**, single-stream, cudagraph `FULL_AND_PIECEWISE` ON, with **`jasl/vllm@428e08e`** + canada-quant's two vendored patches (`kylesayrs-deepseek-ct.patch` + `patch_v4_packed_mapping.py`) on `nvidia 580.142` aligned across the fleet.

### Throughput @ canonical config

| Bench | Pair A (S1+S2) | Pair B (S5+S6) |
|---|---|---|
| c=1 P=1K D=1K | **12.1 tok/s** TPOT 82.9 ms | **11.1 tok/s** TPOT 90.0 ms |
| c=1 P=64K D=2K (long prefix) | 10.6 tok/s TPOT 94.3 ms | 11.1 tok/s TPOT 90.0 ms |

Both pairs serving the SAME vllm-w4a16-dsv4 image, same artifact, same flags. Cross-pair variance under 10% (was +23% on un-aligned drivers — see F006 below).

### Long-reasoning think-max (AIME-2024, max_tokens=65536, reasoning_effort=high)

5 problems each pair; both pairs land 4/5 correct, **0 truncations**, decode stays at 11.5-11.9 tok/s throughout. Problem AIME-I-5 used **14,725 reasoning tokens (~20 min single-stream wall)** and finished cleanly inside the 65K budget — concrete validation that long-reasoning on 2× GB10 doesn't hit the 32K-cap failure mode the model card previously documented at lower max-model-len.

### OOM threshold sweep (Pair B, gpu-memory-utilization 0.90)

| (ctx × seqs) | Result |
|---|---|
| 256K × 1, 512K × 1, **1M × 1** | fits — 1M is the production canonical |
| 1.5M × 1, 2M × 1 | OOM-style fast-exit (~50 s) — **context wall between 1M and 1.5M** at seqs=1 |
| 256K × 2 | OOM-style fast-exit — **single-stream only at ctx ≥ 256K** |

### Why I'm posting on YOUR fork

While running the OOB recipe end-to-end I hit **seven blockers** on stock DGX Spark that the model card / bootstrap doesn't yet handle. Five are bootstrap-side (canada-quant repo fix), but **two are about your `ds4-sm120-preview-dev` branch** specifically that I think you'd want to know about for the mainline-merge push for DSv4 SM 12.x consumer-hardware support:

**F004 — `kylesayrs-deepseek-ct.patch` stale after PR #43004 refactor**

The canada-quant bootstrap vendors a content-pinned successor of the kylesayrs PR #41276 patch. After your 2026-05-19 commit `287471b99` ([Model Refactoring] Migrate DeepSeek V4 to vllm/models/), the patch's file paths (`vllm/model_executor/layers/deepseek_compressor.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/model_executor/models/deepseek_v4.py`) no longer exist. **Bootstrap can only build SHAs ≤ a8887c208 (2026-05-13).** Builds against `0a65d4662` (2026-05-14, "Fuse norm and router"), `a937d4b28` (2026-05-24, "Stabilize SM12x sparse MLA long prefill"), and HEAD `5d6479811` (2026-05-25) all fail at `git apply --check`.

If the kylesayrs functionality was integrated into your post-refactor branch natively, we'd love to know what flag/path to use — happy to refresh the vendored patch against your latest if there's still a gap.

**F007 — Driver upgrade to 595.71.05 blocked by Secure Boot MOK**

Tried to upgrade the cluster from 580 → 595 to test if newer NCCL helped with the cross-pair throughput variance. The Ubuntu 24.04 DGX Spark image has Secure Boot enabled with only the Canonical CA enrolled. DKMS-built `nvidia.ko` is signed with a per-host MOK that isn't in shim's enrolled-key DB → `modprobe: ERROR: could not insert 'nvidia': Key was rejected by service`. MOK enrollment requires firmware UI (`mokutil --import` + reboot to MOK Manager for password entry) — not remotely fixable.

Not your problem directly, but worth flagging if you're recommending newer drivers in future docs: prebuilt-signed module packages (`linux-modules-nvidia-580-open-6.17.0-1014-nvidia` etc.) are Canonical-signed and load fine; DKMS-built ones need MOK enrollment. The 595 packages on Canonical's archive don't yet have prebuilt-signed variants for the GB10 kernel image, so DKMS is the only path on stock DGX Spark — and Secure Boot blocks it.

**F006 — Driver mismatch on the NCCL head↔worker path costs 20% throughput**

This one IS validation data for you. Pre-alignment fleet had S1 on 590.48.01 and the other 5 sparks on 580.142 / 580.126.09. Pair A (S1↔S2, mixed) was running **23% slower than Pair B** (S5↔S6, both 580.142). Same vllm image, same QSFP, same flags. After aligning the entire fleet to 580.142, the gap collapsed to ±5%.

The hypothesis is that mixed-version NCCL between head and worker falls back to a slower transport. Hadn't seen this called out in your docs — might be worth noting in the recommended-config section as a heads-up for users running heterogeneous fleets.

### Build matrix we wanted to test but couldn't (per F004)

Originally targeted SHAs across your recent SM12x sparse-MLA + long-prefill work:
- BASELINE `428e08e` (2026-05-05) — tested ✓
- INTERMEDIATE `a937d4b28` (2026-05-24, "Stabilize SM12x sparse MLA long prefill") — blocked by F004
- HEAD `5d6479811` (2026-05-25, "Protect active decode from very long prefill") — blocked by F004

Happy to re-run the bench against any post-refactor SHA if you can point us at the right patch-set / build pattern. The 2× DGX Spark cluster is on standby for it.

### Raw data + per-cell logs

https://github.com/canada-quant/dsv4-flash-w4a16-fp8/tree/dual-spark-bench-2026-05-26/findings/dual_spark_jasl_sha_regression_2026-05-26/

Includes raw bench JSONs, per-cell vllm_node logs, the diagnosis docs for F001-F007, the suggested upstream-patch sketches, and Markdown summaries.

Thanks for the steady stream of SM12x kernel work on the fork — these are numbers we hope help in the upstream-merge conversation for DSv4 consumer-hardware support.
