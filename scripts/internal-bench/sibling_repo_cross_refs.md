# Cross-reference snippets for sibling repo updates

When the no-MTP dual-spark bench data lands, these snippets get committed to each
sibling repo's `MODEL_CARD.md` / `README.md` on the `dual-spark-bench-2026-05-26`
branch.

---

## For `~/dsv4-flash-w4a16-fp8-mtp/MODEL_CARD.md` (W4A16+MTP)

Add as a subsection under "Hardware":

```markdown
### 2× DGX Spark TP=2 (GB10, SM 12.1a)

The base W4A16+FP8 (no MTP) sibling artifact has been validated on 2× DGX Spark
TP=2 — see [`canada-quant/DeepSeek-V4-Flash-W4A16-FP8`](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-W4A16-FP8)
for full numbers. **This MTP variant adds ~6.6 GB for the BF16 MTP block + spec-decode
workspace overhead**, which on 2× GB10 (242 GiB combined UMA) reduces the usable
context budget at TP=2.

Empirical operating points on 2× GB10 are pending — see the no-MTP sibling for
the dual-spark recipe + OOM characterization. Three OOB blockers from the shared
bootstrap (`bootstrap_dsv4_spark.sh`) carry through to this repo as well:

- F001 — PEP 668 on Ubuntu 24.04 blocks bootstrap step 2's `pip install --user`
- F002 — `huggingface_hub >= 0.35` removed the `huggingface-cli` legacy CLI;
  bootstrap hardcodes it
- F003 — bootstrap step 3 (`sudo ip addr replace`) requires passwordless sudo

See [`canada-quant/dsv4-flash-w4a16-fp8/tree/main/findings/dual_spark_jasl_sha_regression_2026-05-26/`](https://github.com/canada-quant/dsv4-flash-w4a16-fp8/tree/main/findings/dual_spark_jasl_sha_regression_2026-05-26)
for the OOB workarounds.
```

---

## For `~/dsv4-flash-nvfp4-fp8-mtp/MODEL_CARD.md` (NVFP4+MTP)

Add as a new row in the "Hardware validated" table near the top:

```markdown
| 2× NVIDIA DGX Spark GB10 (`sm_121a`) | SM 12.1 | 128 GB UMA per node (242 GB combined) | QSFP 200 Gbps RDMA | NOT validated — 172 GiB artifact + BF16 MTP block + NVFP4 workspace at TP=2 exceeds available headroom on this UMA tier. The smaller [W4A16-FP8 sibling](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-W4A16-FP8) is the recommended Spark SKU. |
```

And add to "Hardware notes" section:

```markdown
### Why not 2× DGX Spark?

This artifact is ~172 GiB (NVFP4 routed experts + FP8 block attention + BF16
preserved MTP). At TP=2 across 2× GB10, that's ~86 GiB of weights per node
against a 121 GiB UMA budget. After subtracting OS + Python runtime + cuda graph
workspace + spec-decode draft state, KV cache for any non-trivial context
window does not fit cleanly at `--gpu-memory-utilization 0.90`. The
[W4A16-FP8-MTP sibling](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP)
at ~165 GiB is the recommended Spark SKU at this hardware tier (or, for the
non-MTP path, the smaller [W4A16-FP8 baseline](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-W4A16-FP8)
at ~143 GiB with documented 1M-context dual-spark recipe).

For 2× DGX Spark deployment with MTP draft-decode, prefer the W4A16+MTP
sibling. For 1M-context production serving without MTP, prefer the W4A16-FP8
baseline.

Future: a `--cpu-offload-gb` configuration on GB10 (where UMA makes host
offload essentially free) may unlock this artifact at modest context. Not yet
characterized.
```

---

## For `~/dsv4-pro-nvfp4-fp8-mtp/MODEL_CARD.md` (V4-Pro NVFP4+MTP)

No update — V4-Pro is ~913 GB, fundamentally won't fit on any reasonable
GB10 cluster. Existing "B300-only" framing stands.

---

## Commit message template for each sibling repo

```
Cross-reference no-MTP dual-spark validation (2026-05-26)

The smaller W4A16+FP8 no-MTP sibling has been validated on 2× DGX Spark TP=2.
This repo's MTP / NVFP4 variants don't fit cleanly at that hardware tier —
adding a "Why not" note + cross-reference to the no-MTP sibling for users
who came here looking for the Spark recipe.

Also noting the 3 OOB bootstrap blockers (F001-F003) that the shared
bootstrap_dsv4_spark.sh hits on stock DGX Spark.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```
