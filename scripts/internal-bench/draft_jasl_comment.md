# Draft jasl/vllm comment (fill in after bench)

Posting target: https://github.com/jasl/vllm/issues/12 or a fresh issue on jasl/vllm
(also possible: https://github.com/jasl/vllm/issues/5 if comparison data is the better
fit there).

---

## 2× DGX Spark TP=2 validation of `ds4-sm120-preview-dev` vs prior baseline

We re-tested the `canada-quant/DeepSeek-V4-Flash-W4A16-FP8` (no-MTP) artifact on a
**2× DGX Spark GB10 (Blackwell SM 12.1a, aarch64, 121 GiB UMA per node, QSFP 200 Gbps)**
cluster at TP=2 across two jasl/vllm SHAs spanning your recent SM12x sparse-MLA + long-prefill
work, with the goal of giving you concrete data for the mainline-merge push for DSv4 SM 12.x
consumer-hardware support.

### Build matrix tested

| Tag | jasl/vllm SHA | Date | Description |
|---|---|---|---|
| BASELINE | `428e08e` | 2026-05-05 | prior production canonical (used by `vllm-w4a16-dsv4:exp` image) |
| HEAD | `5d6479811` (`ds4-sm120-preview-dev`) | 2026-05-25 | latest at test time, post sparse-MLA flashmla + long-prefill protect |
| _(optional)_ INTERMEDIATE | `a937d4b28` | 2026-05-24 | "Stabilize SM12x sparse MLA long prefill" — to localize regression direction |

Build path: `canada-quant/dsv4-flash-w4a16-fp8/scripts/bootstrap_dsv4_spark.sh --vllm-ref <sha>`.
Both pairs (Sparks 1↔2 over enp1s0f1np1 @ 192.168.1.0/30, and Sparks 5↔6 over enp1s0f0np0 @
192.168.101.0/30) used identical OOB recipe; only the vllm ref differed.

### What got better / worse vs baseline

**Decode throughput @ canonical 1M-ctx single-stream (`--max-model-len 1048576 --max-num-seqs 1`):**

| Build | Cold start | bs=1 decode tok/s | TPOT median (ms) | TPOT p95 (ms) |
|---|---|---|---|---|
| BASELINE 428e08e | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| HEAD 5d6479811   | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

**Long-context 64K prompt / 2K decode (`--max-model-len 65536 --max-num-seqs 4`):**

| Build | bs=1 decode tok/s | TPOT median | TPOT p95 |
|---|---|---|---|
| BASELINE 428e08e | _TBD_ | _TBD_ | _TBD_ |
| HEAD 5d6479811   | _TBD_ | _TBD_ | _TBD_ |

**Batched chat c=4 (`--max-model-len 16384 --max-num-seqs 4`):**

| Build | aggregate tok/s | per-req median |
|---|---|---|
| BASELINE 428e08e | _TBD_ | _TBD_ |
| HEAD 5d6479811   | _TBD_ | _TBD_ |

### OOM ceiling on 2× GB10 (242 GiB combined UMA)

We swept `--max-model-len` at `seqs=1` and `seqs` at `ctx=256K` to find the OOM boundary.
This is the missing data the model card needed.

**Context wall at single stream (seqs=1, gpu-mem-util=0.90):**

| max_model_len | boot? | served 256-tok request? | KV cache % @ idle | failure mode |
|---|---|---|---|---|
| 262144 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 524288 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 1048576 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 1572864 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 2097152 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

**Concurrency wall at ctx=256K, gpu-mem-util=0.90:**

| max_num_seqs | boot? | served? | failure mode |
|---|---|---|---|
| 1 | _TBD_ | _TBD_ | _TBD_ |
| 2 | _TBD_ | _TBD_ | _TBD_ |
| 4 | _TBD_ | _TBD_ | _TBD_ |
| 8 | _TBD_ | _TBD_ | _TBD_ |
| 16 | _TBD_ | _TBD_ | _TBD_ |

### Long-reasoning think-max stability

`chat_template_kwargs={"thinking": true, "reasoning_effort": "high"}`, `max_tokens=65536`,
3 AIME-2024 problems:

| Build | problems run | completed | truncated @ 65K | avg wall (s) | avg decode tok/s |
|---|---|---|---|---|---|
| BASELINE 428e08e | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| HEAD 5d6479811   | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

### OOB bug surfaced (separate filing)

`bootstrap_dsv4_spark.sh` fails on stock DGX Spark (Ubuntu 24.04, Python 3.12, modern
huggingface_hub) at step 2:
- PEP 668 blocks `pip install --user`
- `huggingface_hub>=0.35` removed the `huggingface-cli` legacy entry-point module

Workarounds documented; patch sketched against canada-quant/dsv4-flash-w4a16-fp8.

### Production recommendation for dual-DGX-Spark deployment

After this run we will pin our internal production W4A16 (no-MTP) build on dual spark to
**jasl/vllm@`<WINNER_SHA>`** — replacing the prior `428e08e` baseline image
`vllm-w4a16-dsv4:exp`.

### Raw data

Full result JSONs + per-cell vllm_node logs:
https://github.com/canada-quant/dsv4-flash-w4a16-fp8/tree/dual-spark-bench-2026-05-26/findings/dual_spark_jasl_sha_regression_2026-05-26/

Thanks for the steady stream of SM12x kernel work — these are the kinds of numbers that
should help advocate for upstream merge of DSv4 SM 12.x consumer-hardware support.
