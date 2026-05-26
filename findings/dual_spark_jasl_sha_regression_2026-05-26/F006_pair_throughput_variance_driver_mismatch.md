# F006 — 20% throughput variance between two Spark pairs; likely NVIDIA driver mismatch

**Severity:** medium — affects production-canonical decision (which pair to ship as the default DSV4 deployment)
**Discovered:** 2026-05-26 during dual-spark bench on jasl@428e08e baseline
**Hardware:** identical 6× DGX Spark GB10 (Blackwell SM 12.1a, ARM64, 128 GiB UMA)

## Observation

Same artifact (`canada-quant/DeepSeek-V4-Flash-W4A16-FP8`), same vLLM image
(`vllm-w4a16-dsv4:baseline-428e08e` built from `jasl/vllm@428e08e`), same
serve flags (`--max-model-len 1048576 --max-num-seqs 1 --gpu-memory-utilization
0.90`), same QSFP 200 Gbps RDMA between the pair:

| Bench config            | Pair A (S1+S2) | Pair B (S5+S6) | Pair A vs B |
|---|---|---|---|
| c=1 P=1K D=1K throughput | 10.0 tok/s     | 12.3 tok/s     | **-19%** |
| c=1 P=64K D=2K throughput| 9.2 tok/s      | 11.2 tok/s     | **-18%** |
| AIME think-max decode    | 9.9 tok/s      | 12.8 tok/s     | **-23%** |
| AIME think-max wall      | 205 s          | 160 s          | +28% slower |

Consistent ~20% degradation on Pair A across all bench types.

## Hardware deltas between pairs

| Field | Pair A (S1, S2)        | Pair B (S5, S6) |
|---|---|---|
| GB10 GPU model        | identical                  | identical |
| UMA size              | 121 GiB                    | 121 GiB |
| QSFP NIC used         | `enp1s0f1np1` (2nd NIC port) | `enp1s0f0np0` (1st NIC port, prod default) |
| QSFP /30 subnet       | 192.168.1.0/30             | 192.168.101.0/30 |
| QSFP MTU              | 9000                       | 9000 |
| QSFP RTT (ping)       | 0.6–0.8 ms                  | 0.6 ms |
| **Driver version S1** | **590.48.01**              | n/a |
| Driver version S2     | 580.142                    | n/a |
| Driver version S5     | n/a                        | 580.142 |
| Driver version S6     | n/a                        | 580.142 |

**Spark 1 has driver 590.48.01; Sparks 2, 5, 6 have 580.142.** Pair A is the
*only* pair with mixed driver versions on the head ↔ worker NCCL path.

## Hypothesis

NCCL collectives between mixed driver versions (590 head ↔ 580 worker) likely
incur a slow-path negotiation or fall back to a less-optimal protocol than when
both ends run the same driver. The 20% per-token decode hit on a memory-bound
TP=2 workload is consistent with NCCL reduction overhead, not compute.

## Mitigations

1. **Recommended:** uniform driver version across all 6 sparks before dual-spark
   serving. Either align S1 to 580.142 OR upgrade S2,S5,S6 to 590.x. Production-
   canonical at canada-quant uses 580.142 on the DSV4 pair, so leaving Pair B at
   580.x and aligning S1 down is the safer pin.
2. **Diagnostic:** rerun with `NCCL_DEBUG=INFO` and capture which transport is
   selected on each pair. If Pair A picks `SHM` or a TCP fallback while Pair B
   picks `IB` (RoCEv2), that confirms the hypothesis.
3. **Alternative explanation to rule out:** `enp1s0f1np1` could be a different
   PCIe lane / NIC firmware than `enp1s0f0np0`. Verify via `ethtool -i` and
   `lspci -vv`.

## Production-canonical implication

For the canada-quant `dsv4-flash-w4a16-fp8` model card's "Recommended hardware"
section: **Pair B numbers (12.3 tok/s @ bs=1) are the production-canonical**.
Pair A's numbers (10.0 tok/s) reflect the as-deployed mixed-driver fleet and
should be documented as the "currently-measured cross-pair variance" until
driver alignment is performed.

## Test-run impact

This 20% variance is the largest single confounding factor in our test data.
Going forward in the bench: OOM-threshold sweep is on Pair B (the faster pair)
so reported boundaries reflect best-case performance.
