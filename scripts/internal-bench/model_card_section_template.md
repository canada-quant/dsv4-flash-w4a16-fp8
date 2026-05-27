# Model card section template — fills in after bench data lands

## 2× DGX Spark TP=2 dual-spark deployment (2026-05-26 update)

**Run conditions:** `scripts/bootstrap_dsv4_spark.sh` OOB against two jasl/vllm SHAs across two
QSFP-paired Spark configurations (S1↔S2 over `enp1s0f1np1` @ 192.168.1.0/30, S5↔S6 over
`enp1s0f0np0` @ 192.168.101.0/30, both at 200 Gbps RDMA). All numbers measured 2026-05-26.

**Production-canonical SHA recommendation:** _<WINNER_SHA — fill in after STAGE 4>_.

### Throughput at three canonical configs

| Config | jasl@428e08e (BASELINE) | jasl@a937d4b28 (INTERMEDIATE) | jasl@5d6479811 (HEAD) |
|---|---|---|---|
| c=1 P=1K D=1K (single-stream chat) — aggregate tok/s | _<>_ | _<>_ | _<>_ |
| c=1 P=1K D=1K — TPOT median (ms) | _<>_ | _<>_ | _<>_ |
| c=1 P=64K D=2K (long-context retrieval) — aggregate tok/s | _<>_ | _<>_ | _<>_ |
| c=4 P=1K D=512 (batched chat) — aggregate tok/s | _<>_ | _<>_ | _<>_ |
| Cold-start time to /health 200 | _<>_ | _<>_ | _<>_ |

### OOM threshold at `--gpu-memory-utilization 0.90`

**Context wall (single stream, `--max-num-seqs 1`):**

| `--max-model-len` | BASELINE | HEAD |
|---|---|---|
| 262144 (256K) | _<>_ | _<>_ |
| 524288 (512K) | _<>_ | _<>_ |
| 1048576 (1M) — production canonical | _<>_ | _<>_ |
| 1572864 (1.5M) | _<>_ | _<>_ |
| 2097152 (2M) | _<>_ | _<>_ |

**Concurrency wall (at `--max-model-len 262144`):**

| `--max-num-seqs` | BASELINE | HEAD |
|---|---|---|
| 1 | _<>_ | _<>_ |
| 2 | _<>_ | _<>_ |
| 4 | _<>_ | _<>_ |
| 8 | _<>_ | _<>_ |
| 16 | _<>_ | _<>_ |

### Long-reasoning think-max (AIME-2024, `max_tokens=65536`, `reasoning_effort=high`)

3 problems each, single stream. The model card previously documented a "32K cap" failure
mode at `--max-model-len 16384`; this benchmark uses the canonical 1M config so context is
not the bottleneck — only the requested `max_tokens` is.

| SHA | problems run | completed | truncated @ 65K | avg wall (s) | avg decode tok/s | avg completion tokens |
|---|---|---|---|---|---|---|
| BASELINE 428e08e | _<>_ | _<>_ | _<>_ | _<>_ | _<>_ | _<>_ |
| HEAD 5d6479811 | _<>_ | _<>_ | _<>_ | _<>_ | _<>_ | _<>_ |

### OOB bootstrap blockers discovered on stock DGX Spark Ubuntu 24.04

Three blockers in `bootstrap_dsv4_spark.sh` step 2 + step 3 — all hit on a fresh box, all
have one-line workarounds documented in the findings dir:

- **F001** — PEP 668 blocks bootstrap step 2's `pip install --user huggingface_hub`. Use
  `--break-system-packages --ignore-installed rich` system-wide.
- **F002** — `huggingface_hub >= 0.35` removed the legacy `huggingface-cli` entry point.
  Install a 4-line shim at `/usr/local/bin/huggingface-cli` that `exec hf "$@"`.
- **F003** — bootstrap step 3 `sudo ip addr replace` requires passwordless sudo, but stock
  pcozz user doesn't have it. Add `/etc/sudoers.d/pcozz-dsv4-bench` scoped to `/usr/bin/ip
  /usr/sbin/ip /usr/bin/netplan /usr/sbin/netplan`.

Full repros + suggested upstream patches:
[`findings/dual_spark_jasl_sha_regression_2026-05-26/`](https://github.com/canada-quant/dsv4-flash-w4a16-fp8/tree/main/findings/dual_spark_jasl_sha_regression_2026-05-26).

### Production-canonical recommendation

_<TBD — winning SHA + serve flags, decided after STAGE 4 synthesis>_

Replace the prior `vllm-w4a16-dsv4:exp` image (built from jasl@428e08e) with
`vllm-w4a16-dsv4:<WINNER_SHA>` for dual-DGX-Spark deployment if the data supports an upgrade.
Otherwise, retain `428e08e` as the production canonical and pin in `launch_native_v2.py`.

### Watchdog interaction note

The `cozzspark/dsv4-watchdog.timer` systemd unit on the gateway Spark auto-recovers a
`vllm_node` container every 5 minutes if `/health` fails. Before a manual swap (e.g. moving
from baseline to a new SHA), the operator must `systemctl --user stop dsv4-watchdog.timer
minimax-watchdog.timer` on the gateway, OR the watchdog will fight the bench run by
re-launching prod containers on top of the test ones.

This is observed-only on the canada-quant fleet; the upstream model card doesn't (yet) carry
this note because the watchdog is a cozzspark-specific orchestration artifact.
