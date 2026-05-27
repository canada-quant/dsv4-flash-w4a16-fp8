# Dual DGX Spark TP=2 — jasl/vllm SHA regression sweep (2026-05-26)

**Goal:** validate the no-MTP W4A16 artifact on dual DGX Spark TP=2 across recent
`jasl/vllm@ds4-sm120-preview-dev` SHAs, document the OOM threshold + long-reasoning
characteristics that the current model card lacks, and produce data jasl can use to
push DSv4 SM 12.x consumer-hardware support to upstream `vllm-project/vllm` mainline.

## Hardware

- **Pair A**: spark-a5fa (S1, head) ↔ spark-2861 (S2, worker), QSFP 200 Gbps on
  `enp1s0f1np1`, 192.168.1.0/30.
- **Pair B**: spark-7a88 (S5, head) ↔ spark-75bf (S6, worker), QSFP 200 Gbps on
  `enp1s0f0np0`, 192.168.101.0/30 (production canonical pair).
- Each node: NVIDIA GB10 Grace Blackwell, ARM64/aarch64, 128 GiB UMA (121 GiB OS-visible),
  Ubuntu 24.04.4, CUDA 13.1, compute capability sm_121.

## SHAs in the matrix

| Tag | jasl/vllm SHA | Date | Subject |
|---|---|---|---|
| BASELINE | `428e08e` | 2026-05-05 | prior production canonical |
| _candidate_ INTERMEDIATE | `a937d4b28` | 2026-05-24 | Stabilize SM12x sparse MLA long prefill |
| HEAD | `5d6479811` (`ds4-sm120-preview-dev`) | 2026-05-25 | Protect active decode from very long prefill |

Build path: `scripts/bootstrap_dsv4_spark.sh --vllm-ref <SHA>` with `--ssh-user pcozz`.

## OOB findings during this run

- **[F001](F001_pep668_bootstrap_failure.md)** — bootstrap step 2 fails on Ubuntu 24.04
  due to PEP 668 + rich Debian RECORD + `~/.local/bin` not on non-interactive SSH PATH.
  Workaround applied, upstream patch sketch attached.
- **[F002](F002_huggingface_cli_modern_removed.md)** — `huggingface_hub >= 0.35` removed
  the legacy `huggingface-cli` CLI module; bootstrap hardcodes its use. Workaround:
  install a 4-line shim at `/usr/local/bin/huggingface-cli` that exec's `hf`.

## Results (to be filled in after the bench suite completes)

- `pairB_head_5d647/` — Pair B (S5+S6) with jasl HEAD; deep bench (OOM grid + concurrency
  + long-reasoning think-max + GSM8K-style smoke)
- `pairA_baseline_428e08e/` — Pair A (S1+S2) with prior baseline; same suite for comparison
- `pairA_a937d4b28/` — Pair A re-run on intermediate SHA (regression localization)
- `summary.md` — final A/B/HEAD comparison table + production-canonical recommendation
- `draft_jasl_comment.md` — data-grounded comment for the jasl/vllm fork

