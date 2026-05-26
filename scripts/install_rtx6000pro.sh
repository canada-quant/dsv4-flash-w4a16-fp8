#!/usr/bin/env bash
# install_rtx6000pro.sh — RTX PRO 6000 install for Card A (W4A16-FP8, no MTP)
#
# canada-quant/DeepSeek-V4-Flash-W4A16-FP8
#
# **CURRENT SHIPPING STATE (2026-05-26)**: Card A's safetensors weren't
# dequantized (yesterday's attempt crashed on e_score_correction_bias before
# touching weights). On current upstream vLLM Card A hits the same kernel-
# dispatch bug as Card D (FP8 block-quant naming) plus an additional
# architecture-drift bug (ffn.gate.e_score_correction_bias vs ffn.gate.bias).
# Awaiting vllm-project/vllm#43564 resolution.
#
# Card A's published H200 + DGX Spark + RTX PRO 6000 benchmarks remain valid
# on the older jasl/vllm@abad5dc71 build they were measured against.
set -euo pipefail

ARTIFACT="canada-quant/DeepSeek-V4-Flash-W4A16-FP8"
VLLM_REPO="https://github.com/jasl/vllm.git"
VLLM_PIN="a02a3778f"
VLLM_BRANCH="ds4-sm120-preview-dev"
VLLM_SRC="${VLLM_SRC:-$HOME/src/vllm}"
VENV="${VENV:-$HOME/venv-serve}"
SCRATCH="${SCRATCH:-/scratch}"

cat <<EOF
================================================================
Card A (W4A16-FP8, no MTP) RTX PRO 6000 install — $(date)

⚠  CURRENT SHIPPING STATE: vLLM kernel-dispatch issue prevents this
   artifact from running on current upstream vLLM. The build will
   succeed but serve will fail. Card A also has an architecture-drift
   bug (ffn.gate.e_score_correction_bias vs ffn.gate.bias rename
   needed).

   Card A's H200 / DGX Spark / RTX PRO 6000 benchmarks remain valid on
   jasl/vllm@abad5dc71 (the build they were measured against).

================================================================
EOF

SHARED_INSTALL_URL="https://raw.githubusercontent.com/canada-quant/dsv4-flash-nvfp4-fp8-mtp/main/scripts/install_rtx6000pro.sh"
ARTIFACT="$ARTIFACT" REPO_ROOT="$(dirname "$(realpath "$0")")/.." curl -sL "$SHARED_INSTALL_URL" | bash
