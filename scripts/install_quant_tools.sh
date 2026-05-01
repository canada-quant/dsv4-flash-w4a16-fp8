#!/usr/bin/env bash
# Phase 0: install quantization tooling and clone harness/flagos repos in parallel with vLLM build
set -euo pipefail
exec > >(tee -a /workspace/output/logs/quant-tools-install.log) 2>&1
echo "==== install_quant_tools.sh start $(date -u +%FT%TZ) ===="

source /opt/pytorch/bin/activate

pip install -U pip setuptools wheel
# llmcompressor stack
pip install llmcompressor compressed-tensors datasets accelerate
# AutoAWQ as fallback recipe option
pip install autoawq || echo "WARN: autoawq pin failed; will fall back to llmcompressor only"

cd /workspace
if [ ! -d vllm-ds4-sm120-harness ]; then
  git clone https://github.com/jasl/vllm-ds4-sm120-harness.git
fi
cd vllm-ds4-sm120-harness
git pull --ff-only || true
git log -1 --format='%H %s' | tee /workspace/output/harness-commit.txt
pip install -e . || python -m pip install -e . || echo "WARN: harness install -e failed; will run via PYTHONPATH"

cd /workspace
if [ ! -d flagos ]; then
  git clone https://github.com/flagos-ai/DeepSeek-V4-FlagOS.git flagos
fi
cd flagos
git log -1 --format='%H %s' | tee /workspace/output/flagos-commit.txt
if [ -f requirements.txt ]; then
  pip install -r requirements.txt || echo "WARN: flagos requirements partial"
fi

echo "==== verify imports ===="
python -c "import llmcompressor, compressed_tensors, datasets, accelerate; print('llmcompressor', llmcompressor.__version__)"
python -c "import awq; print('awq', awq.__version__)" || echo "awq not importable; ok if llmcompressor recipe sufficient"

echo "TOOLS_DONE_$(date +%s)"
