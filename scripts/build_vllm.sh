#!/usr/bin/env bash
# Phase 0: build vLLM from jasl/ds4-sm120 for SM90 (H200)
set -euo pipefail
exec > >(tee -a /workspace/output/logs/vllm-build.log) 2>&1
echo "==== build_vllm.sh start $(date -u +%FT%TZ) ===="

source /opt/pytorch/bin/activate
# Do NOT override CUDA_HOME / PATH — DLAMI venv activation sets them correctly
# (CUDA_HOME=/opt/pytorch/cuda, nvcc on PATH via that bin dir).
export TORCH_CUDA_ARCH_LIST="9.0a"
export MAX_JOBS=64
export NVCC_THREADS=4
# DLAMI venv ships static cudart/cudadevrt under $CUDA_HOME/lib; expose to ld at build time.
export LIBRARY_PATH="${CUDA_HOME}/lib:${LIBRARY_PATH:-}"
export CMAKE_LIBRARY_PATH="${CUDA_HOME}/lib:${CMAKE_LIBRARY_PATH:-}"
echo "CUDA_HOME=$CUDA_HOME"
echo "LIBRARY_PATH=$LIBRARY_PATH"
which nvcc
ls "$CUDA_HOME/lib/libcudart_static.a" "$CUDA_HOME/lib/libcudadevrt.a"

cd /workspace
if [ ! -d vllm-source ]; then
  git clone https://github.com/jasl/vllm.git vllm-source
fi
cd vllm-source
git fetch origin ds4-sm120
git checkout ds4-sm120
git reset --hard origin/ds4-sm120
git log -1 --format='%H %s' | tee /workspace/output/vllm-commit.txt

python -V
pip --version
nvcc --version | tail -2

pip install -U pip setuptools wheel ninja setuptools_scm cmake pybind11

# Per upstream practice: use precompiled torch from venv, build only kernels.
echo "==== pip install -e . (no build isolation) ===="
pip install -e . --no-build-isolation

echo "==== verify import ===="
python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)"

echo "BUILD_DONE_$(date +%s)"
