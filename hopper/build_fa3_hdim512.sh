#!/usr/bin/env bash
# Build FA3 hopper with head_dim=512 support (Gemma 4 global attention patch).
# GPU is not required; compilation uses CPU + nvcc only.
#
# Usage:
#   ./build_fa3_hdim512.sh              # fast build: only hdim=512 kernels
#   ./build_fa3_hdim512.sh --full       # full FA3 hopper build (all head dims)
#
# Override parallelism (112-core machine example):
#   MAX_JOBS=48 NVCC_THREADS=4 ./build_fa3_hdim512.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parallelism: MAX_JOBS = ninja concurrent .cu compiles; NVCC_THREADS = nvcc --threads per file
export MAX_JOBS="${MAX_JOBS:-48}"
export NVCC_THREADS="${NVCC_THREADS:-4}"
export FLASH_ATTENTION_FORCE_BUILD="${FLASH_ATTENTION_FORCE_BUILD:-1}"

BUILD_MODE="${1:---hdim512-only}"

echo "==> FA3 hopper build"
echo "    MAX_JOBS=$MAX_JOBS  NVCC_THREADS=$NVCC_THREADS"
echo "    mode=$BUILD_MODE"
echo ""

pip install -q ninja einops

case "$BUILD_MODE" in
  --full)
    echo "==> Full hopper build (all head dimensions; slow)"
    pip install -e . --no-build-isolation
    ;;
  --hdim512-only|"")
    echo "==> Minimal build (hdim=512 SM90 only)"
    export FLASH_ATTENTION_DISABLE_HDIM64=1
    export FLASH_ATTENTION_DISABLE_HDIM96=1
    export FLASH_ATTENTION_DISABLE_HDIM128=1
    export FLASH_ATTENTION_DISABLE_HDIM192=1
    export FLASH_ATTENTION_DISABLE_HDIM256=1
    export FLASH_ATTENTION_DISABLE_SM80=1
    export FLASH_ATTENTION_DISABLE_HDIMDIFF64=1
    export FLASH_ATTENTION_DISABLE_HDIMDIFF192=1
    export FLASH_ATTENTION_DISABLE_FP8=1
    rm -rf build *.egg-info
    pip install -e . --no-build-isolation
    ;;
  -h|--help)
    sed -n '2,11p' "$0"
    exit 0
    ;;
  *)
    echo "Unknown option: $BUILD_MODE" >&2
    echo "Use --hdim512-only (default) or --full" >&2
    exit 1
    ;;
esac

echo ""
echo "==> Done. Run tests on an H100/H800 (SM90):"
echo "    export PYTHONPATH=$SCRIPT_DIR"
echo "    pytest test_flash_attn.py -k '512 and gqa' -x --tb=short"