#!/usr/bin/env bash
# Build FA3 hopper with head_dim=512 support (Gemma 4 global attention patch).
# GPU is not required; compilation uses CPU + nvcc only.
#
# Usage:
#   ./build_fa3_hdim512.sh              # quiet: one progress line + errors only
#   ./build_fa3_hdim512.sh --verbose    # full compiler output (use low MAX_JOBS)
#   ./build_fa3_hdim512.sh --full       # full FA3 hopper build (all head dims)
#
# Override parallelism (112-core machine example):
#   MAX_JOBS=48 NVCC_THREADS=4 ./build_fa3_hdim512.sh
#
# Debugging a failed build (parallel logs are messy — lower MAX_JOBS helps):
#   MAX_JOBS=4 ./build_fa3_hdim512.sh --verbose

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export MAX_JOBS="${MAX_JOBS:-48}"
export NVCC_THREADS="${NVCC_THREADS:-4}"
export FLASH_ATTENTION_FORCE_BUILD="${FLASH_ATTENTION_FORCE_BUILD:-1}"

VERBOSE="${VERBOSE:-0}"
BUILD_MODE="--hdim512-only"
LOG_FILE="${LOG_FILE:-$SCRIPT_DIR/build_fa3.log}"

for arg in "$@"; do
  case "$arg" in
    --verbose|-v) VERBOSE=1 ;;
    --full|--hdim512-only|-h|--help) BUILD_MODE="$arg" ;;
    *) echo "Unknown option: $arg" >&2; exit 1 ;;
  esac
done

export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_ROOT_USER_ACTION=ignore
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::DeprecationWarning}"

pip_flags=(--no-build-isolation)
[[ "$VERBOSE" == 0 ]] && pip_flags+=(-q)

# Pull actual errors out of interleaved parallel ninja/nvcc output
show_build_errors() {
  local log="$1"
  echo "==> Build failed. Matched errors in $log:"
  local errs
  errs="$(
    grep -E 'ninja: error|fatal error:|(^|[^a-z])error:|FAILED:|RuntimeError:|CalledProcessError|subprocess-exited-with-error' "$log" \
      | grep -vE 'DEPRECATION|EasyInstallDeprecation|SetuptoolsDeprecation' \
      | sort -u \
      | head -40
  )"
  if [[ -n "$errs" ]]; then
    echo "$errs"
  else
    echo "(no error pattern matched — parallel output may have buried it)"
    echo "    try: MAX_JOBS=4 $0 --verbose"
  fi
  echo ""
  echo "    full log: $log"
}

run_pip() {
  if [[ "$VERBOSE" == 1 ]]; then
    pip install "$@"
    return
  fi

  echo "==> pip install $*"
  echo "    log: $LOG_FILE  (parallel compile — screen stays quiet)"
  : >"$LOG_FILE"

  set +e
  pip install "$@" >>"$LOG_FILE" 2>&1 &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    local n
    n=$(wc -l <"$LOG_FILE" 2>/dev/null || echo 0)
    printf '\r    compiling... %s log lines' "$n"
    sleep 2
  done
  wait "$pid"
  local status=$?
  set -e
  printf '\n'

  if [[ "$status" -ne 0 ]]; then
    show_build_errors "$LOG_FILE"
    exit "$status"
  fi
}

echo "==> FA3 hopper build  MAX_JOBS=$MAX_JOBS  NVCC_THREADS=$NVCC_THREADS  mode=$BUILD_MODE"
[[ "$VERBOSE" == 0 ]] && echo "    quiet mode (--verbose for full output)"

pip install -q ninja einops 2>/dev/null || pip install "${pip_flags[@]}" ninja einops

case "$BUILD_MODE" in
  --full)
    run_pip -e . "${pip_flags[@]}"
    ;;
  --hdim512-only)
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
    run_pip -e . "${pip_flags[@]}"
    ;;
  -h|--help)
    sed -n '2,15p' "$0"
    exit 0
    ;;
esac

echo "==> Done."
[[ "$VERBOSE" == 0 ]] && echo "    log: $LOG_FILE"
echo "    smoke test (H100/H800):"
echo "      PYTHONPATH=$SCRIPT_DIR pytest 'test_flash_attn.py::test_flash_attn_output[256-256-512-False-False-0.0-False-False-False-gqa-bfloat16]' -x -v"
echo "    (do not use -k with hyphens; pytest treats '-' as NOT)"