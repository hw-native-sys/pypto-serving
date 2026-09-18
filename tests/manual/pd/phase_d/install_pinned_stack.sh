#!/usr/bin/env bash
# Install an exported D0 stack in its dedicated container workspace.
set -eo pipefail
stage=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source /home/sj/git/env_all.sh
cd "$stage/pypto"
for repo in . runtime 3rdparty/libbacktrace 3rdparty/msgpack-c; do
    git -C "$repo" init -q
done
source .claude/skills/testing/load-env.sh
if [[ -n ${DSPARK_BUILD_JOBS:-} ]]; then
    PYPTO_BUILD_JOBS="$DSPARK_BUILD_JOBS"
fi
export CMAKE_BUILD_PARALLEL_LEVEL="$PYPTO_BUILD_JOBS"
export MAX_JOBS="$PYPTO_BUILD_JOBS"
python3 -m venv --system-site-packages "$stage/pypto/runtime/.venv"
source "$stage/env_pinned_stack.sh"
mkdir -p "$stage/ascend" "$SIMPLER_ROOT/build"
if [[ ! -e $SIMPLER_ROOT/build/pto-isa ]]; then
    git clone --local /home/sj/git/phase-c-20260908/pypto/runtime/build/pto-isa "$SIMPLER_ROOT/build/pto-isa"
fi
test "$(git -C "$SIMPLER_ROOT/build/pto-isa" rev-parse HEAD)" = "$(tr -d '[:space:]' < runtime/pto_isa.pin)"
python -m pip install 'scikit-build-core>=0.12.2,<2' 'pybind11<3' 'nanobind>=2.4,<3' 'setuptools>=77.0.3' ninja
python -m pip install --no-build-isolation --no-deps -e "$stage/ptoas" \
    --config-settings="build-dir=$stage/ptoas/build" \
    --config-settings="cmake.define.LLVM_DIR=$LLVM_BUILD_DIR/lib/cmake/llvm" \
    --config-settings="cmake.define.MLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir" \
    --config-settings="build.tool-args=-j$PYPTO_BUILD_JOBS"
"$PTOAS_ROOT/ptoas" --version
python -m pip install --no-build-isolation --no-deps -e . \
    --config-settings="build.tool-args=-j$PYPTO_BUILD_JOBS"
python -m pip install --no-build-isolation --no-deps -e runtime \
    --config-settings="build.tool-args=-j$PYPTO_BUILD_JOBS"
if rg -l '/home/sj/git/pypto/runtime' "$SIMPLER_ROOT/build/cache" -g CMakeCache.txt >/dev/null; then
    bash "$stage/rebuild_pinned_runtime.sh"
    exit 0
fi
python -m pip install --no-build-isolation --no-deps -e "$stage/pypto-serving"
python "$stage/verify_d0_environment.py"
