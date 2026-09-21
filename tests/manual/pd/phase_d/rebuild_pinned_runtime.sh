#!/usr/bin/env bash
# Rebuild frozen native libraries after editable mapping selects this checkout.
set -eo pipefail
stage=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$stage/env_pinned_stack.sh"
cd "$stage/pypto"
source .claude/skills/testing/load-env.sh
export CMAKE_BUILD_PARALLEL_LEVEL=${DSPARK_BUILD_JOBS:-$PYPTO_BUILD_JOBS}
python -c 'import simpler_setup; from pathlib import Path; import os; assert Path(simpler_setup.__file__).resolve().is_relative_to(Path(os.environ["SIMPLER_ROOT"])), simpler_setup.__file__'
if rg -l '/workspace/pypto/runtime' "$SIMPLER_ROOT/build/cache" -g CMakeCache.txt >/dev/null; then
    test ! -e runtime/build/cache-bootstrap
    test ! -e runtime/build/lib-bootstrap
    mv runtime/build/cache runtime/build/cache-bootstrap
    mv runtime/build/lib runtime/build/lib-bootstrap
fi
python -m pip install --no-build-isolation --no-deps -e runtime \
    --config-settings="build.tool-args=-j$CMAKE_BUILD_PARALLEL_LEVEL" \
    --config-settings="build.targets=build_package_a2a3"
python -m pip install --no-build-isolation --no-deps -e "$stage/pypto-serving"
python "$stage/verify_d0_environment.py"
