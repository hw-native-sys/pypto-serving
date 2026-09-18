#!/usr/bin/env bash
# Print effective D0 resolution without allocating an NPU device.
set -eo pipefail
stage=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$stage/env_pinned_stack.sh"

variables=(
    DSPARK_STAGE ASCEND_HOME_PATH ASCEND_AICPU_PATH ASCEND_TOOLKIT_HOME
    ASCEND_TOOLKIT_LATEST_HOME TOOLCHAIN_HOME CMAKE_PREFIX_PATH CANN_ROOT
    MOONCAKE_ROOT MOONCAKE_LIB_DIR MOONCAKE_LIB64_DIR
    PYPTO_HOME SIMPLER_ROOT PYPTO_LIB_ROOT PTOAS_ROOT PTO_ISA_ROOT
    SOC_VERSION PYTHONNOUSERSITE PYTHONPATH LD_LIBRARY_PATH PATH
)
for name in "${variables[@]}"; do
    printf '%s=%s\n' "$name" "${!name-<unset>}"
done
printf 'python=%s\n' "$(command -v python)"
printf 'ptoas=%s\n' "$(command -v ptoas)"
printf 'ptoas_version=%s\n' "$(ptoas --version)"

python - <<'PY'
import sys

import mooncake.engine
import pypto
import pypto_serving
import ptoas
import simpler
import torch

for name, module in (
    ("torch", torch),
    ("pypto", pypto),
    ("simpler", simpler),
    ("ptoas", ptoas),
    ("serving", pypto_serving),
    ("mooncake", mooncake.engine),
):
    print(f"{name}_package_version={getattr(module, '__version__', '<unset>')}")
    print(f"{name}_file={module.__file__}")
print("sys_path=" + "|".join(sys.path))
PY
