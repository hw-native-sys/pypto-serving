#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Apply the Python-only owner-service hooks required by the Phase C production bridge.
set -euo pipefail
: "${PYPTO_HOME:?set PYPTO_HOME to the PyPTO source root}"
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
expected_pypto=${PYPTO_OWNER_SERVICE_BASE:-df4dc009368d4318a1a8692ef8a4c80ea5f62b43}
expected_simpler=${SIMPLER_OWNER_SERVICE_BASE:-22385d2b0c08b8f697fe8feede488c87cf83ef84}
actual_pypto=$(git -C "$PYPTO_HOME" rev-parse HEAD)
actual_simpler=$(git -C "$PYPTO_HOME/runtime" rev-parse HEAD)
if [[ $actual_pypto != "$expected_pypto" || $actual_simpler != "$expected_simpler" ]]; then
    printf 'owner-service overlay base mismatch: PyPTO=%s Simpler=%s\n' \
        "$actual_pypto" "$actual_simpler" >&2
    exit 2
fi

patch_args=(--forward --fuzz=0)
patch "${patch_args[@]}" --dry-run -d "$PYPTO_HOME" -p1 \
    < "$script_dir/patches/pypto-owner-service.patch"
patch "${patch_args[@]}" --dry-run -d "$PYPTO_HOME/runtime" -p1 \
    < "$script_dir/patches/simpler-owner-service.patch"
patch "${patch_args[@]}" -d "$PYPTO_HOME" -p1 \
    < "$script_dir/patches/pypto-owner-service.patch"
patch "${patch_args[@]}" -d "$PYPTO_HOME/runtime" -p1 \
    < "$script_dir/patches/simpler-owner-service.patch"
