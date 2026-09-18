#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
d0_stage=${DSPARK_STAGE:-/home/sj/git/phase-d-pypto783-lib216-20260910}
source "$d0_stage/env_pinned_stack.sh"
export ASCEND_PROCESS_LOG_PATH="${PHASE_D_COEXIST_ASCEND_DIR:-$d0_stage/ascend}"
export HCCL_INTRA_ROCE_ENABLE=1
cd "$PYPTO_HOME"
source .claude/skills/testing/load-env.sh
cd "$script_dir"
exec python -u coexist_pair.py "$@"
