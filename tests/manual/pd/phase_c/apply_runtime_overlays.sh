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
patch --dry-run -d "$PYPTO_HOME" -p1 < "$script_dir/patches/pypto-owner-service.patch"
patch --dry-run -d "$PYPTO_HOME/runtime" -p1 < "$script_dir/patches/simpler-owner-service.patch"
patch -d "$PYPTO_HOME" -p1 < "$script_dir/patches/pypto-owner-service.patch"
patch -d "$PYPTO_HOME/runtime" -p1 < "$script_dir/patches/simpler-owner-service.patch"
