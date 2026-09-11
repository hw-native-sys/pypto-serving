#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
artifact_dir=$(realpath "$1")
cd "$repo_root"
source activate.sh

: "${TASK_DEVICE:?task-submit must assign eight devices}"
mkdir -p "$artifact_dir/cann"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYPTO_RUNTIME_LOG=error
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_PROCESS_LOG_PATH="$artifact_dir/cann"

{
  date -u '+started=%Y-%m-%dT%H:%M:%SZ'
  echo "devices=$TASK_DEVICE"
  echo "python=$(command -v python)"
  echo "cann_logs=$ASCEND_PROCESS_LOG_PATH"
  npu-smi info || true
} > "$artifact_dir/task-metadata.txt"

set +e
python .github/scripts/diagnose-sdma-init.py \
  --output "$artifact_dir/sdma-init" 2>&1 | tee "$artifact_dir/sdma-init.log"
test_rc=${PIPESTATUS[0]}
set -e
echo "$test_rc" > "$artifact_dir/diagnostic-exit-code.txt"
exit "$test_rc"
