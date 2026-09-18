#!/usr/bin/env bash
# Run the canonical DSpark K0/K7 guard in the D0 frozen environment.
set -eo pipefail
stage=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$stage/env_pinned_stack.sh"
run_name=${DSPARK_RUN_NAME:-service-d0-01}
test_target=${DSPARK_PYTEST_TARGET:-tests/test_deepseek_dspark_accuracy.py}
[[ $run_name =~ ^[a-zA-Z0-9_-]+$ ]]
[[ $test_target != *$'\n'* ]]
run_dir="$stage/$run_name"
test ! -e "$run_dir"
mkdir "$run_dir"
python "$stage/verify_d0_environment.py" > "$run_dir/environment.json"
cd "$PYPTO_HOME"
source .claude/skills/testing/load-env.sh
bash runtime/.claude/skills/onboard-arch-precheck/check.sh a2a3
npu-smi info > "$run_dir/preflight-npu.log"
if awk '/Process id/ {table=1; next} table && $2 ~ /^[0-9]+$/ {busy=1} END {exit !busy}' "$run_dir/preflight-npu.log"; then
    echo 'NPU process table occupied; no service started.' >&2
    exit 2
fi
trap 'npu-smi info > "$run_dir/postflight-npu.log"' EXIT
mkdir "$run_dir/ascend"
export ASCEND_PROCESS_LOG_PATH="$run_dir/ascend"
export PYPTO_DSV4_DSPARK_MODEL_DIR=/models/dsv4-flash-0731-dspark-w8a8
export TASK_DEVICE=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
cd "$stage/pypto-serving"
pytest_args=("$test_target" -q -s --basetemp "$run_dir/http")
if command -v task-submit >/dev/null 2>&1; then
    printf -v quoted_pytest_args '%q ' "${pytest_args[@]}"
    task-submit --device "$TASK_DEVICE" --max-time 3000 --run \
        "python -m pytest $quoted_pytest_args"
else
    echo 'task-submit unavailable; using the idle process-table preflight without a device lock.'
    python -m pytest "${pytest_args[@]}"
fi
