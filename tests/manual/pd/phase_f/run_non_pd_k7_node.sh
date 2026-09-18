#!/usr/bin/env bash
# Exact-session non-PD K7 baseline for the Phase F correctness comparison.
set -eo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../../../.." && pwd)
stage=$(cd "$repo_root/.." && pwd)
source "$stage/env_pinned_stack.sh"

: "${BASELINE_RUN_ID:?BASELINE_RUN_ID must be set}"
: "${BASELINE_API_PORT:?BASELINE_API_PORT must be set}"
: "${BASELINE_EVIDENCE_NAME:?BASELINE_EVIDENCE_NAME must be set}"
[[ $BASELINE_RUN_ID =~ ^[a-zA-Z0-9_.-]+$ ]]
[[ $BASELINE_EVIDENCE_NAME =~ ^[a-zA-Z0-9_-]+$ ]]

run_dir="$stage/$BASELINE_EVIDENCE_NAME"
test ! -e "$run_dir"
mkdir "$run_dir"
exec > "$run_dir/service.log" 2>&1

session_id=$(ps -o sid= -p $$ | tr -d ' ')
if [[ $session_id != $$ ]]; then
    echo "Phase F baseline launcher requires its own process session" >&2
    exit 2
fi
npu-smi info > "$run_dir/preflight-npu.log"
if awk '/Process id/ {table=1; next} table && $2 ~ /^[0-9]+$/ {busy=1} END {exit !busy}' \
    "$run_dir/preflight-npu.log"; then
    echo "NPU process table occupied; no baseline started." >&2
    exit 3
fi

export PYPTO_DSV4_DSPARK_MODEL_DIR=/models/dsv4-flash-0731-dspark-w8a8
export TASK_DEVICE=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export ASCEND_PROCESS_LOG_PATH="$run_dir/ascend"
export PYPTO_PROG_BUILD_DIR=${PYPTO_PROG_BUILD_DIR:-$repo_root/build_output}
mkdir "$ASCEND_PROCESS_LOG_PATH"

child_pid=""
owned_pids=()
collect_owned_session_children() {
    owned_pids=()
    local proc_stat pid stat rest state ppid pgrp sid
    for proc_stat in /proc/[0-9]*/stat; do
        pid=${proc_stat#/proc/}; pid=${pid%/stat}
        [[ $pid == $$ ]] && continue
        IFS= read -r stat 2>/dev/null < "$proc_stat" || continue
        rest=${stat##*) }
        read -r state ppid pgrp sid _ <<< "$rest"
        [[ $sid == $session_id ]] && owned_pids+=("$pid")
    done
}
wait_owned_session_children() {
    local deadline=$((SECONDS + $1))
    while (( SECONDS < deadline )); do
        collect_owned_session_children
        ((${#owned_pids[@]} == 0)) && return 0
        sleep 1
    done
    collect_owned_session_children
    ((${#owned_pids[@]} == 0))
}
cleanup() {
    if [[ -n $child_pid ]] && kill -0 "$child_pid" 2>/dev/null; then
        kill -TERM "$child_pid" 2>/dev/null || true
        wait "$child_pid" || true
    fi
    if ! wait_owned_session_children 60; then
        echo "Owned chip children exceeded graceful exit budget: ${owned_pids[*]}" >&2
        for pid in "${owned_pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
        if ! wait_owned_session_children 30; then
            echo "Owned chip children exceeded TERM budget: ${owned_pids[*]}" >&2
            for pid in "${owned_pids[@]}"; do kill -KILL "$pid" 2>/dev/null || true; done
            wait_owned_session_children 10 || true
        fi
    fi
    npu-smi info > "$run_dir/postflight-npu.log" || true
}
trap cleanup EXIT INT TERM

cd "$repo_root"
python -m pypto_serving.cli \
    --model "$PYPTO_DSV4_DSPARK_MODEL_DIR" \
    --served-model-name dsv4-flash-dspark-w8a8 \
    --backend npu --platform a2a3 \
    --devices "$TASK_DEVICE" --dp 4 --ep 16 --tp 4 \
    --block-size 32 --max-model-len 1024 --max-num-seqs 8 \
    --max-num-batched-tokens 8192 --long-prefill-token-threshold 128 \
    --speculative-config '{"method":"dspark","num_speculative_tokens":7}' \
    --no-enable-prefix-caching \
    --ring-heap 2147483648,2147483648,4294967296,8589934592 \
    --use-compile-cache \
    --host 0.0.0.0 --port "$BASELINE_API_PORT" --show-startup-logs &
child_pid=$!
echo $$ > "$run_dir/service.pid"
echo "$child_pid" > "$run_dir/python.pid"
wait "$child_pid"
