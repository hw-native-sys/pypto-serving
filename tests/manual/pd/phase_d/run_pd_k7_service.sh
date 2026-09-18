#!/usr/bin/env bash
# Start one side of the fixed 1P1D K7 smoke without touching unrelated processes.
set -eo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../../../.." && pwd)
stage=$(cd "$repo_root/.." && pwd)
source "$stage/env_pinned_stack.sh"

required=(
    PD_ROLE
    PD_NODE_ID
    PD_PEER_NODE_ID
    PD_RUN_ID
    PD_LOCAL_HOST
    PD_PEER_HOST
    PD_CONTROL_PORT
    PD_API_PORT
    PD_EVIDENCE_NAME
)
for name in "${required[@]}"; do
    if [[ -z ${!name:-} ]]; then
        echo "$name must be set" >&2
        exit 2
    fi
done
if [[ $PD_ROLE != prefill && $PD_ROLE != decode ]]; then
    echo "PD_ROLE must be prefill or decode" >&2
    exit 2
fi
if [[ ! $PD_EVIDENCE_NAME =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo "PD_EVIDENCE_NAME contains unsupported characters" >&2
    exit 2
fi
if [[ ${#PYPTO_PD_AUTH_SECRET} -lt 16 ]]; then
    echo "PYPTO_PD_AUTH_SECRET must contain at least 16 characters" >&2
    exit 2
fi

run_dir="$stage/$PD_EVIDENCE_NAME"
test ! -e "$run_dir"
mkdir "$run_dir"
exec > "$run_dir/service.log" 2>&1

session_id=$(ps -o sid= -p $$ | tr -d ' ')
if [[ $session_id != $$ ]]; then
    echo "Phase D launcher requires its own process session" >&2
    exit 2
fi

npu-smi info > "$run_dir/preflight-npu.log"
if awk '/Process id/ {table=1; next} table && $2 ~ /^[0-9]+$/ {busy=1} END {exit !busy}' \
    "$run_dir/preflight-npu.log"; then
    echo "NPU process table occupied; no service started." >&2
    exit 3
fi

export PYPTO_DSV4_DSPARK_MODEL_DIR=/models/dsv4-flash-0731-dspark-w8a8
export TASK_DEVICE=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export ASCEND_PROCESS_LOG_PATH="$run_dir/ascend"
mkdir "$ASCEND_PROCESS_LOG_PATH"

child_pid=""
monitor_pid=""
owned_pids=()

collect_owned_session_children() {
    owned_pids=()
    local proc_stat pid stat rest state ppid pgrp sid
    for proc_stat in /proc/[0-9]*/stat; do
        pid=${proc_stat#/proc/}
        pid=${pid%/stat}
        [[ $pid == $$ ]] && continue
        IFS= read -r stat 2>/dev/null < "$proc_stat" || continue
        rest=${stat##*) }
        read -r state ppid pgrp sid _ <<< "$rest"
        if [[ $sid == $session_id ]]; then
            owned_pids+=("$pid")
        fi
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
    if [[ -n $monitor_pid ]] && kill -0 "$monitor_pid" 2>/dev/null; then
        kill -TERM "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" || true
    fi
    if [[ -n $child_pid ]] && kill -0 "$child_pid" 2>/dev/null; then
        kill -TERM "$child_pid" 2>/dev/null || true
        wait "$child_pid" || true
    fi
    # Simpler chip children are re-parented by the container but retain this
    # launcher's unique session id.  Give them a bounded graceful window, then
    # signal only the exact PIDs still proven to belong to this run/session.
    if ! wait_owned_session_children 60; then
        echo "Owned chip children exceeded graceful exit budget: ${owned_pids[*]}" >&2
        for pid in "${owned_pids[@]}"; do
            kill -TERM "$pid" 2>/dev/null || true
        done
        if ! wait_owned_session_children 30; then
            echo "Owned chip children exceeded TERM budget: ${owned_pids[*]}" >&2
            for pid in "${owned_pids[@]}"; do
                kill -KILL "$pid" 2>/dev/null || true
            done
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
    --backend npu \
    --platform a2a3 \
    --devices "$TASK_DEVICE" \
    --dp 4 \
    --ep 16 \
    --tp 4 \
    --block-size 32 \
    --max-model-len 1024 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 8192 \
    --long-prefill-token-threshold 128 \
    --speculative-config '{"method":"dspark","num_speculative_tokens":7}' \
    --no-enable-prefix-caching \
    --ring-heap 2147483648,2147483648,4294967296,8589934592 \
    --use-compile-cache \
    --pd-role "$PD_ROLE" \
    --pd-node-id "$PD_NODE_ID" \
    --pd-peer-node-id "$PD_PEER_NODE_ID" \
    --pd-run-id "$PD_RUN_ID" \
    --pd-control-host "$PD_LOCAL_HOST" \
    --pd-control-port "$PD_CONTROL_PORT" \
    --pd-peer-host "$PD_PEER_HOST" \
    --pd-transfer-hostname "$PD_LOCAL_HOST" \
    --pd-model-revision dsv4-flash-dspark-w8a8-phase-d \
    --pd-generation "${PD_GENERATION:-1}" \
    --pd-route-epoch "${PD_ROUTE_EPOCH:-1}" \
    --pd-control-incarnation "${PD_CONTROL_INCARNATION:-1}" \
    --pd-connect-timeout-seconds 1800 \
    --pd-request-timeout-seconds 600 \
    --pd-max-pending-handoffs 8 \
    --pd-max-transfer-attempts 2 \
    --pd-journal-path "$run_dir/pd-journal.jsonl" \
    --port "$PD_API_PORT" \
    --show-startup-logs &
child_pid=$!
echo $$ > "$run_dir/service.pid"
echo "$child_pid" > "$run_dir/python.pid"

# This monitor owns only the child PID created above.  A 503 after startup is
# the Serving signal that the PD control/data state became unsafe to reuse.
# Exit the local runtime normally; a fresh run/incarnation is started only
# after an external controller has confirmed both hosts retired.
(
    while kill -0 "$child_pid" 2>/dev/null; do
        health_code=$(curl --max-time 2 --silent --output /dev/null \
            --write-out '%{http_code}' "http://127.0.0.1:$PD_API_PORT/health" || true)
        if [[ $health_code == 503 ]]; then
            echo "PD health became fail-closed; stopping owned runtime $child_pid" >&2
            kill -TERM "$child_pid" 2>/dev/null || true
            exit 0
        fi
        sleep 2
    done
) &
monitor_pid=$!
wait "$child_pid"
