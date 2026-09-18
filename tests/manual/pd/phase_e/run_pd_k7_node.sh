#!/usr/bin/env bash
# Start one independently-ready external-router P or D node.
set -eo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../../../.." && pwd)
stage=$(cd "$repo_root/.." && pwd)
stack_env=${PYPTO_STACK_ENV_FILE:-"$stage/env_pinned_stack.sh"}
if [[ ! -f $stack_env ]]; then
    echo "PYPTO stack environment file does not exist: $stack_env" >&2
    exit 2
fi
source "$stack_env"

# Mooncake AscendDirect must use the device RoCE path when it coexists with
# Serving's compute HCCL domain.  Phase B froze this as a provider requirement;
# keeping it in the launcher prevents a generic workspace env from silently
# falling back to the conflicting non-RoCE channel path.
export HCCL_INTRA_ROCE_ENABLE=${HCCL_INTRA_ROCE_ENABLE:-1}
if [[ $HCCL_INTRA_ROCE_ENABLE != 1 ]]; then
    echo "HCCL_INTRA_ROCE_ENABLE must be 1 for Mooncake AscendDirect PD" >&2
    exit 2
fi

required=(PD_ROLE PD_CONFIG PD_API_PORT PD_EVIDENCE_NAME)
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
evidence_root=${PD_EVIDENCE_ROOT:-$stage}
if [[ $evidence_root != /* || $evidence_root == / || ! -d $evidence_root ]]; then
    echo "PD_EVIDENCE_ROOT must be an existing explicit absolute directory" >&2
    exit 2
fi
if [[ $PD_CONFIG != /* || ! -f $PD_CONFIG ]]; then
    echo "PD_CONFIG must identify the shared absolute JSON config" >&2
    exit 2
fi
max_model_len=${PYPTO_MAX_MODEL_LEN:-1024}
if [[ ! $max_model_len =~ ^[1-9][0-9]*$ ]]; then
    echo "PYPTO_MAX_MODEL_LEN must be a positive integer" >&2
    exit 2
fi

run_dir="$evidence_root/$PD_EVIDENCE_NAME"
test ! -e "$run_dir"
mkdir "$run_dir"
exec > "$run_dir/service.log" 2>&1
cd "$repo_root"

{
    printf 'stack_env=%s\n' "$stack_env"
    sha256sum "$stack_env" || true
    printf 'serving_head='
    git -C "$repo_root" rev-parse HEAD || true
    printf 'pypto_lib_head='
    git -C "$PYPTO_LIB_HOME" rev-parse HEAD || true
    printf 'pypto_head='
    git -C "$PYPTO_HOME" rev-parse HEAD || true
    printf 'simpler_runtime_head='
    git -C "$PYPTO_HOME/runtime" rev-parse HEAD || true
    printf 'pto_isa_env_root=%s\n' "$PTO_ISA_ROOT"
    printf 'pto_isa_env_head='
    git -C "$PTO_ISA_ROOT" rev-parse HEAD || true
    managed_pto_isa_root="$PYPTO_HOME/runtime/build/pto-isa"
    printf 'pto_isa_managed_root=%s\n' "$managed_pto_isa_root"
    printf 'pto_isa_managed_head='
    git -C "$managed_pto_isa_root" rev-parse HEAD || true
    printf 'python=%s\n' "$(command -v python)"
    printf 'ptoas=%s\n' "$(command -v ptoas)"
    ptoas --version || true
    printf 'HCCL_INTRA_ROCE_ENABLE=%s\n' "$HCCL_INTRA_ROCE_ENABLE"
    printf 'max_model_len=%s\n' "$max_model_len"
    python - <<'PY'
import mooncake.engine
import pypto
import pypto_serving
import simpler

for name, module in (
    ("pypto", pypto),
    ("simpler", simpler),
    ("pypto_serving", pypto_serving),
    ("mooncake", mooncake.engine),
):
    print(f"{name}_version={getattr(module, '__version__', '<unset>')}")
    print(f"{name}_file={module.__file__}")
try:
    import _task_interface
except Exception as exc:
    print(f"task_interface_error={type(exc).__name__}: {exc}")
else:
    print(f"task_interface_file={_task_interface.__file__}")
PY
    sha256sum \
        "$MOONCAKE_LIB_DIR/libtransfer_engine.so" \
        "$MOONCAKE_LIB_DIR/libmooncake_common.so" \
        "$SIMPLER_BUILD_LIB_DIR/libsimple_runtime.so" 2>/dev/null || true
} > "$run_dir/version-lock.txt" 2>&1

session_id=$(ps -o sid= -p $$ | tr -d ' ')
if [[ $session_id != $$ ]]; then
    echo "Phase E launcher requires its own process session" >&2
    exit 2
fi

npu-smi info > "$run_dir/preflight-npu.log"
if awk '/Process id/ {table=1; next} table && $2 ~ /^[0-9]+$/ {busy=1} END {exit !busy}' \
    "$run_dir/preflight-npu.log"; then
    echo "NPU process table occupied; no service started." >&2
    exit 3
fi

export PYPTO_DSV4_DSPARK_MODEL_DIR=${PYPTO_DSV4_DSPARK_MODEL_DIR:-/models/dsv4-flash-0731-dspark-w8a8}
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
    if [[ -n $child_pid ]] && kill -0 "$child_pid" 2>/dev/null; then
        kill -TERM "$child_pid" 2>/dev/null || true
        wait "$child_pid" || true
    fi
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

pd_profile_args=()
if [[ ${PD_PROFILE:-false} == true ]]; then
    pd_profile_args=(
        --profile
        --profile-output "$run_dir/profile"
        --profile-level "${PD_PROFILE_LEVEL:-e2e,kernel}"
    )
fi
pd_node_args=()
if [[ -n ${PD_NODE_ID:-} ]]; then
    pd_node_args=(--pd-node-id "$PD_NODE_ID")
fi
python -m pypto_serving.cli \
    --model "$PYPTO_DSV4_DSPARK_MODEL_DIR" \
    --served-model-name dsv4-flash-dspark-w8a8 \
    --backend npu --platform a2a3 \
    --devices "$TASK_DEVICE" --dp 4 --ep 16 --tp 4 \
    --block-size 32 --max-model-len "$max_model_len" --max-num-seqs 8 \
    --max-num-batched-tokens 8192 --long-prefill-token-threshold 128 \
    --speculative-config '{"method":"dspark","num_speculative_tokens":7}' \
    --no-enable-prefix-caching \
    --ring-heap 2147483648,2147483648,4294967296,8589934592 \
    --use-compile-cache \
    --pd-role "$PD_ROLE" \
    --pd-config "$PD_CONFIG" \
    "${pd_node_args[@]}" \
    "${pd_profile_args[@]}" \
    --host 0.0.0.0 --port "$PD_API_PORT" --show-startup-logs &
child_pid=$!
echo $$ > "$run_dir/service.pid"
echo "$child_pid" > "$run_dir/python.pid"
wait "$child_pid"
