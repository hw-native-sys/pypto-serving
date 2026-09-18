#!/usr/bin/env bash
# Start the CPU-only external Router after both protected node APIs are ready.
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

required=(PD_RUN_ID ROUTER_P_URL ROUTER_D_URL ROUTER_PORT ROUTER_EVIDENCE_NAME)
for name in "${required[@]}"; do
    if [[ -z ${!name:-} ]]; then
        echo "$name must be set" >&2
        exit 2
    fi
done
if [[ ${#PYPTO_PD_AUTH_SECRET} -lt 16 || ${#PYPTO_PD_ROUTER_SECRET} -lt 16 ]]; then
    echo "both PD secrets must contain at least 16 characters" >&2
    exit 2
fi
if [[ ! $ROUTER_EVIDENCE_NAME =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo "ROUTER_EVIDENCE_NAME contains unsupported characters" >&2
    exit 2
fi
evidence_root=${PD_EVIDENCE_ROOT:-$stage}
if [[ $evidence_root != /* || $evidence_root == / || ! -d $evidence_root ]]; then
    echo "PD_EVIDENCE_ROOT must be an existing explicit absolute directory" >&2
    exit 2
fi

run_dir="$evidence_root/$ROUTER_EVIDENCE_NAME"
test ! -e "$run_dir"
mkdir "$run_dir"
exec > "$run_dir/router.log" 2>&1

child_pid=""
cleanup() {
    if [[ -n $child_pid ]] && kill -0 "$child_pid" 2>/dev/null; then
        kill -TERM "$child_pid" 2>/dev/null || true
        wait "$child_pid" || true
    fi
}
trap cleanup EXIT INT TERM

cd "$repo_root"
python -m pypto_serving.router \
    --prefill-url "$ROUTER_P_URL" \
    --decode-url "$ROUTER_D_URL" \
    --run-id "$PD_RUN_ID" \
    --model-id dsv4-flash-dspark-w8a8 \
    --host 0.0.0.0 --port "$ROUTER_PORT" \
    --route-epoch "${PD_ROUTE_EPOCH:-1}" \
    --control-incarnation "${PD_CONTROL_INCARNATION:-1}" \
    --request-timeout-seconds 1800 \
    --ticket-ttl-seconds 1200 \
    --max-active-handoffs "${PD_MAX_ACTIVE_HANDOFFS:-1}" \
    --max-pending-handoffs "${PD_MAX_PENDING_HANDOFFS:-8}" \
    --journal-path "$run_dir/router-journal.jsonl" &
child_pid=$!
echo $$ > "$run_dir/router-service.pid"
echo "$child_pid" > "$run_dir/router-python.pid"
wait "$child_pid"
