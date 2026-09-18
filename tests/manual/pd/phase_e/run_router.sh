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

required=(PD_CONFIG ROUTER_PORT ROUTER_EVIDENCE_NAME)
for name in "${required[@]}"; do
    if [[ -z ${!name:-} ]]; then
        echo "$name must be set" >&2
        exit 2
    fi
done
if [[ $PD_CONFIG != /* || ! -f $PD_CONFIG ]]; then
    echo "PD_CONFIG must identify the shared absolute JSON config" >&2
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
    --pd-config "$PD_CONFIG" \
    --host 0.0.0.0 --port "$ROUTER_PORT" &
child_pid=$!
echo $$ > "$run_dir/router-service.pid"
echo "$child_pid" > "$run_dir/router-python.pid"
wait "$child_pid"
