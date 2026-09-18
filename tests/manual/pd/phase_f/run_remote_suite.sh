#!/usr/bin/env bash
# Run the Phase F suite under an independently-owned remote session.  The
# caller polls a durable exit-code file, so a transient SSH disconnect cannot
# orphan or accidentally cancel an otherwise valid system test.
set -uo pipefail

if (($# < 3)); then
    echo "usage: run_remote_suite.sh CONTROL_DIR DRIVER RUN_ARGS..." >&2
    exit 2
fi

control_dir=$1
shift
driver=$1
shift
if [[ $driver != run_suite.py && $driver != run_soak.py && $driver != run_non_pd_suite.py ]]; then
    echo "unsupported Phase F driver: $driver" >&2
    exit 2
fi
test ! -e "$control_dir" || {
    echo "suite control directory already exists: $control_dir" >&2
    exit 2
}
mkdir "$control_dir"
exec > "$control_dir/stdout.log" 2>&1

child_pid=""
cleanup() {
    if [[ -n $child_pid ]] && kill -0 "$child_pid" 2>/dev/null; then
        kill -TERM "$child_pid" 2>/dev/null || true
        wait "$child_pid" || true
    fi
}
trap cleanup EXIT INT TERM

python "tests/manual/pd/phase_f/$driver" "$@" &
child_pid=$!
echo $$ > "$control_dir/suite-service.pid"
echo "$child_pid" > "$control_dir/suite-python.pid"

set +e
wait "$child_pid"
status=$?
set -e
printf '%s\n' "$status" > "$control_dir/exit-code"
exit "$status"
