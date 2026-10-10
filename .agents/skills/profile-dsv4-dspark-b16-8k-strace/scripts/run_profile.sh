#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)

usage() {
  cat <<'EOF'
Usage: run_profile.sh --model-dir DIR [options]

Options:
  --artifact-dir DIR       Output directory (default: artifacts/<timestamp>)
  --use-compile-cache      Reuse kernels for the same locked source/configuration
  --python PATH            Python interpreter (default: python3/python on PATH)
  --devices IDS            Exactly sixteen assigned device IDs
  --served-model-name NAME API model name (default: dsv4-flash-dspark-w8a8)
  --run-id ID              Optional scheduler/task identifier
  -h, --help               Show this help

The workload is fixed: 16 requests, DP4/EP16/TP4,
8192 prompt tokens, 256 output tokens, DSpark k=7, greedy sampling, ignore_eos=1.
EOF
}

PYTHON_BIN=${PYPTO_PROFILE_PYTHON:-}
MODEL_DIR=${PYPTO_DSPARK_MODEL_DIR:-${PYPTO_DSV4_DSPARK_MODEL_DIR:-${PYPTO_DSV4_MODEL_DIR:-}}}
DEVICES=${PYPTO_PROFILE_DEVICES:-${TASK_DEVICE:-${ASCEND_RT_VISIBLE_DEVICES:-}}}
SERVED_MODEL_NAME=dsv4-flash-dspark-w8a8
RUN_ID=${PYPTO_PROFILE_RUN_ID:-${TASKQUEUE_TASK_ID:-unknown}}
ARTIFACT_DIR=
USE_COMPILE_CACHE=${PYPTO_USE_COMPILE_CACHE:-0}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --artifact-dir) ARTIFACT_DIR=$2; shift 2 ;;
    --use-compile-cache) USE_COMPILE_CACHE=1; shift ;;
    --python) PYTHON_BIN=$2; shift 2 ;;
    --model-dir) MODEL_DIR=$2; shift 2 ;;
    --devices) DEVICES=$2; shift 2 ;;
    --served-model-name) SERVED_MODEL_NAME=$2; shift 2 ;;
    --run-id) RUN_ID=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done

if [[ -z ${MODEL_DIR} ]]; then
  echo "error: --model-dir or PYPTO_DSPARK_MODEL_DIR is required" >&2
  exit 2
fi
if [[ -z ${DEVICES} ]]; then
  echo "error: --devices or an allocation device environment variable is required" >&2
  exit 2
fi
if [[ -z ${PYTHON_BIN} ]]; then
  PYTHON_BIN=$(command -v python3 || command -v python)
fi
if [[ -z ${ARTIFACT_DIR} ]]; then
  ARTIFACT_DIR="${REPO_ROOT}/artifacts/dsv4-dspark-b16-8k-profile-$(date +%Y%m%d-%H%M%S)"
fi

ARTIFACT_DIR=$(realpath -m "${ARTIFACT_DIR}")
MODEL_DIR=$(realpath -m "${MODEL_DIR}")
case "${USE_COMPILE_CACHE,,}" in
  1|true|yes|on) USE_COMPILE_CACHE=1 ;;
  0|false|no|off|"") USE_COMPILE_CACHE=0 ;;
  *) echo "error: PYPTO_USE_COMPILE_CACHE must be boolean" >&2; exit 2 ;;
esac

mkdir -p "${ARTIFACT_DIR}"
trap 'printf "Artifacts: %s\n" "${ARTIFACT_DIR}"' EXIT

# Keep the successful DSpark timeouts while enabling Host-only runtime spans.
export PYPTO_RUNTIME_LOG=v9
export SIMPLER_DEVICE_STRACE_ENABLE=0
export PYPTO_DISABLE_DEVICE_LOG=1
export SIMPLER_OP_EXECUTE_TIMEOUT_US=${SIMPLER_OP_EXECUTE_TIMEOUT_US:-400000000}
export SIMPLER_STREAM_SYNC_TIMEOUT_MS=${SIMPLER_STREAM_SYNC_TIMEOUT_MS:-440000}
export SIMPLER_SCHEDULER_TIMEOUT_MS=${SIMPLER_SCHEDULER_TIMEOUT_MS:-320000}
export NO_PROXY=${NO_PROXY:-127.0.0.1,localhost}
export no_proxy=${no_proxy:-127.0.0.1,localhost}
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PTOAS_ROOT=${PTOAS_ROOT:-${REPO_ROOT}/.venv}

PROFILE_ARGS=(
  --artifact-dir "${ARTIFACT_DIR}"
  --model-dir "${MODEL_DIR}"
  --devices "${DEVICES}"
  --served-model-name "${SERVED_MODEL_NAME}"
)
if [[ ${USE_COMPILE_CACHE} == 1 ]]; then
  PROFILE_ARGS+=(--use-compile-cache)
fi

cd "${REPO_ROOT}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/run_profile.py" "${PROFILE_ARGS[@]}" \
  2>&1 | tee "${ARTIFACT_DIR}/server.log" "${ARTIFACT_DIR}/run.log"
"${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_profile.py" "${ARTIFACT_DIR}" \
  --run-id "${RUN_ID}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/render_16lane.py" \
  "${ARTIFACT_DIR}/simpler-swimlane.json" \
  "${ARTIFACT_DIR}/server.log" \
  "${ARTIFACT_DIR}/serving-strace-swimlane.json" \
  --serving-trace "${ARTIFACT_DIR}/serving-trace/trace.json" \
  --profile-summary "${ARTIFACT_DIR}/profile-summary.json"
"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_artifact.py" "${ARTIFACT_DIR}"
