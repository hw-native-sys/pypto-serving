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
  --devices IDS            Exactly eight assigned device IDs
  --served-model-name NAME API model name (default: dsv4-flash-w8a8)
  --run-id ID              Optional scheduler/task identifier
  -h, --help               Show this help

The workload is fixed: GBS32, DP8, per-rank batch4, 32 distinct attraction prompts,
64 tokens per prompt, 256 output tokens, MTP k=1,
temperature=0, top_p=1, top_k=0, ignore_eos=1.
EOF
}

PYTHON_BIN=${PYPTO_PROFILE_PYTHON:-}
MODEL_DIR=${PYPTO_DSV4_MODEL_DIR:-}
DEVICES=${PYPTO_PROFILE_DEVICES:-${TASK_DEVICE:-${ASCEND_RT_VISIBLE_DEVICES:-}}}
SERVED_MODEL_NAME=dsv4-flash-w8a8
RUN_ID=${PYPTO_PROFILE_RUN_ID:-unknown}
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
  echo "error: --model-dir or PYPTO_DSV4_MODEL_DIR is required" >&2
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
  ARTIFACT_DIR="${REPO_ROOT}/artifacts/dsv4-gbs32-profile-$(date +%Y%m%d-%H%M%S)"
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

# Match the skill trace contract: verbose serving spans plus Host STRACE only.
export PYPTO_RUNTIME_LOG=v9
export SIMPLER_DEVICE_STRACE_ENABLE=0
export PYPTO_DISABLE_DEVICE_LOG=1
export NO_PROXY=${NO_PROXY:-127.0.0.1,localhost}
export no_proxy=${no_proxy:-127.0.0.1,localhost}
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

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
"${PYTHON_BIN}" "${SCRIPT_DIR}/render_8lane.py" \
  "${ARTIFACT_DIR}/simpler-swimlane.json" \
  "${ARTIFACT_DIR}/server.log" \
  "${ARTIFACT_DIR}/serving-strace-swimlane.json" \
  --serving-trace "${ARTIFACT_DIR}/serving-trace/trace.json" \
  --profile-summary "${ARTIFACT_DIR}/profile-summary.json"
"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_artifact.py" "${ARTIFACT_DIR}"
