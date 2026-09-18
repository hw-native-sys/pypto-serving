#!/usr/bin/env bash
# Phase F node launcher: preserve the Phase E safety envelope and enable the
# bounded request-aware pipeline through explicit environment defaults.
set -eo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PD_MODEL_REVISION=${PD_MODEL_REVISION:-dsv4-flash-dspark-w8a8-phase-f}
export PD_MAX_ACTIVE_HANDOFFS=${PD_MAX_ACTIVE_HANDOFFS:-4}
export PD_MAX_PENDING_HANDOFFS=${PD_MAX_PENDING_HANDOFFS:-8}
export PD_MAX_INFLIGHT_TRANSFER_BYTES=${PD_MAX_INFLIGHT_TRANSFER_BYTES:-1073741824}
export PD_TRANSFER_POLL_INTERVAL_SECONDS=${PD_TRANSFER_POLL_INTERVAL_SECONDS:-0.005}
export PD_ENABLE_CHUNK_OVERLAP=${PD_ENABLE_CHUNK_OVERLAP:-true}
exec "$script_dir/../phase_e/run_pd_k7_node.sh"
