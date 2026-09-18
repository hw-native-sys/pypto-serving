#!/usr/bin/env bash
# Phase F Router launcher with FIFO active/queued admission bounds.
set -eo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PD_MAX_ACTIVE_HANDOFFS=${PD_MAX_ACTIVE_HANDOFFS:-4}
export PD_MAX_PENDING_HANDOFFS=${PD_MAX_PENDING_HANDOFFS:-8}
exec "$script_dir/../phase_e/run_router.sh"
