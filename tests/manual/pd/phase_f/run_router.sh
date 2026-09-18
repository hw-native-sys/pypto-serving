#!/usr/bin/env bash
# Phase F external Router launcher using the same PD JSON as P and D.
set -eo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "$script_dir/../phase_e/run_router.sh"
