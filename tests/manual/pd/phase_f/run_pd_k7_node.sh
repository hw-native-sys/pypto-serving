#!/usr/bin/env bash
# Phase F K7 node launcher using the shared minimal PD JSON document.
set -eo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "$script_dir/../phase_e/run_pd_k7_node.sh"
