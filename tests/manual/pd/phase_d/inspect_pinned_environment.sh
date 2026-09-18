#!/usr/bin/env bash
# Compatibility entrypoint retained for earlier Phase D instructions.
set -eo pipefail
stage=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$stage/inspect_d0_environment.sh"
