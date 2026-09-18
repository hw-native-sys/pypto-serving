#!/usr/bin/env bash
# Submit the K7-only 128-token closure case to the Router public endpoint.
set -eo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 ROUTER_URL EVIDENCE_DIR" >&2
    exit 2
fi
router_url=${1%/}
evidence_dir=$2
test ! -e "$evidence_dir"
mkdir -p "$evidence_dir"

curl --fail-with-body --max-time 1800 --silent --show-error \
    -H 'Content-Type: application/json' \
    -d '{"model":"dsv4-flash-dspark-w8a8","prompt":"紫禁城","max_tokens":128,"temperature":0.0,"top_p":1.0,"stream":false}' \
    "$router_url/v1/completions" > "$evidence_dir/response.json"

python - "$evidence_dir/response.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    response = json.load(stream)
assert response["choices"][0]["text"], response
assert response["choices"][0]["finish_reason"] == "length", response
assert response["usage"]["completion_tokens"] == 128, response
assert response["usage"]["total_tokens"] == (
    response["usage"]["prompt_tokens"] + 128
), response
print(json.dumps(response["usage"], ensure_ascii=False, sort_keys=True))
PY
