#!/usr/bin/env bash
# Submit a prompt that must cross the configured 128-token Prefill chunk boundary.
set -eo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 ROUTER_URL EVIDENCE_DIR" >&2
    exit 2
fi
router_url=${1%/}
evidence_dir=$2
test ! -e "$evidence_dir"
mkdir -p "$evidence_dir"

prompt=""
for ((index = 0; index < 24; index++)); do
    prompt+="紫禁城见证了明清两代的历史，也保存着丰富的建筑与文化记忆。"
done

printf \
    '{"model":"dsv4-flash-dspark-w8a8","prompt":"%s","max_tokens":128,"temperature":0.0,"top_p":1.0,"stream":false}' \
    "$prompt" > "$evidence_dir/request.json"

curl --fail-with-body --max-time 1800 --silent --show-error \
    -H 'Content-Type: application/json' \
    --data-binary "@$evidence_dir/request.json" \
    "$router_url/v1/completions" > "$evidence_dir/response.json"

python - "$evidence_dir/response.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    response = json.load(stream)
assert response["choices"][0]["text"], response
assert response["choices"][0]["finish_reason"] == "length", response
assert response["usage"]["prompt_tokens"] > 128, response
assert response["usage"]["completion_tokens"] == 128, response
assert response["usage"]["total_tokens"] == (
    response["usage"]["prompt_tokens"] + 128
), response
print(json.dumps(response["usage"], ensure_ascii=False, sort_keys=True))
PY
