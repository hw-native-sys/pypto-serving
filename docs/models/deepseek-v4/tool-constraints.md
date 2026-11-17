# DeepSeek V4 DSpark constrained tool deployment

This guide deploys structural tool-call constraints for DeepSeek V4 Flash DSpark with K7 decoding. The tool names and JSON schemas come from each `/v1/chat/completions` request, not from a server-side allowlist or a particular agent client.

For the common request, tool-result, and streaming behavior, see [Tool Calling](../../user-guide/tool-calling.md). The model's DSML limits are in [Function Tool Format](index.md#function-tool-format).

## Compatible components

- Use a DeepSeek V4 Flash DSpark W8A8 checkpoint and the 16-device `--dp 4 --ep 16 --tp 4` topology described in [DSpark deployment](dspark.md).
- Deploy PyPTO Serving and PyPTO-Lib as a matched pair. Their kernel argument layouts must agree; updating only one repository will not work.
- Install `xgrammar==0.2.7` into the Python environment that runs Serving. This is the version validated with the DeepSeek V4 structural-tag grammar and the checkpoint tokenizer. XGrammar is a host-side dependency; PyPTO-Lib does not import it. See the [XGrammar package](https://pypi.org/project/xgrammar/0.2.7/).
- DSpark currently supports greedy sampling only. Use `temperature: 0`; non-greedy requests are rejected rather than silently changing sampling behavior.

```bash
python -m pip install 'xgrammar==0.2.7'
python -m pip show xgrammar
```

Serving uses PyPTO's validated persistent JIT cache; there is no cache CLI flag. `PYPTO_CACHE=1` enables reuse and `PYPTO_CACHE_DIR` selects the artifact root. Leave `PYPTO_PROG_BUILD_DIR` unset for reuse, because requesting explicit build output bypasses caching. See [JIT Cache](../../developer-guide/jit-cache.md) for the required PyPTO revision and cache controls.

## Start the service

The paths below are deployment choices, not required repository locations. Set the checkpoint and JIT-cache paths for your environment. The 16 listed devices must be free before starting the server.

```bash
PYPTO_CACHE=1 PYPTO_CACHE_DIR=/path/to/shared-jit-cache \
python -m pypto_serving.cli \
  --model /path/to/dsv4-flash-dspark-w8a8 \
  --served-model-name dsv4-flash-dspark-w8a8 \
  --backend npu --platform a2a3 \
  --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --dp 4 --ep 16 --tp 4 --block-size 32 \
  --max-model-len 16384 --max-num-seqs 8 \
  --max-num-batched-tokens 8192 \
  --long-prefill-token-threshold 128 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7}' \
  --enable-prefix-caching \
  --ring-heap 2147483648,2147483648,4294967296,8589934592 \
  --generate-config '{"max_new_tokens":2048,"temperature":0}' \
  --host 127.0.0.1 --port 8000
```

PyPTO compiles on a cache miss. Set `PYPTO_CACHE=0` to disable reuse when a fresh build is required. Check `/health` before sending requests:

```bash
curl --noproxy '*' http://127.0.0.1:8000/health
```

## Verify constrained and ordinary requests

This request requires a `shell` tool call with a `command` string and forbids extra keys. The client still decides whether and how to execute the returned command.

```bash
curl --noproxy '*' http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "dsv4-flash-dspark-w8a8",
    "messages": [{"role":"user","content":"Show the current directory."}],
    "tools": [{"type":"function","function":{
      "name":"shell",
      "description":"Run a shell command",
      "strict":true,
      "parameters":{
        "type":"object",
        "properties":{"command":{"type":"string"}},
        "required":["command"],
        "additionalProperties":false
      }
    }}],
    "tool_choice":"required",
    "temperature":0,
    "max_tokens":256
  }'
```

Expect a completed `message.tool_calls` entry named `shell`, with `function.arguments` as a JSON string containing `command`, and `finish_reason: "tool_calls"`. A length-truncated call is not a successful invocation. For a baseline, repeat the request with `tool_choice: "auto"` and `strict` omitted: it follows the unconstrained path and may answer directly or produce a tool call.

`auto` enters structural constraints only when at least one declared tool has `strict: true`; `required` and a named tool choice also enter the constrained path. A schema needs `required` to require a field and `additionalProperties: false` to exclude unknown fields. Ordinary non-strict `auto` does not guarantee schema-valid parameters. `parallel_tool_calls` is passed to the grammar only for constrained requests.

Missing XGrammar, an unsupported schema, or an unsupported model path yields an HTTP 400 before streaming headers are sent. There is no fallback to unconstrained generation for a request that asked for constraints. Request completion and cancellation release request-local grammar state. Recompute preemption of a constrained request is not yet supported. Under cache pressure, it waits while in-flight work can make progress; if no progress is possible, the scheduler explicitly rejects a stalled request and releases its resources.

## Upgrade and rollback

Roll Serving and PyPTO-Lib forward or backward together, then restart workers so PyPTO validates the updated source and toolchain identity. For an isolated rollout, select a new `PYPTO_CACHE_DIR` or set `PYPTO_CACHE=0`; do not load old named build directories manually or mix the new Serving `TaskArgs` order with old Lib kernels. After a restart, check health, one constrained tool request, and one ordinary chat request before restoring traffic.
