# DeepSeek V4 DSpark constrained tool deployment

This guide deploys structural tool-call constraints for DeepSeek V4 Flash DSpark with fused K7 decoding. The tool names and JSON schemas come from each `/v1/chat/completions` request, not from a server-side allowlist or a particular agent client. Serving compiles the request grammar on the host; the device receives a fixed-layout allowed-token mask and still runs one fused K7 decode dispatch per step.

## Compatible components

- Use a DeepSeek V4 Flash DSpark W8A8 checkpoint and the 16-device `--dp 4 --ep 16 --tp 4` topology described in [DSpark serving](../developer-guide/deepseek-v4-dspark.md).
- Deploy PyPTO Serving and PyPTO-Lib as a matched pair. Their prefill and decode positional ABI now includes `grammar_mask`; fused K7 decode also includes `valid_draft_counts`. Updating only one repository will not work. The mask shape and argument order are fixed regardless of whether a request has constraints.
- Install `xgrammar==0.2.7` into the Python environment that runs Serving. This is the version validated with the DeepSeek V4 structural-tag grammar and the checkpoint tokenizer. XGrammar is a host-side dependency; PyPTO-Lib does not import it. See the [XGrammar package](https://pypi.org/project/xgrammar/0.2.7/).
- DSpark currently supports greedy sampling only. Use `temperature: 0`; non-greedy requests are rejected rather than silently changing sampling behavior.

```bash
python -m pip install 'xgrammar==0.2.7'
python -m pip show xgrammar
```

When upgrading either repository, use a new `PYPTO_PROG_BUILD_DIR` for the changed kernels before enabling `--use-compile-cache`. The compile cache does not validate a previous executable against the new source or positional ABI.

## Start the service

The paths below are deployment choices, not required repository locations. Set the checkpoint and build-cache paths for your environment. The 16 listed devices must be free before starting the server.

```bash
PYPTO_PROG_BUILD_DIR=/path/to/new-compile-cache \
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
  --use-compile-cache --host 127.0.0.1 --port 8000
```

Start without `--use-compile-cache` for the first build if your deployment does not use a persistent compile directory. Check `/health` before sending requests:

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

Expect a completed `message.tool_calls` entry named `shell`, with `function.arguments` as a JSON string containing `command`, and `finish_reason: "tool_calls"`. A length-truncated call is not a successful invocation. For a baseline, repeat the request with `tool_choice: "auto"` and `strict` omitted: it follows the unconstrained path and may answer directly or produce a tool call. `/v1/completions` uses the same fixed kernel ABI with an all-allowed mask.

`auto` enters structural constraints only when at least one declared tool has `strict: true`; `required` and a named tool choice also enter the constrained path. A schema needs `required` to require a field and `additionalProperties: false` to exclude unknown fields. Ordinary non-strict `auto` does not guarantee schema-valid parameters. `parallel_tool_calls` is passed to the grammar only for constrained requests.

Missing XGrammar, an unsupported schema, or an unsupported model path yields an HTTP 400 before streaming headers are sent. There is no fallback to unconstrained generation for a request that asked for constraints. Request completion and cancellation release request-local grammar state. Recompute preemption of a constrained request is not yet supported; under cache pressure it may wait for resources rather than preempt another constrained request.

## Upgrade and rollback

Roll Serving and PyPTO-Lib forward or backward together, then use a build-cache directory compiled from that pair. After a restart, check health, one constrained tool request, and one ordinary chat request before restoring traffic. Do not reuse a compile cache across the changed kernel ABI or mix the new Serving `TaskArgs` order with old Lib kernels.
