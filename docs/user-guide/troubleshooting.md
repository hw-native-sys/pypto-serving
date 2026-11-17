# Troubleshooting

Start with the stage that failed. Keep the exact error message, model path, CLI arguments, and the PyPTO Serving and `pypto-lib` revisions when investigating a reproducible failure.

## CLI or Imports Fail Before Startup

Run `pypto-serving --help` in the same Python environment used to start the server. If the command is missing, install the checkout with `python -m pip install --no-deps -e .` as in [Installation](../get-started/installation.md). This install intentionally does not resolve the Ascend-compatible PyTorch, PyPTO runtime, CANN, or serving dependencies. Import errors should be resolved in that active environment; initialize `pypto-lib` with `git submodule update --init --recursive` after cloning.

## Model Load or Topology Is Rejected

Check `config.json`, tokenizer files, and checkpoint shards in `--model`. Qwen expects a Hugging Face style checkpoint. DeepSeek V4 Flash requires the [converted W8A8 checkpoint](../models/deepseek-v4/mtp.md#checkpoint-conversion); the source FP8/MXFP4 snapshot is not a serving checkpoint. If conversion stopped midway, rerun the converter with `--resume` to validate and skip completed shards.

Compare `--devices`, `--dp`, `--ep`, `--tp`, `--block-size`, and `--speculative-config` with the [model compatibility matrix](../models/index.md). MTP requires eight devices with DP8/EP8/TP1 and block size 128; DSpark accepts eight or 16 devices with TP4 and block size 32. The CLI rejects unsupported combinations before worker startup. `--show-startup-logs` exposes model loading and compilation progress.

| Error text | Check |
| --- | --- |
| `requires the quantized W8A8 compressed-tensors checkpoint` | Convert the DeepSeek source checkpoint and point `--model` at the converted output. |
| `requires --dp 8 --ep 8 with --tp 1` or `requires exactly 8 NPU device ids` | Use the MTP topology, or select DSpark explicitly with `--speculative-config` and its own topology. |
| `kernels require --block-size 128` | Set block size 128 for MTP; DSpark uses 32. |
| `safetensors is required` | Install `safetensors` in the active serving environment. |

## NPU Startup or Capacity Fails

Confirm the requested NPU IDs are visible and free in the active CANN environment. On a capacity failure, reduce `--max-model-len`, `--max-num-seqs`, or `--max-num-batched-tokens` within the model's documented limits, then rerun the same request. Weight residency, KV cache pools, and ring heaps consume device memory before a request can be admitted; the model guides describe their different layouts. Do not treat a host-side unit test as proof that NPU allocation or kernel dispatch succeeds.

## First Start Is Slow or Cache Reuse Is Missing

The first NPU run can compile kernels. Set `PYPTO_CACHE=1` and a writable `PYPTO_CACHE_DIR` for reuse, leaving `PYPTO_PROG_BUILD_DIR` unset. Use `--show-startup-logs` to distinguish compilation from checkpoint staging. A cache hit still has startup overhead, and different specializations may build different artifacts. [JIT Cache](../developer-guide/jit-cache.md) covers the precedence, bypass reasons, and inspection commands.

## HTTP Requests Fail

Wait for `Application startup complete`, then check `/health` before sending generation requests. For the request schema, status codes, and SSE error events, see the [HTTP API](../api-reference/http.md). A request rejected before streaming returns an HTTP error; after SSE headers, a DeepSeek tool-output parsing error is delivered as an event even though the HTTP status is 200. DSpark accepts greedy sampling only (`temperature: 0`), and [constrained tools](../models/deepseek-v4/tool-constraints.md) require a matched Serving/PyPTO-Lib pair and XGrammar.

| Error text | Check |
| --- | --- |
| `the model has no tool-call parser` | Send tools only to a model with a registered parser; see [Tool Calling](tool-calling.md). |
| `xgrammar is required for constrained tool generation` | Install the version in [DSpark Tool Constraints](../models/deepseek-v4/tool-constraints.md#compatible-components). |
| `tool message must reference a preceding assistant tool_call_id` | Include the returned assistant call in history and match the tool result's `tool_call_id`. |

When the server is running, `/metrics` and `/metrics/json` expose engine counters and latency observations. For an internal timing trace, start with [Profiling](../developer-guide/profile.md).
