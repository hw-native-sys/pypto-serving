# pypto-serving

PyPTO Serving is a small local inference stack for running Qwen3-14B and
DeepSeek V4 generation with PyPTO kernels on Ascend NPUs. It includes an
installable Python package, model executor integrations and CLI entry points.

## Layout

```text
pypto_serving/
  cli/                         pypto-serving CLI implementation
  config/                      runtime, generation, and parallel configuration
  serving/                     engine, scheduler, KV cache, HTTP server, workers
  model/                       loading, common runtime, Qwen, and DeepSeek integrations
  worker/                      Simpler worker wrapper for NPU dispatch
  tools/profile/               Chrome-trace profiling support
pypto-lib/                     submodule providing model-specific PyPTO kernels
platform/                      C++ platform-management layer (engine lifecycle, channels, modules)
examples/
  model/qwen3_14b/
    npu_generate.py            NPU generation/profiling example
  model/deepseek_v4/
    npu_generate.py            Eight-NPU offline generation example
scripts/
  convert_deepseek_v4_to_w8a8.py  DeepSeek V4 checkpoint converter
tests/                         host-side unit tests and CI NPU accuracy guards
```

## Platform

The `platform/` subtree is the first-party C++ platform-management layer for
PyPTO Serving. It is separate from the Python model-serving path and manages
distributed-system bootstrap, deployment metadata, channel lifecycle, module
services, and instance lifecycle. Model support keeps ownership of LLM-specific
behavior (batching, KV cache policy, token scheduling, sampling, execution),
while the platform orchestrates and supervises instances without sitting in the
per-token execution hot path.

It is built around `serving::system::Engine`, which owns a set of
`serving::modules::Module` instances and starts, supervises, and finalizes them
across instances over RPC, using host-side channel primitives for control
traffic. See [`platform/docs/README.md`](platform/docs/README.md) for the full
design split, source layout, and runtime shape.

## Quick Checks

Initialize the kernel submodule after cloning:

```bash
git submodule update --init --recursive
python -m pip install --no-deps -e .
```

Show CLI help:

```bash
pypto-serving --help
```

## NPU Generation

One-shot generation:

```bash
python examples/model/qwen3_14b/npu_generate.py \
  --model-dir /path/to/Qwen3-14B \
  --prompt 'Huawei is' \
  --platform a2a3 \
  --device-id 0 \
  --max-seq-len 512 \
  --max-new-tokens 5
```

DeepSeek V4 Flash W8A8 offline generation reuses the serving scheduler and its
grouped KV-cache implementation, but does not start an HTTP server:

```bash
python examples/model/deepseek_v4/npu_generate.py \
  --model-dir /data/models/dsv4-flash-w8a8 \
  --prompt 'Huawei is' \
  --platform a2a3 \
  --devices 0,1,2,3,4,5,6,7 \
  --max-seq-len 512 \
  --max-new-tokens 20
```

Add `--enable-mtp` for speculative decoding or `--num-prompts N` for offline
continuous batching. DeepSeek V4 requires exactly eight devices with overlapped
attention DP=8 and MoE EP=8.

## HTTP Serving (OpenAI-compatible API)

Start the serving server with a multiprocess worker:

```bash
pypto-serving \
  --model /path/to/Qwen3-14B \
  --backend npu \
  --platform a2a3 \
  --device 0 \
  --max-model-len 512 \
  --port 8899
```

Send a generation request after the server logs `Application startup complete`:

```bash
# Health check
curl --noproxy "*" http://127.0.0.1:8899/health

# Completion
curl --noproxy "*" http://127.0.0.1:8899/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Huawei is", "max_tokens": 32, "temperature": 0.0}'

# Streaming
curl --noproxy "*" http://127.0.0.1:8899/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Huawei is", "max_tokens": 32, "stream": true}'

# Chat completion
curl --noproxy "*" http://127.0.0.1:8899/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is 1+1?"}], "max_tokens": 32}'
```

## Local Monitoring

The serving process exposes cumulative engine metrics at `/metrics` in
Prometheus text format and at `/metrics/json` as structured JSON. Metrics cover
request latency, token traffic and throughput inputs, scheduler queues, KV cache
usage, prefix-cache hits, and request outcomes.

For a self-contained local dashboard with SQLite history, start the companion
tool from the repository root:

```bash
python -m tools.monitor \
  --target http://127.0.0.1:8899 \
  --port 9090
```

Open <http://127.0.0.1:9090>. See
[`tools/monitor/README.md`](tools/monitor/README.md) for retention, database, and
timezone options.

## Notes

- All model/device/runtime options are passed via CLI arguments. Run
  `pypto-serving --help` for the full list.
- Parallel serving development notes live in `docs/dev/parallel.md`.
- DeepSeek V4 checkpoint preparation and NPU serving notes live in
  `docs/dev/model/deepseek-v4.md`.
- Generated kernel artifacts are written under `build_output/` and are ignored
  by git.
- This repository expects PyPTO, CANN, torch, safetensors, transformers, and the
  local Ascend runtime environment to be available in the active Python
  environment.
- `pypto-lib/` is not included in the wheel. An editable checkout discovers its
  kernel submodule automatically; for any other installation, set `PYPTO_LIB_ROOT`
  to the root of a `pypto-lib` checkout before loading a model.
