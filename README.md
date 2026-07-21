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
tests/                         CLI, batching, E2E serving, and benchmark tests
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

> Qwen3-14B raises the PTO2 ring-task heap to 512MB automatically — the runtime
> default (256MB) overflows during warmup and aborts with `HEAP_RING_DEADLOCK`.
> Override with `export PTO2_RING_HEAP=<bytes>` only if you need a different size.

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

Run the serving benchmark:

```bash
python tests/bench_serving.py --port 8899 --stream -n 8 -c 4 --max-tokens 16
```

## Notes

- All model/device/runtime options are passed via CLI arguments. Run
  `pypto-serving --help` for the full list.
- This repository expects PyPTO, CANN, torch, safetensors, transformers, and the
  local Ascend runtime environment to be available in the active Python
  environment.
- `pypto-lib/` is not included in the wheel. An editable checkout discovers its
  kernel submodule automatically; for any other installation, set `PYPTO_ROOT`
  to the root of a `pypto-lib` checkout before loading a model.
