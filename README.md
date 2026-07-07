# pypto-serving

PyPTO Serving is a small local inference stack for running Qwen3-14B generation
with PyPTO kernels on Ascend NPUs. It includes a reusable Python runtime,
Qwen3-14B executor glue, CLI entry points, and tests for batching and config
handling.

## Layout

```text
python/
  cli/                         pypto-serving CLI implementation
  core/                        engine, scheduler, KV cache, model loading, async serving
  runtime/                     Simpler worker wrapper for NPU dispatch
pypto-lib/                     submodule providing Qwen3-14B PyPTO kernels
platform/                      C++ platform-management layer (engine lifecycle, channels, modules)
examples/
  pypto-serving                executable CLI wrapper
  model/qwen3_14b/
    npu_generate.py            NPU generation/profiling example
    npu_serving.json           sample serving config
    runner/                    Qwen3 executors and runner glue
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
```

Run the unit tests:

```bash
python -m pytest tests/test_batching.py tests/test_parallel.py
```

Show CLI help:

```bash
./examples/pypto-serving --help
python -m python.cli --help
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

Offline generation does not require the larger PTO2 ring settings used for
concurrent HTTP serving.

Add `--profile` to print timing and write a Chrome trace when `SA_PROFILE_OUTPUT`
or `SA_PROFILE_LEVEL` is set:

```bash
SA_PROFILE_OUTPUT=/tmp/pypto-serving-profile-offline SA_PROFILE_LEVEL=verbose \
python examples/model/qwen3_14b/npu_generate.py \
  --model-dir /path/to/Qwen3-14B \
  --prompt 'Huawei is' \
  --platform a2a3 \
  --device-id 0 \
  --max-seq-len 512 \
  --max-new-tokens 5 \
  --profile
```

## HTTP Serving (OpenAI-compatible API)

Start the serving server with a multiprocess worker:

```bash
python -m python.cli.main \
  --model /path/to/Qwen3-14B \
  --backend npu \
  --platform a2a3 \
  --device 0 \
  --max-model-len 512 \
  --max-new-tokens 16 \
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

Run the serving benchmark:

```bash
python tests/bench_serving.py --port 8899 --stream -n 8 -c 4 --max-tokens 16
```

## Notes

- All model/device/runtime options are passed via CLI arguments. Run
  `python python/cli/main.py --help` for the full list.
- Parallel serving development notes live in `docs/dev/parallel.md`.
- Generated kernel artifacts are written under `build_output/` and are ignored
  by git.
- This repository expects PyPTO, CANN, torch, safetensors, transformers, and the
  local Ascend runtime environment to be available in the active Python
  environment.
- HTTP serving mode additionally requires `fastapi`, `uvicorn`, and `pydantic`.
  The benchmark script requires `aiohttp`.
