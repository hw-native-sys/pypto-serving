# pypto-serving

PyPTO Serving is a small local inference stack for running Qwen3-14B and DeepSeek V4 generation with PyPTO kernels on Ascend NPUs. It includes an installable Python package, model executor integrations and CLI entry points.

## Documentation

The external documentation lives under [`docs/`](docs/index.md). Start with:

- [Installation](docs/get-started/installation.md)
- [Quickstart](docs/get-started/quickstart.md)
- [Online Serving](docs/user-guide/online-serving.md)
- [CLI Reference](docs/cli-reference/pypto-serving.md)
- [Models](docs/models/index.md)
- [GLM-5.3-Flash](docs/models/glm53.md)

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
scripts/
  convert_deepseek_v4_to_w8a8.py  DeepSeek V4 checkpoint converter
tests/                         host-side unit tests and CI NPU accuracy guards
```

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

## NPU Generation (offline)

Offline generation runs through the same engine as serving (scheduler, worker process, KV cache) from the `pypto-serving` CLI, without opening a port:

```bash
pypto-serving \
  --model /path/to/Qwen3-14B \
  --platform a2a3 \
  --device 0 \
  --max-model-len 512 \
  --prompt 'Huawei is' \
  --generate-config '{"max_new_tokens": 5}'
```

DeepSeek V4 Flash W8A8 offline generation on eight devices:

```bash
pypto-serving \
  --model /path/to/dsv4-flash-w8a8 \
  --devices 0,1,2,3,4,5,6,7 \
  --dp 8 --ep 8 \
  --max-model-len 512 \
  --long-prefill-token-threshold 2048 \
  --prompt 'Huawei is' \
  --generate-config '{"max_new_tokens": 20}' \
  --no-enable-prefix-caching \
  --num-speculative-tokens 1
```

Repeat `--prompt` for offline continuous batching. Add `--profile` to capture the generation window. DeepSeek V4 requires exactly eight devices with overlapped attention DP=8 and MoE EP=8.

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

Requests that omit sampling fields use the server-wide `GenerateConfig`
defaults for every model: `temperature=0.0` (greedy decoding) and `top_p=1.0`.
Override them per request or with `--generate-config` when starting the server.

## Local Monitoring

The serving process exposes cumulative engine metrics at `/metrics` in
Prometheus text format, including `vllm:` families for existing vLLM dashboards,
and at `/metrics/json` as structured JSON. Metrics cover
request latency, token traffic and throughput inputs, scheduler queues, KV cache
usage, prefix-cache hits, request outcomes, and MTP/DSpark draft acceptance,
acceptance length, and speculation fallbacks.

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

- All model/device/runtime options are passed via CLI arguments. Run `pypto-serving --help` for the exact arguments available in the installed package. See `docs/cli-reference/pypto-serving.md` for the documented reference.
- Parallel serving notes live in `docs/user-guide/parallel.md`.
- DeepSeek V4 checkpoint preparation lives in [the MTP model guide](docs/models/deepseek-v4/mtp.md#checkpoint-conversion).
- Generated kernel artifacts are written under `build_output/` and are ignored by git.
- This repository expects PyPTO, CANN, torch, safetensors, transformers, and the local Ascend runtime environment to be available in the active Python environment.
- `pypto-lib/` is not included in the wheel. An editable checkout discovers its kernel submodule automatically; for any other installation, set `PYPTO_LIB_ROOT` to the root of a `pypto-lib` checkout before loading a model.
