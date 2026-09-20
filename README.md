# pypto-serving

PyPTO Serving is a small local inference stack for running Qwen3-14B and DeepSeek V4 generation with PyPTO kernels on Ascend NPUs. It includes an installable Python package, model executor integrations and CLI entry points.

## Documentation

The external documentation lives under [`docs/`](docs/index.md). Start with:

- [Installation](docs/get-started/installation.md)
- [Quickstart](docs/get-started/quickstart.md)
- [Online Serving](docs/user-guide/online-serving.md)
- [CLI Reference](docs/cli-reference/index.md)
- [DeepSeek V4](docs/user-guide/deepseek-v4.md)

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
scripts/
  convert_deepseek_v4_to_w8a8.py  DeepSeek V4 checkpoint converter
tests/                         host-side unit tests and CI NPU accuracy guards
```

## Platform

The `platform/` subtree is the first-party C++ platform-management layer for PyPTO Serving. It is separate from the Python model-serving path and manages distributed-system bootstrap, deployment metadata, channel lifecycle, module services, and instance lifecycle. Model support keeps ownership of LLM-specific behavior (batching, KV cache policy, token scheduling, sampling, execution), while the platform orchestrates and supervises instances without sitting in the per-token execution hot path.

It is built around `serving::system::Engine`, which owns a set of `serving::modules::Module` instances and starts, supervises, and finalizes them across instances over RPC, using host-side channel primitives for control traffic. See [`platform/docs/README.md`](platform/docs/README.md) for the full design split, source layout, and runtime shape.

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

## Local Docs Build

The MkDocs site uses a shared theme checked out into `.site-theme/`. CI prepares this directory automatically; local builds need the same checkout before running `mkdocs build --strict`.

Use the commit pinned in `docs/theme-revision.txt`:

```bash
theme_revision=$(cat docs/theme-revision.txt)
git clone https://github.com/hw-native-sys/hw-native-sys.github.io .site-theme
git -C .site-theme checkout "$theme_revision"
python -m pip install -r docs/requirements.txt
mkdocs build --strict
```

## Notes

- All model/device/runtime options are passed via CLI arguments. Run `pypto-serving --help` for the exact arguments available in the installed package. See `docs/cli-reference/pypto-serving.md` for the documented reference.
- Parallel serving notes live in `docs/user-guide/parallel.md`.
- DeepSeek V4 checkpoint preparation lives in `docs/user-guide/deepseek-v4.md#checkpoint-conversion`.
- Generated kernel artifacts are written under `build_output/` and are ignored by git.
- This repository expects PyPTO, CANN, torch, safetensors, transformers, and the local Ascend runtime environment to be available in the active Python environment.
- `pypto-lib/` is not included in the wheel. An editable checkout discovers its kernel submodule automatically; for any other installation, set `PYPTO_LIB_ROOT` to the root of a `pypto-lib` checkout before loading a model.
