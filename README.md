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
examples/
  pypto-serving                executable CLI wrapper
  model/qwen3_14b/
    cpu_generate.py            CPU generation example (FP default; --tq for TurboQuant)
    npu_generate.py            NPU generation/profiling example
    npu_serving.json           sample serving config
    runner/                    Qwen3 executors and runner glue
    src/                       PyPTO kernel/program builders
tests/                         CLI, batching, E2E serving, and benchmark tests
```

## Quick Checks

Initialize the kernel submodule after cloning:

```bash
git submodule update --init --recursive
```

Run the unit tests:

```bash
python -m pytest tests/test_cli.py tests/test_batching.py
```

Show CLI help:

```bash
./examples/pypto-serving --help
python -m python.cli --help
```

## Offline Mode

### One-shot Generation

```bash
task-submit --device auto --max-time 0 --run \
  "PTO2_RING_HEAP=4294967296 PTO2_RING_TASK_WINDOW=1048576 PTO2_RING_DEP_POOL=1048576 \
  python examples/model/qwen3_14b/npu_generate.py \
    --model-dir /path/to/Qwen3-14B \
    --prompt 'Huawei is' \
    --platform a2a3 \
    --max-seq-len 512 \
    --max-new-tokens 5"
```

Add `--profile` to print timing and write a Chrome trace when `SA_PROFILE_OUTPUT`
or `SA_PROFILE_LEVEL` is set:

```bash
task-submit --device auto --max-time 0 --run \
  "PTO2_RING_HEAP=4294967296 PTO2_RING_TASK_WINDOW=1048576 PTO2_RING_DEP_POOL=1048576 \
  SA_PROFILE_OUTPUT=/tmp/pypto-serving-profile-offline SA_PROFILE_LEVEL=verbose \
  python examples/model/qwen3_14b/npu_generate.py \
    --model-dir /path/to/Qwen3-14B \
    --prompt 'Huawei is' \
    --platform a2a3 \
    --max-seq-len 512 \
    --max-new-tokens 5 \
    --profile"
```

### Enabling TurboQuant (Offline)

TurboQuant compresses the KV cache via key/value low-bit quantization with a
residual window and protected layers, reducing KV memory. Enable it offline with
`--kv-quant`:

```bash
task-submit --device auto --max-time 0 --run \
  "PTO2_RING_HEAP=4294967296 PTO2_RING_TASK_WINDOW=1048576 PTO2_RING_DEP_POOL=1048576 \
  python examples/model/qwen3_14b/npu_generate.py \
    --model-dir /path/to/Qwen3-14B \
    --prompt 'Huawei is' \
    --platform a2a3 \
    --max-seq-len 512 \
    --max-new-tokens 5 \
    --kv-quant"
```

Customizable TurboQuant parameters:

| Flag | Default | Description |
|---|---|---|
| `--kv-quant-key-bits` | 4 | Key quantization bits |
| `--kv-quant-value-bits` | 2 | Value quantization bits |
| `--kv-quant-residual-window` | 128 | Residual quantization window size |
| `--kv-quant-protected-layers` | 4 | Number of protected (unquantized) layers from the bottom |
| `--kv-quant-protected-bits` | 8 | Bit width for protected layers |

## HTTP Serving (OpenAI-compatible API)

Start the serving server with a multiprocess worker. When launching through
`task-submit`, keep `--device {}` so the selected NPU ID is substituted into the
server command:

```bash
task-submit --device auto --run \
  "PTO2_RING_HEAP=4294967296 PTO2_RING_TASK_WINDOW=262144 PTO2_RING_DEP_POOL=262144 \
  python -m python.cli.main \
    --model /path/to/Qwen3-14B \
    --backend npu \
    --platform a2a3 \
    --device {} \
    --port 8899"
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

### Benchmark

```bash
python tests/bench_serving.py --port 8899 --stream -n 8 -c 4 --max-tokens 16
```

### Enabling TurboQuant (Online)

Add a `kv_quant` section to your config JSON (same as offline mode):

```json
{
  "model": { ... },
  "runtime": { ... },
  "kv_quant": {
    "enabled": true,
    "key_bits": 4,
    "value_bits": 2,
    "residual_window": 128
  }
}
```

When enabled, `[TurboQuant]` log lines will appear in the worker output,
indicating KV cache compression is active.

## Common CLI Flags

| Flag | Description |
|---|---|
| `--device <id>` | Override NPU device ID from config |
| `--max-num-running-reqs <n>` | Max concurrent running requests |
| `--long-prefill-token-threshold <n>` | Token threshold for chunked prefill |
| `--disable-prefix-cache` | Disable prefix caching |
| `--disable-chunk-prefill` | Disable chunked prefill |

## Notes

- All model/device/runtime options are passed via CLI arguments. Run
  `python -m python.cli.main --help` for the full list.
- Generated kernel artifacts are written under `build_output/` and are ignored
  by git.
- This repository expects PyPTO, CANN, torch, safetensors, transformers, and the
  local Ascend runtime environment to be available in the active Python
  environment.
- HTTP serving mode additionally requires `fastapi`, `uvicorn`, and `pydantic`.
  The benchmark script requires `aiohttp`.
