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
    cpu_generate.py            CPU reference generation example
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

## NPU Generation

One-shot generation:

```bash
task-submit --device auto --max-time 0 --run \
  "python examples/model/qwen3_14b/npu_generate.py \
    --model-dir /path/to/Qwen3-14B \
    --prompt 'Huawei is' \
    --platform a2a3 \
    --max-seq-len 512 \
    --max-new-tokens 5"
```

Offline generation does not require the larger PTO2 ring settings used for
concurrent HTTP serving.

Add `--profile` to print timing and write a Chrome trace when `SA_PROFILE_OUTPUT`
or `SA_PROFILE_LEVEL` is set:

```bash
task-submit --device auto --max-time 0 --run \
  "SA_PROFILE_OUTPUT=/tmp/pypto-serving-profile-offline SA_PROFILE_LEVEL=verbose \
  python examples/model/qwen3_14b/npu_generate.py \
    --model-dir /path/to/Qwen3-14B \
    --prompt 'Huawei is' \
    --platform a2a3 \
    --max-seq-len 512 \
    --max-new-tokens 5 \
    --profile"
```

## HTTP Serving (OpenAI-compatible API)

Start the serving server with a multiprocess worker. When launching through
`task-submit`, use single quotes around the `--run` payload so `$TASK_DEVICE`
expands inside the task:

```bash
task-submit --device auto --max-time 1200 --run \
  'export PTO2_RING_HEAP=4294967296 PTO2_RING_TASK_WINDOW=131072 PTO2_RING_DEP_POOL=131072; \
  python python/cli/main.py \
    --model /data/linyifan/models/Qwen3-14B \
    --backend npu \
    --platform a2a3 \
    --device "$TASK_DEVICE" \
    --dp 1 \
    --tp 1 \
    --max-model-len 512 \
    --max-new-tokens 16 \
    --port 19340'
```

Send a generation request after the server logs `Application startup complete`:

```bash
# Health check
curl --noproxy "*" http://127.0.0.1:19340/health

# Completion
curl --noproxy "*" http://127.0.0.1:19340/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Huawei is", "max_tokens": 32, "temperature": 0.0}'

# Streaming
curl --noproxy "*" http://127.0.0.1:19340/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Huawei is", "max_tokens": 32, "stream": true}'

# Chat completion
curl --noproxy "*" http://127.0.0.1:19340/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is 1+1?"}], "max_tokens": 32}'
```

Run the serving benchmark:

```bash
python tests/bench_serving.py --port 19340 --stream -n 8 -c 4 --max-tokens 16
```

### Parallel Strategy V1

Serving supports a v1 `DP x TP` device topology. Data parallelism creates one
independent serving engine per replica, and tensor parallelism passes one device
group to the PyPTO L3 distributed worker for that replica. Single-device serving
remains the default. Pipeline parallelism and expert parallelism are accepted in
the config surface but rejected until the model kernels support them.

For example, run two data-parallel replicas on the two devices selected by
`task-submit`:

```bash
task-submit --device auto --device-num 2 --max-time 1800 --run \
  'export PTO2_RING_HEAP=4294967296 PTO2_RING_TASK_WINDOW=131072 PTO2_RING_DEP_POOL=131072; \
  python python/cli/main.py \
    --model /data/linyifan/models/Qwen3-14B \
    --backend npu \
    --platform a2a3 \
    --devices "$TASK_DEVICE" \
    --dp 2 \
    --tp 1 \
    --max-model-len 512 \
    --max-new-tokens 16 \
    --port 19339'
```

Send a completion request to the DP=2 server:

```bash
curl --noproxy "*" http://127.0.0.1:19339/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Huawei is","max_tokens":16,"temperature":0.0}'
```

Offline `npu_generate.py` supports `--devices` and `--tp` for one logical TP
replica. It intentionally rejects `--dp > 1`; launch separate offline jobs if
data-parallel offline generation is needed.

Single-request HTTP serving does not require the larger PTO2 ring settings. For
concurrent NPU serving, use topology-specific ring settings.

Single-replica concurrent serving uses the larger task window and dependency
pool:

```bash
task-submit --device auto --run \
  'export PTO2_RING_HEAP=4294967296 PTO2_RING_TASK_WINDOW=1048576 PTO2_RING_DEP_POOL=1048576; \
  python python/cli/main.py \
    --model /path/to/Qwen3-14B \
    --backend npu \
    --platform a2a3 \
    --device "$TASK_DEVICE" \
    --port 8899'
```

DP=2+ concurrent serving should keep the smaller task window and dependency
pool used by the DP=2 command above:

```bash
PTO2_RING_HEAP=4294967296 PTO2_RING_TASK_WINDOW=131072 PTO2_RING_DEP_POOL=131072
```

Without these settings, multi-request serving may return HTTP 200 while
generating no tokens and logging worker runtime failures such as `rtMalloc
failed: 207001`, `507018`, or `507046`.

For DP=2+, setting `PTO2_RING_TASK_WINDOW` and `PTO2_RING_DEP_POOL` to `1048576`
with a 4 GiB heap can reserve about 19 GiB of runtime arena per replica and fail
with `rtMalloc failed: 207001`.

## Notes

- All model/device/runtime options are passed via CLI arguments. Run
  `python python/cli/main.py --help` for the full list.
- Generated kernel artifacts are written under `build_output/` and are ignored
  by git.
- This repository expects PyPTO, CANN, torch, safetensors, transformers, and the
  local Ascend runtime environment to be available in the active Python
  environment.
- HTTP serving mode additionally requires `fastapi`, `uvicorn`, and `pydantic`.
  The benchmark script requires `aiohttp`.
