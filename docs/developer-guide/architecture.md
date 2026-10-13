# Architecture

PyPTO Serving is organized around a small serving stack:

```text
CLI
  -> FastAPI server
  -> AsyncLLMEngine
  -> Scheduler and KV cache manager
  -> Serving worker process
  -> Model executor
  -> PyPTO kernels on Ascend NPUs
```

## Source Layout

| Path | Responsibility |
| --- | --- |
| `pypto_serving/cli/` | CLI entry point and server startup configuration. |
| `pypto_serving/config/` | Runtime, generation, and parallel configuration. |
| `pypto_serving/serving/` | Engine, scheduler, KV cache, HTTP server, and workers. |
| `pypto_serving/model/` | Model loaders, tokenizers, Qwen, and DeepSeek integrations. |
| `pypto_serving/tools/profile/` | Chrome trace profiling utilities. |
| `scripts/` | Conversion and support scripts. |
| `pypto-lib/` | Model-specific PyPTO kernel sources. |

## Runtime Shape

The API process receives HTTP requests and forwards generation work to the async engine. The scheduler allocates KV pages, selects requests, and dispatches prefill or decode work. Worker processes own model executors and device-facing execution state.

The C++ `platform/` subtree is a separate platform-management layer and is not in the per-token Python serving hot path.

## Scheduler and KV Cache

The scheduler tracks request state, enforces runtime limits, allocates KV cache pages, and dispatches work to the engine.

Request flow:

1. A request enters the engine with a prompt and `GenerateConfig`.
2. The tokenizer produces prompt token IDs.
3. The scheduler assigns the request to prefill.
4. KV pages are allocated for the request.
5. The model executor runs prefill and then decode steps.
6. The scheduler releases cache pages when the request finishes.

The generic KV cache uses page IDs and a fixed block size. DeepSeek V4 uses grouped cache specs for model-specific cache families and rank-local partitions.

## Worker and Executor

Serving workers isolate model execution from the API process. A worker owns the model executor, compiled kernel handles, and device-facing runtime state.

Worker responsibilities:

- Load model runtime state.
- Initialize model executors.
- Run prefill and decode commands.
- Participate in profiling start and stop commands.
- Return generated token state to the engine.

Model executors implement the model-specific bridge from serving batches to PyPTO kernels. Qwen and DeepSeek V4 have separate executors and runners because their kernel layouts, cache layouts, and parallel contracts differ.

Changes to request scheduling or cache allocation should include focused unit tests under `tests/unit/serving/` and NPU validation when model behavior or device dispatch is affected.

## PyPTO target configuration

The serving CLI retains `--platform` (`a2a3`, `a2a3sim`, `a5`, or `a5sim`).
The shared executor config builder translates this into PyPTO `backend_type`
and `execution_mode`; the common kernel compiler preserves both when applying
per-model options. PyPTO continues to derive its internal platform encoding
for artifact metadata and caches, including the simulator suffix.
