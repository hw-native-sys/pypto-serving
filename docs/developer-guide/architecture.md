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

The API process receives HTTP requests and forwards generation work to `AsyncLLMEngine` in `serving/engine/async_engine.py`. Each `ReplicaEngineCore` owns a `Scheduler`, a `KvCacheManager`, and a worker process. Replica placement creates one core per replica; overlapped placement creates one core for the model-local rank group.

`ReplicaEngineCore._try_dispatch_step()` asks the scheduler for work, `_build_step_command()` sends a `StepCommand` to the worker, and `_process_step_output()` applies its `StepResult`. The worker's `_make_prefill_batch()` and `_make_decode_batch()` preserve request IDs, token positions, cache partitions, and per-group block IDs when constructing model-facing batches. The executor delegates those batches to its runner, which binds the PyPTO kernel arguments.

Worker startup is also part of the capacity contract: `WorkerProcess.init_device_and_model()` loads the checkpoint through `ModelLoader`, selects the configured executor, and calls `register_model()`. The runner uploads persistent weights and allocates device KV pools before reporting their actual capacity to the engine. Only then does the scheduler use the available page count.

The C++ `platform/` subtree is a separate platform-management layer and is not in the per-token Python serving hot path.

## Scheduler and KV Cache

The scheduler in `serving/sched/scheduler.py` tracks request state, enforces token and request limits, and allocates cache pages through `KvCacheManager` in `serving/memory/kv_cache.py`.

Request flow:

1. A request enters the engine with a prompt and `GenerateConfig`.
2. The tokenizer produces prompt token IDs.
3. The scheduler assigns the request to prefill.
4. KV pages are allocated for the request.
5. The model executor runs prefill and then decode steps.
6. The scheduler releases cache pages when the request finishes.

The generic cache uses page IDs and a fixed block size. For model-specific layouts, `RuntimeConfig.kv_cache_groups` carries `KVCacheGroupSpec` entries with independent page size, compression ratio, rolling-history limit, and partition count. The scheduler owns logical block IDs; the runner owns the device tensors and converts those IDs into kernel block tables and slot mappings. See [MTP Internals](../models/deepseek-v4/mtp-internals.md#cache-layout) and [DSpark Internals](../models/deepseek-v4/dspark-internals.md#how-the-deployment-maps-onto-the-kernels) for the two DeepSeek layouts.

## Worker and Executor

Serving workers isolate model execution from the API process. `WorkerProcess` owns a model executor; `PyptoExecutor` in `model/common/executor/pypto_executor.py` compiles model callables and registers a `ModelRunner` for the loaded model. The runner owns compiled kernel bindings, resident weights, scratch buffers, and device cache tensors.

Worker responsibilities:

- Load model runtime state.
- Initialize model executors.
- Run prefill and decode commands.
- Participate in profiling start and stop commands.
- Return generated token state to the engine.

`ModelExecutor` defines the worker-facing capabilities and prefill/decode methods. `ModelRunner` defines `init_kv_cache()`, `run_prefill()`, and `run_decode()`; model implementations may also support split prepare/dispatch/reclaim methods for asynchronous execution. Qwen, DeepSeek V4 MTP, and DSpark have separate executors and runners because their kernel ABIs, cache layouts, and parallel contracts differ. The [Model Integration](model-integration.md) guide maps these interfaces to the steps for adding a model.

Changes to request scheduling or cache allocation should include focused unit tests under `tests/unit/serving/` and NPU validation when model behavior or device dispatch is affected.

## PyPTO target configuration

The serving CLI retains `--platform` (`a2a3`, `a2a3sim`, `a5`, or `a5sim`).
The shared executor config builder translates this into PyPTO `backend_type`
and `execution_mode`; the common kernel compiler preserves both when applying
per-model options. PyPTO continues to derive its internal platform encoding
for artifact metadata and caches, including the simulator suffix.
