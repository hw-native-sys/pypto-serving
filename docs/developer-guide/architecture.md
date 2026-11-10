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

### Qwen Program chip runtime

For Qwen3-14B Program execution, `EngineConfig(runtime_config=RuntimeConfig(chip_runtime="host_build_graph"))` selects HBG. The default is `"tensormap_and_ringbuffer"`. This is the chip scheduler policy, independent of the compiler backend and onboard/simulation execution mode. Other model executors currently reject HBG configuration at engine construction.

The existing replica configuration, worker model loader and `RuntimeModel.runtime` carry this value. Qwen configures its shared compiler from the loaded model before compiling all three stages; the shared run-config builder passes the selection into `DistributedConfig.runtime`. The existing compiled-program worker consumes that configuration. No additional engine, executor or process environment selection is involved.

HBG prefill uses the `qwen3_prefill_hbg_host` entry from pypto-lib, which takes separate HOST control tensors. Set `PYPTO_LIB_ROOT` to a checkout with that entry if the bundled Lib revision predates it.

For Qwen3-14B on one A3, use chunked prefill and set both `RuntimeConfig.max_num_batched_tokens` and `EngineConfig.max_num_scheduled_tokens` to 80 for the validated long-prompt setup. The worker partitions prefill into single-request calls to bound graph scratch. Decode continues to use the configured batch size.

The Qwen serving CI guard runs the same single-request, multi-batch, chunked-prefill and prefix-cache cases for both runtimes. The HBG invocation sets `PYPTO_QWEN3_RUNTIME=host_build_graph` for the test harness and selects a pinned Lib checkout through `PYPTO_LIB_ROOT`.

The shared `KernelCompiler` activates the selected chip runtime through PyPTO's
public `PassContext` before JIT cache lookup and code generation. The same value
remains in `DistributedConfig` for artifact metadata. A conflicting ambient
runtime is rejected; a matching ambient context retains its instruments and
options, including the existing explicit-option conflict checks. Without an
ambient context, explicit planner/diagnostic options are projected into the
context and the compile configuration is cloned with only those fields neutralized.
Explicit diagnostic requests still force compiler execution. Runtime selection
therefore applies to generated chip and collective configurations as well as the
wrapper metadata. CPU JIT/codegen tests run in the existing unit-tests CI job;
they do not execute SDK/device code or validate model numerics.
