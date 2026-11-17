# Model Integration

A model integration connects checkpoint metadata, a serving topology, scheduler-visible cache groups, worker execution, and PyPTO kernel arguments. The CLI and worker must agree on these contracts before a checkpoint can serve requests.

## Trace an Existing Model

| Stage | Code to inspect | Contract |
| --- | --- | --- |
| Identify the model | `model/model_family.py`: `read_model_config()` and `detect_model_family()` | Detect from `config.json`; the current fallback is Qwen, so a new family needs explicit detection and a validated unsupported-model path. |
| Build the engine | `cli/main.py`: `_build_runtime_config()`, `_executor_cls_for_model_family()`, `_validate_model_topology()` | Select an executor, placement mode, runtime limits, and any model cache groups before worker startup. |
| Load the checkpoint | `model/model_loader.py`: `ModelLoader.load()` | Produce the model config, tokenizer, and `RuntimeModel` used by the worker. |
| Register execution | `serving/server/serving_worker.py`: `WorkerProcess.init_device_and_model()` | Construct the executor and call `register_model()`; return the actual device cache capacity to the engine. |
| Compile and run | `model/common/executor/pypto_executor.py`: `PyptoExecutor`; `model/common/runner/model_runner.py`: `ModelRunner` | Compile callables, allocate cache, stage weights, implement `run_prefill()` and `run_decode()`. |
| Allocate logical pages | `serving/engine/async_engine.py`: `ReplicaEngineCore`; `serving/sched/scheduler.py`: `Scheduler` | The scheduler owns request state and block IDs; the runner maps those IDs to resident cache tensors. |

## Define the Model Contract

1. Add model-family detection and checkpoint validation. Declare the supported format, tokenizer behavior, and kernel variant selection. For DeepSeek V4, the CLI selects MTP or DSpark from `--speculative-config`; these variants use separate executors.
2. Define supported device topology in `ParallelConfig` and validate it before the worker starts. Replica placement has one core per DP replica; overlapped placement shares ranks within a model-local worker group. Do not infer physical device count by multiplying axes for an overlapped model.
3. Set `RuntimeConfig` limits and, when needed, `KVCacheGroupSpec` entries. A group describes the cache family, token-to-storage compression, per-request table depth, and rank-local partitioning. The runner's `init_kv_cache()` must allocate pools from the post-weight memory budget and return the scheduler-visible primary block count.
4. Implement the executor's `_compile_model()` and `_create_runner()` hooks. Implement the runner's `run_prefill()` and `run_decode()` against `PrefillBatch` and `DecodeBatch`. The worker passes `block_ids_by_group` and `cache_partitions`; the runner must lower them to the kernel's block tables and slot mappings.
5. If decoding needs the first sampled output token, implement `finalize_prefill()`. The worker invokes it after terminal prefill, before the first decode. For asynchronous decode, keep prepared metadata independent of an in-flight dispatch and retain mutable buffers until `reclaim_prepared_decode()` finishes.
6. Add weight rules when fused kernels consume whole-model slabs. `model/common/weights/` stages them from a lazy safetensors store; family rules and stack groups live in `model/<family>/weight_spec.py`. See [Weight Staging](weight-staging.md) for ordering and memory invariants.

## Verify the Contract

Add unit tests for family detection, topology rejection, cache-group geometry, and host metadata lowering under `tests/unit/`. Kernel ABI tests should check argument order and shapes against the matching `pypto-lib` entry points. Validate at least one offline generation and one HTTP request on the intended device topology; a passing host-side test cannot establish that device allocation or kernel dispatch works. Document checkpoint preparation and runnable commands under `docs/models/`. Put shared integration contracts in the Developer Guide and model-specific kernel details beside the model deployment guide.
