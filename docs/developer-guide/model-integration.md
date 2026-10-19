# Model Integration

A model integration connects checkpoint metadata, a serving topology, scheduler-visible cache groups, worker execution, and PyPTO kernel arguments. The resolved integration ID and capabilities are shared by the engine, worker, and HTTP server.

## Trace an Existing Model

| Stage | Code to inspect | Contract |
| --- | --- | --- |
| Identify the model | `model/metadata.py`: `read_model_metadata()`; `model/integrations/__init__.py`: `select_model_family()` and `select_integration()` | Require a readable `config.json` and one consistent supported model family before device startup; then select its execution variant. |
| Build the engine | `model/integrations/`: `ModelIntegration.resolve()`; `config/resolve.py`: `resolve_launch_options()` | Validate model topology and feature options, then build effective parallel, runtime, and capability configuration. |
| Load the checkpoint | `model/model_loader.py`: `ModelLoader.load()` | Produce the model config, tokenizer, and `RuntimeModel` used by the worker. |
| Register execution | `serving/server/serving_worker.py`: `WorkerProcess.init_device_and_model()` | Construct the executor and call `register_model()`; return the actual device cache capacity to the engine. |
| Compile and run | `model/common/executor/pypto_executor.py`: `PyptoExecutor`; `model/common/runner/model_runner.py`: `ModelRunner` | Compile callables, allocate cache, stage weights, implement `run_prefill()` and `run_decode()`. |
| Allocate logical pages | `serving/engine/async_engine.py`: `ReplicaEngineCore`; `serving/sched/scheduler.py`: `Scheduler` | The scheduler owns request state and block IDs; the runner maps those IDs to resident cache tensors. |

## Define the Model Contract

1. Register the model's `model_type`, supported `architectures`, tokenizer kind, load format, and execution variants in the model-family registry, then add an integration with a stable ID for each distinct execution contract. Keep variant constraints in that integration; the shared parser checks only the speculative config shape and alias conflict. For DeepSeek V4, `--speculative-config` selects MTP or DSpark after recognizing one shared family. Add checkpoint format and tokenizer support in their existing modules when needed.
2. Define supported device topology in the integration's `resolve()` method and construct `ParallelConfig` before the worker starts. Replica placement has one core per DP replica; overlapped placement shares ranks within a model-local worker group. Do not infer physical device count by multiplying axes for an overlapped model.
3. Return model-specific `RuntimeConfig` values and, when needed, `KVCacheGroupSpec` entries from the integration. Declare static sampling and constrained-tool capabilities there as well. A cache group describes the cache family, token-to-storage compression, per-request table depth, and rank-local partitioning. The runner's `init_kv_cache()` must allocate pools from the post-weight memory budget and return the scheduler-visible primary block count.
4. Implement the executor's `_compile_model()` and `_create_runner()` hooks. Implement the runner's `run_prefill()` and `run_decode()` against `PrefillBatch` and `DecodeBatch`. The worker passes `block_ids_by_group` and `cache_partitions`; the runner must lower them to the kernel's block tables and slot mappings.
5. If decoding needs the first sampled output token, implement `finalize_prefill()`. The worker invokes it after terminal prefill, before the first decode. For asynchronous decode, keep prepared metadata independent of an in-flight dispatch and retain mutable buffers until `reclaim_prepared_decode()` finishes.
6. Add weight rules when fused kernels consume whole-model slabs. `model/common/weights/` stages them from a lazy safetensors store; family rules and stack groups live in `model/<family>/weight_spec.py`. See [Weight Staging](weight-staging.md) for ordering and memory invariants.

## Verify the Contract

Add unit tests for family detection, topology rejection, cache-group geometry, and host metadata lowering under `tests/unit/`. Kernel ABI tests should check argument order and shapes against the matching `pypto-lib` entry points. Validate at least one offline generation and one HTTP request on the intended device topology; a passing host-side test cannot establish that device allocation or kernel dispatch works. Document checkpoint preparation and runnable commands under `docs/models/`. Put shared integration contracts in the Developer Guide and model-specific kernel details beside the model deployment guide.
