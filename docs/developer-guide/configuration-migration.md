# Serving configuration migration

This table records the phase-one migration of serving-owned values and model or runtime capabilities.

| Area | Existing source | Phase-one source of truth | Consumer path | Verification |
| --- | --- | --- | --- | --- |
| Generic batch token default | CLI model-family branch | config/types.py DEFAULT_MAX_NUM_BATCHED_TOKENS | CLI to EngineConfig and RuntimeConfig | CLI parallel option tests |
| V4.1 batch token default | V4.1 CLI branch and runner constant | Existing V4.1 model path and DEFAULT_MAX_NUM_BATCHED_TOKENS | _build_v41_engine_config | V4.1 and CLI tests |
| V4/V4.1 cache geometry | Model runners and Lib bindings | Existing model/Lib contract validation | model builder to cache groups and executor | DeepSeek and V4.1 unit tests |
| Prefill capacity | Model runner constants and Lib binding capacity | Model path validates requested value before runtime construction | CLI to RuntimeConfig and executor | V4.1 capacity tests and compile checks |
| Batch and scheduled token limits | EngineConfig fields | EngineConfig effective runtime and scheduler builder | engine to SchedulerConfig | engine, scheduler and CLI tests |
| Prefill chunk alignment and choices | RuntimeConfig and model builders | RuntimeConfig | EngineConfig.build_scheduler_config to scheduler | scheduler chunk validation tests |
| Prefix cache replay and retention | RuntimeConfig and scheduler construction | RuntimeConfig | engine to scheduler and cache manager | grouped prefix-cache tests |
| KV geometry and page arithmetic | cache managers and model runners | model geometry plus serving.memory.page_math | prefill, cache and model runner paths | page boundary and formula-equivalence tests |
| IPC prefill fields | request and worker IPC structures | existing typed IPC payloads | scheduler to worker to runner | serving and model integration tests |
| Worker and router timeouts | environment helpers and router config | existing timeout config modules | engine, worker and router | router and engine unit tests |
| Device and topology constraints | CLI checks and model runners | existing model contracts and topology validators | CLI/model builder startup | parallel and model topology tests |
| Resource and offload settings | RuntimeConfig and tiering integration | RuntimeConfig | engine to cache tiering | cache tiering tests |

The phase-one implementation reuses the existing ModelConfig, RuntimeConfig, model contract and topology objects. It adds one scheduler construction entry point and checked page arithmetic. A later CLI adapter cleanup can remove the remaining model-family branches without changing the runtime configuration objects.

## Validation record

- DeepSeek unit tests: 120 passed.
- Focused CLI, V4.1, engine, scheduler and memory tests: 273 passed before the scheduler builder change.
- DeepSeek, V4.1, CLI, engine, scheduler and memory tests after the scheduler builder change: 389 passed.
- Page formula equivalence against the previous arithmetic passed across page sizes 1, 2, 4, 16, 128, 256 and 1024.
- Ruff, compileall and git diff --check passed.
- The full tests/unit run reached 503 passed before the unrelated router end-to-end test test_a_new_conversation_is_minted_a_session_id returned HTTP 502. The same router test failed when run alone.

## File audit

### Modified by this PR

- `docs/developer-guide/configuration-migration.md`
- `pypto_serving/cli/main.py`
- `pypto_serving/config/types.py`
- `pypto_serving/model/deepseek_v41/npu_runner.py`
- `pypto_serving/serving/engine/async_engine.py`
- `pypto_serving/serving/memory/page_math.py`
- `tests/unit/serving/memory/test_page_math.py`

### Audited and reused without source changes

These paths remain the source of existing behavior or validation and were checked by the focused tests:

- `pypto_serving/config/parallel.py`
- `pypto_serving/serving/sched/scheduler.py`
- `pypto_serving/serving/memory/kv_cache.py`
- `pypto_serving/serving/utils/env.py`
- `pypto_serving/router/config.py`
- `pypto_serving/model/deepseek/npu_runner.py`
- `pypto_serving/model/deepseek/task_args.py`
- `pypto_serving/model/deepseek/weight_loader.py`
- `pypto_serving/model/deepseek_v41/task_args.py`
- `pypto_serving/model/deepseek_v41/dspark_resources.py`
- `pypto_serving/model/glm53/kernel_api.py`
- `pypto_serving/serving/server/ipc.py`
- `pypto_serving/serving/server/serving_worker.py`
- `pypto_serving/serving/utils/prefill.py`
- existing model, scheduler, router and serving tests

### Deferred

- `pypto_serving/model/contracts/` was not added. The phase uses the existing model and Lib contract paths.
- A separate `EffectiveRuntimeConfig` type was not added. `EngineConfig.build_scheduler_config()` is the single new effective scheduler construction entry point.
- A CLI model adapter registry and removal of model-family branches remain deferred to the later CLI entry-point cleanup.
- New broad config and topology test directories were not added because the existing focused suites cover the current changes.
