# Executor and Runner

The executor and runner are the two runtime components that bridge the serving stack to NPU kernel execution. The executor owns compilation and model registration; the runner owns the per-step prefill and decode loops.

## Overview

```
register_model()          run_prefill() / run_decode()
     │                          │
     ▼                          ▼
┌─────────────┐          ┌──────────────┐
│  Executor   │ ──────▶  │   Runner     │
│  (compile)  │  creates  │  (execution) │
└─────────────┘          └──────────────┘
     │                          │
     │ compile                  │ submit
     ▼                          ▼
┌─────────────┐          ┌──────────────┐
│ pypto-lib   │          │ L3 Worker    │
│ kernel srcs │          │ (NPU)        │
└─────────────┘          └──────────────┘
```

## Executor

### ModelExecutor (ABC)

The base class in `pypto_serving/model/common/executor/executor.py` defines the contract every executor must satisfy. The serving worker interacts with models exclusively through this interface.

**Capability properties** (all opt-in, defaulting to `False`):

| Property | Default | Purpose |
|----------|---------|---------|
| `supports_device_sampling` | `False` | Executor returns already-sampled token IDs |
| `supports_device_stochastic_sampling` | `False` | Sampled IDs use per-request temperature/top-k |
| `device_topk_sampling_k` | `0` | Max top-k candidate width |
| `supports_device_embedding` | `False` | Token embedding inside device kernels |
| `supports_device_decode_embedding` | `supports_device_embedding` | Decode kernels gather embeddings from token IDs |
| `max_prefill_batch_size` | `None` | Executor-specific prefill dispatch limit |
| `supports_async_decode_prepare` | `False` | Decode metadata prepared ahead of execution |
| `supports_async_decode_reclaim` | `False` | Dispatch and host output reclaim run independently |

**Abstract methods** (must implement):

| Method | Signature | Purpose |
|--------|-----------|---------|
| `run_prefill()` | `(model, batch) → PrefillResult` | Run prompt prefill → next-token logits |
| `run_decode()` | `(model, batch) → DecodeResult` | Run one decode step |

### PyptoExecutor (ABC)

The `PyptoExecutor` (`pypto_serving/model/common/executor/pypto_executor.py`) extends `ModelExecutor` for PyPTO backends. It manages the compilation lifecycle and runner registry.

**Key lifecycle — `register_model()`:**

```python
def register_model(self, model_id, record) -> int:
    compiled = self._compile_model(model)          # 1. Compile kernels
    runner = self._create_runner(model_id, compiled)  # 2. Create runner
    runner.init_kv_cache(...)                      # 3. Allocate KV cache
    runner.preflight(record)                       # 4. Materialise weights
    self._runners[model_id] = runner               # 5. Store
    return page_count
```

**Abstract methods for subclasses:**

| Method | Purpose |
|--------|---------|
| `_compile_model(model)` | Load pypto-lib modules, validate shapes, compile L3 callables, build kernel metadata |
| `_create_runner(model_id, compiled)` | Return a model-specific `ModelRunner` from compiled artifacts |

### Qwen Executor Example

`Qwen314BPyptoExecutor` (`pypto_serving/model/qwen/npu_executor.py`):

- **Capabilities:** device sampling (top-k=32), device embedding, single rank
- **Compilation:** loads 3 pypto-lib modules (prefill_fwd, decode_fwd, topk_select), validates shapes against Qwen3-14B constants, compiles 3 L3 callables, builds RoPE tables, pads vocab/weights
- **Runner:** returns `Qwen314BModelRunner(compiled=compiled, device_id=...)`

### DeepSeek V4 Executor Example

`DeepSeekV4PyptoExecutor` (`pypto_serving/model/deepseek/npu_executor.py`):

- **Capabilities:** device sampling, device decode embedding; device stochastic sampling and async decode prepare only when `num_speculative_tokens <= 1`; max prefill batch size = `ranks × max_prefill_requests_per_partition`
- **Compilation:** loads 42 pypto-lib kernel modules, compiles 4 L3 callables (prefill, decode, mtp_prefill, mtp_decode), builds layer plan and weight store, builds SWA + compressed RoPE tables
- **Runner:** returns `DeepSeekV4ModelRunner(compiled=compiled)`

## Runner

### ModelRunner (ABC)

The base class in `pypto_serving/model/common/runner/model_runner.py`:

```python
class ModelRunner(ABC):
    def __init__(self):
        self._kv_caches: dict[str, _KvCachePool] = {}

    def init_kv_cache(self, model_id, config, runtime, *, num_pages=None) -> int:
        """Create paged KV cache in runner-owned device memory. Returns page count."""

    def close_kv_cache(self) -> None:
        """Release all KV cache tensors."""

    def preflight(self, record) -> None:
        """Eagerly materialise weights/buffers before worker signals ready."""

    def warmup(self, model) -> None:
        """Run dummy prefill+decode to warm up device kernels."""

    @abstractmethod
    def _alloc_kv_cache_tensor(self, shape, dtype):
        """Allocate one worker-resident KV cache tensor."""

    @abstractmethod
    def _free_kv_cache_tensor(self, tensor):
        """Free one worker-resident KV cache tensor."""

    @abstractmethod
    def run_prefill(self, model, batch) -> PrefillResult:
        """Run compiled prefill path."""

    @abstractmethod
    def run_decode(self, model, batch) -> DecodeResult:
        """Run compiled decode path."""

    def close(self):
        """Release all resources."""
```

### L3DispatchMixin

Both runners mix in `L3DispatchMixin` (`pypto_serving/model/common/runner/l3_dispatch.py`) for NPU kernel dispatch. The mixin owns:

- **L3 Worker:** a persistent `DistributedWorker` for NPU execution
- **Static tensor cache:** upload-once weights cached across dispatches
- **Dispatch path:** `_run_l3()` for synchronous, `_submit_l3()` / `PendingL3Dispatch.wait()` for async
- **Ring sizing:** per-dispatch `RunConfig` replacing the old process-wide `PTO2_RING_*` env vars

Initialisation in the runner:

```python
class MyModelRunner(L3DispatchMixin, ModelRunner):
    def __init__(self, *, compiled, ...):
        self._init_l3_dispatch(stacked=False)  # or stacked=True for multi-rank
```

The `stacked` parameter controls whether multi-rank (`alloc_stacked_tensor`) or single-rank (`alloc_tensor`) allocation is used.

### TaskArgs

TaskArgs are the centralised argument management mechanism for kernel calls. Each family declares its kernel argument contracts in `task_args.py`:

- **Positional order:** every kernel argument has a fixed position in the call
- **Slot:** describes each argument's type, shape, and placement (`HOST_SHARED` for host-mapped I/O, `DEVICE_RESIDENT` for device scratch)
- **StaticDeviceTensor:** for upload-once weights (never re-uploaded between dispatches)

Qwen's task args (`pypto_serving/model/qwen/task_args.py`) use `_PREFILL_ORDER` and `_DECODE_ORDER` as the source of truth. At this commit, both dispatches register 25 positional arguments. `topk_select_task_args()` owns five shared buffers, and the runner assembles the four-argument top-k dispatch tuple inline.

DeepSeek V4's task args (`pypto_serving/model/deepseek/task_args.py`) declare the main prefill/decode orders in `_PREFILL_FWD_TENSOR_ORDER` and `_DECODE_FWD_TENSOR_ORDER`, with separate MTP prefill/decode order tuples. Treat those order tuples as the canonical kernel contract.

### Runner Design Patterns

**Ping-pong decode slots:** DeepSeek V4 keeps two slots for decode metadata and outputs so the host can prepare step N+1 while the device executes step N. Qwen uses synchronous decode and reuses its normal TaskArgs slots.

**Inactive row replication:** Qwen's decode kernel expects a fixed batch size. Inactive rows replicate row 0 (same KV write target, harmless) rather than using a mask.

**Cache groups:** DeepSeek V4 uses 7 heterogeneous KV cache pools (ori, hca_cmp, csa_cmp, idx, hca_state, csa_state, csa_inner_state) instead of the generic flat KV cache. Each group has its own block size, page layout, and allocation strategy.

## Adding a New Executor and Runner

1. Create `pypto_serving/model/<family>/npu_executor.py`:
   - Subclass `PyptoExecutor`
   - Implement `_compile_model()`: load kernel modules, compile L3 callables
   - Implement `_create_runner()`: return your model-specific runner
   - Set capability properties

2. Create `pypto_serving/model/<family>/npu_runner.py`:
   - Subclass `L3DispatchMixin` and `ModelRunner`
   - Implement `init_kv_cache()`, `run_prefill()`, `run_decode()`, `close()`
   - Define `TaskArgs` for kernel argument contracts

3. Create `pypto_serving/model/<family>/task_args.py`:
   - Declare kernel argument positions and slot specs
   - Build TaskArgs for prefill, decode, and any auxiliary kernels

## Reference: Existing Executors and Runners

| Component | Qwen3-14B | DeepSeek V4 Flash |
|-----------|-----------|-------------------|
| Executor | `qwen/npu_executor.py` (487 lines) | `deepseek/npu_executor.py` (509 lines) |
| Runner | `qwen/npu_runner.py` (~1166 lines) | `deepseek/npu_runner.py` (~4796 lines) |
| Task Args | `qwen/task_args.py` (242 lines) | `deepseek/task_args.py` (836 lines) |
| L3 Callables | 3 (prefill, decode, topk_select) | 4 (prefill, decode, mtp_prefill, mtp_decode) |
| Ranks | 1 | 8 |
| Stacked | No | Yes |
| Device Sampling | Top-K=32 | Full (stochastic when K=1) |
| Async Decode | No | Yes (when K=1) |
| MTP | No | Yes (K=1 or arbitrary) |
