# Testing

Model integration tests live under `tests/unit/model/` and follow a two-tier architecture: family-neutral tests in `tests/unit/model/common/`, model-specific tests in `tests/unit/model/<family>/`.

## Test Organisation

```
tests/unit/model/
├── conftest.py                      # Shared: fingerprint_tensors fixture
├── test_tokenizer.py                # Tokenizer adapter tests
├── common/
│   ├── test_task_args.py            # TaskArgs container tests
│   ├── test_weight_pipeline.py      # Staging policy, global rules, pack_globals
│   └── test_weight_store.py         # LazySafetensorsStore tests
├── deepseek/
│   ├── conftest.py                  # deepseek_checkpoint synthetic fixture
│   ├── test_model_components.py     # Full lifecycle tests (~2000 lines)
│   ├── test_weight_pack_parity.py   # Byte-level parity for stacked weights
│   └── test_weight_sidecar_parity.py # Prepacked sidecar compatibility
└── qwen/
    ├── test_npu_executor.py         # Compilation interface tests
    ├── test_npu_runner_inputs.py    # Runner input preparation tests
    └── test_qwen_weight_staging.py  # Differential parity vs. old manual loop
```

## Key Test Patterns

### Fingerprint Tests (Byte-Level Parity)

The `fingerprint_tensors` fixture (`tests/unit/model/conftest.py`) returns a callable that maps `{name: (shape, dtype, sha256_hex)}` for a dict of tensors. This is the primary oracle for byte-level parity: two pipelines produce the same bytes iff their fingerprints match.

```python
def fingerprint_tensors(tensors: dict[str, torch.Tensor]) -> dict[str, tuple]:
    """Returns {name: (shape, dtype, sha256_hex)} for each tensor."""
```

Why SHA-256 instead of `torch.equal`? A dtype change that round-trips through the same values (e.g., float16 → bfloat16 → float16) would pass `torch.equal` but produces different bytes. The fingerprint catches it.

**What to test with fingerprints:**

- `test_stacked_pack_is_reproducible` — same checkpoint twice → identical bytes
- `test_fingerprint_detects_a_reordered_layer_stack` — swapping layers changes the fingerprint
- `test_group_slabs_only_follow_their_own_attention_kind` — CSA perturbations don't affect HCA slabs
- `test_stacked_slabs_keep_the_rank_axis_and_stay_contiguous` — shape and contiguity invariants

### Differential Parity Tests

Compare the new rule-table-based staging against a known-good reference. For Qwen, `test_qwen_weight_staging.py` compares the rule table output byte-for-byte against a reproduction of the executor's old manual staging loop:

```python
def test_the_rule_table_stages_what_the_executor_stages(self, ...):
    rule_output = pack_using_rule_table(...)
    manual_output = pack_using_manual_loop(...)
    assert fingerprint_tensors(rule_output) == fingerprint_tensors(manual_output)
```

### Fake Objects

Replace expensive or unavailable components with lightweight fakes:

```python
class _FakeCompiler:
    """Captures compile arguments without actually compiling."""
    def __init__(self, *args, **kwargs): ...
    def compile(self, *args, **kwargs):
        return _FakeJitFunction()

class _FakeWorker:
    """Stand-in for DistributedWorker."""
    def run(self, *args, **kwargs): ...
    def submit(self, *args, **kwargs):
        return _FakeHandle()
    def free_tensor(self, tensor): ...
```

### Monkeypatch Patterns

Replace module-level classes and functions to control test environments:

```python
# Replace the real compiler
monkeypatch.setattr(npu_executor, "KernelCompiler", _FakeCompiler)

# Prove torch.cat is never called in the hot path
monkeypatch.setattr(torch, "cat", lambda *args, **kwargs: pytest.fail("torch.cat called"))

# Prove safetensors files are never opened
monkeypatch.setattr(weight_loader, "_default_safe_open", _fail_open)
```

### WeakRef Memory Assertions

Prove that staging releases tensor memory after `stage_and_release` returns:

```python
refs = []
for tensor in layer_tensors:
    refs.append(weakref.ref(tensor))

stage_and_release(...)
gc.collect()
assert [i for i, ref in enumerate(refs) if ref() is not None] == []
```

### Thread Safety Tests

For pooled staging, verify that staging actually runs in parallel and doesn't race:

```python
# Barrier synchronisation: N workers must all reach the barrier to proceed
barrier = threading.Barrier(3)

# Prove single-worker mode runs on the calling thread
main_thread = threading.get_ident()
# (assert that staged layer never changes thread)

# Prove thread pinning is set once, not per worker
sets = []
monkeypatch.setattr(torch, "set_num_threads", sets.append)
# (assert len(sets) == 1, not == num_workers)
```

### Synthetic Checkpoints

The `deepseek_checkpoint` fixture (`tests/unit/model/deepseek/conftest.py`) creates a synthetic on-disk checkpoint for testing:

```python
@pytest.fixture
def deepseek_checkpoint(tmp_path):
    """Returns a factory that writes a synthetic DeepSeekV4 checkpoint to tmp_path."""
    # Parameters: compress_ratios, n_routed_experts, num_hash_layers, ranks, layer_seeds, include_mtp
    # Returns: DeepSeekCheckpoint dataclass with store() and load_stacked() methods
```

Key design: the fixture uses `deepseek_v4_layer_weight_names` from production code to determine which tensors to write, so the contract cannot drift from the loader.

## What to Test

For a new model integration, write tests for:

| Area | Focus |
|------|-------|
| **Weight loading** | Checkpoint detection, tensor loading, shard grouping, error handling for missing files |
| **Weight spec rules** | Each rule type, rule order, optional/default behaviour, padding |
| **Staging** | Serial vs. pooled, thread pinning, memory release, error propagation |
| **TaskArgs** | Argument allocation, idempotency, device memory rollback, cache identity |
| **KV cache** | Allocation, paging, OOM retry, heterogeneous cache groups |
| **Prefill inputs** | Input preparation, block tables, slot mappings, dynamic extent |
| **Decode inputs** | Metadata preparation, inactive row handling, per-rank sharding |
| **Smoke tests** | End-to-end prefill + decode with fake workers (no NPU required) |