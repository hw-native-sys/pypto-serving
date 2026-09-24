# DeepSeek V4.1 text entry

V4.1 integration provides metadata loading, local text tokenization, selective
CPU weight loading and a serving framework for explicit lib composite bindings.
The model loader registers metadata without opening checkpoint payloads. The
CLI routes V4.1 to its own executor, but rejects execution before allocating
devices until `load_composite_bindings()` supplies a verified adapter. The
default adapter is an explicit placeholder; this is not an executable M0 model.

```python
from pypto_serving.model.deepseek_v41.config import load_text_config
from pypto_serving.model.tokenizer import load_tokenizer

model_dir = "/path/to/DeepSeek-V4.1-Flash"
config = load_text_config(model_dir)
tokenizer = load_tokenizer(model_dir)
ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Hello"}], tokenize=True,
)
```

The text config reads dimensions from nested `text_config`, special IDs from
the checkpoint metadata, and compression modes for the target backbone only.
Trailing draft-layer compression entries are not backbone layers. This is
metadata validation, not a guarantee that every shape is supported by kernels.

The tokenizer loads local files through the existing shared fast-tokenizer
loader. Raw `encode` adds no special tokens. Chat encoding inserts the model's
own BOS and assistant prefix once. Supported messages have string content;
tools, images and structured content are rejected. The text prompt format is
derived from the independent DeepSeek encoder at revision
`dba1be0a40aa45a94ad051997016db3960a90277`; the checked-in golden fixture records
its source and digest. A synthetic local tokenizer exercises actual token IDs
and decoding but is not a full-checkpoint tokenizer validation.

## Staged integration

1. Model identification, text config and tokenizer (implemented).
2. Selective checkpoint loading, weight formats and shard contracts (implemented on CPU).
3. Executor/Runner lifecycle and explicit composite contract (framework implemented).
4. Bounded CPU embedding rows from TP shards (implemented); device residual/pre-mix initialization is an adapter callback.
5. Chunked prefill and request/cache ownership (framework implemented); numerical composites remain adapter callbacks.
6. Decode continuity and prefill-to-decode transitions (framework implemented).
7. Rank-owned weights and all backbone layers (dispatch implemented); device TP/EP execution remains in the adapter.
8. Final HC/norm/head boundary and shared greedy sampling (framework implemented); numerical head remains an adapter callback.
9. Model loader, CLI, scheduler metadata and request release (framework implemented); HTTP/device M0 acceptance remains pending.

Engram, vision and speculative decoding are deferred. Optional metadata for
these modules may be present in config.json; this stage does not initialize or
load them. Later numerical acceptance must use a reference with the same
Engram-disabled scope, not claim equality with the complete official model.

Development starts from upstream main. Existing experimental V4.1 code can be
reused selectively with tests; its backend and runtime are not prerequisites.
Kernel stages should track current pypto-lib interfaces and record the revision
used for validation. This integration does not change the lib pin.

## Selective weight loading

The weight path follows V4's declarative source specs and shared
`LazySafetensorsStore`. A constructor reads only config/index JSON and checks
required text names. Requested tensor headers are validated before payload
slicing. It never loads an entire shard or deferred Engram/vision/draft weights.

```python
from pypto_serving.model.deepseek_v41.weight_loader import V41WeightLoader

loader = V41WeightLoader(model_dir, tp_size=4, tp_rank=0, ep_size=8, ep_rank=0)
projection = loader.load("layers.0.attn.wq_b.weight")
expert = loader.load("layers.0.ffn.experts.0.w1.weight")
embedding_rows = loader.load_rows("embed.weight", 0, 32)
```

`load` returns a `WeightBundle` with owned CPU weight/scale tensors, layout,
source names, rank identities and the per-call buffer estimate. Callers retain
global checkpoint names and map them to the chosen composite's parameters in
the Executor stage. This API does not invent an executable whole-layer ABI.

| Source | CPU result |
| --- | --- |
| FP8 block32 `[N,K]` + UE8M0 grid | Contiguous FP8 `[K,N]`, expanded scale codes in MX_B_NN order |
| Routed FP4 `[N,K/2]` bytes | UINT8 tiles `[K*N/256,128]`, with MX_B_NN E8M0 scales |
| `wo_a` FP8 | Dequantized grouped BF16 after selecting this rank's output groups |
| Dense HC/norm weights | Preserved dense layout/dtype |
| Gate and ratio-2 compressor | Required FP32 promotion; compressor/index matrices transposed where required |
| Embedding/head | BF16 vocabulary shard or a bounded local-row range |

Packing matches lib `4c3eab2` host layout helpers; these pure CPU operations do
not import PyPTO or execute small model operators. Runtime integration will
call composite lib entries, following V4 Executor/Runner structure. Verify the
layouts again when lib changes. Current FP4 tiles require K/N multiples of 256;
native FP8 matrices require K divisible by 64 and N by 32. Unsupported geometry
is rejected. Torch/safetensors must support the checkpoint E8M0 dtype.

TP slices grouped/head projections and aligns their scales. Shared experts and
router weights are replicated; EP selects whole routed experts, retaining
global expert IDs. TP and EP ranks are explicit; mapping physical ranks to
these coordinates belongs to the Executor. This loader does not certify a
multi-device execution path.

The default 256 MiB budget covers a conservative estimate for one operation's
tensor buffers, including conversion scratch. It excludes mmap address space,
Python overhead, device storage and previously returned bundles. Large loads
are rejected before payload reads. Use `load_rows` for vocabulary tables and
load expert matrices individually; do not accumulate a full model without a
separate residency budget. `load_rows` offsets are relative to the local TP
vocabulary shard.

Validation uses synthetic checkpoint tensors stored in actual safetensors files,
independent packing-address checks, and shared-store regression tests. These
checks are not real-checkpoint numerical inference or NPU acceptance.

## Executor/Runner preparation

Stage numbers follow serving issue #240. `V41ExecutionPlan` describes the
checkpoint and rank ownership without opening devices. The executor uses one
plan per logical rank and dispatches complete layer entries through its runner.

```python
from pypto_serving.model.deepseek_v41.execution_plan import RankPlacement, V41ExecutionPlan

plan = V41ExecutionPlan(model_dir, RankPlacement(rank=7))
layer = plan.layer(24)  # C1A Reindex: KV producer 20, index producer 24
names = plan.weight_names(24)  # local experts; no duplicate scale loads
bundle = plan.load_weight(24, "layers.24.hc_attn_scale")
```

Logical ranks form contiguous TP groups. With TP4/DP2/EP8, rank 7 is
TP rank 3, DP rank 1 and EP rank 7. These are logical coordinates, not
physical NPU IDs. The checkpoint loader still enforces dimension divisibility.
The plan preserves checkpoint names and the loader's per-operation budget;
it does not invent parameter bindings for an unsupported composite signature.

Layer planning follows lib's `config.layer_config` ownership rules. SWA
windows are layer-local. Compressed layers select the latest preceding
producer of the same compression ratio. Full layers publish KV/index state;
Reindex layers use the KV producer's index-key cache and publish a new Top-K
selection; Reuse layers consume that selection without loading producer
weights. C1A candidates must address the same KV producer as their consumers.
These layer references are not scheduler page IDs or request-global state.

`DeepSeekV41PyptoExecutor` and `V41ModelRunner` follow the shared executor
lifecycle, with one synchronous collective session for TP4/DP2/EP8 on A5.
The runner does not inherit the generic dense K/V allocator. The scheduler
owns page reservations; the adapter owns device pools, compilation, uploads
and completion. Physical device IDs are independent of logical ranks.

## Composite adapter contract

`CompositeBindings` is a serving-owned integration boundary, not a declaration
that current lib functions already have these signatures. A concrete adapter
must identify its tested lib revision and implement every operation below.
All callbacks may enqueue device work; `wait(resources)` must establish
completion across every participating rank before host code reuses storage.

| Operation | Responsibility |
| --- | --- |
| `allocate(plan, device_ids, runtime, build_options)` | Return `(resources, num_pages)` with positive scheduler page capacity. Honor the worker's platform, build directory and compile-cache choice. Allocate the declared cache groups and compressor state, enforce memory budgets, and prepare global embedding/HC/norm/head resources as needed. Clean up partially created resources if allocation raises before returning. |
| `initialize(embeddings, step, resources)` | Upload packed CPU BF16 `[active_tokens, hidden_size]` and initialize the production residual and delayed pre-mix state. No fixture pre-mix value is assumed. |
| `prepare_weights(rank_plans, layer, resources)` | Bind each rank's TP projections and EP experts, retaining packed FP4. Own bounded staging or persistent residency and reuse producer weights correctly. |
| `entries[(phase, mode)](layer, state, step, resources, weights)` | Execute the complete Attention + FFN layer for prefill/decode, preserving the declared residual/pre-mix layout and returning `LayerState`. |
| `output(state, step, resources)` | Execute final HC/norm/head and return CPU float logits `[requests, vocabulary]` in original request order, including a row for each prefill chunk. Only terminal prompt chunks are sampled by the shared worker. |
| `reset_request(resources, key, owner)` | Clear every cache/state location owned by this request. `owner` includes its DP partition, stable compressor block ID, committed length and page tables, including pages touched by a failed step. |
| `wait(resources)` / `close(resources)` | Establish collective completion / release all session resources. A failed wait must never be treated as permission to reuse buffers. |

The runner carries opaque `LayerState` between layers. Both replicated and
TP-local-token residual layouts are supported when all callbacks agree; it
does not insert an unconditional residual AllGather. Every collective must
include the two DP partitions even when one has no active requests. The
adapter lowers logical positions, active tokens and page IDs into the exact
lib ABI, including padding and communication-window lifetimes.

`cache_groups` must describe the same concrete layouts and capacities to the
scheduler and allocator, with two DP partitions. Full-history page tables are
supported by the framework. Rolling page reuse requires a verified lowering
contract and is explicitly rejected for now. Cache payloads must not use a
generic dense K/V substitute. The adapter must bound physical page IDs against
actual allocated pools when a group leaves `num_blocks` unspecified.

At inspected upstream lib revision `1b8caa4`, packed-FP4 MoE and token-local
decode Attention progress do not yet provide a compatible complete-layer
adapter. Decode `stage="block"` remains disabled; the prefill fixture still
uses the older routed-weight/call contract. Initial residual/pre-mix, complete
prefill/decode, cache allocation/reset and final HC/norm/head remain explicit
integration work. These facts do not block testing the serving state machine,
but they do block real-model generation and M0 numerical acceptance.

## Request state and serving lifecycle

`RequestLedger` allocates a stable logical compressor state block per request
and DP partition. This is an ownership ID, not a physical layout: the adapter
must translate it to the selected lib's ring-state block table. Paused or
omitted requests keep their blocks. Batch reordering does not change ownership.
The adapter must provision enough state blocks for the configured request
capacity; no batch row is used as a persistent state address.

Packed prefill metadata carries both the end of the current chunk (`seq_lens`)
and the original full prompt length (`prompt_lens`). A subsequent chunk must
start at the committed position. Decode requires completed prefill and consumes
exactly one supplied token per request. Lengths are committed only after all
layers and output finish. Failed execution waits, resets affected requests and
releases their ownership; failed completion/reset poisons the session instead
of reusing uncertain state. This does not attempt to roll back in-place device
updates to an earlier token position.

The existing worker owns sampling, EOS handling, HTTP request lifecycle and
finished-request notifications. Returned logits own their host storage so an
adapter cannot overwrite them while the worker samples. Prefix caching and
asynchronous scheduling are disabled: restoring a prefix also needs matching
compressor state, and concurrent dispatch needs separate mutable state tickets.

The CLI requires A5, eight distinct device IDs, TP4/DP2/EP8 and 128-row pages.
The shared engine sees one overlapped EP worker; internal TP/DP placement lives
in the rank plan and the two cache partitions. A3 execution, Engram, MTP and
vision are outside this framework's current execution contract.

## Validation boundary

Serving tests use small safetensors checkpoints and explicitly named recording
adapters. They verify ownership, metadata, call order, failures and sampling
integration without inventing numerical results for missing production kernels.
Running these CPU tests on an A5 host is not NPU acceptance. Real M0 still needs
the concrete adapter, original weights, matched-reference 8K + 128-token greedy
results, repeated-request cleanup and device memory observations.

## Test command

```bash
python -m pytest tests/unit/model/deepseek_v41 tests/unit/model/test_tokenizer.py tests/unit/cli/test_parallel_options.py -q
```
