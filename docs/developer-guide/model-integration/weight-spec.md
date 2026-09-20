# Weight Spec Rules

PyPTO Serving's weight pipeline transforms a checkpoint on disk into the fused whole-model slabs that NPU kernels read. The pipeline is driven by declarative rule tables — each family supplies a `weight_spec.py` that describes what its weights are, and the shared packer decides how to produce them.

## Why Declarative Rules?

Before the rule table was introduced, Qwen3 and DeepSeek V4 each had their own ~1000-line staging pipeline that shared nothing. Adding a third family meant writing a third one. The rule table approach lets a family express its weight layout as data, and the shared packer, stacker, and pipeline handle the mechanics.

## Rule Types

Six rule types are available in `pypto_serving/model/common/weights/spec.py`. All are frozen dataclasses.

### LayerWeightRule

A required kernel weight read from the checkpoint, with optional shape-preserving edits:

```python
@dataclass(frozen=True)
class LayerWeightRule:
    name: str          # Kernel weight name (the output key)
    source: str        # Checkpoint tensor suffix (resolved as `{prefix}.{source}`)
    dtype: torch.dtype # Target dtype
    transpose: bool = False           # Transpose dims 0 and 1 after reading
    reshape_groups: int | None = None # Split leading dim into [groups, rows//groups, cols]
    flatten_to_row: bool = False      # Reshape 1-D to [1, dim] for stacking
```

Only `transpose`, `reshape_groups`, and `flatten_to_row` are expressible — all are pure re-orientations of the same bytes. Anything that computes new values belongs in the family's own code.

Qwen example (`pypto_serving/model/qwen/weight_spec.py`):

```python
QWEN_LAYER_RULES: tuple[LayerRule, ...] = (
    LayerWeightRule("decode_input_rms_weight", "input_layernorm.weight", torch.float32, flatten_to_row=True),
    LayerWeightRule("decode_wq", "self_attn.q_proj.weight", torch.bfloat16, transpose=True),
    LayerWeightRule("decode_wk", "self_attn.k_proj.weight", torch.bfloat16, transpose=True),
    LayerWeightRule("decode_wv", "self_attn.v_proj.weight", torch.bfloat16, transpose=True),
    # Optional Q/K norms: absent in some checkpoints, default to ones
    DefaultedWeightRule("decode_q_norm_weight", "self_attn.q_norm.weight", torch.float32,
                        default_shape=("head_dim",), default_fill="ones", flatten_to_row=True),
    DefaultedWeightRule("decode_k_norm_weight", "self_attn.k_norm.weight", torch.float32,
                        default_shape=("head_dim",), default_fill="ones", flatten_to_row=True),
    LayerWeightRule("decode_wo", "self_attn.o_proj.weight", torch.bfloat16, transpose=True),
    LayerWeightRule("decode_post_rms_weight", "post_attention_layernorm.weight", torch.float32, flatten_to_row=True),
    LayerWeightRule("decode_w_gate", "mlp.gate_proj.weight", torch.bfloat16, transpose=True),
    LayerWeightRule("decode_w_up", "mlp.up_proj.weight", torch.bfloat16, transpose=True),
    LayerWeightRule("decode_w_down", "mlp.down_proj.weight", torch.bfloat16, transpose=True),
)
```

### OptionalWeightRule

A weight that exists only for some layers (e.g., different attention kinds), and is zero-filled for the rest:

```python
@dataclass(frozen=True)
class OptionalWeightRule:
    name: str
    source: str
    dtype: torch.dtype
    absent_shape: tuple[Dim, ...]    # Fixed shape for the zero-filled branch
    enabled_ratios: tuple[int, ...]  # Compress ratios that carry this weight
    transpose: bool = False
```

The inactive branch is written, not skipped — every layer must present the same kernel signature. This is why a synthetic checkpoint must use production dimensions for these tensors.

DeepSeek V4 example — compressed KV projection weights enabled only at specific `compress_ratio` values:

```python
# HCA (high compression attention) weights
OptionalWeightRule(name="hca_wq_b",     source="hca_wq_b",     dtype=torch.float16,
                   absent_shape=("ranks", "head_dim", "hidden_size"), enabled_ratios=(128,)),
OptionalWeightRule(name="hca_wo_b",     source="hca_wo_b",     dtype=torch.float16,
                   absent_shape=("ranks", "hidden_size", "head_dim"), enabled_ratios=(128,)),
# CSA (compact compression attention) weights
OptionalWeightRule(name="csa_wq_b",     source="csa_wq_b",     dtype=torch.float16,
                   absent_shape=("ranks", "head_dim", "hidden_size"), enabled_ratios=(4,)),
```

### DefaultedWeightRule

A weight the checkpoint may omit, with a default value and a per-layer requirement flag:

```python
@dataclass(frozen=True)
class DefaultedWeightRule:
    name: str
    source: str
    dtype: torch.dtype
    default_shape: tuple[Dim, ...]
    required_when: str | None = None  # LayerContext flag name; None = genuinely optional
    default_fill: str = "zeros"       # "zeros" or "ones"
    flatten_to_row: bool = False
```

Qwen's optional Q/K norm gammas are the canonical case — a checkpoint without them is not broken:

```python
DefaultedWeightRule(
    "decode_q_norm_weight",
    "self_attn.q_norm.weight",
    torch.float32,
    default_shape=("head_dim",),
    default_fill="ones",
    flatten_to_row=True,
),
DefaultedWeightRule(
    "decode_k_norm_weight",
    "self_attn.k_norm.weight",
    torch.float32,
    default_shape=("head_dim",),
    default_fill="ones",
    flatten_to_row=True,
),
```

The `default_fill` matters: `"ones"` for a norm gamma (zeros would annihilate activations), `"zeros"` for a router placeholder the kernel will not read.

### SyntheticWeightRule

A weight computed rather than read, with the factory looked up by key:

```python
@dataclass(frozen=True)
class SyntheticWeightRule:
    name: str
    dtype: torch.dtype
    factory: str  # Key into a family-supplied factories dict
```

DeepSeek V4 uses this for the CSA Hadamard index and the all-ones MTP projection smoothing rows:

```python
SyntheticWeightRule("csa_hadamard_idx", torch.bfloat16, "hadamard_idx"),
SyntheticWeightRule("e_proj_smooth", torch.float32, "hidden_ones"),
SyntheticWeightRule("h_proj_smooth", torch.float32, "hidden_ones"),
```

### ExpertWeightRule

An expert weight, stacked rank-major over each rank's local expert slice:

```python
@dataclass(frozen=True)
class ExpertWeightRule:
    name: str
    source: str
    dtype: torch.dtype
```

Used with `ExpertParallel` policy. DeepSeek V4's routed MoE weights include int8 weights and FP32 scales:

```python
ExpertWeightRule("routed_w1", "w1.weight", torch.int8),
ExpertWeightRule("routed_w1_scale", "w1.scale", torch.float32),
ExpertWeightRule("routed_w3", "w3.weight", torch.int8),
ExpertWeightRule("routed_w3_scale", "w3.scale", torch.float32),
ExpertWeightRule("routed_w2", "w2.weight", torch.int8),
ExpertWeightRule("routed_w2_scale", "w2.scale", torch.float32),
```

### GlobalWeightRule

A whole-model weight (embedding, LM head, final norm) with tied-weight fallback and padding:

```python
@dataclass(frozen=True)
class GlobalWeightRule:
    name: str
    source: str
    dtype: torch.dtype
    fallback_source: str | None = None    # Tied-weight fallback (e.g., lm_head → embed)
    pad_to_multiple: int | None = None    # Pad vocabulary to this alignment
    pad_fill: str = "zeros"               # "zeros" or "first_row"
    flatten_to_row: bool = False
```

The `pad_fill` choice is critical: the embedding pads with zeros (a padded token id is never looked up), while the LM head pads by replicating row 0 (zero rows would give every padded entry the same logit, which looks like sampling noise).

Qwen example:

```python
QWEN_GLOBAL_RULES: tuple[GlobalWeightRule, ...] = (
    GlobalWeightRule("embed_weight", "model.embed_tokens.weight", torch.bfloat16,
                     pad_to_multiple=512, pad_fill="zeros"),
    GlobalWeightRule("lm_head_weight", "lm_head.weight", torch.bfloat16,
                     fallback_source="model.embed_tokens.weight",
                     pad_to_multiple=512, pad_fill="first_row"),
    GlobalWeightRule("final_norm_weight", "model.norm.weight", torch.float32, flatten_to_row=True),
)
```

## Layer Rules vs Global Rules

| Aspect | Layer Rules | Global Rules |
|--------|------------|--------------|
| Applied | Once per layer, evaluated by `pack_layer()` | Once per model, evaluated by `pack_globals()` |
| Scope | Per-layer tensors (Q, K, V, O, norms, MoE) | Whole-model tensors (embed, norm, lm_head) |
| Stacking | Stacked into slabs across layers | Not stacked |
| Sharding | Subject to rank policy | Subject to explicit padding |

## Rule Order Is Contract

The slab allocator walks the packed mapping to lay out whole-model tensors, and the prepacked sidecar records the resulting name-to-offset map. **Reordering the rules silently invalidates every sidecar already on disk.** New entries go at the end of their group.

## Stacking

pypto-lib fuses every transformer layer into a single kernel, so a layer's weight has no standalone existence on device. The model is uploaded as a few whole-model slabs, each holding all layers back to back.

### Stack Groups

The `StackGroup` dataclass (`pypto_serving/model/common/weights/stacker.py:32`) defines which weights belong together and which layers they cover:

```python
@dataclass(frozen=True)
class StackGroup:
    id: str
    members: tuple[str, ...] | None  # Weight names, or None for the catch-all group
    layer_ids: tuple[int, ...]       # Which layers participate
```

### Stack Axis

The `stack_axis` parameter is family-specific and **not cosmetic**:

- **Qwen** (single rank, no rank axis): `stack_axis=0`
- **DeepSeek V4** (multi-rank, rank axis leads): `stack_axis=1`

Getting it wrong produces a correctly-sized slab holding a transposed model, which no shape check catches.

### Stacking Invariants

- **No `torch.cat` in the stacker.** Slabs are allocated once from one layer's shapes, and every later layer is packed directly into a view of its own slice. Concatenating would hold the whole model twice at peak (~346 GB for DeepSeek V4).
- `torch.cat` does appear in `pack_globals`, where padding grows a weight that has no preallocated destination — but that is a handful of tensors, not the bulk of the model.

## Staging Policy

The `StagingPolicy` (`pypto_serving/model/common/weights/pipeline.py:32`) decides how layers are staged:

```python
@dataclass(frozen=True)
class StagingPolicy:
    workers: int = 1
    pin_torch_threads: bool = True
```

- **Serial (`workers=1`):** DeepSeek V4 uses this. Packing one layer allocates ~8 GB of intermediates (256 routed experts, each stacked and rank-replicated), so overlapping layers multiplies peak memory.
- **Pooled (`workers>1`):** Qwen3 uses this. Its layers are small enough that read latency dominates. Each worker is pinned to one torch thread — without this, N staging threads each fan out into torch's own pool and oversubscribe the machine.

## Reference: Existing Weight Specs

| File | Lines | Rules | Notable Features |
|------|-------|-------|------------------|
| `pypto_serving/model/qwen/weight_spec.py` | 151 | 11 layer + 3 global | Single rank, transposed projections, tied LM head fallback |
| `pypto_serving/model/deepseek/weight_spec.py` | 343 | 49 layer + 5 global + 12 MTP | 8 ranks, 4 compress ratios, expert parallelism, synthetic weights |

## Further Reading

- [Weight Staging](../weight-staging.md) — deep dive into the staging pipeline internals, invariants, and testing strategy
- `pypto_serving/model/common/weights/packer.py` — the `pack_layer()` evaluator
- `pypto_serving/model/common/weights/stacker.py` — `allocate_slabs()`, `destinations_for()`, `stack_layers()`
- `pypto_serving/model/common/weights/pipeline.py` — `stage_and_release()`, `StagingPolicy`
