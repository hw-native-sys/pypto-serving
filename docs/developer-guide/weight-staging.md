# Weight Staging

Weight staging turns checkpoint tensors into the host layout consumed by fused PyPTO kernels before the runner uploads them to the device. The shared code reads only the tensors needed for the current layer, applies declarative packing rules, and writes into preallocated whole-model slabs. A model integration supplies its checkpoint names, rule table, stack groups, sharding policy, and upload requirements.

## Loading Flow

1. `LazySafetensorsStore` in `pypto_serving/model/common/weights/store.py` maps tensor names to checkpoint shards. `load_many()` validates names, groups reads by shard file, and opens each required shard once without materializing unrelated tensors.
2. Rules in `common/weights/spec.py` map checkpoint names to kernel weights. `pack_layer()` in `common/weights/packer.py` evaluates them for a `LayerContext`, applying the requested dtype, orientation, optional/default values, and rank placement. When given destination views, it writes packed values directly into them.
3. `StackGroup` in `common/weights/stacker.py` assigns ordered layers and weights to slabs. A packed template layer determines the output shapes. `allocate_slabs()` reserves the host tensors, and `destinations_for()` gives each layer a view of its own slice. `stack_layers()` fills those slices without concatenating a second complete model.
4. `StagingPolicy` in `common/weights/pipeline.py` selects serial staging or a bounded thread pool. Each layer's raw tensors become releasable after its slice is written; `stage_and_release()` provides this load/write/release pattern when the caller uses it. The chosen worker count bounds how many layers can be live at once.
5. Whole-model weights such as embeddings and the LM head follow `GlobalWeightRule` through `pack_globals()`. The runner then uploads the resulting tensors and initializes its device state.

The rule table describes *what* each kernel weight needs. The store, packer, and stacker implement *how* to read and place it, so adding an integration does not require a separate copy of the staging pipeline.

## Rule and Layout Contracts

| Contract | Why it matters |
| --- | --- |
| Rule order | `pack_layer()` emits weights in rule order. Slab layout and any prepacked sidecar's name-to-offset map depend on that order; changing it can invalidate stored artifacts. |
| Complete destinations | `pack_layer()` skips rules without a destination. Every destination selected for a layer must be written, including zero-filled inactive branches and defaulted weights, so no slab slice contains uninitialized data. |
| Stack geometry | `StackGroup.layer_ids` determines each layer's slice position. The caller selects `stack_axis`, any `new_axis_members`, and the allocator to match the kernel layout and upload process. |
| Bounded host memory | The stacker writes directly into slab views. Layer reads and packing intermediates should be released after each stage; increasing `StagingPolicy.workers` also increases the possible peak. |
| Global weights | `GlobalWeightRule` controls tied-weight fallback and vocabulary padding. Padding values must match the kernel's numerical contract, not just its expected shape. |

`LayerWeightRule` expresses orientation changes such as transpose, group reshape, and NZ packing. `OptionalWeightRule` writes a fixed-shape placeholder when a branch is inactive; `DefaultedWeightRule` fills a missing checkpoint tensor only when its rule permits that variant. `SyntheticWeightRule` delegates computed values to a named factory, and `ExpertWeightRule` applies the caller's expert-parallel placement. Transformations that compute new model values belong in the integration rather than being hidden in a generic rule field.

## Validation

Tests under `tests/unit/model/` should compare staged tensor shape, dtype, order, and bytes against the expected kernel layout. The `(shape, dtype, sha256)` fingerprint helper in `tests/unit/model/conftest.py` catches changes that a value-only comparison can miss. Check that altering one layer changes only its destination slices, and compare a loaded prepacked artifact with a fresh pack when an integration provides one. Synthetic checkpoints must use dimensions compatible with fixed-shape optional branches and the template used to allocate slabs.
