# Weight staging

How a checkpoint on disk becomes the tensors a fused kernel reads. One pipeline, two model
families, and the parts that differ expressed as data rather than as separate code paths.

## Why there is a pipeline at all

pypto-lib fuses every transformer layer into a single kernel, so a layer's weight has no
standalone existence on device: the model arrives as a few **whole-model slabs**, each holding
all of its layers back to back. That makes the stacked layout an *output contract* — not an
implementation detail — and it is why staging is a distinct phase with its own tests rather than
a few lines inside each executor.

Before this pipeline, Qwen3 and DeepSeek V4 each had their own ~1000-line answer to the same
question, sharing nothing. Adding a third family meant writing a third one.

## The four stages

| Module | Owns |
|---|---|
| `common/weights/store.py` | reading named tensors from a safetensors checkpoint, one shard open per group of names |
| `common/weights/spec.py` | the rule vocabulary: what a weight is, where it comes from, how it is conditioned |
| `common/weights/shard.py` | how one packed tensor is distributed — `Replicate`, `ExpertParallel`, `NoShard` |
| `common/weights/packer.py` | evaluating a family's rules against raw tensors |
| `common/weights/stacker.py` | slab geometry: allocate once, hand each layer a view of its slice |
| `common/weights/pipeline.py` | staging order — serial or pooled — and the per-layer release |

A family supplies a rule table (`model/<family>/weight_spec.py`) and nothing else. The shared
code never learns what a weight *means*.

## Rule kinds

- `LayerWeightRule` — a weight read straight from the checkpoint, with the shape-preserving
  edits it needs (`transpose`, `reshape_groups`, `flatten_to_row`). Only re-orientations are
  expressible: anything that computes new values belongs in the family's own code rather than
  hidden behind a field that looks declarative.
- `OptionalWeightRule` — present for some attention kinds, **zero-filled at a fixed shape** for
  the rest. The inactive branch is written, not skipped, because every layer has to present the
  same kernel signature.
- `DefaultedWeightRule` — the checkpoint may omit it. `required_when` names the flag that
  decides whether an absent source is an error or simply a model variant, and `default_fill`
  chooses zeros or ones.
- `SyntheticWeightRule` — computed rather than read, with the factory looked up by key so the
  rule stays data.
- `ExpertWeightRule` — one expert weight, sharded across ranks by the placement the family
  injects.
- `GlobalWeightRule` — embedding, LM head, final norm, with the tied-weight fallback and the
  vocabulary padding.

## Invariants worth knowing before changing any of this

**Rule order is contract.** The slab allocator lays out whole-model tensors in the order the
packed mapping is built, and a prepacked sidecar records the resulting name-to-offset map.
Reordering a rule table silently invalidates every sidecar already on disk. New entries go at
the end of their group.

**Nothing in the stacker calls `torch.cat`.** Slabs are allocated once from one layer's shapes
and every later layer is packed directly into a view of its own slice. Concatenating instead
would hold the whole model twice at the peak — ~346 GB for DeepSeek V4 that it does not have to
pay. `torch.cat` does appear in `pack_globals`, where padding grows a weight that has no
preallocated destination, and that is a handful of tensors rather than the bulk of the model.

**The stack axis differs per family and is not cosmetic.** A DeepSeek V4 weight leads with the
rank axis its upload shards on and stacks its layers on axis 1; a rank-less Qwen weight stacks
on axis 0. Getting it wrong produces a correctly-sized slab holding a transposed model, which no
shape check catches.

**The staging policy differs for a measured reason.** DeepSeek V4 stages serially: packing one
layer allocates ~8 GB of intermediates (256 routed experts, each stacked and rank-replicated),
so overlapping layers multiplies the peak and contends on memory bandwidth instead of hiding
latency. Qwen3 uses a pool, its layers being small enough that read latency dominates — with
each worker pinned to one torch thread, without which N staging threads each fan out into
torch's own pool and the copies run *slower* than serially.

**An LM head does not pad with zeros.** The embedding does, because a padded token id is never
looked up. The LM head pads by replicating row 0: zero rows there would give every padded
vocabulary entry the same finite logit rather than an impossible one, so the mistake survives
review and shows up as sampling noise.

**Qwen's slabs must be shared memory.** Its upload reads them from a forked child, so private
memory is a silent correctness bug rather than a slow path.

## Testing

The staged output has to stay byte-identical across changes, so the tests are written as
*sensitivity* rather than as green checks — each one fails on a specific regression:

- `tests/unit/model/conftest.py` provides a `(shape, dtype, sha256)` fingerprint. Hashing the
  bytes rather than comparing with `torch.equal` catches a dtype change that round-trips through
  the same values.
- layer order: a mis-ordered slab keeps every shape and dtype, so only content shows it.
- group provenance: perturbing a `compress_ratio==4` layer must move the CSA slabs and must not
  touch the HCA ones.
- the sidecar: written, read back, and matched against a fresh pack; the fingerprint recomputed
  against its own definition, since it is what decides whether published files stay valid.

Two traps found by running these rather than by reading them, recorded so they are not
rediscovered:

- **safetensors does not write byte-identical files.** It serializes its metadata map in
  nondeterministic order — eight writes of one dict in a single process produced two different
  key orders — so a whole-file assertion is flaky for reasons unrelated to the payload. Compare
  the header's tensor entries, the metadata as a dict, and the payload bytes.
- **A synthetic checkpoint cannot use toy shapes for the compressor/indexer weights.** The
  packer validates the active branch against fixed model dimensions and zero-fills the inactive
  branch at the same sizes, and slabs are allocated from layer 0's template, so a toy-sized CSA
  tensor cannot match the placeholder.
