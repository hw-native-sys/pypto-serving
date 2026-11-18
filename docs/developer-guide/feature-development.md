# Feature Development

This guide describes the path from a serving feature request to a reviewable,
validated change. A feature that crosses the scheduler, cache manager, model
runner, and PyPTO library needs one contract shared by each layer.

## Start with the contract

Write down the externally visible behavior first: the request lifecycle, the
state that is shared, the state that is private, and the failure behavior. Map
each part to its owner before editing code.

| Concern | Owner | Evidence |
| --- | --- | --- |
| Request and page ownership | scheduler and cache manager | lifecycle tests and page tables |
| Host metadata | model runner | focused host tests and ABI checks |
| Device computation | PyPTO Lib | golden cases and device smoke |
| Configuration and rollout | CLI and CI | pinned revisions and a runnable guard |
| User-facing behavior | documentation | a design note and a reproducible example |

Keep the implementation on one path for the supported configuration. If a
library ABI changes, update its callers in the same change and record the
compatibility boundary in the developer guide.

## Develop in layers

1. Reproduce the behavior with the smallest host-side case. Use explicit
   logical positions, page IDs, request slots, and state generations.
2. Implement the owning layer and update its direct callers. Keep scheduler
   ownership, host metadata, and device publication distinct.
3. Validate boundaries before a device run: page wrap, replay origin, empty
   or padding rows, concurrent requests, and reset/reuse.
4. Run the focused device guard on the development topology. Record the exact
   serving, Lib, PyPTO, and toolchain revisions.
5. Run repository hooks and update the documentation before publishing. The PR
   description should state the behavior, dependency pins, and evidence.

## Case 1: V4.1 local KV-cache migration

Issue #306 changed V4.1 cache addressing while preserving the scheduler's
128-token logical block contract. C2A compressed KV and index data use 64-row
physical pages; C1A and SWA use 128-row pages. The complete addressing and ABI
rules are in [V4.1 Cache Migration](v41-cache-migration.md).

The implementation split state into two classes. Shared compressed KV/index
pages remain immutable during a warm prefix replay. Private SWA, Engram, and
compressor state is recomputed from the replay origin. For replay positions
before the hit boundary, C1A/C2A compressed slots are `-1`; publishers skip
those writes while causal reads, page tables, and state updates continue.
Replay and suffix tokens use the normal prefill budget and may cross the hit in
one dispatch.

The development checks for this case were:

- 64-row C2A and 128-row C1A page-stride golden cases, including page
  permutations and boundary lengths;
- cold/warm long prompts with a replay crossing the hit, concurrent requests,
  reset/reuse, and snapshots of all shared KV/index/scale pages;
- K0 and DSpark K5 HTTP guards on four cards using the pinned compiler;
- a startup ABI check that rejects an incompatible Lib checkout.

The measured cold and warm logits are documented as a precision limitation of
recomputing private rolling state. A passing page-integrity check does not imply
bit-identical logits, so accuracy and state-integrity results are reported
separately.

## Review and publication checklist

- [ ] The contract names shared and private state and covers reset/reuse.
- [ ] Host metadata and device publishers agree on logical and physical page
      strides.
- [ ] Focused boundary tests and the intended device guard pass.
- [ ] CI pins the known-good dependency revisions.
- [ ] Documentation links to the owning source and records limitations.
- [ ] The PR history separates implementation from documentation when the
      repository policy requires it.
