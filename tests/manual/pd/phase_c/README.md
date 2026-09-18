# Phase C production-path smoke

This directory carries the worker-local transfer harness ported to the current
DeepSeek V4 Flash/DSpARK serving baseline. It validates a real PyPTO source
kernel followed by Mooncake AscendDirect D2D writes into peer resident NPU
allocations. It is not the Phase D request coordinator or full model handoff.

## Current serving compatibility

PR223 splits the old `cmp_kv` allocation into `hca_cmp_kv` and `csa_cmp_kv`.
The transfer registry therefore exposes eight physical regions:
`ori`, `hca_cmp`, `csa_cmp`, `idx_k`, `idx_scale`, `hca_state`, `csa_state`,
and `csa_inner_state`. `idx_k` and `idx_scale` remain one atomic logical group.

The production bridge also needs Python-only owner-service hooks in PyPTO and
Simpler. Apply the pinned-source patches before running this harness:

```bash
export PYPTO_HOME=/path/to/pypto
bash tests/manual/pd/phase_c/apply_runtime_overlays.sh
```

The patches are a deliberately narrow compatibility overlay pinned to PyPTO
`df4dc0093` and Simpler `22385d2b`. They add explicit
`prepare(chip_service_factories=...)` propagation, resident allocation
retain/release, chip-child service lifecycle, pin-aware free rejection, and a
binary-cache revision check which ignores only Python control-plane changes.
They do not modify or rebuild native libraries. The apply script uses zero
fuzz and fails closed on a different base revision; porting this small overlay
is the only PyPTO/Simpler work required when the framework stack advances.

## Validation record

On 2026-09-11 the current serving base
`e79a6ec9da716d8ab5f89b2ff430c494babf06d8` passed all 130 selected transfer
tests in both `serving-a` and `serving-b` containers. Run
`phase-c-pr223-one-20260911-03` then passed a one-rank A-to-B hardware smoke:
both roles reported `status=ok`, `components=8`, `payload_validated=true`, and
`source_kernel=true`.

The successful smoke used physical chip id 8 on both hosts because B's physical
chips 0-7 were occupied. `npu-smi` groups two physical chips under each displayed
NPU row: displayed NPU 4 corresponds to physical chip ids 8 and 9. Always select
from the process table by physical chip id and never infer availability from
AICore utilization.

Evidence is stored in each container under:

```text
/home/sj/git/phase-d-pypto783-lib216-20260910/
  phase-c-pr223-one-20260911-03/sender.log
  phase-c-pr223-one-20260911-03/receiver.log
```

A normal pair starts the receiver first, waits for `receiver_ready`, and then
starts the sender with the same run id, port, rank count, and physical chip list.
A successful endpoint alone is insufficient: both `phase_c_result` records must
pass and the post-run NPU process table must contain no process from the smoke.
