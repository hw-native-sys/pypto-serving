# Phase D frozen environment and preflight probes

This directory contains the reproducible D0 environment boundary and the
manual preflight probes added while implementing Phase D.  The D0 artifacts
freeze the known-good DSpark stack, the Phase C transfer overlay, and the
commands used to inspect, rebuild, verify, and run the non-PD model guard.
The later probes and service launcher exercise the D1-D8 control, resource,
K7 continuation, bounded-admission, and recovery contracts.  The final
two-host 16-device evidence is indexed from the Phase D task document.

The container stage is fixed at:

```text
/workspace/phase-d-validation
```

The source and native versions are recorded in `d0-lock.json`.  The lock is
additive: `source-manifest.json` describes the original exported stack, while
`d0-lock.json` lists every intentional Phase C-through-D8 overlay in the closed
Phase D tree.  The verifier checks the original digest for every untouched
source and the final overlay digest for every overlaid or newly added source.
This avoids blessing arbitrary working-tree drift after the PD code is
installed while retaining the immutable D0 repository revisions.

## Container use

Copy the scripts in this directory and `d0-lock.json` to the stage root.  Then
select and verify the environment without allocating an NPU:

```bash
source /workspace/phase-d-validation/env_pinned_stack.sh
bash "$DSPARK_STAGE/inspect_d0_environment.sh"
python "$DSPARK_STAGE/verify_d0_environment.py"
```

When starting from the original `source-manifest.json` export rather than the
already prepared A/B stage, first overlay the current `pypto-serving` files and
apply the Python-only owner hooks exactly once:

```bash
export PYPTO_HOME=/workspace/phase-d-validation/pypto
bash "$PYPTO_HOME/../pypto-serving/tests/manual/pd/phase_c/apply_runtime_overlays.sh"
```

The patch entrypoint performs dry-runs before changing either PyPTO or Simpler;
reapplying it to an already overlaid tree fails instead of silently drifting it.

`env_pinned_stack.sh` sources the container's common CANN/Mooncake setup, then
overrides all PyPTO, Simpler, pypto-lib, Serving, PTOAS, ISA, executable, and
native-library paths with the frozen stage.  Image-provided packages such as
Torch are reported but are not required to have the same version on both
validation images.  All project imports must resolve inside the frozen stage.
The historical `verify_pinned_stack.py` and `inspect_pinned_environment.sh`
names are retained as compatibility entrypoints and delegate to the D0 tools.

For a clean exported bundle, `install_pinned_stack.sh` performs the initial
CPU build and editable installation.  `rebuild_pinned_runtime.sh` rebuilds the
frozen Simpler native target without changing revisions.  Both default to the
build parallelism loaded from PyPTO; set `DSPARK_BUILD_JOBS=16` for the approved
validation-host build setting.

The archived non-PD K0/K7 environment guard is:

```bash
DSPARK_RUN_NAME=<new-run-id> bash "$DSPARK_STAGE/run_pinned_service.sh"
```

It refuses to overwrite an evidence directory, verifies D0 first, and starts
the service only when the NPU process table is empty.  Never reset an NPU or
terminate an unrelated process to make the guard run.

The command above is retained only to explain or reproduce old D0 evidence; it
does not define the current PD acceptance matrix.  All new Phase D development,
PD runs, regressions, failure drills, and acceptance are K7-only.  K0 is
historical evidence, not a supported mode, fallback, test target, or acceptance
gate.  Generic validation may reject a non-K7 configuration, but no new K0 path
is maintained.  In the PD shape, P declares the K7 PD capability but produces
only the target Prefill state; D initializes its request-local drafter state
after target-cache adoption and performs K7 Decode.

The Phase C hardware pair remains under `../phase_c/`.  It proves the owner,
eight-region transfer layout, Mooncake AscendDirect D2D path, and cleanup on
the frozen stack; it is not a substitute for the D1-D8 Serving P/D workflow.

## D1-D5 preflight probes

`host_control_probe.py` is a CPU-only cross-host check for the authenticated
fixed-P/D control session.  It exchanges a synthetic topology `(16, 4)`
registry with all eight physical regions per rank and performs one stable
handoff query/status round trip.  It does not allocate an NPU and does not
prove the data plane.

`cache_registry_probe.py` starts one real 16-device DSpark engine role.  Run
`prefill` and `decode` serially on the same otherwise-idle host, checking the
NPU process table before and after each run.  The probe validates the resident
cache geometry, 16 owner-local Mooncake engines, seven scheduler groups, eight
physical transfer regions and matching registry fingerprint.  The prefill
role also validates a TP4 final-chunk plan and source guard; the decode role
validates an authorized grouped reservation, rejection of premature adoption,
scratch isolation and deterministic abort release.  New probe runs declare the
K7 PD contract: P normalizes its local executor to target-only Prefill, while D
initializes the K7 executor.  The older 2026-09-12 probe evidence remains
historical K0 resource evidence and is not rerun as a product configuration.

The cache probe intentionally stops before peer exchange, P-to-D transfer or
Decode continuation.  A successful result is therefore a D5 preflight result,
not a completed two-host D5 smoke.  Do not run both roles concurrently on one
device set and never reset a device or terminate an unrelated process to make
the probe run.

## D5 K7 service smoke

`run_pd_k7_service.sh` starts exactly one P or D service in the frozen stage.
It requires explicit role, fixed-peer addresses, ports, run identity and a new
evidence-directory name through environment variables.  Both sides declare
K7; the P config normalizes only its local executor to target Prefill, while D
loads the drafter.  The startup handshake permits the two model loads to finish
at different times, and a native transfer attempt has its own bounded deadline.

The script refuses an occupied NPU process table and an existing evidence
directory.  Its pid files identify only the shell and Python process created by
that run.  Stop those exact pids normally; never use a broad process-name match,
and never reset an NPU.  A completed D5 result must come from the P HTTP endpoint
and contain 64 prompt tokens, 128 completion tokens, `finish_reason=length`,
coherent text and D-side K7 acceptance statistics.

The accepted D5 run is `phase-d-k7-d5-20260912-04`: the prefill host ran P,
the decode host ran D, the response was 64+128 tokens with coherent text, and D
reported `verifies=68, matched=61, proposed=469, accepted=129,
mean_len=1.90, fallbacks=1`.  This is the single-request semantic smoke; D8
adds a 192-token prompt to force two Prefill transfer chunks.

## D6-D8 admission, journal, and recovery

The first control session is ordered and has one reply stream.  It therefore
does not pretend to multiplex wire replies: `--pd-max-pending-handoffs` bounds
the active request plus queued HTTP waiters, while one admitted handoff at a
time executes the reserve/prefill/transfer/commit/decode flow.  Requests over
the bound receive HTTP 503 with `Retry-After`; D reservation failure supplies
the device-capacity backpressure.  Request-aware control multiplexing and
performance overlap remain Phase E work.

Every production PD launch requires a new `--pd-journal-path`.  The JSONL
journal is append-only, fsynced, mode 0600, bound to run/node/role/control
incarnation, and protected by a SHA-256 hash chain.  It intentionally excludes
prompts, secrets, native addresses, and provider envelopes.  The hash chain
detects corruption but is not a malicious-tampering proof without an external
anchor.  An existing path, a partial/corrupt chain, or an unresolved handoff
is fail-closed; preserve it for audit and start recovery with a new path and a
strictly newer generation/control incarnation.

Deterministic `NOT_SUBMITTED`/`FAILED_DEFINITE` outcomes may retry up to
`--pd-max-transfer-attempts`.  `UNKNOWN` never retries: the destination stays
quarantined, both services become unhealthy, and the owning packed runtime is
retired before a new generation may reuse the devices.  The launcher watches
`/health`; a post-startup 503 causes normal TERM of only its recorded Python
child.  It does not reset devices or match processes by name.

For an intentional stop, terminate the exact D launcher PID from its
`service.pid`, wait until D's NPU process table is empty, and then do the same
for P.  The launcher has its own session and, after the main service exits,
allows chip children 60 seconds to leave; it then sends TERM and finally KILL
only to PIDs still proven to belong to that session (30-second and 10-second
bounded windows).  Current Mooncake/ADXL teardown can log `DeregisterMem`
errors and leave some chip children for this process-level fallback.  This is
an operational limitation, not permission to reset an NPU or terminate an
unrelated process.

The validated recovery sequence is deliberately external and explicit:

```text
preserve generation-N journals
  -> confirm both old P/D process tables are empty
  -> choose generation N+1 and control incarnation N+1
  -> use new journal paths
  -> start D and P with the same run/route identity
  -> re-submit the request, causing a fresh Prefill and K7 continuation
```

Phase D does not provide automatic routing, automatic process restart, or
transparent replay of an already exposed response.  It provides the safe
fail-closed boundary and the reproducible explicit recovery operation on which
those later capabilities can be built.

The accepted two-chunk closure run is
`phase-d-k7-d8-multichunk-20260912-02`: 192 prompt tokens forced two transfer
chunks, D adopted both and generated 128 tokens with real K7 statistics.  The
preceding diagnostic run exposed an owner-lifetime sequencing bug because each
chunk restarted `attempt_sequence` at zero.  The runner now maintains a
monotonic sequence independently per owner/rank, and a focused unit test keeps
that multi-chunk contract from regressing.

At Phase D closure both hosts pass the same final lock: 12,491 base-manifest
files, 83 additive overlays, and eight stage-root entrypoints.  The overlays
include 78 final `pypto-serving` files plus five already frozen PyPTO/Simpler
Python hooks.  One of those five is the identical A/B `device_runner.py`
startup revision-reporting fix; recording its digest prevents it from becoming
an invisible environment difference.
