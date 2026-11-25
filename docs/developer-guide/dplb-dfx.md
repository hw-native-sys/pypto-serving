# DPLB Host Diagnostics

This increment of [issue #325](https://github.com/hw-native-sys/pypto-serving/issues/325)
records grouped-cache placement, ownership, and submitted Host step load in the
serving process. It builds on the grouped DPLB policy and initial Host collector in
[PR #336](https://github.com/hw-native-sys/pypto-serving/pull/336).
Both the policy (`--enable-dplb`) and diagnostics are opt-in.

## Capture a serving window

Add these options to a supported model's normal serving command:

```bash
pypto-serving --model /path/to/model [MODEL_OPTIONS] \
  --enable-dplb \
  --profile --profile-output /tmp/dplb-profile \
  --dplb-dfx-level load --dplb-dfx-max-seconds 60
```

Start the capture, send requests, and stop it:

```bash
curl -X POST http://localhost:8000/start_profile
# Send the workload through the normal completion endpoints.
curl -X POST http://localhost:8000/stop_profile
```

Offline generation uses the same options and captures its generation window.
Omit `--enable-dplb` to observe the existing count-and-rotation policy. Enabling
diagnostics does not enable the load-balancing policy.

Output lives beside the merged profile trace:

```text
/tmp/dplb-profile/dplb/<capture_id>/
  engine-<replica>-<replica_instance_id>.jsonl
  engine-<replica>-<replica_instance_id>.manifest.json
```

For a JSON profile output path, this directory is under that file's parent.
The coordinator shares one capture ID across replicas. Repeated starts preserve
the capture while it is active on all replicas. If a replica has already ended
its capture, a restart seals any remaining active captures before starting a new
window across all replicas. A new capture waits for previous writers to finish
for up to two seconds. If the wait times out, the start fails.
A new capture starts with a baseline of existing owners and requests.
The Host window starts before the ordered worker profile command. An explicit
stop ends Host observations before sending the stop command. If the Host window
ends while a start or stop command is pending, the manifest waits for the worker
reply. A failed command sets `stop_reason=profile_control_failed` and
`complete=false` in the pending manifest. `profile_control_ack` records a start
acknowledgement received within the Host window.

The duration limit stops only the Host collector, including when no requests
arrive. Call `/stop_profile` to stop and merge ordinary profiling as usual.

## Record contract (schema version 1)

Every event has `capture_id`, `replica`, `replica_instance_id`, `event_seq`,
`pid`, `monotonic_ns`, `clock_domain`, and `clock_instance_id`. Host timestamps
are nanoseconds from the host monotonic clock; the instance identifies the
host boot. They cannot be subtracted from device timestamps.

Request events add `request_id`, a capture-local `request_uid`,
`admission_epoch`, and `identity_status`. Client request IDs can be reused after
termination. A UID survives preemption and prefix fallback within a capture;
its epoch advances on each new cache ownership. Reservation adoption and cache
growth do not advance it. Baseline identities are marked `preexisting`; their
epoch starts at one for observed ownership, without claiming earlier history.
UIDs are not joins between independent captures.

| Event | Meaning |
| --- | --- |
| `capture_start`, `capture_end` | Host capture controls; the manifest remains authoritative if an end event is dropped. |
| `load_baseline`, `request_baseline` | Existing grouped load and scheduler requests at capture start. |
| `request_arrived`, `request_terminal` | Scheduler arrival and final completion, cancellation, rejection, or reservation rollback. |
| `prefix_lookup` | Results of the allocator's actual prefix search, including unexamined candidates. |
| `placement_commit` | Successful cache ownership; `commit_scope=cache_ownership` can precede scheduling or be rolled back. |
| `placement_failed` | Allocation raised before a placement commit; follow release/rollback events if ownership was partially allocated. |
| `owner_change` | Acquisition, growth, baseline ownership, or release, with the resulting group load. |
| `reservation_change` | Reserved, adopted, quarantined, or released reservation state. |
| `step_plan` | The submitted `StepCommand`: participating requests, phase, token counts, speculative proposal budget, logical DP-group topology, and bounded request length snapshots. |
| `step_outcome` | The ordered worker result: returned token counts and speculative proposed/accepted counts, or an error/discard status. |
| `profile_control_ack` | Worker acknowledged the existing ordered start-profile command. |

A placement decision has a replica-local `decision_id`, the actual policy
name/version, incoming logical length, requested allocation length, rotation
start, actual scores, ranked candidates, selected group, and reason. Scores
are sorted lexicographically ascending. Explicit placement and ownership
growth bypass ranking; missing scores are not zero scores.

Each `groups` row includes request count, sum and maximum logical lengths,
request-slot limit, and a capacity entry per KV family: required/free/total
blocks, tokens per block, and table limit. `feasible` and `exclusion_reasons`
describe existing ownership, explicit partition constraints, request-slot
limits, KV capacity, and table capacity. Prefix lookup has
`feasibility_scope=prefix_request_limit_only`: cold-allocation capacity is
evaluated when growing the chosen prefix ownership. `prefix_status` and
`required_status` explicitly distinguish unexamined values from measured zero.
Diagnostics reuse the policy's ranking and prefix lookup; they do not run a
second prefix search.

Logical load is `max(allocation_tokens, full_prompt_tokens)` per owner. It
includes a complete chunked-prefill prompt, then grows with the context.
Reservation admission counts its full prompt rather than reserved future
output capacity. Shared prefixes count once per owning request, so this view
is not physical memory occupancy or executed work. Releasing an owner
recomputes the group's sum/max, including empty groups. Capacity and logical
load are separate fields.

## Bounds and incomplete captures

The disabled path creates no collector, writer, diagnostic snapshots, or DFX
files. Active recording copies and serializes observations in the serving
thread; a separate writer performs file I/O. No device synchronization,
diagnostic kernels, or device-to-host copies are introduced.

The Python `DplbDfxConfig` defaults are a 60-second window, 1,024 queued events,
8 MiB of queued/in-flight serialized bytes, 256 KiB per event, 64 MiB of JSONL
per capture across replicas, and 65,536 tracked live requests per replica.
The coordinator divides the JSONL byte budget between replicas. Manifests
are separate small files outside that budget. Library users can lower the
limits through `EngineConfig.dplb_dfx_config`; the CLI exposes level/duration.

The producer never waits for disk or queue space. Queue or event-size pressure
drops records. Output exhaustion and writer errors stop Host collection;
ordinary inference continues. Shutdown waits at most two seconds for a writer;
repeated captures cannot accumulate blocked writers in one replica. A blocked
writer can produce its manifest later, so a missing manifest means the capture
cannot yet be treated as complete. If the directory is unwritable, even the
manifest may be unavailable and a warning is logged.

Each manifest records per-event attempted/written counts, drop reasons,
written/queued/peak bytes, source revision and dirty state when available,
topology, policy, settings, open request count, stop reason, and writer error.
Existing metrics snapshots are stored as `counters_start` and `counters_end`
for explicitly stopped captures. The live-identity limit emits null UID/epoch
with `identity_status=request_limit`, increments `untracked_requests`, and
marks the capture incomplete. No unbounded identity map is created.

`complete=true` means the Host capture lost no records or tracked identities;
requests can still be open at the boundary. It does not certify full request
execution, a complete model trace, or device timing. Use the manifest and
sequence gaps before analyzing any subset of the JSONL.

## First-increment source baseline and validation

The implementation was integrated against these source snapshots on
2026-11-16. This is a source baseline, not a validated NPU compatibility matrix.

| Component | Revision |
| --- | --- |
| Serving main | `e2d597d5b7e948053317878735ff843404c99c2f` |
| PR #336 policy prerequisite | `7c952b7bbfcce9087ea566015db013d200c4f844` (adapted to main, default off) |
| PR #336 Host reference | `dec9730473c7d5712129d990b1de7ddd87616a86` |
| PyPTO source | `5ae2a2df58225fd58f5ae53192ec1a947b118e3b` |
| Simpler source | `42e0d7b97bead99a5820a18142e3fb3e6414be4a` |
| pypto-lib gitlink | `b67a5fd60ccbc1e7451e8b995293c7f1bc76426b` (not initialized) |

`tests/unit/serving/test_dplb_dfx.py` checks decision equivalence, logical
accounting, prefix fallback, reservations/rollback, request identities,
cancellation, bounds, writer failures, repeated captures, and profile controls.
The metadata-only example requires the normal Host dependencies but no PyPTO
compiler or NPU execution:

```bash
python examples/dfx/capture_dplb_host.py --output /tmp/dplb-host-example
```

`step_plan` is emitted after the exact `StepCommand` is accepted by the engine
input queue. Its `requests` rows distinguish prefill and decode work, record
logical lengths before and after the planned tokens, and expose speculative
proposal capacity separately from returned output. `row_extent.valid_rows` and
`padded_rows` remain null because runner preparation has not been observed yet;
the event records that limitation instead of treating table extent as executed
attention work. `groups` includes logical DP-group topology and TP members when
the serving layout can prove them. `step_outcome` is paired by `step_id` and
`dispatch_ordinal`, and reports worker output before scheduler truncation or
application. It preserves the request UID, admission epoch, and cache partition
frozen at submission, including cancelled and preempted requests. The plan's
`speculative_budget_tokens` describes reserved capacity; the outcome's
`speculative_proposed_tokens` and `speculative_accepted_tokens` come from the
worker and must not be counted as client output. Late results never create new
request identities or enter a different capture. Capture boundaries list
in-flight step IDs at the start and end of a capture, including when the
duration limit is reached. The manifest stores these boundaries as
`in_flight_step_ids_at_start` and `in_flight_step_ids`, respectively.
`invocation_id` joins the step to its Host L3 spans, including asynchronous
submit and wait. `planned_execution_slot` and `planned_slot_generation`
describe the engine's logical dispatch sequence. Actual worker buffer
ownership remains unobserved in the step snapshot.

This increment leaves SWA/HCA/CSA task semantics, per-rank device cost, and
report generation to later work. `--dplb-dfx-level
timing` fails explicitly. NPU smoke tests and measured overhead remain required
before making performance claims. The next device prototype must verify runtime
invocation identity, semantic coverage and clock domains on a matched, built
stack before enabling timing.

The local Host checks used Python 3.10, pytest 9.0.3, msgspec 0.22.0,
FastAPI 0.162.5, NumPy 1.26.4, and Torch 2.6.0+cpu. This environment is not an
NPU runtime baseline. Verification passed 172 placement/collector/engine/
scheduler checks, 85 grouped-prefix/cache-metrics checks, and three DPLB CLI
flag checks. Ruff, copyright headers, and English-content checks passed.
Twenty-one other CLI checks and four program-acceptance checks were blocked
by the missing PyPTO package. A pre-existing external PyPTO installation also
failed to import because its Python source expected a `CommEngine` binding
absent from its native extension; it was not used as validation evidence.
The documentation site was not built because the shared theme was not present.

## Follow-up validation (2026-11-17)

The existing `zm_pypto` Conda environment resolved the earlier PyPTO import
blocker. All 91 CLI and program-acceptance checks passed. An expanded Host
run initially passed 518 of 530 cases; 12 DSpark cases required the missing
Lib sources. Git submodule cloning failed over the network, so the exact
gitlink commit `b67a5fd60ccbc1e7451e8b995293c7f1bc76426b` was obtained from
GitHub's commit archive. The archive SHA256 was
`7d0db48483faf3e348b217eefa676ae5afa0a76a60bab72d3825d84db5c72ef1`.
It supplies source files in `pypto-lib/`; it is not an initialized Git checkout.
No native artifacts were copied and no dependency gitlink was changed.

After adding those sources, all 182 DSpark Host cases passed. Seven new
smoke-verdict checks also passed: the combined, deduplicated result is 537
cases with no remaining failures. This was a staged validation, not one run
of 537 tests. The environment used Python 3.10.22, Torch 2.8.0+cpu,
torch_npu 2.8.0, NumPy 2.2.6, and pytest 9.1.1. Packages were not changed.
PyPTO came from the installed baseline checkout at
`1dedb26c329407360c5c5ce6ecb495565c2f154e`; its runtime was
`355ce3015e3e344d1fc3a61c206ee01823ce292c`. These differ from the original
workspace source snapshots above; NPU compatibility is still unverified.

The manual smoke checks the actual `TASK_DEVICE` allocation, starts a K7
server, warms compilation, runs mixed prompt lengths and a repeated prefix,
and validates two captures with repeated profile-start controls. The verdict
requires actual prefix hits, contiguous event sequences, matching byte/record
counts, terminal events for placed requests, and zero remaining logical load.
It stops only the server process it created. It does not measure overhead or
cover every hardware cancellation, preemption, or in-flight capture scenario.

Configuration can be checked without devices:

```bash
python tests/manual_dplb_dfx_smoke.py --check-only \
  --model /path/to/dspark-checkpoint --ep 8 \
  --output-dir /tmp/dplb-dfx-config-check
```

Real execution must use the matching environment/toolchain and task queue:

```bash
task-submit --device auto --device-num 16 --max-time 0 --timeout 0 --run \
  'PYPTO_WORKER_INIT_TIMEOUT=3600 PYPTO_DSV4_STARTUP_TIMEOUT_SECONDS=4200 \
    python tests/manual_dplb_dfx_smoke.py \
    --model /path/to/dspark-checkpoint --ep 16 --startup-seconds 4200 \
    --output-dir /tmp/dplb-dfx-npu-smoke'
```

The worker-init timeout is independent of the outer HTTP health deadline.
Allow time for the workload and shutdown; the queue limit is disabled above.
These bounds allow a diagnostic retry; they do not guarantee
that initialization will complete on a given dependency stack.

Use `--no-dplb` to exercise count balancing and `--dfx-level off` for a
generation control without DFX files. Use separate output directories per run.
Each smoke also allocates a fresh private build directory under
`PYPTO_PROG_BUILD_DIR`, or under its output directory when that variable is
unset. The selected directory is saved in `configuration.json`. Reusing a
mutable build directory can pair a regenerated kernel table with an older
source-adjacent orchestration binary; separate profile outputs alone do not
prevent that mismatch.
The configuration-only check passed locally. The initial sandbox view showed
only two devices and a stopped queue daemon. An outside-sandbox check corrected
that observation: the host queue was running with sixteen devices. Real
submissions must use that host queue. The smoke accepts task-submit's appended
`--device` option and validates the actual allocation through `TASK_DEVICE`.

The eight-device task allocated devices 0-7 and started model initialization,
but failed before HTTP health and capture start: the post-resident KV budget
was -5,695,729,664 bytes against a 570,425,344-byte minimum. The hardware has
64 GiB per device. The task exited and released all eight locks; a subsequent
device inspection found no remaining NPU processes from the run.

The sixteen-device DP4/EP16/TP4 run loaded all 43 target layers and 42 drafter
banks on the host, and all sixteen chip processes reported ready. It then
reached the worker's independent 1800-second initialization timeout before
HTTP health or capture start. A further run with the coordinated deadlines
above was stopped by the user during resident initialization (exit 130).
All recorded task processes exited and all sixteen device locks were released.
Those initial attempts produced no generation responses or DFX captures.
They do not establish successful NPU generation, capture correctness,
observer overhead or a DPLB performance improvement. Later M2 results follow.

## M2 validation

The smoke now requires `step_plan`/`step_outcome` pairs, stable request identities,
logical DP-group membership and token accounting, and explicit in-flight
boundary explanations. A placement-only capture cannot pass this M2 verdict.
Host integration tests exercise the actual engine scheduling/result paths with a
deterministic fake worker, including two in-flight steps, cancellation,
preemption, errors/timeouts, repeated windows, and identical DFX off/load command
and output replay. Real NPU coverage must be recorded separately; these tests
still do not validate execution slots or L3 invocation correlation.

The November 18-19, 2026 validation accumulated 820 distinct passing Host
cases across staged runs. A real sixteen-device DP4/EP16/TP4 K7 load smoke
completed ten requests and 320 client output tokens. Both captures passed:
28 and 24 planned steps paired with their worker outcomes, four prefix hits,
no event loss, and logical group load returning to zero. Worker-returned token
counts are measured before scheduler application and differ from client output.

The subsequent DFX-off control failed on its first warmup request after a
generic AICPU exception (507018), followed by a failed device drain (507015)
and unconfirmed communication-window cleanup. The underlying cause remains
undetermined. Hardware off/load output equivalence and observer overhead are
therefore unverified. A capture-stop boundary fix added after the load process
started was validated by Host integration tests only. Both task phases exited
and released their device allocation.

### Follow-up controls on November 19, 2026

Fresh, independent build directories let both off and load complete ten
requests and 320 client tokens each. Load produced two complete captures with
49 paired steps and four prefix hits. No DFX files were created in off.

The earlier failure used a regenerated 334-entry kernel table alongside an
older orchestration binary. That binary's executable text matches a fresh
333-entry build, whose function IDs differ from the regenerated table. The
faulting PC maps to HCA merge argument unpacking, consistent with the wrong
kernel receiving another task's arguments. The smoke now always allocates a
private build directory, including when the caller supplies a build base.
This avoids the observed mismatch; source-adjacent binary invalidation in
PyPTO still needs a separate fix.

Strict output comparison passed for nine of ten requests. The first window's
garden response differed; the second load window matched off. Usage and
finish reasons matched for every request. Hardware output equivalence is
therefore still unproven. A serial control with `--concurrency 1` is queued
to investigate batch/cache-history effects. These diagnostic smoke runs do
not establish observer overhead or a DPLB performance improvement.
