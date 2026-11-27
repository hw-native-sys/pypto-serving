# DPLB Host Diagnostics

This first increment of [issue #325](https://github.com/hw-native-sys/pypto-serving/issues/325)
records grouped-cache placement and ownership in the serving process. It builds
on the grouped DPLB policy and initial Host collector in
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
the current capture while any replica is still collecting. Once all replicas
have stopped collecting, retry start if any previous writer is still finishing.
A new capture starts with a baseline of existing owners and requests.
The Host window starts before the ordered worker profile command. An explicit
stop ends Host observations before sending the stop command. If the Host window
ends while a start or stop command is pending, the manifest waits for the worker
reply. For a coordinated start, it waits for all replicas. A failed start marks
the coordinated capture's pending manifests with `stop_reason=profile_control_failed`
and `complete=false`, including on replicas that acknowledged the start.
A failed stop marks the affected replica's pending manifest the same way.
`profile_control_ack` records a start acknowledgement received within the Host window.

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
| `profile_control_ack` | Worker acknowledged the existing ordered start-profile command. |

A placement decision has a replica-local `decision_id`, the actual policy
name/version, incoming logical length, requested allocation length, rotation
start, actual scores, ranked candidates, selected group, and reason. Scores
are sorted lexicographically ascending. Explicit placement and ownership
growth bypass ranking; missing scores are not zero scores.

Each `groups` row includes request count, sum and maximum logical lengths,
request-slot limit, and a capacity entry per KV family: required/free/total
blocks, tokens per block, and table limit. `feasible` and `exclusion_reasons`
describe existing ownership, explicit partition constraints, scheduler partition
limits, request-slot limits, KV capacity, and table capacity. PD Prefill uses
`scheduler_partition_limit` for groups unavailable to the current step.
`candidates` lists the groups eligible for allocation ranking. Prefix lookup has
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

## Validation and scope

This increment provides Host placement and ownership records for issue #325.
Scheduled dispatch snapshots, runner/L3 and execution-slot identities, device
attention timings, analysis reports, and formal overhead measurements are
follow-up work. A complete Host manifest does not imply completion of the
issue's end-to-end acceptance criteria.

The grouped policy is an adapted prerequisite from PR #336. Its CLI is default
off; the diagnostic level is independent of the policy switch. The current
integration preserves main's shared, read-only grouped-prefix selector and
DDR-offload shutdown order.

Host regression checks cover unchanged allocation choices with capture enabled,
full-prompt accounting, prefix fallback, reservations and rollback, cancellation,
preemption, reused request IDs, bounds, writer failures, repeated captures,
and ordered profile controls:

```bash
python -m pytest -q tests/unit/serving/test_dplb.py \
  tests/unit/serving/test_dplb_dfx.py \
  tests/unit/serving/test_dplb_smoke_validation.py \
  tests/unit/serving/test_grouped_prefix_cache.py \
  tests/unit/serving/engine tests/unit/cli
python examples/dfx/capture_dplb_host.py --output /tmp/dplb-host-example
```

The manual NPU smoke starts a K7 server, warms compilation, sends mixed prompt
lengths and a repeated prefix, and validates two capture windows. It requires
actual prefix hits, contiguous event sequences, matching manifest byte/record
counts, terminal events for placed requests, and zero remaining logical load.
Each run uses a private program build directory to avoid pairing a regenerated
kernel table with a previous run's orchestration library.

Check the configuration without allocating devices:

```bash
python tests/manual_dplb_dfx_smoke.py --check-only \
  --model /path/to/dspark-checkpoint --ep 16 \
  --output-dir /tmp/dplb-dfx-config-check
```

Run with the matching compiler, Lib, and runtime environment through the task
queue. The example reserves sixteen devices for DP4/EP16/TP4:

```bash
task-submit --device auto --device-num 16 --max-time 0 --timeout 0 --run \
  'PYPTO_WORKER_INIT_TIMEOUT=3600 PYPTO_DSV4_STARTUP_TIMEOUT_SECONDS=4200 \
    python tests/manual_dplb_dfx_smoke.py \
    --model /path/to/dspark-checkpoint --ep 16 --startup-seconds 4200 \
    --output-dir /tmp/dplb-dfx-npu-smoke'
```

Use `--no-dplb` for count balancing and `--dfx-level off` for a generation control
without DFX files. Use separate output directories per run. The smoke records
functional evidence; it does not establish generation equivalence, observer
overhead, or a DPLB performance improvement.
