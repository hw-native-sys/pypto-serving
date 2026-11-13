---
name: profile-serving-simpler-swimlane
description: Capture serving traces and Simpler host STRACE and combine them into one Perfetto swimlane. Use for a unified view of serving, model host work and runtime execution, or to merge existing traces.
---

# Serving and Simpler Swimlane

Apply these profiling settings to the chosen serving command and workload.
Use a fresh artifact directory and collect both sources from the same run on the
same host. Existing matching artifacts can be converted and merged without rerunning.

## Capture

Enable Simpler host STRACE in the serving process and capture stdout/stderr to
`"$ARTIFACT_DIR/run.log"`:

```bash
export PYPTO_RUNTIME_LOG=v9
export SIMPLER_DEVICE_STRACE_ENABLE=0
```

Add `--profile --profile-level verbose --profile-output "$ARTIFACT_DIR"` to the
serving command. CLI generation automatically starts/stops profiling. For HTTP,
call `POST /start_profile`, run the intended workload, then call `POST /stop_profile`
on the same server; the stop operation flushes and merges serving fragments into
`trace.json`. Keep any intended warmup outside that window.

## Convert and combine

Use the matching Simpler installation's native converter:

```bash
python -m simpler_setup.tools.strace_timing "$ARTIFACT_DIR/run.log" \
  --swimlane "$ARTIFACT_DIR/simpler-host-swimlane.json"

python .agents/skills/profile-serving-simpler-swimlane/scripts/merge_swimlane.py \
  --serving "$ARTIFACT_DIR/trace.json" \
  --simpler "$ARTIFACT_DIR/simpler-host-swimlane.json" \
  --output "$ARTIFACT_DIR/serving-simpler-swimlane.json"
```

Use `--swimlane`, which preserves host process/thread lanes; `--trace-out` instead
creates synthetic invocation lanes. Pass multiple log files to the native converter
if needed. The merger preserves serving events and original timestamps, selects
Simpler events overlapping the serving duration range, and separates source tracks
to avoid PID collisions. Open the combined JSON in Perfetto.

## Check

Confirm the workload succeeded, both traces contain duration events, and the merged
trace contains serving and Simpler lanes with overlapping times. Both sources must
use unshifted host monotonic timestamps in microseconds; overlap alone does not prove
that they came from the same run. Never shift unrelated runs to align them.

Host-only STRACE does not provide device-core lanes or device Effective. Runtime
calls may include device execution and waiting, so their durations are not pure host
CPU time. Do not add nested parent and child durations. Missing model host spans
require instrumentation; enabling `verbose` alone cannot create them.
