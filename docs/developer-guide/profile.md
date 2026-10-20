# Profiling

The `pypto_serving.tools.profile` module records serving and generation activity in the [Chrome Trace Event Format](https://chromium.googlesource.com/catapult/+/HEAD/tracing/README.md). It is disabled by default; when disabled, the helpers return without writing trace events. When enabled, it records spans from the HTTP API, scheduler, engine, worker, executor, and NPU kernel dispatch paths.

Each process writes to a separate JSON Lines fragment. This avoids cross-process writes to one file and preserves the events that were flushed if a run exits before the final merge. On a normal shutdown, the entry points merge the fragments into a single `trace.json` file that can be opened in a trace viewer such as [Perfetto](https://ui.perfetto.dev/).

Profiling explains where time is spent inside a run. To measure externally visible performance, record request rate or concurrency, prompt and output lengths, latency, token throughput, and errors from a repeatable client workload. Keep model, device placement, sampling settings, and runtime capacity fixed when comparing runs; report first-run startup separately from steady-state generation.

## Profile Setup

HTTP serving and offline generation profiling are configured with the `pypto-serving` profiling options documented in [CLI Reference](../cli-reference/pypto-serving.md#profiling-arguments).

In serve mode, recording begins only after `POST /start_profile`. In generate mode, recording begins when the run starts generating and the trace is merged when it finishes. The CLI resolves `--profile-output` to an absolute path before spawning workers, so every process writes to the same location.

The `SA_PROFILE_OUTPUT` / `SA_PROFILE_LEVEL` environment variables configure the recorder for library users and are read by `scripts/merge_profile.sh`; the CLI entries replace them with explicit CLI configuration.

The event levels are:

- `e2e`: request, scheduler, engine, executor, and worker spans.
- `kernel`: NPU kernel dispatch spans.
- `verbose`: enables all levels, including any fine-grained events marked as verbose.

To capture chip-level AICore and AICPU timing as well, add `--profile-chip-level 2` (or `4` for scheduler and orchestrator phases) alongside `--profile`. This enables PyPTO's chip swimlane only during the offline generation window or between HTTP `/start_profile` and `/stop_profile`. Each completed L3 dispatch is saved under `<profile-output>/chip/captures/worker-*/<dispatch>_<program>/rank*/d*/`, including `chip_swimlane_records.json` and, on supported hardware, `merged_swimlane_*.json`. For a `.json` profile output, the chip directory is the sibling path with a `.chip` suffix. PyPTO's normal JIT cache policy remains in effect.

If a dispatch produces no new chip records or its capture cannot be archived, stopping the profile reports an error. The host-side trace is still flushed and merged.

**Diagnostic limitation:** on real hardware, the current PyPTO L3 swimlane runs each profiled dispatch twice to collect the dependency graph and then timing. It does not restore mutable inputs or KV cache between passes. Captured requests can therefore produce different tokens or modify cache state; use a disposable profiling workload, not production traffic. Chip capture also serializes profiled async dispatches and can be expensive. Level 1 records AICore task timing, level 2 adds AICPU dispatch/finish timestamps, level 3 adds scheduler phases, and level 4 adds orchestrator phases. The regular `trace.json` remains the host-side trace; chip captures are separate files.

For a directory output, the module creates:

```text
/tmp/pypto-profile/
├── fragments/
│   ├── trace.<pid-1>.jsonl
│   └── trace.<pid-2>.jsonl
└── trace.json
```

When the output ends in `.json`, that path is used for the merged trace and the fragments are stored in a sibling directory. For example, `--profile-output=/tmp/run.json` produces `/tmp/run.json` and `/tmp/run.json.fragments/`.

Use a different output path for each run. Starting a new main process removes stale `trace.*.jsonl` files from its fragments directory, and merging replaces the existing trace file. Absolute output paths are recommended, especially when using the manual merge script.

## Profile Offline Generation

Pass the profiling options to the generate mode:

```bash
pypto-serving \
  --model /path/to/Qwen3-14B \
  --prompt 'Huawei is' \
  --platform a2a3 \
  --device 0 \
  --max-model-len 512 \
  --generate-config '{"max_new_tokens": 5}' \
  --profile \
  --profile-output /tmp/pypto-profile-offline \
  --profile-level e2e,kernel
```

The CLI wraps the generation window with profile start/stop and merges the fragments when the run finishes, so the completed trace is normally available at `/tmp/pypto-profile-offline/trace.json` even if generation raises an exception.

## Profile HTTP Serving

Start the server with profiling configured:

```bash
pypto-serving \
  --model /path/to/Qwen3-14B \
  --backend npu \
  --platform a2a3 \
  --device 0 \
  --max-model-len 512 \
  --port 8899 \
  --profile \
  --profile-output /tmp/pypto-profile-serving \
  --profile-level e2e,kernel
```

Wait for the server to become ready, start profiling, send the workload, and then stop profiling:

```bash
curl -X POST http://localhost:8899/start_profile
# Send the requests to profile.
curl -X POST http://localhost:8899/stop_profile
```

The endpoints follow the vLLM profiling API. `start_profile` starts recording in the API process and every replica worker. `stop_profile` waits for every worker to flush its fragment and then atomically writes the merged trace to `/tmp/pypto-profile-serving/trace.json`; the serving process keeps running. Repeated `start_profile` calls while active and `stop_profile` calls while inactive are safe.

A graceful server shutdown also attempts to stop active recorders and merge available fragments, so it remains a fallback when profiling was left active.

## Merge Profile Fragments Manually

Use `scripts/merge_profile.sh` when automatic merging did not run, for example after an interrupted serving process. Pass the same path used for `--profile-output`:

```bash
./scripts/merge_profile.sh /tmp/pypto-profile-serving
```

Alternatively, provide the path through the environment:

```bash
SA_PROFILE_OUTPUT=/tmp/pypto-profile-serving \
  ./scripts/merge_profile.sh
```

Stop all profiled processes before running the script so their buffered events are flushed to the fragments.

The script accepts both directory and `.json` output forms. It locates every `trace.<pid>.jsonl` fragment, ignores incomplete or malformed lines, and atomically replaces the merged trace. A successful run prints the event and fragment counts, for example:

```text
Merged 1136 events from 2 fragments into /tmp/pypto-profile-serving/trace.json
```

The fragments are retained after merging, so the script can be run again. It fails without changing the trace if the fragments directory contains no trace fragments.

## Add Instrumentation

The public helpers are available from `pypto_serving.tools.profile`:

```python
from pypto_serving.tools.profile import profile_duration, profile_instant, profile_span


with profile_span(
    "scheduler.schedule",
    cat="scheduler",
    args={"batch_size": batch_size},
):
    schedule_batch()

profile_instant(
    "request.queued",
    cat="request",
    args={"request_id": request_id},
)

profile_duration(
    "kernel.execute",
    dur_us=kernel_time_us,
    cat="kernel",
    level="kernel",
)
```

- `profile_span()` is a context manager that records a complete duration event.
- `profile_instant()` records a point-in-time event.
- `profile_duration()` records an already measured duration in microseconds. Pass `ts_us` to set its start timestamp; otherwise the interval ends when the helper is called.
- `is_enabled(level)` can guard expensive argument construction or optional instrumentation.
- `get_profiler(process_name=...)` initializes the process-local recorder and sets the process name shown in the trace viewer. Pass `initially_active=False` when the process must wait for an external control signal.
- `start_profile()` starts a fresh process-local recording session.
- `stop_profile()` flushes and stops the process-local recording session.
- `merge_profile()` closes the current process recorder and builds the final trace. Call it only after child processes have stopped and flushed their fragments.

The `cat` value groups events in the trace viewer, while `level` controls whether the event is collected. The helpers are no-ops when profiling or their level is disabled. Keep event arguments small and JSON-serializable to limit trace size and recording overhead.
