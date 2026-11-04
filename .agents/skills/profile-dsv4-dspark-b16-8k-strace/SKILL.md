---
name: profile-dsv4-dspark-b16-8k-strace
description: Run and analyze the fixed DeepSeek V4 DSpark 16-request benchmark on the canonical 16-card DP4/EP16/TP4 topology, with 8192 input tokens, 256 output tokens, DSpark k=7, verbose serving profiling, and sixteen Simpler Host STRACE lanes. Use for DSpark B16 long-context profiling or its Perfetto swimlane; use profile-dsv4-dspark-singlebatch-strace for the short single-request case.
---

# Profile DSV4 DSpark B16 8K

Run one 16-request long-context DSpark batch and generate the same artifact family as the
single-request DSpark profiling skill, including a combined serving/Host Perfetto trace.

## Fixed workload

- GBS: 16; four requests assigned to each of four TP groups
- parallelism: DP=4, EP=16, TP=4 on exactly 16 devices
- prompt: identical deterministic prompts, exactly 8192 model tokens each
- output: 256 tokens per request, `ignore_eos=1`
- sampling: `temperature=0`, `top_p=1`, `top_k=null`
- speculation: `method=dspark`, `k=7`
- serving: offline `generate_batch`, 128-token chunked prefill enabled, prefix cache disabled
- profiling: verbose serving profiler plus Simpler Host STRACE
- device diagnostics: Device STRACE and device log disabled

The runner sends one unprofiled B16/8K/256 warmup, waits for the engine to become idle, and
profiles only the second batch. Host STRACE remains enabled for the process; analysis selects
only invocations overlapping the formal profile window.

EP is fixed at 16. The DSpark checkpoint and kernels use the canonical DP4/EP16/TP4 expert
view; do not change this to EP4 even though each TP group contains four ranks.

## Acquire devices

Obtain exactly 16 devices through the environment's scheduler. Do not kill or reuse devices
owned by another task. Pass the scheduler-assigned IDs to the wrapper.

## Run

From the pypto-serving repository root:

```bash
bash .agents/skills/profile-dsv4-dspark-b16-8k-strace/scripts/run_profile.sh \
  --model-dir /path/to/dsv4-flash-dspark-w8a8 \
  --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --use-compile-cache \
  --artifact-dir /path/to/artifacts \
  --run-id <task-id>
```

The wrapper also accepts `PYPTO_DSPARK_MODEL_DIR` (or
`PYPTO_DSV4_DSPARK_MODEL_DIR`), `PYPTO_PROFILE_PYTHON`, `PYPTO_PROFILE_DEVICES`,
`PYPTO_PROFILE_RUN_ID`, and `PYPTO_USE_COMPILE_CACHE=1`. Set `PYPTO_RUNTIME_ROOT` when
`simpler_setup` is not installed in the selected Python environment. `PTOAS_ROOT` defaults
to the repository `.venv`; an explicit value overrides it.

The wrapper defaults the Host operation, stream-sync, and scheduler timeouts to 400, 440,
and 320 seconds respectively. Explicit environment values take precedence. Reuse compile
cache only with identical device mapping, commits, model configuration, and kernel sources.

## Trace contract

The skill produces:

- `serving-trace/trace.json`: scheduler, serving, worker, executor, and kernel spans
- `simpler-swimlane.json`: Host-only Simpler invocations overlapping the profile window
- `serving-strace-swimlane.json`: serving spans plus sixteen Host STRACE device tracks
- `profile-summary.{json,md}`: workload, performance, callable classification, and decode Steps
- `skill-profile-validation.json`: final artifact validation
- `responses.json`, `warmup-responses.json`, `performance_summary.json`, `server.log`, and
  `run.log`: raw evidence

Open `serving-strace-swimlane.json` in Perfetto. Device STRACE is disabled because it perturbs
timing, so device Effective timing is unavailable. Treat profiler-instrumented throughput and
TPOT as diagnostic rather than an official unprofiled performance result.

## Validate

Report success only when all of these hold:

1. Warmup and profiled batches each contain 16 successful responses with 8192 prompt tokens
   and 256 output tokens.
2. The deterministic prompt tokenizes to exactly 8192 IDs with the selected checkpoint.
3. `server.log` proves DSpark speculation progress and contains Host but no Device STRACE.
4. The serving trace contains `scheduler`, `serving`, `worker`, `executor`, and `kernel` spans.
5. The combined trace contains exactly sixteen Host device tracks.
6. Every rank contains the same continuous target-decode Step sequence.
7. `skill-profile-validation.json` reports `valid: true`.

If warmup fails, preserve `server.log` and report it as an execution failure; do not present a
partial or failure-only trace as a successful profiling artifact.

## Reprocess

Rebuild analysis and the combined trace without reserving devices:

```bash
python .agents/skills/profile-dsv4-dspark-b16-8k-strace/scripts/analyze_profile.py \
  <artifact-dir> --run-id <task-id>

python .agents/skills/profile-dsv4-dspark-b16-8k-strace/scripts/render_16lane.py \
  <artifact-dir>/simpler-swimlane.json \
  <artifact-dir>/server.log \
  <artifact-dir>/serving-strace-swimlane.json \
  --serving-trace <artifact-dir>/serving-trace/trace.json \
  --profile-summary <artifact-dir>/profile-summary.json

python .agents/skills/profile-dsv4-dspark-b16-8k-strace/scripts/validate_artifact.py \
  <artifact-dir>
```
