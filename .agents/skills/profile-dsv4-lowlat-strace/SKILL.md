---
name: profile-dsv4-lowlat-strace
description: Capture and validate a DeepSeek V4 lowlat single-batch TP8 K5 native-pipeline profile, combining serving spans and eight Simpler Host STRACE lanes into a Perfetto JSON. Use for current lowlat rolling-cache profiling; DSpark and MTP have separate workflows.
---

# Profile DSV4 Lowlat

Use the public [Lowlat profiling workflow](../../../docs/models/deepseek-v4/lowlat.md#profiling)
and the skill-local scripts below. This workflow serves one active request (B1),
DP1/TP8/EP1, K5, rolling target caches, and pipeline2/launch1. It does not
exercise multi-batch or concurrent requests.

Acquire eight devices through the site's scheduler; use `task-submit` in the
shared environment. Activate the compatible lowlat environment inside the job.
Never pass `--use-compile-cache`. The tool does not acquire devices itself.

```bash
python .agents/skills/profile-dsv4-lowlat-strace/scripts/lowlat_profile.py run \
  --model-dir "$MODEL_DIR" --devices "$TASK_DEVICE" \
  --artifact-dir "$ARTIFACT_DIR" --run-id "$TASK_ID"
```

Use a fresh empty artifact directory. Defaults are prompt `Huawei is` repeated
32 times (128 tokens on the accepted checkpoint), 64 output tokens and chunk512.
The tool records actual prompt IDs. It runs warmup and an unprofiled reference
before starting the on-demand verbose profiler for one formal HTTP request.
Host STRACE is enabled, Device STRACE is disabled. Sources are hashed before
and after the run, and accepted work is drained during normal server shutdown.
The launcher sets both `PYPTO_RUNTIME_LOG` and `PYPTO_SIMPLER_LOG_LEVEL` so
CLI/worker logging cannot suppress Host spans after an early runtime import.
Benchmark-only mode explicitly sets both thresholds to ERROR.

The main output is `serving-strace-swimlane.json`, readable in Perfetto.
`decode-only-swimlane.json` retains all eight native Host lanes and serving
events in the native decode window, excluding prefill and its first output
token. Spans crossing the window are clipped and marked. Raw
`serving-trace/trace.json`, `simpler-swimlane.json`, `server.log`, responses,
`profile-summary.{json,md}` and `skill-profile-validation.json` accompany it.
Only report success when the command exits zero and the validation says
`valid: true`; preserve failures and retry into a new artifact directory.

The analyzer joins each submit's pinned native run ID to all eight rank roots,
and checks explicit reservation/frame identity and actual emitted tokens.
Do not assume one parent means one token or classify programs from call counts.
Parent lifetimes may overlap while device launch depth remains one. Do not sum
those lifetimes, or call Host `runner_run` pure kernel time. Instrumented TPOT
is diagnostic; compare the separate unprofiled reference before reporting cost.

Rebuild and validate without acquiring devices:

```bash
python .agents/skills/profile-dsv4-lowlat-strace/scripts/lowlat_profile.py analyze "$ARTIFACT_DIR"
```

For a representative native-pipeline acceptance, inspect the reported
`cycles_seen`, `frames` and `max_accepted_parents`: the default accepted input
exercises one/two cycles, both frames and depth two. A different prompt or
output budget may legitimately exercise less of the pipeline.


## Long-context TPOT without trace collection

For a TPOT-only request, use `run --benchmark-only --warmup-count 1`
with a short `--warmup-prompt`, the requested long `--prompt-file`,
`--max-model-len`, `--output-tokens`, and explicit memory utilization.
This executes one short warmup and one full long request, with the verbose
recorder inactive and Host STRACE disabled. Prefix caching must be off when
using a different warmup prompt. This mode produces responses and a
`run-summary.json` with `mode: benchmark`; it does not produce a validated
swimlane. Check exact prompt/output token counts, `finish_reason: length`,
source integrity, and the benchmark measurement before reporting TPOT.

For a long-context profile, omit `--benchmark-only` and keep a short
`--warmup-prompt --warmup-count 1 --prefix-cache-mode off`. The formal request
is profiled once, including prefill; the analyzer also exports its decode-only
window. A different warmup is not a comparable unprofiled reference: baseline
TPOT and instrumentation delta are null, and its text is not compared against
the long request. Use a separate identical unprofiled workload for performance
comparisons. A benchmark-only log cannot reconstruct eight Host lanes.

## Prefix state acceptance diagnostics

For a separate cold/hot correctness run, add `--prefix-cache-mode hot`
`--warmup-count 2 --audit-prefix-state`. The tool selects the skill-local
`lowlat_prefix_audit.py` server entry; it adds no public serving CLI flag.
The entry installs read-only diagnostics in the serving worker before model
initialization. Readbacks reuse the prefix copier's pre-fork registered host
buffer, use owning allocation bases plus explicit offsets, and run after whole
prefill completion with no accepted native parent.

The run writes `prefix-audit/<request-id>.json` and `validation.json`. Require
one cold checkpoint, two hot restores, complete output IDs, matching complete
logical cache pages across eight ranks, retained checkpoint bytes after prior
decode, real terminal history taps, and the expected native seed on every rank.
The rank0 history ring holds the captured seed; other ranks start at zero.
Communication epochs may advance across requests and must not be compared as
literal cold/hot state equality.

These additional copies and file writes affect timings. Keep this mode separate
from TPOT measurements. Its receipt does not qualify partial-page bytes,
independent numerical goldens, cancellation/eviction behavior, or long contexts;
those require their own cases and evidence. An offline recheck is available with
`lowlat_prefix_audit.validate(<prefix-audit-directory>)`.
