# Lowlat Deployment

Lowlat serves the 43-layer DeepSeek V4 Flash W8A8 checkpoint with one TP8
replica and one active request. Prefill attention uses CP8; decode attention,
MoE intermediate dimensions and the vocabulary head use TP8. Every rank retains
all expert IDs, so the serving topology is DP1 / TP8 / EP1.

The public lowlat implementation uses native K5 with
one/two-cycle parents, two output frames and rolling target caches. Sampling
is greedy. The serving group handles one active request. Rolling-cache prefix reuse and
context capacities up to 1,048,576 positions are implemented; their device
validation scopes are documented below. Preemption, PD separation and multiple
TP8 replicas remain unsupported.

## Reproducible environment

Use PyPTO `01edaa33b3c60d67f32e11cc97925f95611758c9`, its runtime gitlink
`355ce3015e3e344d1fc3a61c206ee01823ce292c`, PTOAS 0.72 from PyPTO's
`toolchain/versions.env`, and PTO-ISA
`5487a6b0e8d9bd3653dbee8d7c874c667ee8db2f` from runtime's `pto_isa.pin`.
The kernel source is pypto-lib PR #1747 commit
`72a6501f` plus the submodule's local integration fixes. A different PR head requires
ABI, weight coverage and device validation again.

The local validation environment uses Python 3.10, torch 2.10.0+cpu,
torch-npu 2.10.0, CANN 9.0.0 and A2/A3 devices. Rebuild runtime when its pin
changes and verify `_task_interface.__build_commit__` against the gitlink.
Lowlat forces compilation without persistent caching; do not pass
`--use-compile-cache`.

## Launch

Set `MODEL_DIR` to the converted Flash W8A8 directory. Source the environment
inside the submitted shell. `TASK_DEVICE` must contain eight distinct devices
allocated by the scheduler, in one supported communication group.

```bash
# In a task-submit-owned shell after activating the environment:
pypto-serving --model "$MODEL_DIR" --platform a2a3 \
  --devices "$TASK_DEVICE" --dp 1 --tp 8 --ep 1 --lm-head-tp 8 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5}' \
  --max-num-seqs 1 --block-size 128 --max-model-len 16384 \
  --max-num-batched-tokens 512 --long-prefill-token-threshold 512 \
  --no-enable-prefix-caching --enable-chunked-prefill \
  --ring-dep-pool 131072 --ring-task-window 131072 \
  --ring-heap 1073741824,268435456,1073741824,1073741824 \
  --generate-config '{"max_new_tokens":64,"temperature":0,"top_p":1}' \
  --host 127.0.0.1 --port 8000
```

Submit the launch script with `task-submit --device auto --device-num 8
--max-time 14400 "bash /path/to/launch.sh"`. Offline inference uses the same
command with `--prompt 'Huawei is'` and exits after generation.

The compiler capacity follows the configured runtime context, aligned to 128
positions. CLI accepts contexts from 1 through 1,048,576 and prefill chunks
from 1 through 8,192; contexts above 16,384 require rolling caches. Accepted configuration
limits do not establish device validation: the initial successful smoke used
context 512 and chunk 128; the final acceptance passed at context 16,384 and chunk 512. An extra
16,376-token prompt plus eight outputs also passed, with seven actual decode
inputs at positions 16,376 through 16,382. An 8,192-token single chunk remains
unvalidated. Consult the validation scope before changing chunk settings. CP padding of
the final prompt chunk must fit within the compiled RoPE capacity.

The internal target-weight compaction removes 4,328,521,728 bytes (4.03125 GiB)
of duplicate full-WO banks per rank while retaining their TP shards and the
prefill projection workspace. Resident target weights require approximately
36.3364 GiB per rank; the original M0 required 40.3676 GiB. M0 reserves
all seven cache groups for the configured full context; cache storage is about
169.5 MiB per rank at context 512 and 5.30 GiB at context 16,384, plus scratch,
communication windows and runtime allocations. Startup checks the actual free
memory against `--npu-memory-utilization`, allocates the complete fixed B1 cache
pool before uploading weights, and checks the remaining weight/scratch budget.
All ranks must have enough memory.

## HTTP contract

Use `/health` for readiness, `/v1/completions` for text completion and
`/v1/chat/completions` for messages. Both generation routes support SSE.
Requests arriving while another request runs wait in the existing queue.

```bash
curl http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Huawei is","max_tokens":8,"temperature":0,"top_p":1,"stream":true}'
```

Greedy requires `temperature=0`, `top_p=1`, `top_k` omitted, null, -1 or 1,
and `min_tokens=0`. Lowlat rejects unsupported sampling and unknown JSON
options before SSE headers, including penalties, logit bias, logprobs,
structured generation and tool constraints. Completion follows the existing
server convention of ignoring EOS; chat exposes `ignore_eos`. Stop strings
and maximum output length use the shared output path. The matched stop string
remains in the output under the existing server convention; `finish_reason` is
`stop`. A disconnected streaming
request drains all accepted native parents before its pages are reused.

## Validation and timing

On 2026-11-14, task `task_20261114_050603_377346325918` passed all 48 main
acceptance cases, all 10 device subchecks, and an extra 16K tail decode case on
devices 8-15, then exited zero and released all locks. This covers the real
43-layer checkpoint, HTTP/chat/SSE/EOS/stop, two-request reuse, cancellation
and long-context boundaries. Compilation caching was disabled.

The first eight-card smoke loaded the real 43-layer checkpoint and generated
eight tokens, including the terminal prefill prediction exactly once. Subsequent
acceptance covers HTTP/chat/SSE, EOS, stop strings, request reuse, cancellation,
compression/chunk boundaries and numerical consistency across CP/TP execution.
The workspace acceptance report records individual results; a running or queued
task must not be counted as a pass.

`LOWLAT_DECODE_METRIC` is a host-observed breakdown. `prepare_ms` includes cache
metadata, initialization and H2D staging; `submit_wait_ms` includes host dispatch,
device execution and completion wait; `capture_ms` includes sampled-ID D2H.
`submit_wait_ms` is not an isolated device-kernel measurement. Client TTFT and
SSE text-event intervals also include scheduler, IPC and HTTP overhead. Report
cold compilation/upload separately and use repeated warm runs with fixed prompt
and output lengths for decode comparisons.

## Historical M0 K0 baseline

The following M0/M1 sections preserve earlier validation evidence. Their K0
and synchronous lowlat launch modes are not public options. Public DSpark K0
uses the existing DSpark implementation; K5 selects lowlat's fixed native path.

The following host-observed runner measurements use two warmups and three
trials per prompt length, 64 output tokens per request, and 63 decode steps per
trial. All 756 measured steps, including page-boundary outliers, are retained.

| Prompt tokens | Runner mean ms/step | Runner decode steps/s |
| --- | --- | --- |
| 4 | 170.799 | 5.855 |
| 128 | 181.386 | 5.513 |
| 512 | 181.265 | 5.517 |
| 8192 | 183.834 | 5.440 |

Host preparation accounts for about 87-88% of runner time; submit/wait averages
about 20-22 ms and sample capture about 1 ms. These figures exclude the outer
scheduler/IPC/HTTP work. The mean is larger than the median because new page
initialization causes periodic preparation spikes.

CP versus TP full-vocabulary logits at 7/129 tokens passed the fixed cosine
>=0.99 and relative-L2 <=0.10 gate with equal argmax. Observed relative L2 was
5.42%/9.85%; this is not an independent Torch accuracy guarantee. Chunk 128/512
at 513 tokens produced identical logits. The full CPU suite has 1067 passes
and one pre-existing DSpark positional-ABI failure against PR1747; lowlat's
57 focused tests pass.

## Accepted incremental updates (2026-11-15)

P1 reuses registered host metadata/readback storage and stable decode bindings.
P2 uses bounded registered cache-initialization templates. Both passed full
eight-card serving acceptance. Metadata/readback staging allocations are zero;
a steady decode step performs 24 metadata H2D calls across eight ranks.

P3 optionally primes the exclusive B1 cache pool before prefill. It remains off
by default: the measured initialization cost averages 7.75 seconds per request.
Its lower decode TPOT does not establish a shorter end-to-end request latency.

Compact K0 was accepted before M1. Task
`task_20261115_113026_563663604` exited zero with 49/49 HTTP cases, 10/10 device
checks, unchanged frozen source, and a real healthy prefill parent tested with
an injected host wait exception. The accepted handle and cache pages remain
owned until the parent completes; a poisoned worker rejects the next request.
Compact changes decode/native ABIs to 81/153 parameters; prefill remains 94.
Use the local validated pypto-lib delta with serving, rather than substituting
the unmodified PR1747 checkout.

| Prompt tokens | Client TPOT ms | TTFT ms | Total request ms |
| --- | --- | --- | --- |
| 4 | 43.790 | 715.8 | 3474.6 |
| 128 | 45.251 | 1527.8 | 4378.6 |
| 512 | 45.507 | 4208.4 | 7075.3 |
| 8192 | 49.565 | 66912.8 | 70035.5 |

These O64 results use two warmups and three trials, with diagnostic counters
enabled and request pool priming off. Separate stage resource windows do not
constitute a controlled causal performance comparison. Main lowlat unit tests
pass 88 cases. The current main full suite has 1098 passes and the same existing
DSpark ABI failure.
Rolling caches and dual-frame execution were subsequently accepted together;
the final configuration is below.

## Accepted M1 native K5 (2026-11-15)

M1 was accepted before the final combination using synchronous
scheduling and full-context cache allocation. Current native decode requests allow up
to 2,048 generated tokens and reserve six physical lookahead positions within
the configured context. A one-output request uses the prefill prediction and
can use the prefill-only context boundary. The server checks each request's
actual prompt and output budget before execution.

Task `task_20261115_162002_273981217500` on
`liteserver-hps-f7e5-00002.novalocal`, devices 8-15, exited zero with 49/49 cases,
complete frozen-source checks before/after execution, and P127/O132 plus P513/O8
K0/K5 semantic gates. Token IDs and committed cache rows match; each diagnostic
cycle matches the CPU commit/history oracle and all eight device states.
It uses real draft/Markov weights and standard generated W2. Independent
full-model Torch accuracy remains outside this validation.

| Prompt tokens | Client TPOT ms | TTFT ms | Total request ms |
| --- | --- | --- | --- |
| 4 | 27.590 | 863.4 | 2601.6 |
| 128 | 16.697 | 1679.5 | 2731.4 |
| 512 | 16.787 | 4094.7 | 5152.3 |
| 8192 | 18.183 | 63054.4 | 64199.9 |

The O64 client measurements use two warmups and three trials. Native acceptance
counts actual emitted tokens rather than assuming one token per parent.
Diagnostic API counters are included; overlapping parent intervals must not
be summed into request latency. This host differs from the earlier Compact
measurements, so these results do not establish a controlled causal speedup.
Main full unit regression: 1169 passed and the same existing DSpark ABI failure;
159 focused lowlat tests are included.

## Accepted M2 with rolling caches: final configuration

Task `task_20261115_215243_39495218053` on host
`liteserver-hps-f7e5-00002.novalocal`, devices 0-7, exited zero. All 55 HTTP
cases and all three K0/K5 semantic comparisons passed: P127/O132, P513/O25 and
P8192/O25. Comparisons cover exact IDs, committed live-cache bytes, the CPU
commit/history oracle, all eight states, and successor completion before
predecessor packet capture. Source manifests matched before and after execution.

The public worker has pipeline depth 2, launch depth 1, and two actual resource
sets on all eight ranks. Each parent owns a separate output frame. K remains 5;
a parent executes one or two cycles, with at most 12 raw outputs. The engine
accounts for actual retirements and drains accepted successors on cancellation
or EOS before reusing pages.

The 207 production parents made zero explicit `Worker.copy_to` calls; host
staging create/release checks passed. Prefill initialization, runtime argument
transport, device collectives and output readback still occur. This counter
establishes removal of serving's per-parent uploads, not zero hardware traffic.

| Prompt tokens | Client TPOT ms | TTFT ms | Total request ms |
| --- | --- | --- | --- |
| 4 | 11.911 | 875.3 | 1625.8 |
| 128 | 4.863 | 1646.8 | 1953.1 |
| 512 | 4.883 | 3975.5 | 4283.1 |
| 8192 | 5.433 | 59011.8 | 59354.1 |

These O64 results use two warmups and three trials with diagnostic API counters.
Scope statistics are enabled only in semantic diagnostics, and disabled for
performance samples. Parent intervals overlap and must not be summed as request
latency; pure-device STRACE timing and an independent full-model Torch reference
are outside this validation. The full final main suite has 1219 passes and the
same existing DSpark positional-ABI failure; 209 focused lowlat tests are included.

At chunk 512 and context 16,384, the rolling target pool uses 151,109,632 bytes
per rank, versus 5,688,868,864 bytes for the full target pool. This is capacity
saving, not a causal latency measurement. Native draft weights, scratch and
runtime windows remain in the actual startup memory admission.

The validated two-cycle ring heaps are
`1073741824,268435456,1073741824,1073741824`. A 256 MiB ring 0 caused a proven
open-scope allocator deadlock in an earlier attempt. The 1 GiB ring 0 passed
actual memory admission and final acceptance at utilization 0.9. Retained scope
samples show at least 302.6 MiB live in ring 0; they are not exhaustive allocator
high-water measurements. Request pool priming remains off by default.

The shared workspace provides `run-lowlat-serving-final.sh` with this validated
K5, async native pipeline and rolling configuration. Launch it through the queue:

```bash
cd /mnt/data/pyptouser/zhouchunxu/pypto_dm0
task-submit --device auto --device-num 8 --max-time 14400 \
  'bash /mnt/data/pyptouser/zhouchunxu/pypto_dm0/run-lowlat-serving-final.sh'
```

The launcher creates separate build/log directories per host and invocation;
compilation caching remains disabled. Its default endpoint is 127.0.0.1:8000.
Set `LOWLAT_SERVING_HOST` or `LOWLAT_SERVING_PORT` in the submitted environment
when needed. The launcher passes shell syntax and CLI configuration checks.

## Profiling

The single-batch lowlat profiler captures the current TP8/K5 native pipeline
with rolling caches. It keeps one active request, pipeline depth two and
launch depth one. Use the compatible environment and eight scheduler-assigned
devices. Compilation caching remains disabled.

Enable the shared verbose profiler with `--profile --profile-level verbose`.
Use the `/start_profile` and `/stop_profile` endpoints to bound the formal
request. Enable Simpler Host STRACE before server startup and keep Device
STRACE disabled for this measurement.

The artifact directory must be empty. The default prompt is `Huawei is`
repeated 32 times, with 64 output tokens and chunk512. The tool saves actual
prompt IDs, performs warmup and an unprofiled reference, then profiles one
HTTP streaming completion using the on-demand verbose recorder. It enables
Host STRACE before server startup and disables Device STRACE.

Open `serving-strace-swimlane.json` in Perfetto. It combines scheduler, serving,
worker and executor spans with native prepare/submit/wait/capture/retire,
eight Host device lanes, and two frame lifetime tracks. Submit spans carry
the pinned runtime's native run ID; the analyzer joins that identity to every
rank rather than estimating association from overlapping timing windows.
Native capture spans include actual cycle counts, physical advancement,
accepted drafts and emitted token snapshots. No additional device reads or
waits are introduced by these spans. Disabled profiling preserves the original
execution path.

`decode-only-swimlane.json` exports the native decode window with all eight
Host lanes, excluding prefill and its first output token. Long-context profiling
can use one different short warmup with prefix caching off. Such a warmup is
not a comparable performance baseline; the analyzer reports null baseline
TPOT and instrumentation delta. Benchmark-only logs contain no Host lanes.

The tool also saves raw `serving-trace/trace.json`, `simpler-swimlane.json`,
`server.log`, responses, source fingerprints, `profile-summary.json` and
`profile-summary.md`. `skill-profile-validation.json` must contain
`valid: true` before treating the trace as successful. Validation rejects
missing/duplicate rank invocations, mismatched request/frame/reservation
identity, broken token/physical/epoch continuity, incomplete budgets, failed
operations, torn or dropped logs, and changed greedy output or source files.
Observed cycle/frame coverage is reported separately from trace validity.

Saved evidence can be reprocessed without NPU access. Reprocessing must
preserve native run IDs and validate all eight Host lanes before reporting
a successful profile.

Host `runner_run` includes dispatch, execution and waiting; it is not pure
device kernel time. Parent elapsed lifetimes may overlap and must not be
summed. The summary reports unprofiled and instrumented TPOT separately.
The run ID is an artifact label; native runtime run IDs are distinct numeric
correlation keys. Their collection depends on the pinned runtime handle
layout and is validated explicitly when processing the trace.


## Updated prefill integration

The updated lib prefill ABI has 100 parameters; target-only and native ABIs
retain 81 and 153. Default projection, causal and ECHO controls are initialized
explicitly. Experimental prefill mechanisms retain the lib defaults.

A pre-fork registered HC input buffer broadcasts one physical FP32 copy across
rank views. Typed metadata arenas from lib separate stable page tables from
per-chunk IDs, positions, RoPE and slot mappings. Stable tables are uploaded
only when the request generation or source-table revision changes. Device roots
own their borrowed field views; allocation accounting includes alignment.
Resident weight bindings are prepared once, and token IDs are uploaded once
per chunk. The chunk kernel derives local request IDs itself. The adapter still requires
the host descriptor for layout validation; it receives no device allocation
or upload.

A pending native submission owns its output frame and a completion future
before entering the runtime. Blocking submit does not retain the native-state
lock. The previous completed parent may be captured while the successor waits
inside the runtime; cancellation retains both leases until they settle. Runtime
launch ordering is fixed to the validated depth-one configuration.

## Confirmed prefix reuse

Enable `--enable-prefix-caching`. Prefix reuse supports one active request and
greedy K5 native decoding. The lowlat implementation fixes the rolling cache
layout and on-demand private-cache initialization internally.

The cacheable boundary is `floor(max(0, P - 128) / 128) * 128`. A hit must
satisfy all seven groups and leaves at least 128 actual prompt tokens to replay.
Full-history pages remain shared and immutable. Rolling tail state is restored
into a stable request-owned ring before replay; checkpoint tails are copied to
separate retained pages after confirmed prefill completion. Allocation epochs
and bounded publication proofs prevent stale-page reuse. Native decode never
writes the retained checkpoint, and replay rebuilds real history taps before
seeding a fresh native request. No compiled model kernel changes are needed for
prefix reuse.

The allocator owns page references, partitions, snapshot pins and reclamation.
The lowlat policy owns replay boundaries, valid state ranges and K5 receipt
interpretation. Shared engine and worker code transport model metadata and
execution receipts as opaque bytes; they do not interpret native cycle state.

The updated-lib short-context gate used P1536/O64/C16384/chunk512 and verified
cold/hot text, H1408/replay128, both frames and eight Host trace lanes. This is
not complete numerical, cancellation or long-context qualification. Dedicated
state diagnostics additionally compare complete logical cache pages, output
IDs, history seed and native seed ranks; their extra readbacks affect timing.

## Extended context capacity

`--max-model-len 1048576` binds the same capacity to target, prefill and native
compilation, page-table depths and target RoPE.
Drafter table depth may cover a larger target profile. Output length and native
lookahead must fit the admitted context; a full-capacity input cannot also
request additional positions outside that capacity.

Logical table depth and full compressed history scale with context. Raw KV and
compressor state retain bounded active rings. Prefix caching adds independent
checkpoint storage. Startup reserves the cache before weights and checks the
actual rank budget; accepting a CLI value does not prove device memory fit.
Full-context RoPE construction uses bounded FP32 intermediates.

1M program generation and CPU position/shape/byte-budget checks have passed.
Complete long-input device correctness and long-context TPOT remain pending;
short inputs under a 1M configuration would not qualify those claims. The
requested future TPOT case is P850000/O512, separate from correctness audits.
A larger outer prefill chunk changes memory requirements and requires its own
admission and correctness checks.

## 850K input performance evidence (2026-11-19)

Task `task_20261119_101857_64099120929` completed one B1 request with exactly
850,000 input tokens and 512 output tokens at context capacity 1,048,576,
chunk512, memory utilization 0.93 and prefix caching off. HTTP TTFT was
1,849.100 seconds and TPOT was 8.065895 ms/token; the first-to-last nonempty
text-delta interval was 4.121672 seconds. Model loading took about 2,167 seconds
before readiness and is outside this HTTP TTFT.

The formal request retired 43 native parents, 86 cycles and 511 tokens after
the prefill prediction. All 430 attempted drafts were accepted. Consecutive
parent completion log timestamps were 95 ms apart at the median (83-123 ms),
compared with 39-49 ms for the few short-warmup two-cycle intervals. This is a
host completion interval, not the gap between rank runner calls. Earlier
short-context measurements are not a controlled baseline for this workload.

With chunk512, the prompt requires 1,661 prefill chunks. TTFT divided by that
count is about 1.113 seconds per chunk, including all request overhead; it is
not a measured kernel average. The lib prefill indexer scores and ranks the
visible compressed history. Decode indexer also expands scoring and top-k
work with visible history, capped by configured capacity. At 850K positions,
a compression ratio of four exposes about 212,500 candidates. Sparse final
attention therefore does not imply constant indexer cost. The serving path
also pays per-chunk scheduling, metadata, embedding and transport costs;
post-prefill native input-bank preparation can contribute to TTFT. Their
individual contributions have not been measured in this run.

This was a benchmark-only run: serving trace had zero events and Host STRACE
was disabled. Its log-derived parent timeline cannot establish eight-rank
runner gaps or split lib/runtime and serving costs. Reprofile the same workload
with one short warmup, prefix caching off and benchmark-only disabled, then
inspect decode-only-swimlane.json. Use the full trace to compare early/middle/
late prefill chunk costs, per-rank indexer/runner spans, chunk-boundary overhead
and post-prefill native setup. Host runner spans still include runtime waits;
operator-level attribution requires further instrumentation. Profile TPOT
must not replace the separate uninstrumented benchmark.

## Serving/model boundary

Lowlat native execution is carried through a typed model extension. The common
engine owns request admission, token scheduling, ordinary KV block lifetime and
IPC ordering. The lowlat adapter owns native cycle/frame reservations, physical
lookahead, retirement receipts and rolling checkpoint publication.

`RequestExecutionContext` and `ModelStepExtension` live in
`pypto_serving.serving.model_extension`. Prefill batches convert legacy byte
metadata into a versioned opaque extension at construction. Executor features
are resolved once into `ModelExecutionCapabilities` during worker
initialization; the worker hot path does not probe executor attributes for
async or sampling behavior. Lowlat decode receipts are exposed through a
`deepseek_v4_flash_lowlat` result extension while the legacy receipt fields
remain wire-compatible for older workers.

The compatibility fields in the common dataclasses are transitional input
aliases. New model adapters should use `model_extensions` and the capability
contract. The generic scheduler never interprets native reservation payloads.
