# DPLB DFX P0 progress and handoff

This is an initial implementation for [issue #325](https://github.com/hw-native-sys/pypto-serving/issues/325),
validated on the explicitly selected source matrix below. The issue remains open: semantic device
attention timing and several parts of the broader acceptance criteria are unfinished.

## Implemented

- Migrate the existing grouped DPLB policy to the selected serving revision: short requests rank by
  request count then logical KV length, long requests rank by logical KV length then request count,
  and exact ties rotate. Longest prefix hits retain priority. Chunked prefill accounts for the full
  prompt, ownership stays fixed, decode growth updates load, and release removes it.
- Add opt-in `load` capture through the existing HTTP profiling controls. Capture policy score tuples,
  candidate order, prefix lookup results, owner baselines, ownership epochs and changes, planned
  group loads, worker command mirrors, output application status, and cumulative output metrics.
- Carry capture and step identifiers explicitly through IPC. Retain slot/generation attribution
  across asynchronous reclaim and preserve late outcomes after stop within a bounded drain.
- Bound writer queues and capture output, report loss and I/O failure, and leave capture disabled
  by default. `timing` rejects configuration rather than substituting host time for device time.
- Generate JSON/Markdown summaries and a Perfetto trace, with missing joins and partial captures
  visible. Compute token-weighted observed TPOT from cumulative ITL sum/count deltas, including
  observed output from unfinished requests.

No new cost formula is implemented. The [cost proposal](dplb-cost-policy-proposal.md) is a record of
an earlier idea; its TP1 coefficients must not be applied to TP4 without new measurements.

## Validated source matrix

| Component | Revision |
| --- | --- |
| pypto-serving base | `95c8280982491d25459bd10cb3128ab7616d6106` |
| pypto | `70da9baed1edd18308420d0ac2d5b9f4ede4bb90` |
| pypto-lib | `604d4eade07f6cfffe1bf7821692d5cea1f3061e` |
| pypto runtime | `c1f89ea431f5306f78c0e4ab9573c829756d7f1e` |

The implementation branch deliberately preserves this requested baseline. It is not a tested port
to current `main`; mainline integration and validation remain follow-up work. Do not infer NPU
compatibility with a newer Lib/runtime from this matrix. No dependency gitlinks are changed by
these feature commits.

## Eight-card validation on 2026-11-16

CPU verification passed all 828 unit tests, headers, English-only checks, and Ruff. The analyzer's
numeric Perfetto track IDs were also checked by the 14 targeted DFX tests after that full run.
Documentation navigation/public-link checks are included in the PR preparation checks.

NPU development used at most eight allocated cards at a time, DP2/EP8/TP4, K=7, fused one-L2
bootstrap and device bootstrap metadata. Both device tasks completed successfully and released
the devices. The first smoke generated 64 tokens across four mixed-length requests and produced
six complete sampled command joins with no drops. The final smoke generated 1152 tokens across
12 requests, excluding warmup, and exercised capture start during traffic, stop with four long
requests unfinished, and a second capture after the requests completed.

| Final capture | Sampled commands | Existing owners at start | Outcomes applied after stop | Missing joins | Dropped events |
| --- | ---: | ---: | ---: | ---: | ---: |
| Start during traffic | 8 | 4 | 2 | 0 | 0 |
| Repeated capture | 6 | 0 | 1 | 0 | 0 |

All sampled commands included both logical groups, including empty groups. Slot and generation
identifiers matched between preparation and reclaim. Both capture reports marked host correlation
complete and device timing unavailable. This smoke does not measure observer overhead or DPLB
performance gains. The earlier AgentX benchmark was not restarted.

Queue pressure, output caps, disk failure, capture deadlines, late outcomes, prefix priority,
ownership reuse, unchanged placement choices, live multi-token accounting, and preparation ahead
of the device control barrier are covered by CPU tests.

## Reproduction

Prepare the selected Python/compiler/runtime stack and allocate eight compatible devices. Keep the
model path and assigned device list specific to the deployment; the following are the development
smoke settings, not the production AgentX settings:

```bash
export PYPTO_DSPARK_BOOTSTRAP_ONE_L2=1
export PYPTO_DSPARK_BOOTSTRAP_METADATA_DEVICE=1
export PYPTO_DSPARK_PREFILL_ASYNC=1
export PYPTO_DSPARK_DRAFTER_RING_HEAP=1073741824,1073741824,1073741824,1073741824

pypto-serving \
  --model /path/to/dsv4-flash-dspark-w8a8 \
  --served-model-name dsv4-flash-dspark-w8a8 \
  --backend npu --platform a2a3 --devices "$TASK_DEVICE" \
  --dp 2 --ep 8 --tp 4 \
  --block-size 32 --max-model-len 32768 --max-num-seqs 8 \
  --max-num-batched-tokens 8192 --long-prefill-token-threshold 4096 \
  --npu-memory-utilization 0.93 --enable-prefix-caching --enable-dplb \
  --generate-config '{"ignore_eos":true}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7}' \
  --ring-heap 1073741824,1073741824,1073741824,1073741824 \
  --profile --profile-output ./profile_out --profile-level e2e \
  --dplb-dfx-level load --dplb-dfx-sample-every 1 \
  --dplb-dfx-max-steps 128 --dplb-dfx-max-seconds 600 \
  --host 127.0.0.1 --port 8232
```

1. Warm up with one eight-token completion. Use `temperature=0` and seed `20260707`.
2. Launch four concurrent 256-token completions with distinct prompts and mixed prompt lengths.
   The final smoke used prompts with 61, 229, 453 and 901 tokens.
3. Start capture while these requests are running with `POST /start_profile`.
4. Add four concurrent 16-token completions. After the first short request completes, stop capture
   with `POST /stop_profile` while long requests remain unfinished. Let all requests finish.
5. Start another capture, send four concurrent 16-token mixed-length completions, and stop capture.
6. Analyze each directory with `python scripts/analyze_dplb_dfx.py profile_out/dplb/CAPTURE_ID`.
   Confirm complete host joins, both group entries, zero drops and retained late outcomes.

See [usage and record semantics](dplb-dfx.md) for boundaries and limits. In particular,
`step_outcome.token_counts` contains worker-returned counts before engine finish-limit trimming;
use cumulative output-metric deltas for client-visible service outcomes.

## Remaining work, in order

1. Integrate with current mainline's worker command IDs, profiler controls, grouped reservations and
   executor changes, preserving that code's lifecycle contracts. Re-run unit/lint checks and an
   eight-card smoke on the resulting source matrix; the results above apply to the pinned branch.
2. Complete P0 attribution below the command boundary. Carry immutable diagnostic context through
   batches/prepared inputs to each L3 submission, including multiple prefill/bootstrap/decode
   invocations in one command. Capture actual prepared descriptor membership and physical extents;
   the current worker mirror is not device-confirmed participation. Enrich placement records with
   per-candidate free/required blocks and exclusion reasons, maximum logical length, and explicit
   cancellation/preemption/rejection causes. Current capacity snapshots and filtered candidate
   lists do not provide all of those per-decision explanations.
3. Add semantic SWA/HCA/CSA device timing using runtime task/function metadata, without splitting
   fused bootstrap. Define the attention envelope and exclude MoE. Preserve each rank's clock and
   reduce measured rank durations with max per logical DP group. HCA/CSA's SWA component must not
   be double-counted as another pure SWA layer. Validate runtime loss and timing coverage.
4. Quantify default-off, P0 and P1 observer overhead independently. Then run a controlled AgentX
   DPLB-off/on comparison, using all observed output-token interval sums/counts. Eight-card
   development cannot establish the final sixteen-card deployment's performance.
5. Only after measured cost is reliable, evaluate a new cost policy as separate work. Keep issue
   #325 open until the relevant attribution, timing, outcome and overhead criteria are satisfied.

The runtime exposes 16 selective task-timing slots and detailed chip task streams. A merged slot
envelope is not additive attention compute time, and kernel-name matching alone is insufficient
semantic attribution. Host command envelopes and planned length imbalance cannot establish waits,
attention cost, or a DPLB speedup. See the [design](dplb-dfx-design.md) for the full target.
