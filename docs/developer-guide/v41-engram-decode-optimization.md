# V4.1 Engram and decode buffer optimization

## Problem and investigation

The four-card V4.1 K5 comparison exposed substantial host preparation and decode
overhead. Inspection found repeated safetensors shard opens and Python slicing,
cloning and concatenation for Engram rows. The A5 DeepInfer CPU Engram adapter
used in the comparison provided a reference for retaining mapped tensor views
and gathering rows in bulk.

Inspection of the DSpark dispatch path also found that intermediate outputs
were bound to CPU tensors even though corresponding device allocations already
existed. All 23 selected arguments were checked against the lib signatures and
were declared Out on both tested lib revisions. This established which buffers
could remain resident while sampling outputs and mutable request state followed
their existing handling.

## Implementation

The PR keeps two commits with the following responsibilities.

| Change | Code boundary | Behavior |
| --- | --- | --- |
| Mapped Engram gather | `weight_loader.py`, safetensors row reader | Retain mmap views for the loader lifetime and gather bytes through one `index_select`, preserving FP8 bytes, duplicates, row order and independent result storage |
| Mapping lifecycle | `weight_loader.py`, mapped-view initialization | Serialize initialization and allow failed opens to be retried |
| Active-row capacity | `npu_runner.py`, `HostEngramResources.stage_lookup` | Size temporary results to active tokens and validate token capacity and request stride |
| Resident decode intermediates | `npu_runner.py`, `V41ResidentResources.dispatch_dspark` | Bind 23 intermediate Out arguments to existing device buffers and preserve that binding when assigning the communication epoch |
| Explicit readback | `npu_runner.py`, `V41ResidentResources._read_dspark_output` | Read requested intermediate results from their resident device storage |

The active-row change overlaps the corresponding hunk in PR316. The resident
decode change was moved from the subsequently closed PR333 into the second
commit of PR332.

The active-row lookup contract passes the active token count to the host reader.
The staging boundary validates that count against the selected topology, while
the reader defaults its allocation capacity to the number of supplied tokens.
PR316's companion capacity test must use this active-count contract when the
two changes are combined.

## Experiments and decisions

The mapped-reader experiment alternated old/new/new/old functions within one
initialized engine. Each block warmed the first case and measured all 16 cases,
giving 64 formal requests. Concurrent compilation affected the shared host.
Block TPOT medians were 65.13, 44.40, 41.01 and 50.14 ms/token, so the pooled
result includes substantial host-load variation.

| Mapped-reader experiment, medians | Previous reader | Mapped reader |
| --- | --- | --- |
| TTFT including bootstrap | 3185.92 ms | 1221.78 ms |
| TPOT | 54.93 ms/token | 41.99 ms/token |
| CPU staging lookup, 6 active tokens with capacity 80 | 30.41 ms | 5.63 ms |
| CPU staging lookup, full 80-token chunk | 211.98 ms | 5.53 ms |

The two CPU rows above are staging measurements. Separate direct lookup
microbenchmarks used capacities equal to active counts, two warmups and ten
measurements. For 6/80/128/512 tokens, old/new medians were respectively
18.92/0.67, 195.20/2.96, 312.49/6.03 and 1161.96/14.00 ms. Their BF16 results
matched byte for byte.

Subsequent experiments evaluated each additional direction separately.

| Direction | Evidence | Decision |
| --- | --- | --- |
| Active-row temporary allocation | 32 paired requests, paired TPOT delta -0.231 ms/token, 95% case-bootstrap interval [-1.743, +0.423] | Retained for smaller temporary allocation at the user's request, incremental latency benefit remains inconclusive |
| Resident decode buffers | 32 adjacent pairs, pooled TPOT medians 43.40 to 40.49 ms/token, paired decrease 2.562 ms/token, 95% interval [0.625, 5.256] | Retained, paired median improvement 6.25%, TTFT improvement remains inconclusive |
| Skip CPU scratch clearing | Paired TPOT interval crossed zero, one pair had different draft acceptance sequences despite identical final output | Withdrawn |
| Reduce Torch threads from 2 to 1 | Paired TPOT interval crossed zero, paired TTFT increased 11.28 ms | Retained the 2-thread configuration |

The reader and residency results come from separate experiments. Their
percentages cannot be added to predict a combined speedup. For both retained
latency experiments, all paired input IDs, output IDs, stops and per-round draft
acceptance sequences matched. Each variant produced 3478 tokens and accepted
2422 of 5320 proposed drafts over 1064 decode rounds.

Resident decode initialization now cross-checks the selected scratch names
against the current resource layout, lifecycle workspace keys, compiled decode
parameter names and host tensor shape/dtype. Decode staging retains the existing
host clearing for those fields because the separate skip-clear experiment had an
inconclusive TPOT interval and one draft-acceptance sequence difference. Readback
requires the original decode workspace identity and selects the resident device
value for a bound name, preventing a stale host mirror from being returned. The
runtime's `data_ptr` callable case and non-contiguous mapped tensor case still fail
explicitly at their existing copy/view boundary, so no compatibility wrapper is
added for those paths.

## Measurement configuration

Both experiments used a real DeepSeek V4.1-Flash checkpoint, four Ascend 950DT
cards, TP1/DP4/EP4, K5, concurrency one and CPU Engram. The reader experiment
used cards 1,2,4,6 and the residency experiment used cards 1,5,6,7. Inputs were
the fixed first 16 GSM8K cases with five-shot prompts of 603-684 tokens.
Generation used greedy sampling, seed123 and a 256-token cap, stopping on
`Question`, `Assistant:` or `<|separator|>`. Context capacity was 8320, KV block
size128, prefill chunk80, prefix caching disabled and Torch intra-op threads2.

TTFT is the interval from request start to the first nonempty streamed event,
including synchronous bootstrap. TPOT is the interval between the first and
last token events divided by output token count minus one. An event can deliver
multiple tokens. Startup, compilation, weight upload and warmups were excluded.
Formal measurement disabled module timers and diagnostic tensor readback.
Paired intervals used case-cluster bootstrap over 16 cases with 10000 resamples
and seed123, retaining the repeated measurements within each case.

Device tests replayed the candidate methods in serving63654c91/liba658d93f with
the native whole-round L2 boundary, CANN9.3, Python3.10.19, Torch2.10.0,
PyPTO950a30ec, simpler31f9e72, PTOAS0.70 and PTO ISA f83d4b7.

## Validation and review boundaries

Historical CPU validation covered mapped-view lifetime, concurrent first access,
failed-open retry, FP8 byte preservation, duplicate and noncontiguous indices,
owned outputs, decode binding and explicit readback. Added regression test files
were subsequently removed at the user's request. Historical focused checks and
external harness results remain recorded in the PR description.

The November17 rebase uses serving main6333bab and its pinned libb67a5fd6.
Profiling-wrapper conflicts were resolved by preserving upstream spans and the
existing optimization logic. The three affected methods have identical ASTs to
the pre-rebase versions after removing profiling wrappers. Syntax, Ruff, header
and diff checks passed. Local pytest could not load its conftest because Torch
was unavailable. Full-model device performance has yet to be revalidated on the
rebased head. Historical measurements apply to the versions recorded above.

Review should check the mapped-view lifetime and result ownership, the 23 lib
Out contracts, sampling and request-state handling, explicit readback and the
active-row overlap with PR316. Required CI must pass before merging. The
historical device evidence covers the recorded single-request fixture and
configuration. Multi-request concurrency and other context lengths require
separate validation.
