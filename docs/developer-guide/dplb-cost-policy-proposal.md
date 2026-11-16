# Proposed TP1 DSpark DPLB cost policy

Status: recorded for future work; not implemented. The user requested that the
current policy remain unchanged while its enabled/disabled AgentX performance
is measured first.

## Proposed admission score

Replace the short/long-request branch with one score for each feasible cache
partition after admitting the incoming request:

```text
score_after = sum_length + incoming_length
            + 55,000 * (request_count + 1)
            + 0.30 * max(max_length, incoming_length)
```

Choose the partition with the lowest score after admission. Compare resulting
load rather than only incremental cost. Lengths are logical source-token
counts, including the complete prompt during chunked prefill, and grow during
decode. Request count includes requests whose admission has already reserved
an owner, so concurrent admissions do not ignore pending load.

The initial coefficients apply to the measured TP1 DSpark K=7 configuration:
16 local request rows, 128 local query rows and 8192-column compressed cache
tables. They come from combining the SWA/HCA/CSA branch fits with layer counts
2/20/21. In token-equivalent units, the measured coefficients are approximately
55,386 per request and 0.2998 per maximum-length token; the proposed defaults
round these to 55,000 and 0.30. The common fixed attention cost cancels when
comparing nonempty post-admission candidates.

Keep the coefficients configurable. They describe attention branch timing,
not full-model latency: MoE, communication and complete serving execution still
need to be measured before treating them as production calibration. Do not
apply these coefficients to V4.1, TP4, another query width or another tensor
layout without new measurements.

## Integration boundaries

- Update `KvCacheManager._rank_group_partitions` to track count, total logical
  length and maximum logical length per partition.
- Preserve physical cache capacity and request-slot constraints before scoring.
- Preserve existing ownership for admitted requests; do not migrate KV pages
  or speculative state.
- Preserve longest-prefix-hit priority; use the cost score only to resolve
  equally long prefix hits in the first version.
- Preserve rotating ties and the existing policy when DPLB is disabled.
- Recompute load correctly after decode growth, completion, cancellation and
  preemption. An empty partition has no active attention cost; do not charge
  it a fixed cost when reporting existing load.

## Evidence and evaluation order

The serving-interface experiment covered 16,384 through 850,000 input tokens
and 1/2/4/8/16 active requests, including mixed-length batches. Training on
uniform points and holding mixed batches out, count-only models had HCA/CSA
MAPE of approximately 14.1%/19.6%; adding total length reduced these to
4.6%/5.6%, and adding maximum length reduced them to 2.9%/3.7%. SWA was
approximately constant with this fixed local execution tile. Maximum length
supplements total length and does not fully describe mixed-length behavior.

Local experiment evidence is under `artifacts/dplb-serving-tp1/`, including
`conclusions.md`, `report.md`, `analysis.json` and frozen metadata. Rejected
numerical samples were excluded from all timing fits.

First measure the **current** DPLB policy enabled versus disabled on the same
AgentX trace workload. Do not enable this proposed score in that baseline
comparison. After implementation, compare the proposal against both baselines
with identical topology, checkpoint, scheduling settings, cache policy and
workload seed, recording TTFT, inter-token latency, throughput, completed turns,
errors and per-partition load.
