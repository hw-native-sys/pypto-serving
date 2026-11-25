# DeepSeek V4.1-Flash prefill windows and drafter RoPE tables

## Sliding-window indices during prefill

For a prefill segment with scratch storage and more than one query token,
`prepare_swa_window_metadata` builds one sequence of cache addresses. The
sequence starts with the visible history in the ring and continues with the
current tokens in scratch storage. Adjacent queries share most of these
addresses in their 128-token causal windows.

Short windows at the beginning are left-aligned and padded with `-1`.
Once a full window is available, each subsequent query takes a 128-entry
slice starting one position later. The final window of up to 128 scratch
entries carries the destinations used to restore the ring. Single-token
segments and segments without scratch storage use the existing per-query path.

## Resident RoPE tables during DSpark decode

The worker uploads the drafter's cosine and sine tables during initialization.
When recurrent decode receives those same two host tensor objects, it binds
the resident device tables directly. If either input is a custom tensor, or
resident DSpark buffers are unavailable, decode binds both supplied inputs.
Both optimizations are selected automatically when their conditions hold.

## Performance and scope

On four Ascend 950DT devices with DSpark K5 and prefill chunk 128, reusing
window addresses reduced median cumulative address preparation per request
from 46.20 to 9.86 ms. Median time to first token (TTFT) fell from 970.05 to
957.38 ms; 14 of 16 prompts had lower TTFT. All 288 CPU metadata comparisons
matched the baseline.

In a separate run, resident RoPE binding reduced median time per output token
after the first (TPOT) from 27.13 to 26.48 ms/token. The paired median change
was -0.15 ms/token, with a 95% interval from -1.31 to +1.05. This establishes
that the binding avoids resupplying the original tables; the measurements
leave the size and direction of its latency effect uncertain.

The dynamic verification head belongs to the companion Lib change. Both
arms of its chunk-512 comparison used the Serving optimizations above.
At nine concurrent requests, the dynamic head reduced median TPOT from
59.43 to 56.93 ms/token and increased median TTFT from 8283.98 to
8356.03 ms. For a deployment with both first-token and decode latency
requirements, compare both metrics at its expected concurrency. A combined
end-to-end gain for the two Serving changes has yet to be established under
matched conditions.

[Experiment results and methodology](https://github.com/hw-native-sys/pypto-serving/issues/369)
and [Lib verification head](https://github.com/hw-native-sys/pypto-lib/pull/1916).
