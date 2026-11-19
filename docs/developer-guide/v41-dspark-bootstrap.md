# V4.1 DSpark terminal prefill

Terminal prefill prepares the first proposed tokens and the draft KV state
used by the next decode step. Serving now binds the completed main prefill
outputs to the Lib bootstrap composite. Each rank submits one L2 program for
this handoff. The main prefill and the recurrent decode have separate entries.

The Lib update also changes the draft projection to packed MXFP8 and divides
compressed C2A pages into 64 storage rows for 128 logical tokens. The serving
adapter binds the projection scales, drafter state and compact page views.
Unit tests cover crossing the C2A page boundary and sharing the packed pool
with C1A pages.

The four-rank synthetic bootstrap comparison on Lib base 873c2452 produced
identical request state, draft IDs, confidence values and draft KV. Median
bootstrap time across four warmed measurements was 19.92 ms for the five
submissions and 12.93 ms for the fused submission.

On Lib base f4dbcb5a and Serving base ffab1ab0, the real checkpoint ran
16 serial GSM8K 5-shot prompts per variant at TP1/DP4/EP4 with greedy K5,
80-token prefill chunks and prefix caching disabled. Each process ran one
excluded warmup. All input IDs, generated IDs, stop reasons and per-round
accepted counts matched (1,532 generated tokens and 463 decode rounds).

| Median | Five submissions | One submission per rank |
| --- | ---: | ---: |
| TTFT including bootstrap | 1506.27 ms | 1423.22 ms |
| TPOT | 41.89 ms/token | 39.74 ms/token |

The scheduler assigned devices 0,2,3,5 to the control and 0,2,4,7 to
the candidate. There was one process per variant. Steady decode uses the
same kernel, so this comparison cannot isolate a causal decode gain.
The K0 path still needs to bind the output head of the backbone-only Lib
decode entry.
