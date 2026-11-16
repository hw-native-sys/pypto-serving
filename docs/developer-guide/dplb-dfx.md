# DPLB diagnostics: initial P0 implementation

DPLB DFX records grouped KV placement and follows sampled scheduler commands through
worker preparation and host output reclaim. It observes the existing policy without
changing placement, prefix matching, model kernels, or fused bootstrap.

Enable it on a profiled server:

```bash
pypto-serving ... --profile --profile-output ./profile_out \
  --dplb-dfx-level load --dplb-dfx-sample-every 32 \
  --dplb-dfx-max-steps 256 --dplb-dfx-max-seconds 60
curl -X POST http://localhost:8232/start_profile
# Send inference requests through the usual serving endpoints.
curl -X POST http://localhost:8232/stop_profile
python scripts/analyze_dplb_dfx.py profile_out/dplb/CAPTURE_ID
```

The default level is `off`: no DFX recorder, scans, files, or device readbacks are
created. Optional IPC metadata is omitted when unset. Level `timing` fails explicitly:
semantic SWA/HCA/CSA device timing has not been connected. `--profile` is required
for `load`. The analyzer writes `summary.json`, `report.md`, and Perfetto `trace.json`.
`--compare OTHER_CAPTURE` writes and prints both summaries; it does not infer causality
or normalize incompatible scenarios.

Each capture has separate engine and worker JSONL files. Records carry capture,
process instance, replica, local host clock domain, and step IDs. Placement ranking
records include the actual candidate order, score tuples, logical sequence counts,
logical token sums, branch and rotation state. Prefix hits come from the existing
lookup pass. Ownership records cover admission, growth, baseline owners, release,
and admission epochs when an ID is reused. Decision IDs are local to a capture.
Requests admitted before capture have an unknown original decision.

A complete sampled command includes all logical DP groups, including empty ones.
`scheduler_planned` records precede optimistic scheduler advancement. Worker records
use `worker_command_mirror`, with explicit buffer slot and generation retained until
that command's output is reclaimed. This is a host command view; prepared device
descriptors, per-rank execution and effective speculative lengths are not confirmed.
`step_outcome.token_counts` records worker-returned counts before engine finish-limit
trimming and cancellation handling; it is not a count of client-visible output.
Use cumulative output-metric deltas for the service outcome window.
A command may contain multiple prefill/bootstrap/decode runtime invocations. Their
individual IDs and semantic device timings remain a follow-up to this initial P0.
The report never treats command envelopes as per-group device attention cost.

Start snapshots existing owners and cumulative output metrics. The worker sink starts on
receipt of the input control command so preparation ahead of the device control barrier
is retained; `worker_control_applied` marks the later acknowledgement boundary. Stop ends new samples
and snapshots the outcome window before the worker control barrier. Previously sampled
commands may report late outcomes under their original capture ID. The worker barrier
drains its output lane; the engine allows up to two seconds for queued results to be
applied, then seals its sink. Missing outcomes remain explicit if this drain times out.
Capture duration limits new records and sampling, while metric boundaries are the
actual start/stop snapshots. Repeated start during an active capture is idempotent.

The analyzer computes observed token-weighted TPOT from cumulative ITL sum/count deltas
for each replica. This includes observed tokens from unfinished requests and the existing
zero intervals within multi-token output updates. The first token has no ITL; unobserved
tails contribute no interval. Missing metrics and zero denominators produce `null`.
This metric is independent of sampled steps and does not use inverse global throughput.

Each process uses a bounded background file writer: at most 1024 queued events, 256 KiB
per event, and 64 MiB per capture by default. JSON serialization occurs on the emitting
host thread to enforce the queued-byte bound; file writes occur on the writer thread.
Queue pressure, oversized events and output caps drop diagnostics rather than block
inference. A per-file manifest reports loss and I/O failures. Control-path close waits
at most two seconds. Interrupted/open captures, drops, missing joins and malformed JSONL
are reported as partial. A missing manifest must not be interpreted as zero drops.

Development uses eight allocated cards, DP2/EP8/TP4, with fused bootstrap. This validates
two logical groups; it cannot establish production DP4/EP16/TP4 performance. Observer
overhead has not yet been quantified with the duration-based AgentX workload. Do not
use planned length imbalance alone to claim a DPLB speedup. The broader target and P1
requirements are in [the design](dplb-dfx-design.md).

See [progress and handoff](dplb-dfx-handoff.md) for the validated version matrix, eight-card
reproduction, test results, and remaining mainline/P1 work.
