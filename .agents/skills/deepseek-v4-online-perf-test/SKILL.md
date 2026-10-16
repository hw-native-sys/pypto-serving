---
name: deepseek-v4-online-perf-test
description: Test and analyze the ONLINE serving performance of DeepSeek V4 (Flash W8A8) on the pypto-serving HTTP server. DeepSeek has NO offline generation entry — online serving is the only path, so this skill is the DeepSeek counterpart of qwen3-14b-online-perf-test. It starts an 8-NPU overlapped DP=8 / EP=8 profiled server using the built-in SA_PROFILE Chrome-trace recorder, drives load, and produces a per-operator / per-kernel time breakdown — including DeepSeek-specific prefill sub-phases (prepare / ensure_l3_shared_buffers / prepare_inputs / prepare_fwd_args / l3_dispatch) and MTP speculative decode. Requires the W8A8 compressed-tensors checkpoint, --dp 8 --ep 8 --tp 1, --block-size 128, and exactly 8 device ids. For end-to-end-only throughput/TTFT numbers use vllm-bench-perf; this skill measures WHERE time goes at operator granularity.
---

# DeepSeek V4 online serving performance profiling

`SA_PROFILE` is the built-in Chrome trace-event recorder in `pypto_serving.tools.profile`. It is disabled by default with low overhead, and when enabled it records duration spans from the HTTP API, scheduler, engine, executor, worker, and NPU kernel-dispatch paths. Each process writes its own JSON Lines fragment; on a graceful shutdown the entry point merges them into a single `trace.json` for a trace viewer such as Perfetto.

**DeepSeek V4 has no offline generation entry.** The only supported launch is the online HTTP server, so this skill always profiles `pypto-serving`. Unlike Qwen (single process, `--dp 1`), DeepSeek serving runs an **8-NPU overlapped** placement where the same eight physical ranks carry both attention DP=8 and MoE EP=8 — one model replica, not eight independent ones. DeepSeek also runs **MTP speculative decoding** (`--enable-mtp`), so decode spans produce multiple tokens per step and TPOT is not simply `mean(decode_fwd)`.

This skill measures **where time goes** at operator granularity. The canonical references are `docs/dev/model/deepseek-v4.md` (verified launch command) and `docs/dev/profile.md` (profiling semantics); re-read them if the env or topology below disagrees with the checkout.

Do not hard-code a specific commit, user directory, device id, port, model path, or one-off output path. Use the model dir, output path, devices, port, and workload the user provides, then print what actually ran.

---

## 1. Prerequisites

- A Conda environment with `pypto-serving` installed so the `pypto-serving` console script is on `PATH` (it calls `pypto_serving.cli.main:main`).
- The **quantized W8A8 compressed-tensors** checkpoint (e.g. `/data/models/dsv4-flash-w8a8`). The original checkpoint is too large for 8 NPUs; the CLI rejects it with `DeepSeekV4 serving requires the quantized W8A8 compressed-tensors checkpoint`.
- **Eight** available NPU devices. If the box gates device access through a queue wrapper such as `task-submit`, run the server inside that wrapper.
- A free TCP port (the docs example uses `8225`; common ports are taken by teammates — pick a free one).

## 2. Enable profiling

Profiling turns on when **either** `SA_PROFILE_OUTPUT` or `SA_PROFILE_LEVEL` is present. For operator timing use:

| Variable | Value for this skill |
| --- | --- |
| `SA_PROFILE_OUTPUT` | An **absolute** directory path, fresh per run. A new main process clears stale `trace.*.jsonl` in its `fragments/` dir, so reusing a path overwrites the prior run. |
| `SA_PROFILE_LEVEL` | `verbose` (all levels) or `e2e,kernel`. The `kernel` level is required — without it no `kernel.*_fwd` spans are recorded and there is no operator breakdown. |

Output layout for a directory output:

```text
<SA_PROFILE_OUTPUT>/
├── fragments/trace.<pid>.jsonl   # one JSONL fragment per process (API + workers)
└── trace.json                    # merged trace, written on graceful shutdown
```

Keep the `PTO2_*` runtime env vars the launch template uses (ring heap/pool/window, op-execute / stream-sync / scheduler timeouts, `SERVING_WORKER_STEP_TIMEOUT`). They are unrelated to profiling but required for the multi-rank NPU runtime on this box — DeepSeek needs much larger ring/timeout budgets than Qwen because of the 8-rank overlapped collective.

## 3. Start the server

DeepSeek serving enforces a fixed topology (`_validate_model_topology` in `cli/main.py`): the W8A8 checkpoint, `--dp 8 --ep 8 --tp 1`, exactly 8 device ids, `--block-size 128`, overlapped placement, and `--max-num-seqs` / `--max-model-len` within the pypto-lib decode CSA state limits. Mismatches fail fast with an explicit `ValueError`, so the command below is not a suggestion — every one of those flags is required.

Template (queue-wrapped; replace the 8 device ids, model path, served name, and port):

```bash
task-submit --device <d0>,<d1>,<d2>,<d3>,<d4>,<d5>,<d6>,<d7> \
  --max-time 0 --timeout 0 --ptoas 0.48 --run "\
SA_PROFILE_OUTPUT=/abs/path/dsv4-profile-out \
SA_PROFILE_LEVEL=verbose \
PYPTO_RUNTIME_LOG=error \
PTO2_RING_DEP_POOL=131072 PTO2_RING_TASK_WINDOW=131072 PTO2_RING_HEAP=2147483648 \
PTO2_OP_EXECUTE_TIMEOUT_US=400000000 PTO2_STREAM_SYNC_TIMEOUT_MS=440000 \
PTO2_SCHEDULER_TIMEOUT_MS=320000 SERVING_WORKER_STEP_TIMEOUT=1800 \
pypto-serving \
    --model /path/to/dsv4-flash-w8a8 \
    --served-model-name dsv4-flash-w8a8 \
    --backend npu --platform a2a3 \
    --devices <d0>,<d1>,<d2>,<d3>,<d4>,<d5>,<d6>,<d7> \
    --dp 8 --ep 8 --tp 1 \
    --block-size 128 \
    --max-model-len 512 \
    --max-num-seqs 32 \
    --max-num-batched-tokens 512 \
    --long-prefill-token-threshold 2048 \
    --enable-mtp --no-enable-prefix-caching \
    --port <port> --show-startup-logs"
```

Notes:

- `--enable-mtp` enables speculative decoding; `--no-enable-prefix-caching` is forced by the CLI for DeepSeek anyway. To profile **non-MTP** decode, drop `--enable-mtp` (this also raises the per-rank decode batch and the `--max-num-seqs` ceiling).
- `--max-model-len 512` matches the verified command. The real ceiling is `prefill_csa_state_max_blocks * c4_state_block_size` from `pypto_serving.model.deepseek.npu_runner.DeepSeekV4CacheLayout`; the CLI rejects anything larger until the pypto-lib decode CSA state table is enlarged.
- `--max-num-seqs 32` is the MTP ceiling (`ranks(8) × decode_batch(4)`). Without MTP it doubles.

Wait for `INFO: Application startup complete.` / `Uvicorn running on http://0.0.0.0:<port>` before sending traffic. The workers print `Worker entering busy loop` and the engine prints `Engine loop started` once the weights, KV/state pools, and kernels are ready.

## 4. Confirm ready, then drive workload

**Workload spec — `128/128/16`, capped at `--max-model-len 512`.** The config is `input/output/num-prompts` (token length and request count = concurrency). Concurrency 16 stays within `--max-num-seqs 32`; raise `-c` toward 32 to saturate the batch.

| Config | input | output | num-prompts (= concurrency) | Regime |
| --- | --- | --- | --- | --- |
| `128/128/16` | 128 | 128 | 16 | balanced (short prefill) |

Run this config. Use the user's values if they specify different lengths, keeping `input + output <= --max-model-len` (512) and concurrency `<= --max-num-seqs`.

Confirm the endpoint serves (a healthy `/v1/models` alone does not prove generation works):

```bash
PORT=<your-port>
curl -sf http://localhost:$PORT/health                                # {"status":"ok"}
curl -sf http://localhost:$PORT/v1/models | grep -o '"id":"[^"]*"'    # the served-model-name
curl -sf http://localhost:$PORT/v1/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"<served-model-name>\",\"prompt\":\"ping\",\"max_tokens\":4,\"temperature\":0}" >/dev/null && echo gen OK
```

Drive the workload with **`tests/bench_serving.py` (default — no install)**. The repo's own async benchmark drives the endpoint with aiohttp (already in the env) and prints TTFT, per-token decode interval, throughput (req/s, tok/s), and latency p50/p99. Run the config:

```bash
PORT=<your-port>
python tests/bench_serving.py --host localhost --port "$PORT" \
    --input-len 128 --max-tokens 128 -n 16 -c 16 --stream
```

`--stream` enables TTFT + per-token decode measurement. `-n` is the request count, `-c` the concurrency, `--input-len` builds a fixed-length synthetic prompt (approximate; the exact count shows in the server log and the SA_PROFILE `prefill_fwd` span).

Under MTP, `bench_serving.py`'s tok/s reflects **accepted** tokens (main + committed draft), which is the throughput the user perceives — it does not need MTP-aware correction.

## 5. Flush and merge fragments

The recorder writes fragments with default buffering and flushes on process close. To get a complete merged `trace.json`:

- **Preferred:** stop the server gracefully (`SIGTERM`/`SIGINT` to the `pypto-serving` process). The application-shutdown path waits for the workers and calls `merge_profile()`, producing `<SA_PROFILE_OUTPUT>/trace.json`.
- **If the server was killed ungracefully:** run `./scripts/merge_profile.sh <SA_PROFILE_OUTPUT>` (or `SA_PROFILE_OUTPUT=<dir> ./scripts/merge_profile.sh`). Stop all profiled processes first so buffered events are flushed.

Fragments are retained after merging, so aggregation also works directly on `fragments/trace.*.jsonl` without a merged file — useful for an interim read while the server is still running (some recent events may still be buffered).

## 6. Read operator timing from the trace

The trace uses the **same Chrome trace-event schema and the same kernel lane names** as the Qwen skill — `kernel.prefill_fwd` / `kernel.decode_fwd` are produced by `_kernel_trace_name()` in `pypto_serving/model/deepseek/npu_runner.py`, which maps any prefill L3 callable to `kernel.prefill_fwd` and any decode callable to `kernel.decode_fwd` so the lane name is stable across model/kernel changes. DeepSeek adds executor-level **prefill sub-phase** spans that Qwen does not have.

Each line in a fragment is one Chrome trace event. Duration events are `{"ph":"X","name":...,"cat":...,"ts":...,"dur":<us>,"pid":...,"tid":...,"args":{...}}`. To get operator timing, filter `ph=="X"`, group by `name`, and sum `dur` (microseconds); sort by total descending.

Categories and the names that matter most for DeepSeek:

| `cat` | Representative `name` | What it measures |
| --- | --- | --- |
| `kernel` | `kernel.prefill_fwd`, `kernel.decode_fwd`, `kernel.greedy_sample_fwd` | NPU kernel dispatch per prefill / decode-step / sampling. **The operator breakdown lives here.** |
| `kernel` | `<name>.worker_run` | The inner `worker.run` device dispatch (a sub-span of the above). |
| `executor` | `DeepSeekV4ModelRunner.prefill.prepare`, `...ensure_l3_shared_buffers`, `...prepare_inputs`, `...prepare_fwd_args`, `...l3_dispatch` | **DeepSeek-specific prefill decomposition** — host-side arg prep and the L3 dispatch that uploads/clears state pools. Use these to split a slow `prefill_fwd` into its real phases. |
| `executor` | `DeepSeekV4ModelRunner.decode.prepare` | Decode host prep. |
| `scheduler` | `scheduler.schedule`, `scheduler.wait_worker_output`, `scheduler.process_step_output` | Scheduling and the host wait for the device each step. |
| `worker` | `WorkerProcess.execute_step`, `WorkerProcess.batch_prefill`, `WorkerProcess.batch_decode` | Worker-side step/prefill/decode wrappers. |
| `request` | `http.completions`, `http.stream_completion` | End-to-end per-request latency. |

For kernel events, prefer the **device-side** time carried in `args`: `device_wall_us` (device run time) and `host_wall_us` (host-side wall time), added by the runner after each dispatch. Aggregating `args.device_wall_us` by kernel name gives the true device cost.

Compact aggregator (writes nothing into the repo; run from anywhere). It is identical to the Qwen skill's because the trace schema is shared:

```python
import glob, json, os
from collections import defaultdict
D="<SA_PROFILE_OUTPUT>"; tot=defaultdict(lambda:[0.0,0,"",0.0,0])  # dur,count,cat,dev_us,dev_n
def events():
    m=os.path.join(D,"trace.json")
    if os.path.isfile(m):
        for e in json.load(open(m)).get("traceEvents",[]): yield e
    for f in sorted(glob.glob(os.path.join(D,"fragments","trace.*.jsonl"))):
        for line in open(f):
            line=line.strip()
            if line:
                try: yield json.loads(line)
                except json.JSONDecodeError: pass
for e in events():
    if e.get("ph")!="X": continue
    r=tot[e.get("name","?")]; r[0]+=float(e.get("dur",0) or 0); r[1]+=1; r[2]=e.get("cat","")
    a=e.get("args") or {}
    if isinstance(a,dict) and a.get("device_wall_us") is not None:
        r[3]+=float(a["device_wall_us"]); r[4]+=1
for name,(d,n,c,dev,dn) in sorted(tot.items(),key=lambda kv:kv[1][0],reverse=True)[:25]:
    print(f"{d/1000:10.3f}ms  n={n:<5} {c:<10} {name}")
```

**Exclude startup spans** when analyzing the request period: `AsyncLLMEngine.start`, `WorkerProcess.init_device_and_model`, weight upload / `register_model`, `_ensure_l3_shared_buffers` during init, and any kernel-compile spans are one-time and dominate totals if included.

## 7. Interpreting the results

- `kernel.prefill_fwd` fires once per prefill; its mean ≈ per-prefill cost and dominates **TTFT**. If a prefill looks slow, break it open with the `DeepSeekV4ModelRunner.prefill.*` executor spans — `l3_dispatch` (state-pool upload / page clear) vs `prepare_fwd_args` (host arg staging) usually tells you whether the cost is device-side or host-side.
- `kernel.decode_fwd` fires **once per decode iteration** (batched across all in-flight rows). Under **MTP** each iteration can emit a main token plus a draft token, so its `count` ≈ decode steps, **not** output tokens. Read TPOT from `bench_serving.py`'s per-token interval, not from `mean(decode_fwd)` — `mean(decode_fwd)` is the cost per *step*, and accepted-draft tokens make effective per-token latency lower than that.
- Compare `args.device_wall_us` to the span `dur`: a large gap means host dispatch / scheduling / L3-arg prep overhead, not device compute.
- `scheduler.wait_worker_output` ≈ `kernel.decode_fwd` means the host is blocked on the device each step (little host idle). If `wait_worker_output` ≫ `decode_fwd`, the host is stalling elsewhere (arg staging, scheduling, or cross-rank sync).
- Decode throughput ≈ `accepted_tokens / total_decode_time`; the per-step ceiling is `1000 / mean(kernel.decode_fwd ms)` steps/s, multiplied by the accepted-tokens-per-step (1 or 2 under MTP).

## 8. Troubleshooting

**a) Startup raises a topology `ValueError`** (`DeepSeekV4 serving requires --dp 8 --ep 8 with --tp 1`, `requires exactly 8 NPU device ids`, `DeepSeekV4 kernels require --block-size 128`, or the W8A8 message). These are hard CLI checks in `_validate_model_topology`. Fix the flag rather than the check — every required flag is load-bearing for the 8-rank overlapped placement.

**b) `--max-model-len` larger than the CSA state limit is rejected.** The ceiling comes from `DeepSeekV4CacheLayout` in pypto-lib; the CLI message tells you to enlarge the decode CSA state table in pypto-lib. Stay at `512` unless you intentionally extend it.

**c) `--max-num-seqs` larger than `ranks × decode_batch` is rejected.** MTP caps this at 32 (8×4); without MTP it is higher. Lower `--max-num-seqs` or drop `--enable-mtp`.

**d) Warmup crashes with `AICore error 507018` / `bounded device drain failed`.** The known NPU device-drain flake during warmup. The card auto-resets; retry the same command.

**e) `[Errno 98] address already in use` on bind.** Pick another `--port` and point the workload at it.

**f) No `kernel.*_fwd` events, only `e2e`/scheduler spans.** `SA_PROFILE_LEVEL` does not include `kernel`. Set it to `verbose` (or `e2e,kernel`) and confirm it is exported **before** launching so the worker processes inherit it.

**g) `trace.json` is missing or much smaller than the fragments.** The server was killed ungracefully so the merge did not run. Stop all profiled processes, then run `./scripts/merge_profile.sh <SA_PROFILE_OUTPUT>`, or aggregate the fragments directly.

**h) Every request returns 422.** The `model` field does not match the served-model-name from `/v1/models`, or `prompt_tokens + max_tokens > --max-model-len`. A single isolated 422 under concurrency is usually transient — retry that request.

## 9. Checklist

1. Pick a fresh absolute `SA_PROFILE_OUTPUT` and set `SA_PROFILE_LEVEL=verbose` (must include `kernel`).
2. Start `pypto-serving` (queue-wrapped) for the W8A8 checkpoint with `--dp 8 --ep 8 --tp 1`, exactly 8 devices, `--block-size 128`, `--enable-mtp`, `--max-model-len 512`; wait for `Application startup complete`.
3. Confirm `/health`, the served-model-name from `/v1/models`, and that one `/v1/completions` returns a completion.
4. Drive the workload with `tests/bench_serving.py`: run the spec config `128/128/16` via `--input-len` / `--max-tokens` / `-n` / `-c` / `--stream`.
5. Stop the server gracefully (or run `scripts/merge_profile.sh`) to produce `trace.json`.
6. Aggregate `ph=X` spans by `name`; for kernel rows prefer `args.device_wall_us`; exclude the one-time startup spans. Use the `DeepSeekV4ModelRunner.prefill.*` spans to split a slow prefill into phases.
7. Report: per-kernel total/count/mean (prefill vs decode vs sample), the prefill sub-phase breakdown, TPOT from the bench (not `mean(decode_fwd)` under MTP), the device-vs-host gap, and the port/model/workload actually used.
