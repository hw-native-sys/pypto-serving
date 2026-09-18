# Phase F system validation

This directory is the K7-only validation driver for the external-Router fixed
1P1D system. F0 passed on 2026-09-18 as run
`phase-f-f0-pr237-20260918-03` against PR237 plus PD commit `6526b48`.
It extends the Phase E launchers with bounded multi-handoff
admission, request correlation, optional closed-page compute/transfer overlap,
metrics, correctness cases, fault recovery, and repeatable soak evidence.

Use a fresh stage, run id, generation, control incarnation, ports, journals,
and evidence directory for every run. Never reuse an arena after an UNKNOWN
transfer. A recovery run must prove the exact old launcher/runtime processes
dead before incrementing generation and control incarnation. The scripts never
reset an NPU and may stop only PIDs recorded by the current run.

`run_pd_k7_node.sh` and `run_router.sh` retain the Phase E launcher safety
checks. Router、P、D share one strict JSON document; its required product
surface is only the P/D HTTP endpoints, while run ID, node IDs, control/RoCE
addresses and observability root are explicit in this reproducible harness:

```json
{
  "runtime": {
    "prefill": [{"host": "192.169.0.173", "port": 8111}],
    "decode": [{"host": "192.169.0.85", "port": 8112}]
  }
}
```

On the control host, the complete fixed-pair matrix is one command. The
orchestrator writes the same config into both containers, starts D, P, and
Router independently, waits for all health gates, runs the exact-token matrix,
stops only the launcher PIDs recorded by this run, and copies five evidence
directories locally:

```bash
python tests/manual/pd/phase_f/run_ab_system.py \
  --run-id phase-f-functional-01 \
  --repo /workspace/pypto-serving \
  --run-root /workspace/phase-f-runs \
  --env-file /home/sj/git/env_all.sh \
  --local-evidence-dir /absolute/new/evidence/path \
  --concurrency 2 --repeat 1
```

Add `--keep-services` for an interactive acceptance run.  The option retains
Router/P/D only after the suite passes and verifies both the launcher and
service child PIDs are still alive; a failed run still performs exact-PID
cleanup so the next run does not inherit an ambiguous generation.  The suite
process itself always exits.  The retained Router remains available on the
configured `--router-port` until the recorded run-owned PIDs are explicitly
stopped.

`--repo` is the immutable code location used by both nodes; `--run-root` is a
separate evidence/control directory and must have a basename beginning with
`phase-f-`.  This keeps the current `/workspace` deployment independent from
the legacy `/home/sj/git/phase-f-*` staging trees.  The environment file is
explicit because its compatibility path may differ from the code location.

The launcher owns Mooncake's provider requirement
`HCCL_INTRA_ROCE_ENABLE=1`; it does not rely on a machine-wide environment
file to select the AscendDirect RoCE path. Version locks separately record the
PyPTO checkout, nested Simpler runtime, environment PTO-ISA checkout, and the
managed PTO-ISA checkout actually used by binary compilation.

Use `--profile` for the overlap evidence run. It enables the protected P/D
profile endpoints and archives each node's merged trace. Analyze the P trace
without overwriting an earlier audit:

```bash
python tests/manual/pd/phase_f/analyze_overlap.py \
  --trace-dir /absolute/prefill/profile \
  --output /absolute/new/overlap-audit.json
```

Run the bounded functional matrix after Router, P, and D report ready:

```bash
python tests/manual/pd/phase_f/run_suite.py \
  --router-url http://127.0.0.1:8110 \
  --prefill-url http://127.0.0.1:8111 \
  --decode-url http://192.169.0.85:8112 \
  --evidence-dir /absolute/new/evidence/path \
  --concurrency 2 --repeat 1
```

For a soak, increase `--repeat` only after the functional matrix passes. Archive
Router/P/D journals, launcher logs, version locks, process tables, and any
device timeline next to the generated manifest, responses, metrics, and audit.
