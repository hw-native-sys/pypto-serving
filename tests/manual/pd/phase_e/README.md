# Phase E external Router smoke

This directory is the reproducible K7-only entrypoint for the Phase E control-plane
inversion.  It keeps the Phase D Mooncake/owner data plane unchanged while running
three processes: an independently-ready P node, an independently-ready D node, and
the CPU-only public Router.

The node launcher refuses an occupied NPU process table, creates a new evidence
directory, and stops only children proven to belong to its own process session. It
never resets a device. P/D expose only `/health` and `/internal/pd/*`;
the Router is the only process exposing `/v1/*`.

The accepted topology is one P host and one D host, with 16 physical devices
per host, TP4/DP4/EP16, target-only P
Prefill, D-local K7 drafter initialization, and Mooncake AscendDirect D2D.

Launch D, then P, wait for both `/health` endpoints, and finally launch the Router.
Use a new run ID, evidence name, and shared PD config on every run.
Run `run_smoke.sh` for the 128-token baseline and
`run_multi_chunk_smoke.sh` for a prompt that crosses the configured 128-token
Prefill chunk boundary. Terminate only the exact PID recorded by each launcher, in
Router, D, P order.

## Required environment

The scripts expect the Phase D frozen dependencies plus the Phase E serving tree
under one stage. The verified 2026-09-13 stage was:

```text
/workspace/pd-serving-validation/
  env_pinned_stack.sh
  pypto-serving/
  pypto-serving/build_output -> frozen Phase D build output
  pypto -> frozen PyPTO
  ptoas -> frozen PTOAS
```

Choose a new shared config for every run and place the same file path/content in
both containers:

```json
{
  "runtime": {
    "run_id": "replace-with-new-run-id",
    "prefill": [{"host": "prefill.example.internal", "port": 8111, "node_id": "prefill-0", "control_port": 29931}],
    "decode": [{"host": "decode.example.internal", "port": 8112, "node_id": "decode-0", "control_port": 29931}]
  }
}
```

## Launch order

Run each node launcher in its own process session. The following values match the
verified A/B topology; ports and evidence names must be changed if already used.

On the decode host, launch D:

```bash
export PD_ROLE=decode
export PD_NODE_ID=decode-0
export PD_CONFIG=/absolute/path/pd.json
export PD_API_PORT=8112
export PD_EVIDENCE_NAME=replace-with-d-evidence-name
setsid bash /workspace/pd-serving-validation/pypto-serving/tests/manual/pd/phase_e/run_pd_k7_node.sh
```

On the prefill host, launch P independently:

```bash
export PD_ROLE=prefill
export PD_NODE_ID=prefill-0
export PD_CONFIG=/absolute/path/pd.json
export PD_API_PORT=8111
export PD_EVIDENCE_NAME=replace-with-p-evidence-name
setsid bash /workspace/pd-serving-validation/pypto-serving/tests/manual/pd/phase_e/run_pd_k7_node.sh
```

After both health endpoints return 200, launch the CPU-only Router on the control host:

```bash
export PD_CONFIG=/absolute/path/pd.json
export ROUTER_PORT=8110
export ROUTER_EVIDENCE_NAME=replace-with-router-evidence-name
setsid bash /workspace/pd-serving-validation/pypto-serving/tests/manual/pd/phase_e/run_router.sh
```

Submit both acceptance cases from the control container:

```bash
bash tests/manual/pd/phase_e/run_smoke.sh \
  http://127.0.0.1:8110 replace-with-baseline-evidence-dir
bash tests/manual/pd/phase_e/run_multi_chunk_smoke.sh \
  http://127.0.0.1:8110 replace-with-multichunk-evidence-dir
```

## Endpoint and shutdown checks

An external node must return 404 for `/v1/models`; the Router owns `/v1/*`.
Before shutdown, confirm the launcher
PID and command from its evidence directory. Send TERM in Router, D, P order and
wait for each launcher to finish its owned-session cleanup and write
`postflight-npu.log`. Do not reset an NPU and do not terminate a process selected
only by name. A run is clean only when both postflight NPU process tables contain
no task row from the run.
