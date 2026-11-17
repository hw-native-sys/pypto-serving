# Device Fault Triage Table

Symptom chains observed on this stack, ordered by how often they masquerade
as model-code bugs. Device logs live under `ASCEND_PROCESS_LOG_PATH` (set it
before launch — see the last row); the `device-N/device-<pid>_*.log` files
carry AICPU scheduler state and the `plog/plog-<pid>_*.log` files carry the
orchestrator view. Match log files to the run by PID, not by timestamp alone.

The canonical reference for device error codes (orchestration and scheduler
error codes, scheduler sub-classes, the ring/timeout knobs they map to) is
the simpler wiki:
https://github.com/hw-native-sys/simpler/wiki/Device-Error-Codes_zh —
consult it for any error code or knob not covered by the rows below.

| # | Symptom | Actual cause | Action |
| --- | --- | --- | --- |
| 1 | Run dies ~30 s after first dispatch. AICPU scheduler `sub_class=S1:running-stalled` on some chips, then `get_tensor_data FATAL(code=8) producer (ring=N) not completed` on others; host side shows 21008 / 507018 / `TENSOR_WAIT_TIMEOUT`. | First dispatch on a fresh world (especially 16-card) takes minutes of host build-graph warm-up; the default AICPU watchdogs kill it at ~30 s. Not a hang, not a deadlock. | Re-run with the timeout envs: `SIMPLER_OP_EXECUTE_TIMEOUT_US=400000000 SIMPLER_STREAM_SYNC_TIMEOUT_MS=440000 SIMPLER_SCHEDULER_TIMEOUT_MS=320000`. The `PTO2_*` spellings are dead names (retired when the runtime renamed its env prefix); only `SIMPLER_*` is read. |
| 2 | `EE9999 ... cross-server / ExportToShareableHandleV2` lines during HCCL bring-up in plog. | HCCL symmetric-memory init noise, present in healthy runs. | Ignore. Do not build a theory around these lines. |
| 3 | Prefill passes; the first dispatch carrying a new buffer SMMU-faults: `DDR address of the MTE instruction is out of range`, `isSmmuFault=1` in debug plog; peers then hit `orch_error_code=8 TENSOR_WAIT_TIMEOUT`. | A host-shared tensor was allocated *after* the L3 worker's lazy fork. `_shared_l3_worker()` forks at the first device-side call with `inherited_host_tensors` as a fork-time snapshot; `.share_memory_()` never retroactively maps into forked children. The buffer passes the shared-memory wire check but the children never map it. | Move the allocation into the pre-fork window: in the runner's `_ensure_l3_shared_buffers` keep the order — all `allocate_host_shared` / `shared_empty` staging first, then weight upload (`_materialize_resident_weights`), then device scratch. Never allocate host-shared buffers at dispatch time. `npu_runner.py` guards this with `_ensure_shared_host_allocation_before_worker`. |
| 4 | Generation returns all-zero token ids (`'!!!!!!!!'`, `finish_reason=length`); no fault in CANN logs. | One-off host-class first-dispatch flake; seen once ever. | Re-run once. If it recurs, then investigate. |
| 5 | Output text differs between runs of the same prompt/config. | Accepted kernel nondeterminism. | Assert the output contract, never exact text: a legal finish reason (`stop`/EOS, or `length` when `max_new_tokens` caps the run — the repo accuracy guards assert `finish_reason == "length"` for capped runs, see `tests/test_deepseek_v4_accuracy.py`), non-empty output, token accounting closes. Guards compare contracts, not golden samples. |
| 6 | Warmup dies with AICPU fault `ERR99999`; kernel wants more sync-start blocks than the local 910B3 dies expose (24 vs 20). | Historical stack-level limitation (pypto-lib issue #1384); resolved by the 2026-10-08 stack generation even though the block constant itself is unchanged. | If it reappears, suspect stack skew: re-verify env-stack pin agreement (see `env-stack.md`) before touching model code. |
| 7 | Executor cannot find kernel modules; `pypto-lib/` is empty. | Fresh worktree without submodules. | `git submodule update --init --recursive` in the worktree (each worktree initializes its own submodule checkout). |
| 8 | Every run recompiles kernels; persistent JIT cache never hits. | Cache persistence is fail-closed: dirty simpler runtime checkout or missing build evidence bypasses it. Also `PYPTO_PROG_BUILD_DIR` set explicitly forces output/rebuild. | Check `git -C <pypto>/runtime status --porcelain`; unset `PYPTO_PROG_BUILD_DIR` unless a named build dir is intended. Details in `env-stack.md`. |
| 9 | Device logs missing or mixed with other runs' under `~/ascend/log`. | Default log location is shared per host. | Set `ASCEND_PROCESS_LOG_PATH=<run-dir>/ascend-logs` before launch (e.g. alongside `task-submit --run`). `ASCEND_SLOG_DIR` does **not** relocate these files (tested); `ASCEND_GLOBAL_LOG_LEVEL` only changes verbosity. Verify files appear under the target dir before relying on it. |
